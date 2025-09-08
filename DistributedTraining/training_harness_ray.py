
import os
import time

import ray.train.torch
from ray.train.torch import TorchTrainer

import torch
import torch.distributed as dist

from models import ViTModel, CNNModel, HybridModel, ClassSegmentationModel, compare_models
from utils.data import *
from utils.loss import *


def run_train_ray(config):
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 16)
    lr = config.get("lr", 0.001)
    data_load_func = config.get("data_load_func", animals_load)
    criterion = config.get("loss_func", torch.nn.CrossEntropyLoss())

    model_name = config.get("model_name", "ViT")
    if model_name == "ViT":
        model = ViTModel()
    elif model_name == "CNN":
        model = CNNModel()
    elif model_name == "hybrid":
        model = HybridModel()
    elif model == "seg":
        model = ClassSegmentationModel()
    else:
        raise ValueError(f"Model {model_name} not recognized. Choose from ['ViT', 'CNN', 'hybrid', 'seg'].")

    data_path = os.path.expanduser(config.get("data_path", "~/animals"))

    # move model to respective gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context = ray.train.get_context()
    world_size = context.get_world_size()
    rank = context.get_world_rank()

    model = model.to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True  # PyTorch DDP argument
        )
    else:
        model = ray.train.torch.prepare_model(model)


    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # sharded dataset to gpu
    trainloader, testloader = data_load_func(data_path, world_size, rank, batch_size)

    losses = []
    accuracies = []

    for epoch in range(epochs):
        if ray.train.get_context().get_world_size() > 1:
            trainloader.sampler.set_epoch(epoch)


        # batch size
        for images, labels in trainloader:  
            images, labels = images.to(device), labels.to(device)  

            optimizer.zero_grad()

            preds = model(images)

            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()


        # metrics
        metrics = {"loss": loss.item(), "epoch": epoch}
        # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        #     torch.save(
        #         model.module.state_dict(),
        #         os.path.join(temp_checkpoint_dir, "model.pt")
        #     )
        #     ray.train.report(
        #         metrics,
        #         checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
        #     )
        ray.train.report(metrics)
        
        # test
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Aggregate the results from all workers
            correct_tensor = torch.tensor(correct).to(device)
            total_tensor = torch.tensor(total).to(device)
            
            # Check if the world size is greater than 1 before calling all_reduce
            if world_size > 1:
                dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)


        if rank == 0:
            total_accuracy = 100 * correct_tensor.item() / total_tensor.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {total_accuracy:.2f}%")
            accuracies.append(total_accuracy)
            losses.append(loss.item())

        model.train()
    
    if rank == 0:
        print(f"\nFinal Training Losses: {losses}")
        print(f"Final Training Accuracies: {accuracies}")

if __name__ == "__main__":
    compare_models()

    start = time.time()

    ray.init("ray://localhost:10001",
             runtime_env={
            "working_dir": os.path.dirname(__file__)
        }
        )  # Change URI to your cluster address

    scaling_config = ray.train.ScalingConfig(num_workers=4, use_gpu=True)
    run_config = ray.train.RunConfig(storage_path="~/", name="training_harness_run")

    train_config = {
        "epochs": 30,
        "batch_size": 16,
        "lr": 0.0001,
        "data_path": "~/animals",
        "data_load_func": animals_load,
        "loss_func": torch.nn.CrossEntropyLoss(),
        "model_name": "hybrid"
    }

    trainer = TorchTrainer(
        train_loop_per_worker=run_train_ray,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config
    )
    result = trainer.fit()

    print("time to train with ray:", time.time() - start)

