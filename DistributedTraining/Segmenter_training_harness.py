
import os
import tempfile
import time

import ray.train.torch
from ray.train.torch import TorchTrainer

import torch
import torch.distributed as dist

if __name__ == '__main__':
    from models import ViTModel, CNNModel, HybridModel, ClassSegmentationModel, compare_models
    from utils.data import *
    from utils.loss import *
else:
    from DistributedTraining.models import ViTModel, CNNModel, HybridModel, ClassSegmentationModel, compare_models
    from DistributedTraining.utils.data import *
    from DistributedTraining.utils.loss import *


def run_train_ray(config):
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 16)
    lr = config.get("lr", 0.001)
    data_load_func = config.get("data_load_func", animals_load)
    data_path = os.path.expanduser(config.get("data_path", "~/animals"))
    model_name = config.get("model_name", "ViT")

    # Get model
    if model_name == "ViT":
        model = ViTModel()
        criterion = torch.nn.CrossEntropyLoss()
    elif model_name == "CNN":
        model = CNNModel()
        criterion = torch.nn.CrossEntropyLoss()
    elif model_name == "hybrid":
        model = HybridModel()
        criterion = torch.nn.CrossEntropyLoss()
    elif model_name == "seg":
        cfg = config.get("model_config", {})
        model = ClassSegmentationModel(cfg)
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Model {model_name} not recognized. Choose from ['ViT', 'CNN', 'hybrid', 'seg'].")
    
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

    if rank == 0:
        print(f"Using model: {model_name}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # sharded dataset to gpu
    trainloader, testloader = data_load_func(data_path, world_size, rank, batch_size)
    if rank == 0:
        print(f"Data loaded")

    losses = []
    accuracies = []


    # training loop
    for epoch in range(epochs):
        if ray.train.get_context().get_world_size() > 1:
            trainloader.sampler.set_epoch(epoch)

        # decaying learning rate
        if epoch % 10 == 0 and epoch > 0:
            lr = lr * .95
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # batched loop
        for images, labels in trainloader:  

            images, labels = images.to(device), labels.to(device)  

            optimizer.zero_grad()

            preds = model(images)
            
            loss = criterion(preds.float(), labels)
            if rank == -1:
                print(preds)
                print(labels)
                print('\n')

            loss.backward()

            optimizer.step()

        # metrics
        metrics = {"loss": loss.item(), "epoch": epoch}
        ray.train.report(
            metrics,
        )
        
        # test
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = (torch.sigmoid(outputs) > 0.5).long()    
                
                correct += (1 - (labels - predicted).abs()).sum()
                total += labels.numel()

            correct_tensor = torch.tensor(correct).to(device)
            total_tensor = torch.tensor(total).to(device)
            
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

        
    if rank == 0:
        save_path = os.path.expanduser("~/final_model.pt")

        torch.save(
            model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            save_path
        )

        ray.train.report(metrics)
    else:
        ray.train.report(metrics)


def run_training(train_config):
    seg_cfg = train_config.get("model_config", {})
    compare_models(seg_cfg)

    start = time.time()

    ray.init("ray://localhost:10001",
            runtime_env={
                "working_dir": os.path.dirname(__file__)
            }
        ) 

    scaling_config = ray.train.ScalingConfig(num_workers=4, use_gpu=True)
    run_config = ray.train.RunConfig(
        storage_path="~/", 
        name="training_harness_run",
        checkpoint_config=ray.train.CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min"
        )
    )


    trainer = TorchTrainer(
        train_loop_per_worker=run_train_ray,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config
    )
    result = trainer.fit()
    

    print("time to train with ray:", time.time() - start)

    #ray.shutdown()
    
    return result


if __name__ == "__main__":

    model_config = {
        'input_shape':(3, 256, 256),
        'patch_size':4,
        'num_classes':2,
        'num_dims':256, 
        'num_heads':8, 
        'num_layers':3
    }

    train_config = {
        "epochs": 100,
        "batch_size": 16,
        "lr": 0.0001,
        "data_path": "~/human_seg/data",
        "data_load_func": seg_human_load,
        "model_name": "seg",
        "model_config": model_config
    }

    result = run_training(train_config)