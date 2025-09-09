import torch.nn.functional as F
import torch

import os, tempfile


import ray.train.torch
from ray.train.torch import TorchTrainer
from torch.utils.data.distributed import DistributedSampler


from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch.distributed as dist


import time

from models import *


def run_train_cpu(epochs:int=30, batch_size:int=8, lr:float=0.0001):
    model = HybridModel()
    path = "~/Desktop/animals"

    dataset = datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],  # standard ImageNet means/stds
                                 std  = [0.229, 0.224, 0.225]),
        ])
    )

    train_dataset, test_dataset = random_split(dataset, [int(0.9*len(dataset)), int(0.1*len(dataset))])


    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )


    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )


    print("Classes:", dataset.classes)
    print("Total samples:", len(dataset))
    

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # batch size
        for images, labels in trainloader:    

            optimizer.zero_grad()

            preds = model(images)

            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs},\tLoss: {loss.item():.4f}")

        # test
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in testloader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"\nAccuracy: {100 * correct / total:.2f}%\n")

        model.train()


def run_train_ray(config):
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 16)
    lr = config.get("lr", 0.001)
    data_path = os.path.expanduser(config.get("data_path", "~/animals"))

    # move model to respective gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context = ray.train.get_context()
    world_size = context.get_world_size()
    rank = context.get_world_rank()

    model = HybridModel().to(device)

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True  # PyTorch DDP argument
        )
    else:
        model = ray.train.torch.prepare_model(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # sharded dataset to gpu
    dataset = datasets.ImageFolder(
        data_path,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],  # standard ImageNet means/stds
                                 std  = [0.229, 0.224, 0.225]),
        ])
    )

    train_dataset, test_dataset = random_split(dataset, [int(0.9*len(dataset)), int(0.1*len(dataset))])

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler
    )
    
    trainloader = ray.train.torch.prepare_data_loader(trainloader)
    testloader = ray.train.torch.prepare_data_loader(testloader)

    

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
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                model.module.state_dict(),
                os.path.join(temp_checkpoint_dir, "model.pt")
            )
            ray.train.report(
                metrics,
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )
        
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
        
        

        model.train()


if __name__ == "__main__":

    start = time.time()

    ray.init("ray://localhost:10001", 
            runtime_env={
                "working_dir": os.path.dirname(__file__)
            }
        )  # Change URI to your cluster address

    scaling_config = ray.train.ScalingConfig(num_workers=4, use_gpu=True)
    run_config = ray.train.RunConfig(storage_path="~/", name="test_run_CNN")

    train_config = {
        "epochs": 30,
        "batch_size": 8,
        "lr": 0.0001,
        "data_path": "~/animals"
    }

    trainer = TorchTrainer(
        train_loop_per_worker=run_train_ray,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config
    )
    result = trainer.fit()

    print("time to train with ray:", time.time() - start)

    ray.shutdown()


    start = time.time()
    run_train_cpu()
    print("time to train with cpu:", time.time() - start)



