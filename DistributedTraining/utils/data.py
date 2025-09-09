from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset

from ray.train.torch import prepare_data_loader

import ray

import os
import numpy as np
from PIL import Image
import torch


def animals_load(data_path, world_size, rank, batch_size):
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

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler
    )
    
    trainloader = prepare_data_loader(trainloader)
    testloader = prepare_data_loader(testloader)

    return trainloader, testloader


class SegmentationDataset(Dataset):
    def __init__(self, data_list, image_transform=None, mask_transform=None):
        self.data_list = data_list
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, mask_path = self.data_list[idx]
        
        # Load the image and apply transforms
        img = Image.open(img_path).convert("RGB")
        if self.image_transform:
            img = self.image_transform(img)

        # Load the mask and apply transforms
        mask = Image.open(mask_path).convert("L")
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return img, mask
    
def seg_human_load(data_path, world_size, rank, batch_size, num_workers=4):
    image_dir = os.path.join(data_path, "images")
    mask_dir = os.path.join(data_path, "masks")
    
    # Define transforms
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t > 0).float())
    ])

    # Build the list of file paths (this is still a fast, lightweight operation)
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    paired = []
    for img_name in image_files:
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            paired.append((os.path.join(image_dir, img_name), mask_path))

    # Create dataset instance
    full_dataset = SegmentationDataset(paired, image_transform, mask_transform)

    # Train/test split
    train_len = int(0.9 * len(full_dataset))
    test_len = len(full_dataset) - train_len
    train_dataset, test_dataset = random_split(full_dataset, [train_len, test_len])

    if rank == 0:
        print(f"Dataset size: {len(paired)}")
        print(f" - training {train_len}")
        print(f" - training {test_len}")

    # Distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # DataLoaders with parallel workers
    trainloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        num_workers=num_workers,
        pin_memory=True
    )
    testloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        sampler=test_sampler, 
        num_workers=num_workers,
        pin_memory=True
    )
    trainloader = prepare_data_loader(trainloader)
    testloader = prepare_data_loader(testloader)

    return trainloader, testloader