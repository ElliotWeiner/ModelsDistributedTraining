from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from ray.train.torch import prepare_data_loader


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


def seg_load():
    pass