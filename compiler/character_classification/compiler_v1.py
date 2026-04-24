import os
import pickle
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2

"""GLOBAL VARIABLES"""

BATCH_SIZE = 64
NUM_WORKERS = 4

# Get Dataset dir
path = os.getcwd()
path = os.path.abspath(os.path.join(path, '..'))
project_root = os.path.abspath(os.path.join(path, '..'))

TARGET = os.path.join(project_root, 'datasets', 'character_classification', 'raw')
DUMP = os.path.join(project_root, 'datasets', 'character_classification', 'pickled')

""""""

""" DATASET CLASS"""

class CharacterDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.subset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.subset)


""""""

"""FUNCTIONS"""

def calc_norm(path):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0

    raw_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    raw_dataset = datasets.ImageFolder(path, transform=raw_transform)
    loader = DataLoader(raw_dataset, batch_size=64, shuffle=False)

    ## Accumulate running totals
    for images, _ in loader:
        batch_size = images.size(0)

        images = images.view(batch_size, 3, -1)

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += batch_size

    ## Avg.
    mean /= n_samples
    std /= n_samples

    return mean, std

def create_dataloader(dataset, shuffle = False):
    return DataLoader(dataset, shuffle=shuffle, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

""""""

"""MAIN"""

def compile_data():


    data_mean, data_std = calc_norm(TARGET)


    # Random Augmentation of Images
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomRotation(degrees=15),
        v2.RandomPerspective(distortion_scale=0.5,),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=data_mean.tolist(), std=data_std.tolist()),
    ])

    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=data_mean.tolist(), std=data_std.tolist()),
    ])


    # Compile datasets
    full_dataset = datasets.ImageFolder(TARGET)

    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

    # Apply transformations
    train_set = CharacterDataset(train_set, train_transform)
    val_set = CharacterDataset(val_set, test_transform)
    test_set = CharacterDataset(test_set, test_transform)

    # Assemble Dataloaders
    training_loader = create_dataloader(train_set, shuffle=True)
    val_loader = create_dataloader(val_set)
    test_loader = create_dataloader(test_set)

    return training_loader, val_loader, test_loader


if __name__ == '__main__':
    compile_data()