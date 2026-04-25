import os

import torch
from torch.utils.data import random_split, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from compiler.compiler_helper.cnn_functions import calc_norm, create_dataloader
from compiler.compiler_helper.CharacterDataset import CharacterDataset

"""GLOBAL VARIABLES"""

# Get Dataset dir

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TARGET = os.path.join(project_root, 'datasets', 'character_classification', 'raw')

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
    train_loader = create_dataloader(train_set, shuffle=True)
    val_loader = create_dataloader(val_set)
    test_loader = create_dataloader(test_set)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    compile_data()