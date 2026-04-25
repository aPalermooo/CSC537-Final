import os

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import v2

"""
Notes:

Uses precompiled dataset from github tutorial

"""


"""GLOBAL VARIABLES"""

# Get Dataset dir

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TARGET = os.path.join(project_root, 'datasets', 'character_classification', 'data')

""""""

"""MAIN"""

def compile_data():

    data_mean = torch.tensor([0.5])
    data_std = torch.tensor([0.5])

    train_transform = transforms.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((224, 224)),
        v2.RandomRotation(degrees=15),
        v2.RandomPerspective(distortion_scale=0.3),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=data_mean.tolist(), std=data_std.tolist()),
    ])

    val_transform = transforms.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((224, 224)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=data_mean.tolist(), std=data_std.tolist()),
    ])

    train_dir = torchvision.datasets.ImageFolder(root=os.path.join(TARGET, 'train'), transform=train_transform)
    val_dir = torchvision.datasets.ImageFolder(root=os.path.join(TARGET, 'val'), transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=4, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dir, batch_size=4, shuffle=True, num_workers=2)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = compile_data()
