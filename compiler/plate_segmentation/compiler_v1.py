import os

import torch
from torchvision import datasets, transforms
from torchvision.transforms import v2

from compiler.compiler_helper.LicenseDataset import LicenseDataset
from compiler.compiler_helper.cnn_functions import calc_norm, create_dataloader

"""GLOBAL VARIABLES"""

# Get Dataset dir
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TARGET = os.path.join(project_root, 'datasets', 'plate_segmentation', 'raw')

""""""

"""MAIN FUNCTION"""

def compile_data():

    # data_mean, data_std = calc_norm(TARGET)


    # random Augmentation
    #
    #  Unneeded for nontrainable models
    # train_transform = transforms.Compose([
    #     v2.ToImage(),
    #     v2.RandomRotation(degrees=15),
    #     v2.RandomPerspective(distortion_scale=0.5, ),
    #     v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize(mean=data_mean.tolist(), std=data_std.tolist()),
    # ])

    # test_transform = v2.Compose([
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize(mean=data_mean.tolist(), std=data_std.tolist()),
    # ])

    # Compile datasets
    train_dataset = LicenseDataset(os.path.join(TARGET, 'train'))
    val_dataset = LicenseDataset(os.path.join(TARGET, 'valid'))
    test_dataset = LicenseDataset(os.path.join(TARGET, 'test'))

    # Assemble Dataloaders

    train_loader = create_dataloader(train_dataset, shuffle=True, batch_size=1)
    val_loader = create_dataloader(val_dataset)
    test_loader = create_dataloader(test_dataset)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    compile_data()