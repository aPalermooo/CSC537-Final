
"""Constants"""
import os
import shutil

import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2

"""

Notes:
Carves out dataloaders from directories instead of random partitions

"""

"""GLOBAL VARIABLES"""

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TARGET = os.path.join(PROJECT_ROOT, 'datasets', 'character_classification', 'raw')

OUTPUT = os.path.join(PROJECT_ROOT, 'datasets', 'character_classification', 'output')
IMAGES_DIR_TRAIN = os.path.join(OUTPUT, 'train')
IMAGES_DIR_VAL = os.path.join(OUTPUT, 'val')

"""Mean/STD"""
def get_stats():
    raw_dataset = torchvision.datasets.ImageFolder(root=IMAGES_DIR_TRAIN, transform=transforms.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((64, 64)),
        transforms.ToTensor(),
    ]))

    l = torch.utils.data.DataLoader(raw_dataset, batch_size=64, shuffle=True, num_workers=2)

    all_pixels = []

    for imgs, _ in l:
        all_pixels.append(imgs.view(-1))  # flatten everything into 1D

    all_pixels = torch.cat(all_pixels)

    mean = all_pixels.mean().item()
    std  = all_pixels.std().item()

    return [mean], [std]



"""DataLoader Compilation"""

def compile_data():
    """
    Compiles Datasets from directories to dataloader objects

    Returns:
        2 Dataset objects, one for training and one for validation
    """

    data_mean, data_std = get_stats()

    train_transform = transforms.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((64, 64)),
        v2.RandomRotation(degrees=15),
        v2.RandomPerspective(distortion_scale=0.3),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=data_mean, std=data_std),
    ])

    val_transform = transforms.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((64, 64)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=data_mean, std=data_std),
    ])

    train_set   = torchvision.datasets.ImageFolder(root=IMAGES_DIR_TRAIN, transform=train_transform)
    val_set     = torchvision.datasets.ImageFolder(root=IMAGES_DIR_VAL, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader


"""Main Function"""

def main():
    """
    Walks raw dataset directory and creates directories spilt between training and validation
    """
    with os.scandir(TARGET) as entries:
        classes = [entry.name for entry in entries if entry.is_dir()]

    for c in classes:
        if not os.path.isdir(os.path.join(IMAGES_DIR_TRAIN, c)):
            os.makedirs(os.path.join(IMAGES_DIR_TRAIN, c))
        if not os.path.isdir(os.path.join(IMAGES_DIR_VAL, c)):
            os.makedirs(os.path.join(IMAGES_DIR_VAL, c))

        source_dir = os.path.join(TARGET, c)
        c_train_dir = os.path.join(IMAGES_DIR_TRAIN, c)
        c_val_dir = os.path.join(IMAGES_DIR_VAL, c)

        imgs = [img for img in os.listdir(source_dir)]

        MAX_TRAIN = 1000
        MAX_VAL = 1000

        train_size = min(int(len(imgs) * 0.8), MAX_TRAIN)
        val_size = min(len(imgs) - train_size, MAX_VAL)

        print(f"Class: {c}")
        print(f"\tTrain size: {train_size}")
        print(f"\tVal size: {val_size}")

        for img in imgs[:train_size]:
            img_name = os.path.basename(img)
            shutil.copyfile(os.path.join(source_dir, img), os.path.join(IMAGES_DIR_TRAIN, c, img_name))

        for img in imgs[train_size:train_size + val_size]:
            img_name = os.path.basename(img)
            shutil.copyfile(os.path.join(source_dir, img), os.path.join(IMAGES_DIR_VAL, c, img_name))
    return

if __name__ == "__main__":

    """Create File Structure"""

    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)

    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
    if not os.path.exists(IMAGES_DIR_TRAIN):
        os.makedirs(IMAGES_DIR_TRAIN)
    if not os.path.exists(IMAGES_DIR_VAL):
        os.makedirs(IMAGES_DIR_VAL)

    main()

