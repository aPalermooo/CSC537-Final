import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


def calc_norm(path):
    mean = 0.0
    std = 0.0
    total_pixels = 0

    raw_transform = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    dataset = datasets.ImageFolder(path, transform=raw_transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    for images, _ in loader:
        pixels = images.numel()
        mean += images.sum()
        std += (images ** 2).sum()
        total_pixels += pixels

    mean /= total_pixels
    std = torch.sqrt(std / total_pixels - mean ** 2)

    return torch.tensor([mean]), torch.tensor([std])

def create_dataloader(dataset, shuffle = False, batch_size = 256, num_workers = 4):
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())