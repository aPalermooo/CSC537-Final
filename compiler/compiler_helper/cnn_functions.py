import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


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

def create_dataloader(dataset, shuffle = False, batch_size = 256, num_workers = 4):
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())