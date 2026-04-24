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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TARGET = os.path.join(project_root, 'datasets', 'character_classification', 'raw')
DUMP = os.path.join(project_root, 'datasets', 'character_classification', 'pickled')

def compile_data() -> None:
    test_loader = None

    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=data_mean.tolist(), std=data_std.tolist()),
    ])



    return test_loader