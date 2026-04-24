import torch
import os
from torchvision import models
from torchvision.models import ResNet18_Weights

# Locate dataset
path = os.getcwd()
path = os.path.abspath(os.path.join(path, '..'))
project_root = os.path.abspath(os.path.join(path, '..'))

dataset_path = os.path.join(project_root, 'datasets', 'character_classification', 'raw')

num_classes = len([d for d in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, d))])

# Construct Model
def build_model():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model