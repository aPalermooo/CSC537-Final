import os

import torch
from torch import nn
from torchvision import models

# Locate dataset
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

train_dir = os.path.join(project_root, 'datasets', 'character_classification', 'data', 'train')

num_classes = 35

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            torch.nn.init.constant_(m.bias, 0)

# Construct Model
def build_model():
    model = models.mobilenet_v2(weights=None)
    model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.apply(initialize_weights)
    return model