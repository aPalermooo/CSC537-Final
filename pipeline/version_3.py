
"""
Haar Cascade edge detection
v
YOLO8 Detection
v
Mobile Net
"""
import os
import random
from string import digits, ascii_uppercase
from typing import Any

import cv2
import torch
from PIL import Image
from matplotlib import pyplot as plt
from numpy import ndarray
from torchvision import models
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor
from ultralytics import YOLO

from models.plate_segmentation.ver_2 import detect_plate
from models.character_segmentation.ver_2.model import detect_characters

"""
Notes

"""


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

dataset_path = os.path.join(project_root, 'datasets', 'character_classification', 'raw')

num_classes = len([d for d in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, d))])

classification_model_path = os.path.join(project_root, 'results', 'character_classification', 'training_loop_v2', 'checkpoint', 'checkpoint_80.pth')
segmentation_model_path = os.path.join(project_root, 'results', 'character_segmentation', 'training_loop_v1', 'yolov8_v1-6', 'weights', 'best.pt')

chars = digits + ascii_uppercase

char2idx = {c :i for i ,c in enumerate(chars)}
idx2char = {i :c for i ,c in enumerate(chars)}

inference_transform = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5])
])

def load_segmentation_model():
    return YOLO(segmentation_model_path)

def load_classification_model():
    model = models.mobilenet_v2(weights=None)
    model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    checkpoint = torch.load(classification_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def main(img_path):

    image = cv2.imread(img_path)

    # Plate segmentation
    plate = detect_plate(image)

    if plate is None:
        print("No plate detected", end='\t')
        return None

    # Character Segmentation
    segmentation_model = load_segmentation_model()
    characters = detect_characters(plate, segmentation_model)

    for character in characters:
        plt.imshow(character)
        plt.show()

    # Character Classification
    representation = []

    for index in range(len(characters)):
        characters[index] = Image.fromarray(characters[index]).convert('L')

    classification_model = load_classification_model()

    for character in characters:
        input_tensor = inference_transform(character)
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = classification_model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            representation.append(predicted_class)


    if len(representation) == 0:
        print("No characters readable", end='\t')
        return None
    else:
        print(len(characters))
        plt.imshow(plate, cmap='gray')
        plt.show()
        plt.imshow(image)
        plt.show()

    return ''.join([idx2char[rep] for rep in representation])


if __name__ == "__main__":
    test_dir = os.path.join("..", "datasets", "plate_segmentation", "raw", "train")
    images = os.listdir(test_dir)
    random.shuffle(images)

    for test_image in images:
        test_image = os.path.join(test_dir, test_image)
        output = main(test_image)
        print(output)
        if output is not None:
            break
