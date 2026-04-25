

"""
Haar Cascade edge detection
v
OpenCV Contour Detection
v
ResNet
"""
import os
import random

import cv2
import torch
from matplotlib import pyplot as plt
from torchvision import models

from models.character_segmentation.ver_1 import detect_characters
from models.plate_segmentation.ver_2 import detect_plate

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

dataset_path = os.path.join(project_root, 'datasets', 'character_classification', 'raw')

num_classes = len([d for d in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, d))])

model_path = os.path.join(project_root, 'results', 'character_classification', 'training_loop_v1', 'checkpoint', 'checkpoint_30.pth' )

def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def main(img_path):

    image = cv2.imread(img_path)

    plate = detect_plate(image)

    if plate is None:
        print("No plate detected", end='\t')
        return None

    characters = detect_characters(plate)

    if len(characters) == 0:
        print("No characters detected", end='\t')
        return None

    representation = []

    model = load_model()

    for character in characters:
        input_tensor = torch.tensor(character, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        input_tensor = input_tensor.unsqueeze(0).repeat(1, 3, 1, 1)  # (1, 3, H, W)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            representation.append(predicted_class)


    if len(representation) == 0:
        print("No characters readable", end='\t')
        return None
    else:
        print(len(characters))
        plt.imshow(plate, cmap='gray')
        plt.show()

    return representation

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
