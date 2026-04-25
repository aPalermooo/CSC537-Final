"""
Haar Cascade edge detection
v
YOLO8 Detection
v
YOLO8 Classification
"""
import os
import random

import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO

from models.plate_segmentation.ver_2 import detect_plate
from models.character_segmentation.ver_2.model import detect_characters

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

segmentation_model_path = os.path.join(project_root, 'results', 'character_segmentation', 'training_loop_v1', 'yolov8_v1-6', 'weights', 'best.pt')
classification_model_path = os.path.join(project_root, 'results', 'character_classification', 'training_loop_v5', 'yolov8_cls_v1-2', 'weights', 'best.pt')

def load_segmentation_model():
    return YOLO(segmentation_model_path)

def load_classification_model():
    return YOLO(classification_model_path)

def classify_characters(characters, classification_model):
    labels = []
    for character in characters:
        result = classification_model(character)
        predicted_class = result[0].probs.top1
        label = result[0].names[predicted_class]
        labels.append(label)
    return labels

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

    if len(characters) == 0:
        print("No characters readable", end='\t')
        return None

    # Character Classification
    classification_model = load_classification_model()
    labels = classify_characters(characters, classification_model)

    plt.imshow(plate, cmap='gray')
    plt.show()
    plt.imshow(image)
    plt.show()

    print(f"Characters found: {len(characters)}")
    return ''.join(labels)


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