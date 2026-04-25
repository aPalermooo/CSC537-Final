from typing import Any

from numpy import ndarray
from ultralytics import YOLO


def detect_characters(plate: ndarray[tuple[Any, ...], Any], segmentation_model: YOLO) -> list[Any]:
    characters_loc = segmentation_model(plate)

    boxes = sorted(characters_loc[0].boxes, key=lambda b: b.xyxy[0][0])

    characters = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        character = plate[y1:y2, x1:x2]
        characters.append(character)
    return characters