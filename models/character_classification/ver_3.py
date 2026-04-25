from ultralytics import YOLO


def classify_characters(characters: list, classification_model: YOLO) -> list[str]:
    labels = []
    for character in characters:
        result = classification_model(character)
        predicted_class = result[0].probs.top1
        label = result[0].names[predicted_class]
        labels.append(label)
    return labels