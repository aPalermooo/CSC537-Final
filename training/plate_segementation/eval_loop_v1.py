import cv2
import matplotlib.pyplot as plt

from compiler.plate_segmentation.compiler_v1 import compile_data
from models.plate_segmentation.ver_1 import detect_plate

def tensor_to_mat(tensor):
    tensor = tensor.cpu().permute(1, 2, 0).numpy()
    return cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)

def evaluation():

    print("Compile Data...")
    training_data, validation_data, testing_data = compile_data()

    img, label = next(iter(training_data))

    detect_plate(tensor_to_mat(img[0]))



if __name__ == "__main__":
    evaluation()