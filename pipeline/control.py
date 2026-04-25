import os
from random import choice

import cv2
import pytesseract
from matplotlib import pyplot as plt

"""
Notes:

Segmentation is weak, so the model fails to consistently find license plates

"""


"""
Citations:

https://www.geeksforgeeks.org/machine-learning/license-plate-recognition-with-opencv-and-tesseract-ocr/

complete pipeline
"""


pytesseract.pytesseract.tesseract_cmd = r"C:\Users\apale\Documents\NeuralNetwork\env\Library\bin\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Users\apale\Documents\NeuralNetwork\env\Library\share\tessdata"

def detect_plate_number(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 100, 200)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    plate_contour = None
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is not None:
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_image = gray[y:y + h, x:x + w]

        ROI = image[y:y + h, x:x + w]
        plt.imshow(ROI)
        plt.show()

        _, thresh = cv2.threshold(plate_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        plate_number = pytesseract.image_to_string(thresh, config='--psm 6')  # Treat it as a single word
        return plate_number.strip()
    return None


if __name__ == "__main__":
    test_dir = os.path.join( "..", "datasets", "plate_segmentation", "raw", "test")
    test_image = choice(os.listdir(test_dir))  # Pull random image from testing group
    # test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    plate_num = detect_plate_number(os.path.join(test_dir, test_image))
    print(plate_num)