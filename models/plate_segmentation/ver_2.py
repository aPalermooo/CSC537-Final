#####################################
#   name:   ver_2.py
#   desc:   segment license plate using openCV and pretrained haar cascade edge detection
#   author: Xander Palermo
#   Date:   April 2026
#
#   Class:  CSC537 - Deep Learning
#####################################

import os
from random import choice

import cv2
import matplotlib.pyplot as plt


"""
Notes:

Model doesnt adapt to different lighting conditions
Often selects an area too big around the license plate

"""

"""
Citations:
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_russian_plate_number.xml

pretrained by openCV on russian license plates

https://www.geeksforgeeks.org/python/detect-an-object-with-opencv-python/

Model primarily based off of this tutorial
"""
CASCADE_PATH = os.path.join(os.path.dirname(__file__), "pretrained_models", "haarcascade_russian_plate_number.xml")

def detect_plate(image: cv2.Mat):
    """
    Identifies license plate using a pretrained Haar Cascade model and openCV

    Args:
        image: Image being processed

    Returns:
        Cropped image over region of interest
        If no license plate is found, return None

    """
    # Load image

    plate_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    def identify_plate(img, n):
        return plate_cascade.detectMultiScale(
            img,
            scaleFactor=1.2,
            minNeighbors=n,
            minSize=(60, 20)
        )

    # Convert color channels
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    img_gray = cv2.bilateralFilter(img_gray, 11, 17, 17)

    # Pass to Classifier

    num_neighbors = 5

    found = identify_plate(img_gray, n=num_neighbors)

    if len(found) < 1:
        img_gray = cv2.convertScaleAbs(img_gray, alpha=3, beta=50)
        found = identify_plate(img_gray, n=num_neighbors)
    while len(found) > 1 and num_neighbors < 50:
        num_neighbors += 1
        found = identify_plate(img_gray, n=num_neighbors)
    if len(found) > 1:
        found = [max(found, key=lambda b: b[2] * b[3])]

    for (x, y, w, h) in found:
        ROI = img_rgb[y:y + h, x:x + w]
        # plt.imshow(ROI)
        # plt.show()
        return ROI
    return None

if __name__ == "__main__":
    """Static Testing of function"""
    for _ in range(10):
        test_dir = os.path.join("..", "..", "datasets", "plate_segmentation", "raw", "test")
        test_image = choice(os.listdir(test_dir))   # Pull random image from testing group
        test_image = cv2.imread(os.path.join(test_dir, test_image))
        if detect_plate(test_image) is None:
            print(None)
