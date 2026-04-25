#####################################
#   name:   ver_1.py
#   desc:   segment license plate using openCV and general edge detection
#   author: Xander Palermo
#   Date:   April 2026
#
#   Class:  CSC537 - Deep Learning
#####################################
import os
from random import choice

import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

"""
Citations:
Adapted from GitHub user pragatiunna

https://github.com/pragatiunna/License-Plate-Number-Detection/blob/main/1.%20License%20Plate%20Detection%20(using%20Contours).ipynb
"""

"""
Notes:

Model is too simple; 
- Detects areas that are not of interest, such as grills and car logos

"""

def detect_plate(image):
    """
    Identifies license plate using a pretrained Haar Cascade model and openCV

    Args:
        image: Image being processed

    Returns:
        Cropped image over region of interest
        If no license plate is found, return None

    """
    # Convert color channels
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.bilateralFilter(img_gray, 11, 17, 17)
    img_edge = cv2.Canny(img_gray, 170, 200)
    # plt.imshow(img_edge, cmap='gray')
    # plt.show()        # Show Original Image

    # Find Contours
    contours = cv2.findContours(img_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    # loop over contours to find the best possible approximate contour

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:                # 4 corners in contour
            x, y, w, h = cv2.boundingRect(contour)

            ROI = img_rgb[y:y+h, x:x+w]

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