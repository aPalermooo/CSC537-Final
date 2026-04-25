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
import numpy as np
import matplotlib.pyplot as plt


"""
Citations:

https://github.com/pragatiunna/License-Plate-Number-Detection/blob/main/1.%20License%20Plate%20Detection%20(using%20Contours).ipynb?short_path=00c6cfb

Code directly taken from.
"""

# Match contours to license plate or character template
def find_contours(dimensions, img):
    # Find all contours in the image
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or  15 contours for license plate or character respectively
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    x_cntr_list = []
    img_res = []
    for cntr in contours:
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        # checking the dimensions of the contour to filter out the characters by contour's size
        if lower_width < intWidth < upper_width and lower_height < intHeight < upper_height:
            x_cntr_list.append(
                intX)  # stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44, 24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)  # List that stores the character's binary image (unsorted)

    # Return characters on ascending order with respect to the x-coordinate (most-left character first)

    plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])  # stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

def detect_characters(image):
    # Preprocess cropped license plate image
    img = cv2.resize(image, (333, 75))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    img_binary = cv2.erode(img_binary, kernel)
    img_binary = cv2.dilate(img_binary, kernel)

    HEIGHT = img_binary.shape[0]
    WIDTH = img_binary.shape[1]

    # Make borders white
    img_binary[0:3,:] = 255
    img_binary[:,0:3] = 255
    img_binary[HEIGHT - 3:HEIGHT, :] = 255
    img_binary[:, WIDTH - 3:WIDTH] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [WIDTH/6,
                       WIDTH/2,
                       HEIGHT/10,
                       2*HEIGHT/3]

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary)

    return char_list

if __name__ == "__main__":
    for _ in range(10):
        test_dir = os.path.join("..", "..", "datasets", "character_segmentation", "raw", "images")
        test_image = choice(os.listdir(test_dir))  # Pull random image from testing group
        test_image = cv2.imread(os.path.join(test_dir, test_image))
        # test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        if len(detect_characters(test_image)) == 0:
            print(None)