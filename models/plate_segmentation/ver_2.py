#####################################
#   name:   ver_1.py
#   desc:   segment license plate using openCV and general edge detection
#   author: Xander Palermo
#   Date:   April 2026
#
#   Class:  CSC537 - Deep Learning
#####################################

import os
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

"""
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_russian_plate_number.xml

pretrained by openCV on russian license plates
"""
CASCADE_PATH = os.path.join("pretrained_models", "haarcascade_russian_plate_number.xml")

def detect_plate(img_path) -> None:
    """
    Identifies license plate using a pretrained Haar Cascade model and openCV

    :param img_path: path to image being processed
    """
    #Load image
    image = cv2.imread(img_path)

    # Convert color channels
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Pass to Classifier
    plate_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    found = plate_cascade.detectMultiScale(img_gray, minSize=(30, 30))

    for (x,y,w,h) in found:
        cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,255,0),2)

    plt.imshow(img_rgb)
    plt.show()



if __name__ == "__main__":
    """Static Testing of function"""
    detect_plate(r"C:\Users\apale\PycharmProjects\CSC537Final\datasets\plate_segmentation\raw\test\test\00d9db3d2c186504_jpg.rf.5a493e083834aa4b4748f09a073cc200.jpg")
    detect_plate(r"C:\Users\apale\PycharmProjects\CSC537Final\datasets\plate_segmentation\raw\test\test\02a3ba4c3886fe9a_jpg.rf.3e8359561ec92f1bbbe9f7ca5454eb44.jpg")
