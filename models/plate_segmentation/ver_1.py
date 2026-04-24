#####################################
#   name:   ver_1.py
#   desc:   segment license plate using openCV and general edge detection
#   author: Xander Palermo
#   Date:   April 2026
#
#   Class:  CSC537 - Deep Learning
#####################################

import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt


"""
Notes:

Cannot detect 

"""

"""
Adapted from GitHub user pragatiunna

https://github.com/pragatiunna/License-Plate-Number-Detection/blob/main/1.%20License%20Plate%20Detection%20(using%20Contours).ipynb
"""
def detect_plate(img_path) -> None:
    """
    Identifies license plate using contours identified by openCV

    :param img_path: path to image being processed
    """
    #Load image
    image = cv2.imread(img_path)

    # Convert color channels
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.bilateralFilter(img_gray, 11, 17, 17)
    img_edge = cv2.Canny(img_gray, 170, 200)
    plt.imshow(img_edge, cmap='gray')
    plt.show()        # Show Original Image

    # Find Contours
    cnts = cv2.findContours(img_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlatesCnt = None

    # loop over contours to find the best possible approximate contour
    count = 0
    ROI = None

    img_area = np.prod(img_gray.shape)

    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:                # 4 corners in contour
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            if area > img_area * .8:
                continue
            if area < img_area * .01:
                continue

            NumberPlatesCnt = approx        # Approximate contour
            ROI = img_rgb[y:y+h, x:x+w]
            break

    if NumberPlatesCnt is not None:
        # Draw on original image
        cv2.drawContours(image, [NumberPlatesCnt], -1, (0, 255, 0), 3)

    # D-bug show images
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    #
    # plt.imshow(ROI)
    # plt.show()



if __name__ == "__main__":
    """Static Testing of function"""
    detect_plate(r"C:\Users\apale\PycharmProjects\CSC537Final\datasets\plate_segmentation\raw\test\test\00d9db3d2c186504_jpg.rf.5a493e083834aa4b4748f09a073cc200.jpg")
    detect_plate(r"C:\Users\apale\PycharmProjects\CSC537Final\datasets\plate_segmentation\raw\test\test\02a3ba4c3886fe9a_jpg.rf.3e8359561ec92f1bbbe9f7ca5454eb44.jpg")