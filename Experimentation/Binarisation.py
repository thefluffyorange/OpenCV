"""
This is a basic set of coding that applies a simple binarisation to the grid image. It applies different types of binarisation to the image and plots them as an array of plots using matplotlib
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
# img = cv.imread('Archive of images/rock.png', 0)
img = cv2.imread("Max's Code/identity3.jpg", 0)
ret, thresh1 = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
""" ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
titles = ['Original Image', 'BINARY',
          'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show() """
cv2.imshow('Binary', thresh1)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("Archive of images/Report/Binary image.png", thresh1)

""" contours, hierarchy = cv2.findContours(
    thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

canvas = np.zeros(thresh1.shape, np.uint8)
canvas.fill(255)

canvas2 = np.zeros(thresh1.shape, np.uint8)
canvas2.fill(255)

canvas = cv2.drawContours(canvas, contours, -1, (0, 0, 0), 3)

cv2.imshow('Canvas', canvas)
cv2.waitKey()
cv2.destroyAllWindows() """
