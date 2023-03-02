import cv2
import numpy as np

img = cv2.imread("Max's Code/Binary image.png", 0)

contours, hierarchy = cv2.findContours(
    img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

canvas = np.zeros(img.shape, np.uint8)
canvas = cv2.drawContours(canvas, contours, -1, (0, 0, 0), 3)

cv2.imshow('Canvas', canvas)
cv2.waitKey()
cv2.destroyAllWindows()
