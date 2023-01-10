# This is some code from the internet that allows you to take a reference image then convert it to another plane of view
# code from https://learnopencv.com/homography-examples-using-opencv-python-c/

import cv2
import numpy as np

# Read source image.
im_src = cv2.imread('rock.png')
# Four corners of the book in source image
pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])

# Read destination image.
im_dst = cv2.imread('rock.png')
# Four corners of the book in destination image.
pts_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

# Display images
cv2.imshow("Source Image", im_src)
cv2.imshow("Destination Image", im_dst)
cv2.imshow("Warped Source Image", im_out)

cv2.waitKey(0)
cv2.destroyAllWindows()
