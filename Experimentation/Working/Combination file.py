"""
This file currently imports a grid image from the input files folder, then applys a grey scale transform to it and then 
a homography transform to the birds eye view. 

To view the photo you imported, uncomment line 49 to add plt.show(), then you can pick the points to insert into A,B,C,D

It outputs the file to the output files folder and the naming convention is at the bottom
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


# Import the image in regular view
originalImage = cv2.imread('Finalised Code/Input Files/grid_simulation.png', 1)

# Import the image, 0 converts it to greyscale
img_greyscale = cv2.imread('Finalised Code/Input Files/grid_simulation.png', 0)

# check to see if the images were imported correctly
if originalImage is None:
    print('original image not imported correctly')
    exit()
else:
    pass

if img_greyscale is None:
    print('Image not imported correctly for greyscaling')
    exit()
else:
    pass


# this is the function that will do the gamma transformation
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# set the gammaValue
gammaValue = 0.5

# creates the gamma adjusted image
gamma_adjusted_img = adjust_gamma(img_greyscale, gammaValue)


# Code for the homography transformations


# Create a copy of the image
img_copy = np.copy(gamma_adjusted_img)


# Convert to RGB so as to display via matplotlib
# Using Matplotlib we can easily find the coordinates
# of the 4 points that is essential for finding the
# transformation matrix
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
plt.imshow(img_copy)

# toggle this to display the matplotlib in order to find the coordinates
# plt.show()

# All points are in format[cols, rows]
pt_A = [42, 681]
pt_B = [1518, 666]
pt_C = [1347, 75]
pt_D = [266, 83]

# Here, I have used L2 norm. You can use L1 also.
width_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
width_DC = np.sqrt(((pt_D[0] - pt_C[0]) ** 2) + ((pt_D[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AB), int(width_DC))


height_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
height_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxHeight = max(int(height_AD), int(height_BC))

# define the input points in an array
input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])

# define the output points in an array
output_pts = np.float32([[0, maxHeight],
                        [maxWidth, maxHeight],
                        [maxWidth, 0],
                        [0, 0]])

# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts, output_pts)

# do the projective transformation
out = cv2.warpPerspective(
    img_copy, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

# code below is simply to save the output picture
# cv2.imwrite('corrected_grid.png', out)


# plotting images

cv2.imshow('originalImage', originalImage)


cv2.putText(gamma_adjusted_img, "g={}".format(gammaValue), (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
cv2.imshow('adjusted', np.hstack(
    [img_greyscale, gamma_adjusted_img]))


cv2.imshow('corrected_grid', out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# writing the images to an output
cv2.imwrite('Finalised Code/Output Files/corrected_grid.png', out)
# cv2.imwrite('transformed_single_line.png', out)
