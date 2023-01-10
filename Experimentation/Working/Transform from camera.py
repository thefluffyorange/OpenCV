# This code works by finding the 4 corners of the grid using the pixels identified using matplotlib.
# Then it finds the relative distances between these points
# then it uses these distances to create the 4 corners of the image
# computes the homography matrix between the input and output points
# Applies the matrix to all the points and displays the result

# code from https://theailearner.com/tag/cv2-warpperspective/


import matplotlib.pyplot as plt
import cv2
import numpy as np

# Code for the homography transformations

# Import the image, 0 converts it to greyscale
img = cv2.imread('Experimentation/grid_simulation.png', 0)


# Create a copy of the image
img_copy = np.copy(img)


# Convert to RGB so as to display via matplotlib
# Using Matplotlib we can easily find the coordinates
# of the 4 points that is essential for finding the
# transformation matrix
# img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
plt.imshow(img_copy)

# toggle this to display the matplotlib in order to find the coordinates
plt.show()

# All points are in format[cols, rows]
pt_A = [502, 448]
pt_B = [1033, 444]
pt_C = [1003, 148]
pt_D = [561, 151]

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
    img, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

# code below is simply to display and save the output picture
cv2.imshow('corrected_grid', out)
cv2.imwrite('corrected_grid.png', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
