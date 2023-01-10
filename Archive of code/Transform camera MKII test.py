# This code works by finding the 4 corners of the grid using the pixels identified using matplotlib.
# Then it finds the relative distances between these points
# then it uses these distances to create the 4 corners of the image
# computes the homography matrix between the input and output points
# Applies the matrix to all the points and displays the result

# code from https://theailearner.com/tag/cv2-warpperspective/


import matplotlib.pyplot as plt
import cv2
import numpy as np

# To open matplotlib in interactive mode
# %matplotlib inline

# Import the image
img = cv2.imread('Experimentation/grid_simulation.png')
if img is None:
    print('Could not read image file')

# Create a copy of the image
img_copy = np.copy(img)


# Convert to RGB so as to display via matplotlib
# Using Matplotlib we can easily find the coordinates
# of the 4 points that is essential for finding the
# transformation matrix
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
plt.imshow(img_copy)

# toggle this to display the matplotlib in order to find the coordinates
plt.show()

# All points are in format[cols, rows]
x4LT = [574, 512]
x4LB = [503, 761]
x2LT = [690, 514]
x2LB = [646, 723]
origin = [805, 513]
x2RT = [923, 512]
x2RB = [937, 763]
x4RT = [1038, 511]
x4RB = [1081, 760]


# define the input points in an array
input_pts = np.float32([x4LT, x2LT, origin, x2RT, x4RT])

# Calculating output points
new_4Lx, new_4Ly = np.sqrt((x4LT[0] - x4LB[0]) ** 2)/8, x4LB
new_2Lx, new_2Ly = np.sqrt((x2LT[0] - x2LB[0]) ** 2)/8, x2LB
new_2Rx, new_2Ry = np.sqrt((x2RT[0] - x2RB[0]) ** 2)/8, x2RB
new_4Rx, new_4Ry = np.sqrt((x4RT[0] - x4RB[0]) ** 2)/8, x4RB

# new_2L = np.sqrt(((x2LT[0] - x2LB[0]) ** 2) + ((x2LT[1] - x2LB[1]) ** 2))/8
# new_4R = np.sqrt(((x4RT[0] - x4RB[0]) ** 2) + ((x4RT[1] - x4RB[1]) ** 2))/8
# new_2R = np.sqrt(((x2RT[0] - x2RB[0]) ** 2) + ((x2RT[1] - x2RB[1]) ** 2))/8

width = np.sqrt(((x4LT[0] - x4RT[0]) ** 2) + ((x4LT[1] - x4RT[1]) ** 2))/8
height_L = np.sqrt(((x4LT[0] - x4LB[0]) ** 2) + ((x4LT[1] - x4LB[1]) ** 2))/8
height_R = np.sqrt(((x4RT[0] - x4RB[0]) ** 2) + ((x4RT[1] - x4RB[1]) ** 2))/8

maxWidth = (int(width))
maxHeight = max(int(height_L), int(height_R))

# define the output points in an array
output_pts = np.float32([[new_4Lx, new_4Ly], [new_2Lx, new_2Ly], [origin[0], origin[1]],
                        [new_2Rx, new_2Ry], [new_4Rx, new_4Ry]])

# width_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
# width_DC = np.sqrt(((pt_D[0] - pt_C[0]) ** 2) + ((pt_D[1] - pt_C[1]) ** 2))
# maxWidth = max(int(width_AB), int(width_DC))


# height_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
# height_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
# maxHeight = max(int(height_AD), int(height_BC))

# Compute homography matrix
h, status = cv2.findHomography(input_pts, output_pts)
print(h)

# # do the projective transformation
# out = cv2.warpPerspective(
#     img, h, (1920, 1080), flags=cv2.INTER_LINEAR)

# Compute the perspective transform M
# M = cv2.getPerspectiveTransform(input_pts, output_pts)

# do the projective transformation
# out = cv2.warpPerspective(
#     img, h, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

# # code below is simply to display and save the output picture
# cv2.imshow('corrected_grid', out)
# cv2.imwrite('corrected_grid.png', out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
