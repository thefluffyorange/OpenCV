# This code works by finding the 4 corners of the grid using the pixels identified using matplotlib.
# Then it finds the relative distances between these points
# then it uses these distances to create the 4 corners of the image
# computes the homography matrix between the input and output points
# Applies the matrix to all the points and displays the result

import matplotlib.pyplot as plt
import cv2
import numpy as np

# To open matplotlib in interactive mode
# %matplotlib inline

# Import the image
img = cv2.imread('warped_grid.png')

# Create a copy of the image
img_copy = np.copy(img)

# Convert to RGB so as to display via matplotlib
# Using Matplotlib we can easily find the coordinates
# of the 4 points that is essential for finding the
# transformation matrix
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
plt.imshow(img_copy)
plt.show()

# toggle this to display the matplotlib in order to find the coordinates
# plt.show()

# All points are in format[cols, rows]
pt_A = [104, 422]
pt_B = [451, 422]
pt_C = [493, 104]
pt_D = [62, 104]

# Here, I have used L2 norm. You can use L1 also.
width_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
width_DC = np.sqrt(((pt_D[0] - pt_C[0]) ** 2) + ((pt_D[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AB), int(width_DC))

print("max wisth is :")
print(maxWidth)


height_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
height_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxHeight = max(int(height_AD), int(height_BC))
print("max height is :")
print(maxHeight)
# define the input points in an array
input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])

# define the output points in an array
output_pts = np.float32([[0, maxHeight],
                        [maxWidth, maxHeight],
                        [maxWidth, 0],
                        [0, 0]])

# output_pts = np.float32([[0, 0],
#                         [maxWidth, 0],
#                         [maxWidth, maxHeight],
#                         [0, maxHeight]])


# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts, output_pts)

# do the projective transformation
out = cv2.warpPerspective(
    img, M, (1024, 1024), flags=cv2.INTER_LINEAR)

# code below is simply to display and save the output picture
cv2.imshow('corrected_grid', out)
cv2.imwrite('corrected_grid.png', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
