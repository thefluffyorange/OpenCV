import cv2
import numpy as np

# image = cv2.imread("grid_simulation_gamma_converted1.png")
image = cv2.imread('Experimentation/reference_photo.png')

# Print error message if image is null
if image is None:
    print('Could not read image')

# Apply identity kernel
kernel1 = np.array([[1, 0, -1],
                    [1, 0, -1],
                   [1, 0, -1]])

"""
The kernel will iterate across the image 
"""

# Apply identity kernel
kernel2 = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])

# identity1 = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)
identity2 = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)
# identity3 = cv2.filter2D(src=identity1, ddepth=-1, kernel=kernel2)

cv2.imshow('Original', image)
# cv2.imshow('Identity1', identity1)
cv2.imshow('Identity2', identity2)
# cv2.imshow('Identity3', identity3)

cv2.waitKey(0)
# cv2.imwrite("Max's Code/identity1.jpg", identity1)
cv2.imwrite("Max's Code/identity2.jpg", identity2)
# cv2.imwrite("Max's Code/identity3.jpg", identity3)
cv2.destroyAllWindows()


"""
import cv2
import numpy as np

img = cv2.imread('images/input.jpg')
rows, cols = img.shape[:2]

src_points = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
dst_points = np.float32([[0,0], [cols-1,0], [int(0.33*cols),rows-1], [int(0.66*cols),rows-1]]) 
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
img_output = cv2.warpPerspective(img, projective_matrix, (cols,rows))

cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey()
"""
