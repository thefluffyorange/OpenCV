####################################
#
#    Find contours of an image
#       and mask backgroung
#
#                by
#
#         Code Monkey King
#
####################################

# packages
import cv2
import numpy as np

# open source image file
image = cv2.imread(
    'Archive of images/Report/convex.png', 0)

# convert image to grayscale
image_gray = image

# onvert image to blck and white
thresh, image_edges = cv2.threshold(image_gray, 60, 255, cv2.THRESH_BINARY)

cv2.imshow('binary', image_edges)
cv2.waitKey()
cv2.destroyAllWindows()

# create canvas
canvas = np.zeros(image.shape, np.uint8)
canvas.fill(255)

# create background mask
mask = np.zeros(image.shape, np.uint8)
mask.fill(255)

# create new background
new_background = np.zeros(image.shape, np.uint8)
new_background.fill(255)

# get all contours
contours_draw, hierachy = cv2.findContours(
    image_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# get most significant contours
contours_mask, hierachy = cv2.findContours(
    image_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# draw all contours
#cv2.drawContours(canvas, contours_draw, 1, (0, 0, 0), 3)

# contours traversal
for contour in range(len(contours_draw)):
    # draw current contour
    cv2.drawContours(canvas, contours_draw, contour, (0, 0, 0), 3)
    # cv2.imshow('original', canvas)
    # cv2.waitKey()


# most significant contours traversal
for contour in range(len(contours_mask)):
    cv2.drawContours(mask, contours_mask, contour, (0, 0, 0), 3)

    # cv2.imshow('mask', mask)
    # cv2.waitKey()

    # cv2.fillConvexPoly(mask, contours_mask[contour], (0, 0, 0))
    # cv2.imshow('mask', mask)
    # cv2.waitKey()

    # create mask
    if contour == 5:
        cv2.fillConvexPoly(mask, contours_mask[contour], (0, 0, 0))

    # # create background
    # if contour != 1:
    #     cv2.fillConvexPoly(new_background, contours_mask[contour], (0, 0, 10))

# display the image in a window
cv2.imshow('Original', image)
cv2.imshow('Contours', canvas)
cv2.imshow('Background mask', mask)
cv2.imshow('New background', new_background)
cv2.imshow('Output', cv2.bitwise_and(image, new_background))

# write images
cv2.imwrite('contours.png', canvas)
cv2.imwrite('mask.png', mask)
cv2.imwrite('background.png', new_background)
cv2.imwrite('output.png', cv2.bitwise_and(image, new_background))

# escape condition
cv2.waitKey(0)

# clean up windows
cv2.destroyAllWindows()
