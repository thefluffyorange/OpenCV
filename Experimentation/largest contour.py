import cv2
import numpy as np

# Read image
img = cv2.imread('Archive of images/Report/4.png')
hh, ww = img.shape[:2]

# threshold on black
# Define lower and uppper limits of what we call "white-ish"
lower = np.array([0, 0, 0])
upper = np.array([0, 0, 0])

# Create mask to only select black
thresh = cv2.inRange(img, lower, upper)

# invert mask so shapes are white on black background
thresh_inv = 255 - thresh

# get the largest contour
contours = cv2.findContours(
    thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv2.contourArea)

# draw white contour on black background as mask
mask = np.zeros((hh, ww), dtype=np.uint8)
cv2.drawContours(mask, [big_contour], 0, (255, 255, 255), cv2.FILLED)

# invert mask so shapes are white on black background
mask_inv = 255 - mask

# create new (blue) background
bckgnd = np.full_like(img, (255, 255, 255))

# apply mask to image
image_masked = cv2.bitwise_and(img, img, mask=mask)

# apply inverse mask to background
bckgnd_masked = cv2.bitwise_and(bckgnd, bckgnd, mask=mask_inv)

# add together
result = cv2.add(image_masked, bckgnd_masked)

# save results
cv2.imwrite('shapes_inverted_mask.jpg', mask_inv)
cv2.imwrite('shapes_masked.jpg', image_masked)
cv2.imwrite('shapes_bckgrnd_masked.jpg', bckgnd_masked)
cv2.imwrite('shapes_result.jpg', result)

cv2.imshow('mask', mask)
cv2.imshow('image_masked', image_masked)
cv2.imshow('bckgrnd_masked', bckgnd_masked)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
