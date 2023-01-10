import cv2

# OpenCV basics

# cv2 can open jpg and png and possibly svg files
# by default cv2 is going to load your image in BGR colour pallete
# all we do is need to tell python where the image is located and format we want it in
# we can choose between greyscale, regular coloured image, without considering transparency

# Importing the image
# this is loading the image
img = cv2.imread('Tutorials/Assets/image.jpeg', 0)

"""
cv2.IMREAD_COLOUR (-1) loads a colour image, any transparency will be neglected
cv2.IMREAD_GREYSCALE (0) loads an image in grey scale mode
cv2.IMREAD_UNCHANGED (1) loads an image as such including alpha channel (if it has transparency it will honour that)
"""

# Displaying the image
img = cv2.resize(img, (400, 400))  # this resizes the image to 400 by 400
# allows for rotation of the image
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


cv2.imshow('image', img)  # displays the image to the output window
cv2.imwrite('new_img.jpg', img)  # writes the image to a new file


cv2.waitKey(10000)
# this will wait an amount of time specified (or if a key is pressed then skip to the next line,
# if we put 0 it waits until key is pressed
cv2.destroyAllWindows()  # destroys all the windows so they don't keep building up
