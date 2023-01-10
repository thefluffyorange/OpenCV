import cv2
import random

img = cv2.imread('Tutorials/Assets/image.jpeg')

# when an image is loaded it imports the pixels and imputs them into a numpy array
# the term shape tells me the number of rows, number of columns, and the number of channels (colour space BGR)
print(img.shape)

# accessing pixel values
# this indexes the first row then the pixels from the 45th to 100th place in the first row
print(img[0][45:100])

print(img[0][100])  # looks at the first row 100th place


# modifying an image
# the code below works via the first loop picking the 1st row then goes through all the coloums and assigns random BGR values
for i in range(int(img.shape[0]/4)):  # loops from 0 to 100
    for j in range(img.shape[1]):  # here we are looking at the columns (width)
        img[i][j] = [random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255)]  # 3 random values for BGR
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
