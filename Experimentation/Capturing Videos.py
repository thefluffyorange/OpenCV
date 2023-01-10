import numpy as np
import cv2

# the number tells the programme which camera you want to access
cap = cv2.VideoCapture(0)

# so what this code below is doing is accessing the camera then every millisecond it is grabbing the information from the camera and displaying it on the screen

while True:  # loops infinitely
    # returns the frame which is the image itself(numpy array), ret tells you if it worked properly/accessed webcam
    ret, frame = cap.read()

    # we are creating an array of zeros here, but we want it to be the same shape (no. rows, no. columns, channels) as the camera therefore we just use frame.shape
    image = np.zeros(frame.shape)

    cv2.imshow('frame', frame)

    # this will wait one millisecond, but if we press a key within that time then it will return the ordinal value (askey value)
    # we are checking to see if the key we pressed was actually q
    if cv2.waitKey(1) == ord('q'):
        break

# allows the camera to be released so another programme can use it
cap.release()
cv2.destroyAllWindows()
