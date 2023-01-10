import numpy as np
import cv2

# video capture source camera (Here webcam of laptop)
# cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
ret, frame = cap2.read()  # return a single frame in variable `frame`

while (True):
    cv2.imshow('img1', frame)  # display the captured image
    if cv2.waitKey(1) & 0xFF == ord('y'):  # save on pressing 'y'
        cv2.imwrite('Experimentation/photo.png', frame)
        cv2.destroyAllWindows()
        break

# cap1.release()
cap2.release()
