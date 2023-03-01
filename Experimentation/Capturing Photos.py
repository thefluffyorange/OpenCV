import numpy as np
import cv2

""" # video capture source camera (Here webcam of laptop)
# remeber to set the camera as the primary device before utilising this code
cap2 = cv2.VideoCapture(0)
ret, frame = cap2.read()  # return a single frame in variable `frame`

while (True):
    cv2.imshow('img1', frame)  # display the captured image
    if cv2.waitKey(1) & 0xFF == ord('y'):  # save on pressing 'y'
        cv2.imwrite('Experimentation/photo.png', frame)
        cv2.destroyAllWindows()
        break

cap2.release()
 """


def take_photo(input):
    """
    This function will utilise the paired camera to take a photo, if the variable viewImage is set as 1 then the image will be shown. If it is set as 0 then the image will be stored but not shown.
    """
    cap1 = cv2.VideoCapture(0)
    ramp_frames = 30
    for i in range(ramp_frames):
        temp = cap1.read()

    ret, frame = cap1.read()

    if input == 1:
        # displays the captured image
        #cv2.imshow('Image from camera', frame)
        # cv2.waitKey(0)
        cv2.imwrite("Archive of images/Report/1.png", frame)
        cv2.destroyAllWindows()
    if input == 0:
        cv2.imwrite("Finalised Code/Output Files/Camera Image.png", frame)
        cv2.destroyAllWindows()
    else:
        print("The variable viewImage has not been defined correctly")
        cap1.release()


if __name__ == '__main__':
    # viewImage = 1
    take_photo(1)
