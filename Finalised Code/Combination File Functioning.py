"""
This is the combination file for the steps from taking the photo, into greyscale, gamma transform, then homogrophy transform to then be ready for edge detection
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
Defining the functions used within this code.
"""


def take_photo(takePhoto, viewImage, device=0):
    """
    This function will check the paired camera is functioning correctly then take a photo.

    Parameters
    -------------------
    takePhoto: set to True to take the photo, false to not
    viewImage: set to True to view the photo, set to False to save the image but not view it
    device: the number representing the camera device you want to use (the default is 0)
    """
    if takePhoto == True:
        cap = cv2.VideoCapture(device)  # opens the device to capture the video
        if cap is None or not cap.isOpened():
            # checks if the device was accessed correctly
            print('Warning: unable to open video source: ', device)

        ramp_frames = 30
        for i in range(ramp_frames):
            temp = cap.read()
        ret, frame = cap.read()  # takes a photo

        if viewImage == True:
            # displays the captured image
            cv2.imshow('Image from camera', frame)
            cv2.waitKey(0)  # waits for a key press
            # writes the photo to the output location
            cv2.imwrite(outputImageLocation, frame)
            cv2.destroyAllWindows()  # destroys the image window
        if viewImage == False:
            # writes the photo to the output location
            cv2.imwrite(outputImageLocation, frame)
        else:
            print("The variable viewImage has not been defined correctly")
        cap.release()  # releases the camera to allow it to be utilised in other applications
    if takePhoto == False:
        pass
    else:
        print("Please enter true of false to take the photo")


def check_empty_img(url):

    # will check if the image location provided in url,
    # will provide a working image and is read correctly

    image = cv2.imread(url)

    if image is None:
        print('Image not imported correctly')
        exit()
    else:
        pass


def import_image(showImage, url="Finalised Code/Output Files/Camera_Image.png"):
    """
    Imports the desired camera image. Leave it blank to import from the camera file.
    Or you can specify a custom location.

    It also runs a check to ensure the image has been imported correctly.

    The import is then transformed into 3 distinct images (details in the output)

    Parameters
    ----------
    url: specifiy the location of the intended image
    showImage: set to 1 to get an output of the grey scale image in order to adjust the gamma transform, otherwise pick 0

    Outputs
    ----------
    original_image: the imported image unchanged
    grey_scale: the imported image converted to grey scale, with the correct grey scale applied
    RGB_Image: an RGB version of the photo for analysis in matplotlib
    """

    if url is None:
        print('Image not imported correctly')
        exit()
    else:
        pass

    original_image = cv2.imread(url, 1)
    grey_scale = cv2.imread(url, 0)
    RGB_Image = cv2.cvtColor(grey_scale, cv2.COLOR_BGR2RGB)

    if showImage == 1:
        cv2.imshow('Grey Scale', grey_scale)
        cv2.waitKey()
        cv2.imwrite('Finalised Code/Output Files/Grey_Scale.png', grey_scale)
        cv2.destroyAllWindows()
    if showImage == 0:
        cv2.imwrite('Finalised Code/Output Files/Grey_Scale.png', grey_scale)
    else:
        pass

    return original_image, grey_scale, RGB_Image


def adjust_gamma(image, gamma=1.0):
    # this is the function that will do the gamma transformation
    invGamma = 1.0/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# Code for the homography transformations

def homography_transformation(original_image, input_pts):
    """
    Expects the original, untransformed image and a list of 4 coordinates [A], [B], [C], [D] which it then unpacks and
    finds the maximum height between the top and bottom of the grid, as well as
    the maximum width between the left and right sides. Once these are calculated
    they become the outer coordinates for the new plot. Then the projective
    transform matrix is calculated and returned.
    """
    pt_A, pt_B, pt_C, pt_D = input_pts

    width_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    width_DC = np.sqrt(((pt_D[0] - pt_C[0]) ** 2) + ((pt_D[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AB), int(width_DC))

    height_AD = np.sqrt(((pt_A[0] - pt_C[0]) ** 2) +
                        ((pt_A[1] - pt_C[1]) ** 2))
    height_BC = np.sqrt(((pt_B[0] - pt_D[0]) ** 2) +
                        ((pt_B[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AD), int(height_BC))

    output_pts = np.float32([[0, maxHeight],
                             [maxWidth, maxHeight],
                             [maxWidth, 0],
                             [0, 0]])

    input = np.float32([pt_A, pt_B, pt_C, pt_D])

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input, output_pts)

    # do the projective transformation
    out = cv2.warpPerspective(
        original_image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    return out


if __name__ == '__main__':

    """
    This section is the image input and preliminary checks along with converting the image into
    greyscale for later and RGB to use in matplotlib
    """
    outputImageLocation = "Archive of images/Report/2.png"  # set the output location of the camera

    # utilises the camera to take and save a photo, set the value to true to see the camera, false to not
    take_photo(True, False)

    # imports the image and creates 3 distinct versions, set to 1 or 0 depending on if you want to view greyscale image for gamma adjustment
    original, greyscale, RGB_Image = import_image(
        0, "Archive of images/Report/2.png")
    gammaValue = 1.5  # set the gamma value based on the above grey scale image

    """
    This section opens the image using matplotlib, so the you can select the 4 corners of the grid
    and modify the gamma value to make it brighter or darker.
    """
    plt.imshow(RGB_Image)
    # plt.show()  # toggle this to display the matplotlib in order to find the coordinates

    # All points are in format[cols, rows]
    bottom_left = [700, 665]
    bottom_right = [1118, 662]
    top_right = [1145, 189]
    top_left = [695, 188]

    """
    This section now takes the inputs from above and applies a gamma correction and as well as the homography transform
    """

    input_points = [bottom_left, bottom_right, top_right, top_left]

    # creates the gamma adjusted image
    gamma_adjusted_original = adjust_gamma(original, gammaValue)
    gamma_adjusted_grey = adjust_gamma(greyscale, gammaValue)

    # out = homography_transformation(gamma_adjusted_img, input_points)
    corrected_original = homography_transformation(original, input_points)
    corrected_greyscale_gamma = homography_transformation(
        gamma_adjusted_grey, input_points)

    """
    Below is outputting the images for user viewing as well as saving them to the folder
    """
    """
    cv2.putText(gamma_adjusted_grey, "g={}".format(gammaValue), (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)  # adds some formatting to the gamma modified image
    cv2.imshow('adjusted', np.vstack(
        [greyscale, gamma_adjusted_grey]))  # displays the gamma transformed image
        """

    # displays the corrected grid from a birds eye view
    # cv2.imshow('corrected_grid', corrected_greyscale_gamma)
    # waits for an input from the keyboard to signify user has finished analysing the images
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  # destroys all of the plots ready for a new runthrough

    # outputs gamma corrected image (original colours)
    cv2.imwrite('Archive of images/Report/gamma original.png',
                gamma_adjusted_original)
    # outputs gamma corrected image (original colours)
    cv2.imwrite('Archive of images/Report/gamma greyscale.png',
                gamma_adjusted_grey)
    cv2.imwrite('Archive of images/Report/original.png',
                original)  # outputs original image
    cv2.imwrite('Archive of images/Report/greyscale.png',
                greyscale)  # outputs greyscale image
    cv2.imwrite('Archive of images/Report/corrected original.png',
                corrected_original)  # outputs the transformed grid in original colours
    cv2.imwrite('Archive of images/Report/corrected greyscale gamma.png',
                corrected_greyscale_gamma)  # outputs the transformed grid in greyscale with gamma correction
