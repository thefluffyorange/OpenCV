"""
This file currently imports a grid image from the input files folder, then applys a grey scale transform to it and then 
a homography transform to the birds eye view. 

To view the photo you imported, uncomment line 49 to add plt.show(), then you can pick the points to insert into A,B,C,D

It outputs the file to the output files folder and the naming convention is at the bottom
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def check_empty_img(url):

    # will check if the image location provided in url,
    # will provide a working image and is read correctly

    image = cv2.imread(url)

    if image is None:
        print('Image not imported correctly')
        exit()
    else:
        pass


def import_image(url):
    originalImage = cv2.imread(url, 1)
    grey_scale = cv2.imread(url, 0)

    images = originalImage, grey_scale

    return images


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

    height_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) +
                        ((pt_A[1] - pt_D[1]) ** 2))
    height_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) +
                        ((pt_B[1] - pt_C[1]) ** 2))
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

    # enter the location of the file/image you want to import
    img_location = 'Archive of images/Test.png'

    # check to see if the image has loaded correctly and got the right file path
    check_empty_img(img_location)

    # imports the images into OpenCV
    original, greyscale = import_image(img_location)

    # Convert to RGB so as to display via matplotlib
    # Using Matplotlib we can easily find the coordinates
    # of the 4 points that is essential for finding the
    # transformation matrix
    greyscale_copy = cv2.cvtColor(greyscale, cv2.COLOR_BGR2RGB)
    plt.imshow(greyscale_copy)

    """
    This section opens the image using matplotlib, so the you can select the 4 corners of the grid
    and modify the gamma value to make it brighter or darker.
    """

    plt.show()  # toggle this to display the matplotlib in order to find the coordinates

    # All points are in format[cols, rows]
    bottom_left = [42, 681]
    bottom_right = [1518, 666]
    top_right = [1347, 75]
    top_left = [266, 83]

    cv2.imshow('originalImage', original)  # displays the input image

    # waits for an input from the keyboard to signify user has finished analysing the images
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # destroys all of the plots ready for a new runthrough

    # set the gammaValue wanted to transform the photo
    gammaValue = 1  # a larger number than 1 is brighter, smaller is darker

    """
    This section now takes the inputs from above and applies a gamma correction and as well as the homography transform
    """

    input_points = [bottom_left, bottom_right, top_right, top_left]

    # creates the gamma adjusted image
    gamma_adjusted_img = adjust_gamma(greyscale, gammaValue)

    out = homography_transformation(gamma_adjusted_img, input_points)

    """
    Below is outputting the images for user viewing as well as saving them to the folder
    """

    cv2.imshow('originalImage', original)  # displays the input image

    cv2.putText(gamma_adjusted_img, "g={}".format(gammaValue), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)  # adds some formatting to the gamma modified image
    cv2.imshow('adjusted', np.hstack(
        [greyscale, gamma_adjusted_img]))  # displays the gamma transformed image

    # displays the corrected grid from a birds eye view
    cv2.imshow('corrected_grid', out)
    # waits for an input from the keyboard to signify user has finished analysing the images
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # destroys all of the plots ready for a new runthrough

    cv2.imwrite('Finalised Code/Output Files/corrected_grid.png',
                out)  # outputs the transformed grid
