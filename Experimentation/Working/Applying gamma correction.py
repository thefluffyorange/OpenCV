import cv2
import numpy as np
from matplotlib import pyplot as plt
original = cv2.imread('corrected_grid.png', 0)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


gammaValue = 0.3

adjusted = adjust_gamma(original, gammaValue)
cv2.putText(adjusted, "g={}".format(gammaValue), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
cv2.imshow('adjusted', np.hstack([original, adjusted]))
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('grid_simulation_gamma_converted.png', adjusted)


# code below iterates the image through various gamma values just to show the differences

# for gamma in np.arange(0.0, 3.5, 0.5):
#     # ignore when gamma is 1 (there will be no change to the image)
#     if gamma == 1:
#         continue
#     # apply gamma correction and show the images
#     gamma = gamma if gamma > 0 else 0.1
#     adjusted = adjust_gamma(darkImage, gamma=gamma)
#     cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
#     cv2.imshow("Images", np.hstack([darkImage, adjusted]))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
