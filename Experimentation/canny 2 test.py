import cv2
image = cv2.imread("Archive of images/Report/original.png")
edge_image = cv2.Canny(image, 230, 255)
cv2.imshow("edgeDetection", edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
