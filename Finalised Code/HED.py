import cv2
img = cv2.imread("Finalised Code/Input Files/reference_photo.png")
(H, W) = img.shape[:2]
blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(W, H),
                             swapRB=False, crop=False)
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", "hed_pretrained_bsds.caffemodel")
net.setInput(blob)
hed = net.forward()
hed = cv2.resize(hed[0, 0], (W, H))
hed = (255 * hed).astype("uint8")
cv2.imshow("Input", img)
cv2.imshow("HED", hed)
cv2.waitKey(0)
