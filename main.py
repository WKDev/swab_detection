import cv2
import numpy as np
import os, time, sys
from datetime import datetime

# load image
src = cv2.imread('images/mb_001_A.jpg', cv2.IMREAD_COLOR)
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)



cv2.namedWindow("image", cv2.WINDOW_NORMAL)

# preprocessing
# preprocessed = preprocess(img)
#   convert to gray
#   밝기 조절
#   hsv 수정
#   gaussian blur

# attach trackbar
cv2.createTrackbar("brightness", "image", 0, 100, lambda x: None)
cv2.createTrackbar("contrast", "image", 0, 100, lambda x: None)
cv2.createTrackbar("gamma", "image", 0, 100, lambda x: None)

# set default value
cv2.setTrackbarPos("brightness", "image", 50)
cv2.setTrackbarPos("contrast", "image", 50)
cv2.setTrackbarPos("gamma", "image", 50)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    clahed_gray = clahe.apply(src)

    brightness = cv2.getTrackbarPos("brightness", "image")
    contrast = cv2.getTrackbarPos("contrast", "image")
    gamma = cv2.getTrackbarPos("gamma", "image")

    if contrast == 0:
        contrast = 1
    if gamma == 0:
        gamma = 1

    adjusted = cv2.convertScaleAbs(src, alpha=contrast/50, beta=brightness-50)
    adjusted = cv2.addWeighted(adjusted, gamma/50, np.zeros(src.shape, src.dtype), 0, 0)

    circles = cv2.HoughCircles(adjusted, cv2.HOUGH_GRADIENT, 1,75, param1 = 10, param2 = 20, minRadius = 30, maxRadius = 40)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0]:
            cv2.circle(adjusted, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 5)

    else:
        cv2.putText(adjusted, "No circles detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow("image", adjusted)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()


# detection
# https://stackoverflow.com/questions/58109962/how-to-optimize-circle-detection-with-python-opencv
#   circle detection
#   hough circle transform
#   watershed algorithm

# clahed_gray = cv2.equalizeHist(gray)




# validation
