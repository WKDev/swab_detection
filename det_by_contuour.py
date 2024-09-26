import cv2

image = cv2.imread('images/mb_001_A_500.jpg')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('image')



cv2.createTrackbar('Threshold Min', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('Threshold Max', 'image', 0, 255, lambda x: None)
cv2.setTrackbarPos('Threshold Min', 'image', 180)
cv2.setTrackbarPos('Threshold Max', 'image', 255)

# trackbar for gaussian blur
cv2.createTrackbar('ThresholdBlur', 'image', 0, 10, lambda x: None)
cv2.setTrackbarPos('ThresholdBlur', 'image', 1)

while True:

    thresh_min = cv2.getTrackbarPos('Threshold Min', 'image')
    thresh_max = cv2.getTrackbarPos('Threshold Max', 'image')

    blur = cv2.getTrackbarPos('ThresholdBlur', 'image')
    ksize = 2 * blur + 1

    bl = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    thresh = cv2.threshold(bl, thresh_min, thresh_max, cv2.THRESH_BINARY_INV)[1]
    otsu = cv2.threshold(bl, thresh_min, thresh_max, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    bin_otsu = cv2.threshold(bl, thresh_min, thresh_max, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    stacked = cv2.hconcat([gray, bl])


    cv2.imshow('image', cv2.resize(stacked, (0, 0), fx=0.5, fy=0.5))
    if cv2.waitKey(50) == ord('q'):
        break

