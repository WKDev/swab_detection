import cv2

while True:
    img=  cv2.imread("ret.jpg")

    cv2.imshow("ret", img)

    key = cv2.waitKey(10) & 0xF

    if key == ord('q'):

        break

cv2.destroyAllWindows()

