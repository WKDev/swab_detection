import cv2

cnt = 50
scale: float = 1/cnt
img= cv2.imread("images/mb_001_A_482.jpg")

for i in range(1,cnt+1):
    print(f"resized/{i*scale:.2f}_mb_001_A_482")
    resized=cv2.resize(img, (0,0),fx=i*scale, fy=i*scale)

    cv2.imwrite(f"resized/{i*scale:.2f}_mb_001_A_482.jpg", resized)