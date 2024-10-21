import cv2
import numpy as np

# 파라미터 업데이트 함수
def update_parameters(val):
    global min_dist, param1, param2, min_radius, max_radius, threshold_value
    global lower_bound, upper_bound, exposure, brightness, highlights, shadows, contrast, black_point, sharpness, definition
    min_dist = cv2.getTrackbarPos('Min Dist', 'Control Panel')
    param1 = cv2.getTrackbarPos('Param1', 'Control Panel')
    param2 = cv2.getTrackbarPos('Param2', 'Control Panel')
    min_radius = cv2.getTrackbarPos('Min Radius', 'Control Panel')
    max_radius = cv2.getTrackbarPos('Max Radius', 'Control Panel')
    threshold_value = cv2.getTrackbarPos('Threshold', 'Control Panel')
    exposure = cv2.getTrackbarPos('Exposure', 'Control Panel')
    brightness = cv2.getTrackbarPos('Brightness', 'Control Panel')
    highlights = cv2.getTrackbarPos('Highlights', 'Control Panel')
    shadows = cv2.getTrackbarPos('Shadows', 'Control Panel')
    contrast = cv2.getTrackbarPos('Contrast', 'Control Panel')
    black_point = cv2.getTrackbarPos('Black Point', 'Control Panel')
    sharpness = cv2.getTrackbarPos('Sharpness', 'Control Panel')
    definition = cv2.getTrackbarPos('Definition', 'Control Panel')

# CLAHE 적용 및 히스토그램 평준화 함수
def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist_equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(hist_equalized)
    return cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

# 이미지 조정 함수
def adjust_image(img):
    img = apply_clahe(img)  
    adjusted = cv2.convertScaleAbs(img, alpha=exposure / 50.0, beta=brightness - 50)
    adjusted = cv2.addWeighted(adjusted, 1 + (highlights - 50) / 100.0, adjusted, 0, shadows - 50)
    adjusted = cv2.convertScaleAbs(adjusted, alpha=contrast / 50.0, beta=0)
    
    black_point_scale = black_point / 50.0
    adjusted = np.clip(adjusted * black_point_scale, 0, 255).astype(np.uint8)
    
    if sharpness > 50:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        adjusted = cv2.filter2D(adjusted, -1, kernel)
    if definition < 50:
        ksize = int((50 - definition) / 10) * 2 + 1
        adjusted = cv2.GaussianBlur(adjusted, (ksize, ksize), 0)
    
    return adjusted

# 처리 함수
def apply_processing(adjusted_image):
    gray = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    _, binary_img = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    circles = cv2.HoughCircles(binary_img, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
                                 param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

    img_result = img.copy()
    green_circle_count = 0
    red_circle_count = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))

        filtered_circles = []
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            mask = np.zeros(binary_img.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)

            masked_img = cv2.bitwise_and(binary_img, binary_img, mask=mask)
            white_pixels = cv2.countNonZero(masked_img)
            total_pixels = np.pi * (radius ** 2)
            black_pixels = total_pixels - white_pixels

            # if black_pixels <= white_pixels * (3 / 2):
            if white_pixels / total_pixels >= 0.6:
                # 이 조건 다르게 표현하면
                filtered_circles.append(i)
                green_circle_count += 1
            else:
                cv2.circle(img_result, (i[0], i[1]), radius, (0, 0, 255), 3)
                red_circle_count += 1

        for i in filtered_circles:
            cv2.circle(img_result, (i[0], i[1]), i[2], (0, 255, 0), 3)

    print(f'녹색 원의 개수: {green_circle_count}, 빨간 원의 개수: {red_circle_count}')

    # height, width = img_result.shape[:2]
    # max_size = 400  # 정사각형 크기 설정

    # scale = min(max_size / height, max_size / width)
    # new_size = (int(width * scale), int(height * scale))
    # resized_img = cv2.resize(img_result, new_size)
    # resized_binary_img = cv2.resize(binary_img, new_size)

    # cv2.imshow('Detected Circles', resized_img)
    # cv2.imshow('Binary Image', resized_binary_img)

    scaled_imshow('Detected Circles', img_result)

# 이미지 로드 및 초기화
img = cv2.imread('mb_010.jpg')
if img is None:
    print("이미지를 로드할 수 없습니다. 파일 경로를 확인하세요.")
    exit()

# 컨트롤 패널 윈도우 생성 및 크기 조정
cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Control Panel', 400, 600)

# 트랙바 생성
cv2.createTrackbar('Min Dist', 'Control Panel', min_dist, 100, update_parameters)
cv2.createTrackbar('Param1', 'Control Panel', param1, 100, update_parameters)
cv2.createTrackbar('Param2', 'Control Panel', param2, 100, update_parameters)
cv2.createTrackbar('Min Radius', 'Control Panel', min_radius, 100, update_parameters)
cv2.createTrackbar('Max Radius', 'Control Panel', max_radius, 100, update_parameters)
cv2.createTrackbar('Threshold', 'Control Panel', threshold_value, 255, update_parameters)
cv2.createTrackbar('Exposure', 'Control Panel', exposure, 100, update_parameters)
cv2.createTrackbar('Brightness', 'Control Panel', brightness, 100, update_parameters)
cv2.createTrackbar('Highlights', 'Control Panel', highlights, 100, update_parameters)
cv2.createTrackbar('Shadows', 'Control Panel', shadows, 100, update_parameters)
cv2.createTrackbar('Contrast', 'Control Panel', contrast, 100, update_parameters)
cv2.createTrackbar('Black Point', 'Control Panel', black_point, 100, update_parameters)
cv2.createTrackbar('Sharpness', 'Control Panel', sharpness, 100, update_parameters)
cv2.createTrackbar('Definition', 'Control Panel', definition, 100, update_parameters)

cv2.namedWindow('Detected Circles', cv2.WINDOW_NORMAL)
cv2.namedWindow('Binary Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Adjusted Image', cv2.WINDOW_NORMAL)  

# 모든 이미지 윈도우 크기 조정 
cv2.resizeWindow('Detected Circles', 400, 400)
cv2.resizeWindow('Binary Image', 400, 400)
cv2.resizeWindow('Adjusted Image', 400, 400)

# 무한 루프를 통해 이미지 업데이트
while True:
    adjusted_image = adjust_image(img)  # 먼저 이미지 조정 호출
    cv2.imshow('Adjusted Image', adjusted_image)  # 조정된 이미지를 독립된 윈도우로 표시
    apply_processing(adjusted_image)      # 조정된 이미지를 처리 함수에 전달
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 루프 종료
        break

cv2.destroyAllWindows()
