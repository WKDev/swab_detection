import glob
import traceback
import cv2
import numpy as np
import yaml
import threading
from utils.adjust_image import adjust_image
from utils.estimator import hough_circles_operation
from utils.misc import odd_maker, scaled_imshow
import PIL
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

def hist_equalization(img, params, **kwargs):
    return cv2.equalizeHist(img)

def bgr_to_gray(img, params,**kwargs):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def adaptive_threshold(img, params,**kwargs):
    ksize = odd_maker(params['adaptive_threshold']['value']['block_size'])
    c = params['adaptive_threshold']['value']['c']

    ret = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ksize, c)
    return ret

def hough_circle(img, params, **kwargs):
    global hough_result
    DETECTION_TIMEOUT = 0.5
    options = params['hough_circle']['value']
    min_dist = options['min_dist']
    param1 = options['param1']
    param2 = options['param2']
    min_radius = options['min_radius']
    max_radius = options['max_radius']

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # hough_thread = threading.Thread(target=hough_circles_operation, 
    #                                 args=(img_gray, min_dist, param1, param2, min_radius, max_radius))
    # hough_thread.start()

    # hough_thread.join(timeout=DETECTION_TIMEOUT)

    with ThreadPoolExecutor() as executor:
        future = executor.submit(hough_circles_operation, img_gray, min_dist, param1, param2, min_radius, max_radius)
    try:
        circles = future.result(timeout=DETECTION_TIMEOUT)
    except concurrent.futures.TimeoutError:
        circles = None
        print(f"HoughCircles operation timed out after {DETECTION_TIMEOUT} seconds")
        cv2.putText(img, f"TIMEOUT: took more than{DETECTION_TIMEOUT}s", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)

    # overlay_src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    valid_cnt = 0
    total_detected = len(circles[0]) if circles is not None else -1

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0]:
            cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 5)

            x_center, y_center = int(i[0]), int(i[1])

    cv2.putText(img, f"Detected {total_detected} circles", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
    scaled_imshow(img,"hough")
    return img

def threshold(img, params,**kwargs):
    ret = cv2.threshold(img, params['threshold']['value']['min'], params['threshold']['value']['max'], cv2.THRESH_BINARY)[1]
    return ret

def threshold2(img, params,**kwargs):
    p = params['thresh_2']['value']
    ret = cv2.threshold(img, p['min'], p['max'], cv2.THRESH_BINARY)[1]
    return ret

def thresh_otsu(img, params,**kwargs):
    ret = cv2.threshold(img, params['threshold']['value']['min'], 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    scaled_imshow(ret,"thresh_otsu")

    return ret

def gaussian_blur(img, params,**kwargs):
    ksize = odd_maker(params['gaussian_blur']['value']['ksize'])
    return cv2.GaussianBlur(img, (ksize,ksize), 0)

def def_preprocess(img, params, **kwargs):
    img = adjust_image(img, shadows=params['_adj']['value']['shadows'] -100,
                          highlights=params['_adj']['value']['highlights'] -100,
                          brilliance=params['_adj']['value']['brilliance'] -100,
                          exposure=params['_adj']['value']['exposure'] -100,
                          contrast=params['_adj']['value']['contrast'] -100,
                          brightness=params['_adj']['value']['brightness'] -100,
                          black_point=params['_adj']['value']['black_point'] -100,
                          sharpness=params['_adj']['value']['sharpness'] -100,
                          noise_reduction=params['_adj']['value']['noise_reduction'] -100)
    return img

def def_preprocess2(img, params, **kwargs):
    img = adjust_image(img, shadows=params['_adj2']['value']['shadows'] -100,
                          highlights=params['_adj2']['value']['highlights'] -100,
                          brilliance=params['_adj2']['value']['brilliance'] -100,
                          exposure=params['_adj2']['value']['exposure'] -100,
                          contrast=params['_adj2']['value']['contrast'] -100,
                          brightness=params['_adj2']['value']['brightness'] -100,
                          black_point=params['_adj2']['value']['black_point'] -100,
                          sharpness=params['_adj2']['value']['sharpness'] -100,
                          noise_reduction=params['_adj2']['value']['noise_reduction'] -100)
    # scaled_imshow(img,"preprocessed2")
    return img

def find_contours(img, params):
    # 컨투어 찾기 ---①
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    # 전체 둘레의 0.05로 오차 범위 지정 ---②
    epsilon = 0.05 * cv2.arcLength(contour, True)

    # 근사 컨투어 계산 ---③
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 각각 컨투어 선 그리기 ---④

    colored_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ret = cv2.drawContours(colored_img, [contour], -1, (0,255,0), 3)
    ret = cv2.drawContours(colored_img, [approx], -1, (0,255,0), 3)

    return ret

def convex_hull(img, params):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #        · mode : 컨투어를 찾는 방법

                    # cv2.RETR_EXTERNAL: 컨투어 라인 중 가장 바깥쪽의 라인만 찾음

                    # cv2.RETR_LIST: 모든 컨투어 라인을 찾지만, 상하구조(hierachy)관계를 구성하지 않음

                    # cv2.RETR_CCOMP: 모든 컨투어 라인을 찾고, 상하구조는 2 단계로 구성함

                    # cv2.RETR_TREE: 모든 컨투어 라인을 찾고, 모든 상하구조를 구성함

    #        · method : 컨투어를 찾을 때 사용하는 근사화 방법

                    # cv2.CHAIN_APPROX_NONE: 모든 컨투어 포인트를 반환

                    # cv2.CHAIN_APPROX_SIMPLE: 컨투어 라인을 그릴 수 있는 포인트만 반환

                    # cv2.CHAIN_APPROX_TC89_L1: Teh_Chin 연결 근사 알고리즘 L1 버전을 적용해 컨투어 포인트

    colored_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
    for i in contours:
        hull = cv2.convexHull(i, clockwise=True)
        cv2.drawContours(colored_img, [hull], 0, (0, 0, 255), 2) 
        # 컨투어 그리기 (0, 0, 255): 선 색상, 2: 선 두께
        #print(hull)

    return colored_img
        
def detect_circles(img, params, **kwargs):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    colored_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    circles = []
    
    for contour in contours:
        # 최소 외접원 찾기
        (x, y), radius = cv2.minEnclosingCircle(contour)

        if radius >500:
            center = (int(x), int(y))
            radius = int(radius)
            
            
            # 원 그리기
            ret = cv2.circle(kwargs['org_img'], center, radius, (0, 255, 0), 2)
            
            # 원 정보 저장
            circles.append((center, radius))
            
            # 정보 출력
            print(f"원 중심: ({x:.2f}, {y:.2f})")
            print(f"반지름: {radius:.2f}")
            print("---")

    scaled_imshow(ret,"detected")
    
    return colored_img

def detect_ellipses(img, params, **kwargs):

    # dilation
    kernel = np.ones((3,3),np.uint8)
    # img = cv2.dilate(img,kernel,iterations = 3)

    # erode
    img = cv2.dilate(img,kernel,iterations = 3)

    org = kwargs['org_img'].copy()

    # 컨투어 찾기
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # 가장 큰 컨투어 선택 (타원이라 가정)
    try:
        largest_contour = max(contours, key=cv2.contourArea)
    except:
        print("No contours found")
        return org


    # 컨투어 그리기
    ctr = cv2.drawContours(org, [largest_contour], -1, (255, 0, 0), 2)

    
    # 타원 피팅
    ellipse = cv2.fitEllipse(largest_contour)
    
    # 타원 중심, 축, 각도 추출
    (xc, yc), (d1, d2), angle = ellipse
    
    # 장축과 단축 계산
    major_axis = max(d1, d2) / 2
    minor_axis = min(d1, d2) / 2

    # 타원 그리기
    ret = cv2.ellipse(org, ellipse, (0, 255, 0), 2)

    # 타원의 장축이 x축과 평행하도록 회전 후, 장축과 단축이 동일하도록 이미지 스케일링하여 원으로 변환
    if major_axis < minor_axis:
        major_axis, minor_axis = minor_axis, major_axis

    angle = - angle

    scale_x = major_axis/minor_axis

    # 회전 변환 행렬 생성
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    result = cv2.warpAffine(org, M, (org.shape[1], org.shape[0]))

    # 스케일링 변환 행렬 생성
    M = cv2.getRotationMatrix2D((xc, yc), 0, scale_x)
    result = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]))

    # 원 중심 그리기
    result= cv2.circle(result, (int(xc), int(yc)), 5, (0, 0, 255), -1)

    # 원 중심 좌표 값이 1200이 되도록 외곽에 까만색 padding 추가
    pad_x = 1400 - int(xc)
    pad_y = 1400 - int(yc)
    result = cv2.copyMakeBorder(result, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))


    # 원 반지름 구하기
    radius = int(major_axis * scale_x)

    x_new = int(xc) + pad_x
    y_new = int(yc) + pad_y

    # 원 심에서 반지름 길이만큼 선 그리기
    # result = cv2.line(result, (int(x_new), int(y_new)), (int(x_new+radius), int(y_new)), (0, 0, 255), 2)
    # result = cv2.line(result, (int(x_new), int(y_new)), (int(x_new), int(y_new+radius)), (0, 0, 255), 2)

    # 원 부분만 자르기
    result = result[y_new-radius:y_new+radius, x_new-radius:x_new+radius]

    # 이미지 해상도 2000x2000 고정
    result = cv2.resize(result, (2000, 2000))

    # scaled_imshow(result,"ellipses-->circles")

    return result

def find_edge(img, params, **kwargs):
    raw = img.copy()
    img2 = img.copy()

    # 이미지 리사이즈
    hor = 1700 # 변환될 가로 픽셀 사이즈
    ver = 1600 # 변환될 세로 픽셀 사이즈
    img = cv2.resize(img, (hor, ver)) 

    # 그레이스케일과 바이너리 스케일 변환
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # 히스토그램 평활화
    #imgray = cv2.equalizeHist(imgray)

    # 이미지 블러 / 이미지 샤프닝
    filter_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    imgray_sharp_1 = cv2.filter2D(imgray, -1, filter_sharp)

    imgray_blur = cv2.blur(imgray, (21, 21))
    #imgray_blur  = cv2.GaussianBlur(imgray, (21, 21), sigmaX = 0, sigmaY = 0)

    # 원본 이미지 - 블러 이미지
    result = imgray - imgray_blur

    # 이진화
    ret, img_binary = cv2.threshold(result, 150, 255, cv2.THRESH_BINARY_INV)

    # # 결과 출력
    # cv2.imshow('imgray', imgray)
    # cv2.imshow('imgray_blur', imgray_blur)
    # cv2.imshow('imgdiff', result)



    # cv2.waitKey()
    # cv2.destroyAllWindows()

    scaled_imshow(img_binary,"edge")

    img = raw

    return img

def hough_circle_mean(img, params, **kwargs):
    DETECTION_TIMEOUT = 0.5
    options = params['hough_circle']['value']
    min_dist = options['min_dist']
    param1 = options['param1']
    param2 = options['param2']
    min_radius = options['min_radius']
    max_radius = options['max_radius']
    rotated_imgs = []

    for i in range(0, 360, 10):
        rotated = PIL.Image.fromarray(img).rotate(i)

        rotated_imgs.append(np.array(rotated))

    detected_list = []

    detected_imgs = []

    for i in rotated_imgs:
        img_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        with ThreadPoolExecutor() as executor:
            future = executor.submit(hough_circles_operation, img_gray, min_dist, param1, param2, min_radius, max_radius)
        try:
            circles = future.result(timeout=DETECTION_TIMEOUT)
        except concurrent.futures.TimeoutError:
            circles = None
            print(f"HoughCircles operation timed out after {DETECTION_TIMEOUT} seconds")
            cv2.putText(i, f"TIMEOUT: took more than{DETECTION_TIMEOUT}s", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)

        # overlay_src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        valid_cnt = 0
        total_detected = len(circles[0]) if circles is not None else -1
        # print(f"Detected {total_detected} circles")
        detected_list.append(total_detected)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for k in circles[0]:
                detected_imgs.append(cv2.circle(i, (int(k[0]), int(k[1])), int(k[2]), (0, 255, 0), 5))


    img = detected_imgs[detected_list.index(max(detected_list))]
    scaled_imshow(img,"hough_circle_mean")

    print(f"Detected avg circles done")
    print(f"Detected {sum(detected_list)/len(detected_list)} circles")

    return img

def imdiff(img, params, **kwargs):
    p = params['imdiff']['value']

    # 이미지 그레이 스케일 변환
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이미지 블러 / 이미지 샤프닝
    filter_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    imgray_sharp_1 = cv2.filter2D(imgray, -1, filter_sharp)

    imgray_blur = cv2.blur(imgray, (p['blur'], p['blur']))
    #imgray_blur  = cv2.GaussianBlur(imgray, (21, 21), sigmaX = 0, sigmaY = 0)

    # 원본 이미지 - 블러 이미지
    result = imgray - imgray_blur

    scaled_imshow(result,"imdiff_result")

    # 이진화
    ret, img_binary = cv2.threshold(result, p['min'], p['max'], cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 결과 출력
    scaled_imshow(img_binary,"imdiff_result_threshold")

    return img_binary

# CLAHE 적용 및 히스토그램 평준화 함수
def sj_apply_clahe(img, params, **kwargs):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist_equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(hist_equalized)
    return cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

def sj_adjust_image(img, params, **kwargs):
    options = params['sj']['value']
    exposure = options['exposure']
    brightness = options['brightness']
    highlights = options['highlights']
    shadows = options['shadows']
    contrast = options['contrast']
    black_point = options['black_point']
    sharpness = options['sharpness']
    definition = options['definition']

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

def sj_apply_processing(img, params, **kwargs):
    options = params['sj']['value']
    threshold_value = options['threshold_value']
    min_dist = options['min_dist']
    param1 = options['param1']
    param2 = options['param2']
    min_radius = options['min_radius']
    max_radius = options['max_radius']

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
                filtered_circles.append(i)
                green_circle_count += 1
            else:
                cv2.circle(img_result, (i[0], i[1]), radius, (0, 0, 255), 3)
                red_circle_count += 1

        for i in filtered_circles:
            cv2.circle(img_result, (i[0], i[1]), i[2], (0, 255, 0), 3)

    print(f'녹색 원의 개수: {green_circle_count}, 빨간 원의 개수: {red_circle_count}')

    height, width = img_result.shape[:2]
    max_size = 400  # 정사각형 크기 설정

    scale = min(max_size / height, max_size / width)
    new_size = (int(width * scale), int(height * scale))
    resized_img = cv2.resize(img_result, new_size)
    resized_binary_img = cv2.resize(binary_img, new_size)

    cv2.imshow('Detected Circles', resized_img)
    cv2.imshow('Binary Image', resized_binary_img)

class SwabDetector:
    def __init__(self,path='images/*.jpg',window_name='image', config_file='config.yaml', show_result=True, interactive=False):
        self.interactive = interactive
        self.imgs=glob.glob(path)
        self.processed_imgs = {}
        self.window_name = window_name
        self.config_file = config_file
        self.config = {}
        self.show_result = show_result

        self.current_img = 0
    
        assert len(self.imgs) > 0, "No images found in the images directory"
        if self.interactive:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.load_values()
            self.init_trackbars()

        self.pipelines = []
        self.processed_imgs = []


    def add_pipeline(self, pipeline):
        '''
        이미지 처리를 수행할 파이프라인 함수 추가합니다.
        반드시 foo_bar(img, params, **kwargs) 형태여야 하고, img를 반환해야 합니다.
        params는 설정값을 전달하며, kwargs는 이미지 처리에 필요한 추가적인 정보를 전달합니다.
        '''
        self.pipelines.append(pipeline)

    def load_values(self):
        '''
        Load values from config file
        '''
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f) 

        if self.config is None:
            self.config = {}

    def init_trackbars(self):
        '''
        Initialize trackbars
        '''
        self.add_trackbar('img_idx', (0, len(self.imgs)-1), 0)

        for k, v in self.config.items():
            if v['show']:
                rng = v['range']
                values = v['value']


                for k2,v2 in values.items():
                    self.add_trackbar(f"{k}.{k2}", (rng['min'], rng['max']), v2)

    def add_trackbar(self, trackbar_name, rng, def_val):
        cv2.createTrackbar(trackbar_name,self.window_name, rng[0], rng[1], self.save_config)
        cv2.setTrackbarPos(trackbar_name,self.window_name, def_val)

    def save_config(self, val):
        # self.config.update(trackbar_name, val)
        # save values to config file
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)

    def run(self):
        '''
        Run the pipeline
        '''
        while True:

            # Load values from trackbars
            for k,v in self.config.items():
                if v['show']:
                    values = v['value']
                    for k2,v2 in values.items():
                        self.config[k]['value'][k2] = cv2.getTrackbarPos(f"{k}.{k2}", self.window_name)

            # Load image
            img_path = self.imgs[self.current_img]
            img = cv2.imread(img_path)
            org_img = img.copy()

            # Run the pipeline
            # 여기가 중요한 부분, __main__에서 pipeline을 추가하면 여기서 실행됨
            for p in self.pipelines:
                img = p(img, self.config, org_img=org_img)
                self.processed_imgs.append(img)


            if not self.interactive:
                break

            
            if self.show_result:
                cv2.imshow(self.window_name, img)
    

            # Handle keyboard input
   
            current_input = cv2.waitKey(1) & 0xFF
            if current_input == ord('q'):
                break

            elif current_input == ord(']'):
                self.current_img +=1

            elif current_input == ord('['):
                self.current_img -=1

            if self.current_img < 0:
                self.current_img = len(self.imgs) - 1
            elif self.current_img >= len(self.imgs):
                self.current_img = 0

            


        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    # 이미지 로드 및 설정값 불러오기
    options= {
        'interactive':True,
        'path':'images/*.jpg', # 이미지 경로
        'window_name':'image', # 윈도우 이름
        'config_file':'config.yaml', # 설정값 파일
        'show_result':True # 결과 보여줄지 여부
    }
    sd = SwabDetector(**options)

    sd.add_pipeline(bgr_to_gray) # 원본 이미지를 흑백으로 변환
    sd.add_pipeline(hist_equalization) # 히스토그램 평활화
    sd.add_pipeline(def_preprocess) # 이미지 전처리
    sd.add_pipeline(adaptive_threshold)
    sd.add_pipeline(gaussian_blur) # 가우시안 블러
    sd.add_pipeline(threshold) # 이진화
    sd.add_pipeline(detect_ellipses) # 타원 검출 및 warping
    # sd.add_pipeline(def_preprocess2)
    # sd.add_pipeline(imdiff)
    # sd.add_pipeline(threshold2)
    # sd.add_pipeline(find_edge)

    # sd.add_pipeline(hough_circle)
    # sd.add_pipeline(hough_circle_mean)

    # trackbar 및 결과 이미지 출력을 위한 루프 
    sd.run()

