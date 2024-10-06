import glob
import cv2
import numpy as np
import yaml

from utils.adjust_image import adjust_image
from utils.misc import odd_maker, scaled_imshow

def bgr_to_gray(img, params,**kwargs):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def threshold(img, params,**kwargs):
    return cv2.threshold(img, params['threshold']['value']['min'], params['threshold']['value']['max'], cv2.THRESH_BINARY)[1]

def thresh_otsu(img, params,**kwargs):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

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
    scaled_imshow(img,"preprocessed")
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
    kernel = np.ones((5,5),np.uint8)
    # img = cv2.dilate(img,kernel,iterations = 3)

    # erode
    img = cv2.erode(img,kernel,iterations = 3)

    scaled_imshow(img,"dilated")


    org = kwargs['org_img'].copy()

    # 컨투어 찾기
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # 가장 큰 컨투어 선택 (타원이라 가정)
    try:
        largest_contour = max(contours, key=cv2.contourArea)
    except:
        print("No contours found")s
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
    
    # 회전 행렬 계산
    rotation_matrix = cv2.getRotationMatrix2D((xc, yc), angle, 1)

    # 타원 그리기

    ret = cv2.ellipse(org, ellipse, (0, 255, 0), 2)
    
    # 이미지 회전
    rotated = cv2.warpAffine(ret, rotation_matrix, (img.shape[1], img.shape[0]))
    
    # 스케일링 행렬 계산
    scale_x = major_axis / minor_axis
    scale_y = 1.0
    scaling_matrix = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])
    
    # 이미지 스케일링
    result = cv2.warpAffine(rotated, scaling_matrix, (img.shape[1], img.shape[0]))
    
    # 원의 반지름 계산 (스케일링 후 장축 길이)
    radius = int(major_axis * scale_x)
    
    # 원을 포함하는 정사각형 영역 크롭
    x1 = max(int(xc - radius), 0)
    y1 = max(int(yc - radius), 0)
    x2 = min(int(xc + radius), result.shape[1])
    y2 = min(int(yc + radius), result.shape[0])

    # 검은색 배경 500픽셀 추가    

    # result = cv2.addWeighted(result, 0.8, org, 0.2, 0)
    cropped = result[y1:y2, x1:x2]

    scaled_imshow(ctr,"ctr")
    scaled_imshow(cropped,"detected")

    return cropped

class SwabDetector:
    def __init__(self,path='images/*.jpg',window_name='image', config_file='config.yaml', show_result=True):
        self.imgs=glob.glob(path)
        self.processed_imgs = {}
        self.window_name = window_name
        self.config_file = config_file
        self.config = {}
        self.show_result = True

        self.current_img = 0
    
        assert len(self.imgs) > 0, "No images found in the images directory"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.load_values()

        self.pipelines = []

        self.init_trackbars()

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

if __name__ == '__main__':
    
    # 이미지 로드 및 설정값 불러오기
    options= {
        'path':'images/*.jpg', # 이미지 경로
        'window_name':'image', # 윈도우 이름
        'config_file':'config.yaml', # 설정값 파일
        'show_result':True # 결과 보여줄지 여부
    }
    sd = SwabDetector(**options)

    sd.add_pipeline(bgr_to_gray) # 원본 이미지를 흑백으로 변환

    sd.add_pipeline(def_preprocess) # 이미지 전처리
    sd.add_pipeline(gaussian_blur) # 가우시안 블러

    sd.add_pipeline(threshold) # 이진화
    sd.add_pipeline(detect_ellipses) # 타원 검출 및 warping

    # trackbar 및 결과 이미지 출력을 위한 루프 
    sd.run()

