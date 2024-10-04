from functools import wraps
import cv2
import time

def Text(text):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 원본 함수 실행
            result_image = func(*args, **kwargs)
            
            # 결과가 이미지인지 확인
            if not isinstance(result_image, np.ndarray):
                raise ValueError("The decorated function must return an image (numpy array)")
            
            # 이미지가 흑백인 경우 컬러로 변환
            if len(result_image.shape) == 2:
                result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
            
            # 텍스트 추가
            position=(10, 30)
            font=cv2.FONT_HERSHEY_SIMPLEX
            font_scale=5
            color=(255, 255, 255)
            thickness=3


            cv2.putText(result_image, text, position, font, font_scale, color, thickness)
            
            return result_image
        return wrapper
    return decorator


def odd_maker(num):
    if num % 2 == 0:
        return num + 1
    else:
        return num

class Perf:
    def __init__(self):
        self.start_time = time.time()
        self.end = None
        self.job_name=""

    def start_perf(self, job_name):
        self.job_name = job_name
        self.start_time = time.time()

    def show_perf(self):
        self.end = time.time()
        print(f"[{self.job_name}] Elapsed: {(self.end - self.start_time)*1000:.3f}ms")

        self.start = time.time()
