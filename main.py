import traceback
import cv2
import numpy as np
import os, time, sys, glob
from datetime import datetime
import threading

from utils.estimator import estimate_cnt
from utils.misc import Perf,Text, odd_maker

import yaml

DETECTION_TIMEOUT = 0.5

# TODO:
# 1. 면봉케이스 바깥쪽 검출된 원 제거하기,  - 손
# 2. 검정부분 헷갈리지 않게 아예 0으로 만들기- 송
# 3. 원의 중심점이 까만 영역에 있으면 제거하기 - 윤
# 4. 머신러닝 gridsearch 만들기 - 손

# presentation 
# 1. 반전 했더니 잘 된다.
# 2. 허프 변환은 진한 색을 먼저 잡는다. 
# 3. 이진화는 보조용이더라.(오탐 제거용)

def load_config():
    if os.path.exists('configuration.yaml'):
        with open('configuration.yaml', 'r') as file:
            return yaml.safe_load(file)
        
def save_config():
    saved_config = {}
    for key, value in config.items():
        saved_config[key] = value[2]

    with open('configuration.yaml', 'w') as file:
        yaml.dump(saved_config, file)

def update_value(value, config_key):
    if config_key in config:
        config[config_key][2] = value

    save_config()





# Load images
imgs=glob.glob('images/*.jpg')
assert len(imgs) > 0, "No images found in the images directory"

pf = Perf()

config = {
    "img_idx": [[0, len(imgs)-1], 0, 0],
    "brightness": [[0,100], 50, 50],  # range, default, current_value
    "contrast": [[0,100], 62, 62],    # range, default, current_value
    "gamma": [[0,100], 52, 52],       # range, default, current_value
    "min_radius": [[20,50], 30, 30],  # range, default, current_value
    "max_radius": [[20,50], 40, 40],  # range, default, current_value
    "min_dist": [[50, 100], 75, 75],       # range, default, current_value
    "param1": [[0,100], 10, 10],      # range, default, current_value
    "param2": [[0,100], 20, 20],       # range, default, current_value

    "threshold_min": [[0, 255], 0, 0],
    "threshold_max": [[0, 255], 255, 255],
    "max_value": [[0, 255], 255, 255],
    "block_size": [[3, 100], 11, 11],

    "c": [[0, 255], 2, 2],
    "kernel_size": [[1, 10], 5, 5],
    "iterations": [[1, 10], 1, 1],
    "blur": [[1, 100], 1, 1],
    "after_blur": [[1, 100], 1, 1],
    "canny_min": [[0, 255], 0, 0],
    "canny_max": [[0, 255], 255, 255],
    "at_dilate": [[1, 100], 1, 1],
    "at_th_min": [[0, 255], 0, 0],
    "at_th_max": [[0, 255], 255, 255],
    "bin_blur": [[1, 250], 1, 1],

}

saved_config = load_config()

if saved_config:
    for key, value in saved_config.items():
        try:
            config[key][1] = value
            config[key][2] = value
            update_value(value, key)
        except:
            pass


# Global variable to store the result of HoughCircles
hough_result = None

def hough_circles_operation(image, min_dist,param1, param2, min_radius, max_radius):
    global hough_result
    hough_result = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, minDist=min_dist,
                                    param1=param1, param2=param2, 
                                    minRadius=min_radius, maxRadius=max_radius)

cv2.namedWindow("image", cv2.WINDOW_NORMAL)

# Attach trackbars
for key, value in config.items():
    add_trackbar(key, "image", value[0], value[1], key)

processed_imgs={}

load_config()

while True:
    pf.start_perf("run")
    
    # Use the current values from the config dictionaries
    img_idx = config["img_idx"][2]
    brightness = config["brightness"][2]
    contrast = config["contrast"][2]
    gamma = config["gamma"][2]
    min_radius = config["min_radius"][2]
    max_radius = config["max_radius"][2]
    min_dist = config["min_dist"][2]
    param1 = config["param1"][2]
    param2 = config["param2"][2]

    threshold_min = config["threshold_min"][2]
    threshold_max = config["threshold_max"][2]
    max_value = config["max_value"][2]
    block_size = config["block_size"][2]
    c = config["c"][2]

    block_size = odd_maker(block_size)
    c = odd_maker(c)
    
    kernel_size = config["kernel_size"][2]
    iterations = config["iterations"][2]

    blur = config["blur"][2]
    after_blur = config["after_blur"][2]

    canny_min = config["canny_min"][2]
    canny_max = config["canny_max"][2]

    at_dilate = config["at_dilate"][2]
    at_th_min = config["at_th_min"][2]
    at_th_max = config["at_th_max"][2]

<<<<<<< HEAD
=======
    bin_blur = config["bin_blur"][2]



>>>>>>> 5b0a3f46ca2795393ff70c3bb36ecc33cb58e725
    # Load image
    src = cv2.imread(imgs[img_idx], cv2.IMREAD_COLOR)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    ksize = 2 * blur + 1
        # apply gaussian blur
<<<<<<< HEAD
    src = cv2.GaussianBlur(src, (ksize,ksize), 0)
=======
    src = cv2.GaussianBlur(src, (ksize,ksize), 0)q
    # add src to processed_imgs dictionary 
    # ex) processed_imgs.update({(name of variable src)): src})
    # processed_imgs.update({f"src": src})


>>>>>>> 5b0a3f46ca2795393ff70c3bb36ecc33cb58e725

    # Histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hist_eq = clahe.apply(src)

    # histogram equalization
    # hist_eq = cv2.equalizeHist(src)

    adjusted = cv2.convertScaleAbs(hist_eq, alpha=contrast/50, beta=brightness-50)
    adjusted = cv2.addWeighted(adjusted, gamma/50, np.zeros(src.shape, src.dtype), 0, 0)

    processed_imgs.update({"adjusted": adjusted})


    try:
        
        adaptive_thresh_image = cv2.adaptiveThreshold(adjusted, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
        thresh_bin_image = cv2.threshold(adjusted, threshold_min, threshold_max, cv2.THRESH_BINARY)[1]
        thresh_otsu_image = cv2.threshold(adjusted, threshold_min, threshold_max, cv2.THRESH_BINARY_INV)[1]


        canny = cv2.Canny(adjusted, canny_min, canny_max)
        canny= cv2.dilate(canny, None, iterations=5)
        canny= cv2.bitwise_not(canny)
        canny= cv2.GaussianBlur(canny, (45,45), 0)

        kernel = np.ones((7,7), np.uint8)

        # Opening 연산 수행
        adaptive_thresh_image = cv2.morphologyEx(adaptive_thresh_image, cv2.MORPH_OPEN, kernel, iterations=5)


        processed_imgs.update({"adaptive_thresh_image": adaptive_thresh_image})

        after_blur = odd_maker(after_blur)


        adaptive_thresh_image = cv2.GaussianBlur(adaptive_thresh_image, (after_blur,after_blur), 0)


        adaptive_thresh_image = cv2.dilate(adaptive_thresh_image, None, iterations=2)

        adaptive_thresh_image = cv2.erode(adaptive_thresh_image, None, iterations=at_dilate)

        at_morp_thesh_image = cv2.threshold(adaptive_thresh_image, at_th_min, at_th_max, cv2.THRESH_BINARY)[1]

<<<<<<< HEAD
        bin_blur = odd_maker(blur)

        res = cv2.GaussianBlur(at_morp_thesh_image, (bin_blur,bin_blur), 0)
=======
        bin_blur_k_size = 2 * bin_blur + 1

        blurred = cv2.GaussianBlur(at_morp_thesh_image, (bin_blur_k_size,bin_blur_k_size), 0)
>>>>>>> 5b0a3f46ca2795393ff70c3bb36ecc33cb58e725

    
        processed_imgs.update({"adaptiveThreshold+blur": blurred})

        blended = cv2.addWeighted(blurred, 0.2, adjusted, 0.8, 0)

        processed_imgs.update({"blended": blended})


        bin_image = at_morp_thesh_image

    except Exception as e:
        print(f"Error: {e}")
        cv2.putText(src, f"Error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    detect_src = blended
    filter_src = at_morp_thesh_image
    circles = None
    # Start the HoughCircles operation in a separate thread
    hough_thread = threading.Thread(target=hough_circles_operation, 
                                    args=(detect_src, min_dist, param1, param2, min_radius, max_radius))
    hough_thread.start()

    # Wait for the thread to complete or timeout
    hough_thread.join(timeout=DETECTION_TIMEOUT)

    if hough_thread.is_alive():
        print(f"HoughCircles operation timed out after {DETECTION_TIMEOUT} seconds")
        cv2.putText(detect_src, f"TIMEOUT: took more than{DETECTION_TIMEOUT}s", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)

        hough_result = None
    else:
        circles = hough_result


    overlay_src = cv2.cvtColor(detect_src, cv2.COLOR_GRAY2BGR)

    valid_cnt = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0]:
            x_center, y_center = int(i[0]), int(i[1])

<<<<<<< HEAD
            try:

                if detect_src[y_center, x_center] == 255: # if the center of the circle is black,
                    
                    cv2.circle(overlay_src, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 5)
                    valid_cnt += 1
                else:
                    cv2.circle(overlay_src, (int(i[0]), int(i[1])), int(i[2]), (0, 0, 255), 2)

            except Exception as e:
                print(traceback.format_exc())
=======
            if filter_src[y_center-1, x_center-1] == 255: # if the center of the circle is black,
                cv2.circle(overlay_src, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
                valid_cnt += 1
            else:
                cv2.circle(overlay_src, (int(i[0]), int(i[1])), int(i[2]), (0, 0, 255), 2)
>>>>>>> 5b0a3f46ca2795393ff70c3bb36ecc33cb58e725
            


        det_cnt = valid_cnt
        ref_cnt = estimate_cnt(imgs[img_idx], det_cnt)


        cv2.putText(overlay_src, f"{det_cnt}/{ref_cnt}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 5)
        cv2.putText(overlay_src, f"Acc: {(det_cnt/ref_cnt)*100:.2f}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 5)
    else:
        cv2.putText(overlay_src, "No circles detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)

    processed_imgs.update({"hough_circles": overlay_src})

    stacked = []

    for k, v in processed_imgs.items():
        if len(v.shape) == 2:
            processed_imgs[k] = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

        processed_imgs[k] = cv2.putText(processed_imgs[k], k, (50, processed_imgs[k].shape[1]-100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
        stacked.append(processed_imgs[k])

    stacked = cv2.hconcat(stacked)

    # save img
    cv2.imwrite("output.jpg", stacked)



    cv2.imshow("ret", cv2.resize(stacked, (0, 0), fx=0.5, fy=0.5))

    if cv2.waitKey(200) & 0xFF == ord('q') :
        break

cv2.destroyAllWindows()