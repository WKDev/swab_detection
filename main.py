import cv2
import numpy as np
import os, time, sys, glob
from datetime import datetime
import threading
from matplotlib import pyplot as plt

from utils.estimator import estimate_cnt
from utils.misc import Perf,Text

# TODO:
# 1. 면봉케이스 바깥쪽 검출된 원 제거하기,  - 손
# 2. 검정부분 헷갈리지 않게 아예 0으로 만들기- 송
# 3. 원의 중심점이 까만 영역에 있으면 제거하기 - 윤
# 4. 머신러닝 gridsearch 만들기 - 손

# presentation - 손찬혁
# 1. 반전 했더니 잘 된다.
# 2. 허프 변환은 진한 색을 먼저 잡는다. 
# 3. 이진화는 보조용이더라.(오탐 제거용)

def update_value(value, config_key):
    if config_key in config:
        config[config_key][2] = value
    elif config_key in binary_config:
        binary_config[config_key][2] = value
    elif config_key in morphological_config:
        morphological_config[config_key][2] = value

def add_trackbar(trackbar_name, window, rng, def_val, config_key):
    cv2.createTrackbar(trackbar_name, window, rng[0], rng[1], lambda x: update_value(x, config_key))
    cv2.setTrackbarPos(trackbar_name, window, def_val)

# Set the timeout duration (in seconds)
TIMEOUT_DURATION = 0.2

# Load images
imgs=glob.glob('images/*.jpg')
assert len(imgs) > 0, "No images found in the images directory"

pf = Perf()

config = {
    "img_idx": [[0, len(imgs)], 0, 0],
    "brightness": [[0,100], 50, 50],  # range, default, current_value
    "contrast": [[0,100], 50, 50],    # range, default, current_value
    "gamma": [[0,100], 50, 50],       # range, default, current_value
    "min_radius": [[20,50], 30, 30],  # range, default, current_value
    "max_radius": [[20,50], 40, 40],  # range, default, current_value
    "min_dist": [[50, 100], 75, 75],       # range, default, current_value
    "param1": [[0,100], 10, 10],      # range, default, current_value
    "param2": [[0,100], 20, 20]       # range, default, current_value
}
binary_config = {
    "threshold_min": [[0, 255], 0, 0],
    "threshold_max": [[0, 255], 255, 255],
    "max_value": [[0, 255], 255, 255],
    "block_size": [[3, 100], 11, 11],
    "c": [[0, 255], 2, 2]
}

morphological_config = {
    "kernel_size": [[1, 10], 5, 5],
    "iterations": [[1, 10], 1, 1]
}

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

for key, value in binary_config.items():
    add_trackbar(key, "image", value[0], value[1], key)

for key, value in morphological_config.items():
    add_trackbar(key, "image", value[0], value[1], key)

processed_imgs={}

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

    threshold_min = binary_config["threshold_min"][2]
    threshold_max = binary_config["threshold_max"][2]
    max_value = binary_config["max_value"][2]
    block_size = binary_config["block_size"][2]
    c = binary_config["c"][2]
    
    kernel_size = morphological_config["kernel_size"][2]
    iterations = morphological_config["iterations"][2]

    # Load image
    src = cv2.imread(imgs[img_idx], cv2.IMREAD_COLOR)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # add src to processed_imgs dictionary 
    # ex) processed_imgs.update({(name of variable src)): src})
    processed_imgs.update({f"src": src})



    # Histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahed_gray = clahe.apply(src)

    adjusted = cv2.convertScaleAbs(clahed_gray, alpha=contrast/50, beta=brightness-50)
    adjusted = cv2.addWeighted(adjusted, gamma/50, np.zeros(src.shape, src.dtype), 0, 0)

    processed_imgs.update({"adjusted": adjusted})

    try:
        adaptive_thresh_image = cv2.adaptiveThreshold(adjusted, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
        thresh_bin_image = cv2.threshold(adjusted, threshold_min, threshold_max, cv2.THRESH_BINARY)[1]
        thresh_otsu_image = cv2.threshold(adjusted, threshold_min, threshold_max, cv2.THRESH_OTSU)[1]

        processed_imgs.update({"createCLAHE": clahed_gray})
        processed_imgs.update({"adaptiveThreshold": adaptive_thresh_image})
        processed_imgs.update({"threshold_binary": thresh_bin_image})
        processed_imgs.update({"threshold_otsu": thresh_otsu_image})

        bin_image = adjusted

        # morphological operations
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        for i in range(iterations):
            bin_image = cv2.morphologyEx(adjusted, cv2.MORPH_CLOSE, kernel)
            bin_image = cv2.morphologyEx(adjusted, cv2.MORPH_OPEN, kernel)

        processed_imgs.update({"morphology": bin_image})

    except Exception as e:
        print(f"Error: {e}")
        cv2.putText(src, f"Error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if contrast == 0:
        contrast = 1
    if gamma == 0:
        gamma = 1


    

    inversed = cv2.bitwise_not(bin_image)

    # apply gaussian blur
    inversed = cv2.GaussianBlur(inversed, (5, 5), 100)

    processed_imgs.update({"blur": inversed})

    # Start the HoughCircles operation in a separate thread
    hough_thread = threading.Thread(target=hough_circles_operation, 
                                    args=(inversed, min_dist, param1, param2, min_radius, max_radius))
    hough_thread.start()

    # Wait for the thread to complete or timeout
    hough_thread.join(timeout=TIMEOUT_DURATION)

    if hough_thread.is_alive():
        print(f"HoughCircles operation timed out after {TIMEOUT_DURATION} seconds")
        cv2.putText(adjusted, f"TIMEOUT: took more than{TIMEOUT_DURATION}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        hough_result = None
    else:
        circles = hough_result

    colored = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0]:
            cv2.circle(colored, (int(i[0]), int(i[1])), int(i[2]), (0, 0, 255), 5)
    else:
        cv2.putText(colored, "No circles detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)

    processed_imgs.update({"hough_circles": colored})

    # black = np.zeros_like(colored)
    # colored = cv2.hconcat([colored, black,black,black])
    
    # # resize th.ret to fit th same size as colored
    # resized = cv2.resize(th.ret, (colored.shape[1], colored.shape[0]), interpolation=cv2.INTER_AREA)
    # resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    # stacked = cv2.vconcat([colored,resized])

    stacked = []

    for k, v in processed_imgs.items():
        if len(v.shape) == 2:
            processed_imgs[k] = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

        processed_imgs[k] = cv2.putText(processed_imgs[k], k, (50, processed_imgs[k].shape[1]-300), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
        stacked.append(processed_imgs[k])

    stacked = cv2.hconcat(stacked)

    # pf.show_perf()

    cv2.imshow("image", stacked)

    if circles is not None:
        estimate_cnt(imgs[img_idx], len(circles[0]))



    if cv2.waitKey(200) & 0xFF == 27:
        break

cv2.destroyAllWindows()