import cv2
import numpy as np
import os, time, sys, glob
from datetime import datetime
import threading

from utils.adjust_image import adjust_image
from utils.estimator import estimate_cnt
from utils.misc import Perf,Text

import yaml

DETECTION_TIMEOUT = 0.5
CONFIG = 'enhanced_thresh.yaml'
def load_config():
    if os.path.exists(CONFIG):
        with open(CONFIG, 'r') as file:
            return yaml.safe_load(file)
        
def save_config():
    saved_config = {}
    for key, value in config.items():
        saved_config[key] = value[2]

    with open(CONFIG, 'w') as file:
        yaml.dump(saved_config, file)

def update_value(value, config_key):
    if config_key in config:
        config[config_key][2] = value

    save_config()


def add_trackbar(trackbar_name, window, rng, def_val, config_key):
    cv2.createTrackbar(trackbar_name, window, rng[0], rng[1], lambda x: update_value(x, config_key))
    cv2.setTrackbarPos(trackbar_name, window, def_val)


# Load images
imgs=glob.glob('images/*.jpg')
assert len(imgs) > 0, "No images found in the images directory"

config = {
    "img_idx": [[0, len(imgs)-1], 0, 0],
    "shadows": [[0, 200], 100, 100],
    "highlights": [[0, 200], 100, 100],
    "brilliance": [[0, 200], 100, 100],
    "exposure": [[0, 200], 100, 100],
    "contrast": [[0, 200], 100, 100],
    "brightness": [[0, 200], 100, 100],
    "black_point": [[0, 200], 100, 100],
    "sharpness": [[0, 200], 100, 100],
    "noise_reduction": [[0, 200], 100, 100],


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
    "bin_blur": [[1, 100], 1, 1],

}

saved_config = load_config()

if saved_config:
    for key, value in saved_config.items():
        if key in config:
            config[key][1] = value
            config[key][2] = value
            update_value(value, key)
        else:
            print(f"Unknown key: {key}")


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

    if key in ["img_idx","shadows", "highlights", "brilliance", "exposure", "contrast", "brightness", "black_point", "sharpness", "noise_reduction"]:
        add_trackbar(key, "image", value[0], value[1], key)

processed_imgs={}

load_config()

while True:    
    # Use the current values from the config dictionaries

    
    img_idx = config["img_idx"][2]

    shadows = config["shadows"][2]-100
    highlights = config["highlights"][2]-100
    brilliance = config["brilliance"][2]-100
    exposure = config["exposure"][2]-100
    contrast = config["contrast"][2]-100
    brightness = config["brightness"][2]-100
    black_point = config["black_point"][2]-100
    sharpness = config["sharpness"][2]-100
    noise_reduction = config["noise_reduction"][2]-100

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
    
    kernel_size = config["kernel_size"][2]
    iterations = config["iterations"][2]

    blur = config["blur"][2]
    after_blur = config["after_blur"][2]

    canny_min = config["canny_min"][2]
    canny_max = config["canny_max"][2]

    at_dilate = config["at_dilate"][2]
    at_th_min = config["at_th_min"][2]
    at_th_max = config["at_th_max"][2]

    



    # Load image
    src = cv2.imread(imgs[img_idx], cv2.IMREAD_COLOR)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


    ksize = 2 *( blur)+ 1
        # apply gaussian blur
    src = cv2.GaussianBlur(src, (ksize,ksize), 0)


    # Histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hist_eq = clahe.apply(src)

    # Adjust shadows and highlights
    adjusted = adjust_image(src, shadows=shadows, highlights=highlights, brilliance=brilliance, exposure=exposure, contrast=contrast, brightness=brightness, black_point=black_point, sharpness=sharpness, noise_reduction=noise_reduction)

    processed_imgs.update({"adjusted": adjusted})


    try:
        adaptive_thresh_image = cv2.adaptiveThreshold(adjusted, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
        thresh_bin_image = cv2.threshold(adjusted, threshold_min, threshold_max, cv2.THRESH_BINARY)[1]
        thresh_otsu_image = cv2.threshold(adjusted, threshold_min, threshold_max, cv2.THRESH_BINARY_INV)[1]

        # processed_imgs.update({"createCLAHE": clahed_gray})
        # processed_imgs.update({"thresh_otsu_image": thresh_otsu_image})

        # cv2.imshow("adaptiveThreshold", cv2.resize(adaptive_thresh_image, (0, 0), fx=0.5, fy=0.5))
        kernel = np.ones((3, 3), np.uint8)

        # adaptive_thresh_image = cv2.erode(thresh_otsu_image, kernel, iterations=iterations)

        # kernel = np.ones((3, 3), np.uint8)

        # opening = cv2.morphologyEx(adaptive_thresh_image, cv2.MORPH_OPEN, kernel)
        # closing = cv2.morphologyEx(adaptive_thresh_image, cv2.MORPH_CLOSE, kernel)

        canny = cv2.Canny(adjusted, canny_min, canny_max)
        processed_imgs.update({"canny": canny})

        after_blur_k_size= 2 * blur + 1
        adaptive_thresh_image = cv2.GaussianBlur(adaptive_thresh_image, (after_blur,after_blur), 0)

        adaptive_thresh_image = cv2.dilate(adaptive_thresh_image, None, iterations=2)

        adaptive_thresh_image = cv2.erode(adaptive_thresh_image, None, iterations=at_dilate)

        at_morp_thesh_image = cv2.threshold(adaptive_thresh_image, at_th_min, at_th_max, cv2.THRESH_BINARY)[1]

        bin_blur_k_size = 2 * blur + 1

        res = cv2.GaussianBlur(at_morp_thesh_image, (bin_blur_k_size,bin_blur_k_size), 0)



        processed_imgs.update({"adaptiveThreshold+blur": res})





        # border = cv2.dilate(adaptive_thresh_image, None, iterations=iterations) - cv2.erode(adaptive_thresh_image, None, iterations=iterations)

        # dt = cv2.distanceTransform(adaptive_thresh_image, cv2.DIST_L2, 3)

        # dt= ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)

        # ret, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
        

        # processed_imgs.update({"distance_transform": dt})

        # processed_imgs.update({"border": border})
        # cv2.imshow("border", cv2.resize(border, (0, 0), fx=0.5, fy=0.5))

        bin_image = at_morp_thesh_image

        # morphological operations
        # kernel = np.ones((kernel_size, kernel_size), np.uint8)
        for i in range(iterations):
            bin_image = cv2.morphologyEx(adjusted, cv2.MORPH_CLOSE, kernel)
            bin_image = cv2.morphologyEx(adjusted, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((3, 3), np.uint8)

        # 1. 침식 연산 (Erosion)
        erosion = cv2.erode(at_morp_thesh_image, kernel, iterations=1)

        # 2. 팽창 연산 (Dilation)
        dilation = cv2.dilate(at_morp_thesh_image, kernel, iterations=iterations)

        # # 3. 열림 연산 (Opening: 침식 후 팽창)
        # opening = cv2.morphologyEx(thresh_otsu_image, cv2.MORPH_OPEN, kernel)


        # processed_imgs.update({"thresh_bin_image": thresh_bin_image})
        # processed_imgs.update({"dilation": dilation})

        # processed_imgs.update({"morphology": bin_image})

        contours, _ = cv2.findContours(at_morp_thesh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 각 윤곽선을 확인
        for contour in contours:
            # 윤곽선 근사화로 모양 확인
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            # 원 모양에 가까운지 판단 (다각형의 꼭짓점이 많을수록 원에 가까움)
            if len(approx) > 8:  # 꼭짓점이 8개 이상이면 원으로 간주
                # 원형의 윤곽선을 그리기
                cv2.drawContours(at_morp_thesh_image, [approx], 0, (0, 255, 0), 3)

        # '닫기(closing)' 연산을 통해 열린 원 메우기
        kernel = np.ones((5,5), np.uint8)
        closed = cv2.morphologyEx(at_morp_thesh_image, cv2.MORPH_CLOSE, kernel)

        processed_imgs.update({"closed": closed})

    except Exception as e:
        print(f"Error: {e}")
        cv2.putText(src, f"Error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    inversed = cv2.bitwise_not(adjusted)

    detect_src = at_morp_thesh_image
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

            if detect_src[y_center, x_center] == 255: # if the center of the circle is black,
                cv2.circle(overlay_src, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 5)
                valid_cnt += 1
            else:
                cv2.circle(overlay_src, (int(i[0]), int(i[1])), int(i[2]), (0, 0, 255), 2)
            


        det_cnt = valid_cnt
        ref_cnt = estimate_cnt(imgs[img_idx], det_cnt)


        cv2.putText(overlay_src, f"{det_cnt}/{ref_cnt}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 5)
        cv2.putText(overlay_src, f"Acc: {(det_cnt/ref_cnt)*100:.2f}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 5)
    else:
        cv2.putText(overlay_src, "No circles detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)

    processed_imgs.update({"hough_circles": overlay_src})

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

        processed_imgs[k] = cv2.putText(processed_imgs[k], k, (50, processed_imgs[k].shape[1]-100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
        stacked.append(processed_imgs[k])

    stacked = cv2.hconcat(stacked)

    # save img
    cv2.imwrite("output.jpg", stacked)



    cv2.imshow("image", cv2.resize(stacked, (0, 0), fx=0.5, fy=0.5))

    if cv2.waitKey(200) & 0xFF == ord('q') :
        break

cv2.destroyAllWindows()