import cv2
import numpy as np
import os, time, sys, glob
from datetime import datetime
import threading

from utils.estimator import estimate_cnt
from utils.misc import Perf,Text

import yaml

from utils.preprocess import adjust_image

DETECTION_TIMEOUT = 0.5


def load_config():
    if os.path.exists('config_simple.yaml'):
        with open('config_simple.yaml', 'r') as file:
            return yaml.safe_load(file)
        
def save_config():
    saved_config = {}
    for key, value in config.items():
        saved_config[key] = value[2]

    with open('config_simple.yaml', 'w') as file:
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

pf = Perf()

config = {
    "img_idx": [[0, len(imgs)-1], 0, 0],

    "shadows": [[0, 200], 0, 0],  # range, default, current_value
    "highlights": [[0, 200], 0, 0],  # range, default, current_value
    "brilliance": [[0, 200], 0, 0],  # range, default, current_value
    "exposure": [[0, 200], 0, 0],  # range, default, current_value
    "contrast": [[0, 200], 0, 0],  # range, default, current_value
    "brightness": [[0, 200], 0, 0],  # range, default, current_value
    "black_point": [[0, 200], 0, 0],  # range, default, current_value
    "sharpness": [[0, 200], 0, 0],  # range, default, current_value
    "noise_reduction": [[0, 200], 0, 0],  # range, default, current_value

    "min_radius": [[20,50], 30, 30],  # range, default, current_value
    "max_radius": [[20,50], 40, 40],  # range, default, current_value
    "min_dist": [[50, 100], 75, 75],       # range, default, current_value
    "param1": [[0,100], 10, 10],      # range, default, current_value
    "param2": [[0,100], 20, 20],       # range, default, current_value
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

    try:

        img_idx = config["img_idx"][2]
        min_dist = config["min_dist"][2]
        param1 = config["param1"][2]
        param2 = config["param2"][2]
        min_radius = config["min_radius"][2]
        max_radius = config["max_radius"][2]

        shadows = config["shadows"][2]- 100
        highlights = config["highlights"][2]- 100
        brilliance = config["brilliance"][2]- 100
        exposure = config["exposure"][2]- 100
        contrast = config["contrast"][2]- 100
        brightness = config["brightness"][2]- 100
        black_point = config["black_point"][2]- 100
        sharpness = config["sharpness"][2]- 100
        noise_reduction = config["noise_reduction"][2]- 100
        
        # Use the current values from the config dictionaries
        img_idx = config["img_idx"][2]


        # Load image
        src = cv2.imread(imgs[img_idx], cv2.IMREAD_COLOR)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


        adjusted = adjust_image(src, shadows, highlights, brilliance, exposure, contrast, brightness, black_point, sharpness, noise_reduction)

        processed_imgs.update({"adjusted": adjusted})



        
    except Exception as e:
        print(f"Error: {e}")
        cv2.putText(src, f"Error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    detect_src = adjusted
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

            if True or filter_src[y_center-1, x_center-1] == 255 : # if the center of the circle is black,
                cv2.circle(overlay_src, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
                valid_cnt += 1
            else:
                cv2.circle(overlay_src, (int(i[0]), int(i[1])), int(i[2]), (0, 0, 255), 2)
            


        det_cnt = valid_cnt
        ref_cnt = estimate_cnt(imgs[img_idx], det_cnt)


        cv2.putText(overlay_src, f"{det_cnt}/{ref_cnt}, Acc: {(det_cnt/ref_cnt)*100:.2f}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 255), 3)
    else:
        cv2.putText(overlay_src, "No circles detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)

    processed_imgs.update({"hough_circles": overlay_src})

    stacked = []

    for k, v in processed_imgs.items():
        if len(v.shape) == 2:
            processed_imgs[k] = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

        processed_imgs[k] = cv2.putText(processed_imgs[k], k, (50, processed_imgs[k].shape[1]-300), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
        stacked.append(processed_imgs[k])

    stacked = cv2.hconcat(stacked)

    # save img
    cv2.imwrite("output.jpg", stacked)



    cv2.imshow("ret", cv2.resize(stacked, (0, 0), fx=0.5, fy=0.5))

    if cv2.waitKey(200) & 0xFF == ord('q') :
        break

cv2.destroyAllWindows()