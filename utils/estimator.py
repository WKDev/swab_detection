import os, sys, time, glob
import cv2




def get_ref_cnt(src):
    filename = os.path.basename(src)
    ref_ = int(filename.split('_')[3].split('.')[0])
    return ref_

def estimate_cnt(src, detection_cnt):
    print(f"src: {src}")
    ref_cnt = get_ref_cnt(src)
    print(f"det: {detection_cnt}, ref: {ref_cnt}, accuracy: {detection_cnt/ref_cnt*100:.2f}%")

    return ref_cnt

def hough_circles_operation(image, min_dist,param1, param2, min_radius, max_radius):
    hough_result = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, minDist=min_dist,
                                    param1=param1, param2=param2, 
                                    minRadius=min_radius, maxRadius=max_radius)
    
    return hough_result
