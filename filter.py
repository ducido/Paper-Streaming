import cv2
import numpy as np

def thresh_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.bilateralFilter(image,9,8,8)
    th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,45,2)

    return th3


def remove_shadow(img):
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((17,17), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 9)
        
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    processed_image = cv2.merge(result_planes)
    return processed_image

def add_noise_border(image):
    h, w = image.shape[:2]
    color = (0, 0, 0) 
    thickness = 1
    image = cv2.rectangle(image, (0, 0), (w, h), color, thickness)

    return image  