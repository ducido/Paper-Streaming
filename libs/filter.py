import cv2
import numpy as np
import time

class FilterImage:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))

    def run(self, image, hand_area):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        copy = image.copy()

        image = cv2.GaussianBlur(image, (7,7),0)

        canny = cv2.Canny(image, 15,11)
        canny = cv2.dilate(canny, self.kernel, iterations=10)

        out = cv2.bitwise_and(copy, copy, mask= canny)

        h = out.shape[0]
        w = out.shape[1]
        mask = np.zeros((h+2, w+2), np.uint8)

        cv2.floodFill(out, mask, (5, 5), 255)
        cv2.floodFill(out, mask, (w-5, 5), 255)
        cv2.floodFill(out, mask, (w-20, h-20), 255)

        cv2.floodFill(out, mask, (w//3, h-5), 255)
        cv2.floodFill(out, mask, (w-w//3, h-5), 255)
        cv2.floodFill(out, mask, (w//2, 2), 255)
        cv2.floodFill(out, mask, (5, h - 5), 255)

        return out


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