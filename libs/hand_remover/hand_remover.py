from sklearn.cluster import KMeans
from collections import Counter
import cv2
import numpy as np
from threading import Thread, Lock
import time

class HandRemover(object):

    def __init__(self):
        self.lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        self.upper_HSV_values = np.array([25, 255, 255], dtype="uint8")

        self.lower_YCbCr_values = np.array((0, 138, 67), dtype = "uint8")
        self.upper_YCbCr_values = np.array((255, 173, 133), dtype = "uint8")

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))

        self.background = None
        self.image_list = []
        self.check_hand = False
        
    def process(self, image, is_cropped):
        if is_cropped == False:
            return image

        if self.background is None:
            self.background = image

        self.image_list.append(image)
        hand_mask = None
        try:
            hand_mask, hand_dilate = self.__get_hand_mask(image)
        except:
            pass


        if hand_mask is not None:
            if np.sum(hand_mask) > 0:
                self.image_list.pop(-1)

            background_area = np.where(hand_mask==0)
            hand_area = np.where(hand_dilate==255)

            if len(self.image_list) > 10:
                self.background[background_area] = image[background_area]
                self.background[hand_area] = self.image_list[-10][hand_area]

        return self.background
    
    def __get_hand_mask(self, image):
        HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        YCbCr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

        mask_YCbCr = cv2.inRange(YCbCr_image, self.lower_YCbCr_values, self.upper_YCbCr_values)
        mask_HSV = cv2.inRange(HSV_image, self.lower_HSV_values, self.upper_HSV_values)

        foreground_mask = cv2.add(mask_HSV, mask_YCbCr)
        # Morphological operations
        background_mask = ~foreground_mask
        # background_mask = cv2.erode(background_mask, self.kernel, iterations=50)
        background_mask[background_mask==255] = 128

        marker = cv2.add(foreground_mask, background_mask)
        marker = np.int32(marker)
        cv2.watershed(image, marker)

        m = cv2.convertScaleAbs(foreground_mask)
        m[m < 200] = 0
        m[m >= 200] = 1
        m_dilate = cv2.dilate(m*255, self.kernel, iterations=10)


        # cv2.imshow('m', m*255)
        # print('--------', np.sum(m))
        if np.sum(m) < 5000:
            m = np.zeros_like(m)
        
        return m, m_dilate