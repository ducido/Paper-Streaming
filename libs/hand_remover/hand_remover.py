from sklearn.cluster import KMeans
from collections import Counter
import cv2
import numpy as np
from threading import Thread, Lock
import time

class DominantColor(object):
    def __init__(self, name='thread dominant color'):
        self.image = None 
        self.dominant_color = None
        self.stopped = False

        self.thread = Thread(target=self.update, name=name, args=())
        self.thread.daemon = True

    def update(self):
        while not self.stopped:
            if self.image is None:
                continue
            self.dominant_color = self.__update_dominant_color(self.image)
            # time.sleep(1)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped = True

    def update_image(self, image):
        self.image = image

    def get_color(self):
        return self.dominant_color

    def __update_dominant_color(self, image, k=2):
        #reshape the image to be a list of pixels
        image = cv2.resize(image, (128, 128))
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        #cluster and assign labels to the pixels 
        clt = KMeans(n_clusters = k)
        labels = clt.fit_predict(image)

        #count labels to find most popular
        label_counts = Counter(labels)

        #subset out most popular centroid
        dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
        dominant_color = list(dominant_color)
        
        return dominant_color


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
        m_dilate = cv2.dilate(m*255, self.kernel, iterations=20)


        # cv2.imshow('m', m*255)
        # print('--------', np.sum(m))
        if np.sum(m) < 30000:
            m = np.zeros_like(m)

        # self.fill_hand(m)

        # print(m_dilate.max())
        
        return m, m_dilate
    
    def fill_hand(self, img):
        img = img.astype('uint8')
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        max_area = 0
        biggest = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10000:
                # peri = cv2.arcLength(cnt, True)
                # approx = cv2.approxPolyDP(cnt, 0.01*peri, True)                   

                print(cnt)