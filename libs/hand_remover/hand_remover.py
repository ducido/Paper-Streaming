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
                self.check_hand = True
                self.image_list.pop(-1)
            else:
                self.check_hand = False
            
            background_area = np.where(hand_mask==0)
            hand_area = np.where(hand_dilate==255)

            if len(self.image_list) > 5:
                # print(1)
                self.background[background_area] = image[background_area]
                self.background[hand_area] = self.image_list[-5][hand_area]

        return self.background

    def __remove_noise_border(self, image):
        h, w = image.shape[:2]
        color = (255, 255, 255) 
        thickness = 2
        image = cv2.rectangle(image, (0, 0), (w, h), color, thickness)

        return image  


    def __flood_fill(self, image, hand_mask):
        image = self.__remove_noise_border(image)
        test = cv2.Canny(image, 90, 90)
        test = cv2.dilate(test, self.kernel, iterations=4)

        # cv2.imshow('test', test)
        # cv2.waitKey(1)
        # Copy the thresholded image
        im_floodfill = test.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = test.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        # cv2.floodFill(im_floodfill, mask, (30, 20), 255)
        # cv2.floodFill(im_floodfill, mask, (w-10,20), 255)
        cv2.floodFill(im_floodfill, mask, (20,h - 20), 255)

        # cv2.imshow("im_floodfill", im_floodfill)
        # cv2.waitKey(1)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # cv2.imshow("im_floodfill_inv", im_floodfill_inv)
        # cv2.waitKey(1)

        # Combine the two images to get the foreground.
        im_out = test | im_floodfill_inv

    
        image = cv2.bitwise_and(image, image, mask=im_out)


        color = self.dominant_color.get_color()
        if color is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("gray", gray)
            

            # print(hand_mask.max())
            hand_area = np.where(hand_mask == 1)
            hand_mask[hand_area] = im_out[hand_area]//255
            hand_mask = cv2.dilate(hand_mask, self.kernel, iterations=20)
            # cv2.imshow("hand_mask", hand_mask*255)
            # cv2.waitKey(1)

            image[im_out == 0] = color
            # image[gray == 255] = color
            # image[hand_mask == 1] = color

        # hi = self.process_imout(image)
        return image
    
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
        m_dilate = cv2.dilate(m*255, self.kernel, iterations=40)

        # print(np.sum(m))
        if np.sum(m) < 10000:
            m = np.zeros_like(m)

        # self.fill_hand(m)
        # cv2.imshow('m', m)

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