import cv2
import numpy as np
import time
from ..utils.common import *

class PaperProcessor():
    points = []
    pred = None
    size = 144
    center = []
    biggest_list = []

    def reorder(self, myPoints):
        myPoints = myPoints.reshape((4,2))
        myPointsNew = np.zeros((4,2),np.int32)
        add = myPoints.sum(1)
        #print("add", add)             
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints,axis=1)
        myPointsNew[1]= myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
        #print("NewPoints",myPointsNew)

        array_axis0 = myPointsNew[:, 0] * (self.frame.shape[1]/self.size)  # Multiply only the first column

        # Multiply along axis 1 with 'b'
        array_axis1 = myPointsNew[:, 1] * (self.frame.shape[0]/self.size)  # Multiply only the second column

        # Create the output array
        myPointsNew = np.column_stack((array_axis0, array_axis1))
        
        return myPointsNew


    def getContours(self, img, draw):
        img = img.astype('uint8')
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        max_area = 0
        biggest = None
        # cv2.drawContours(draw, contours, -1, (255,0,0), 10)
        # cv2.imshow('draw', draw)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # print('-----------------', area)
            if area > 5000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)                   
                if area > max_area and len(approx) in [4]:
                    max_area = area
                    biggest = approx
                    self.points.append(biggest)

        if self.points:
            cx, cy = 0,0
            for i in range(4):
                cx += self.points[-1][i][0][0]
                cy += self.points[-1][i][0][1]

            if self.center:
                a = 10
                if abs(cx - self.center[-1][0]) < a and abs(cy - self.center[-1][1]) < a:
                    return self.biggest_list[-1]
                else:
                    print(2)
                    self.center.append([cx,cy])
                    biggest = self.reorder(self.points[-1])
                    biggest = np.round(biggest).astype('int').reshape(4,1,2)
                    self.biggest_list.append(biggest)
            else:
                self.center.append([cx,cy])
                biggest = self.reorder(self.points[-1])
                biggest = np.round(biggest).astype('int').reshape(4,1,2)
                self.biggest_list.append(biggest)
            # biggest = self.reorder(self.points[-1])
            # biggest = np.round(biggest).astype('int').reshape(4,1,2)
            # self.biggest_list.append(biggest)
        

        return biggest

    def get_warp(self, img, biggest):
        biggest = biggest.reshape(4,2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0], [img.shape[1],0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])
        # print(pts1)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))

        a = 40
        h = imgOutput.shape[0] - a
        w = imgOutput.shape[1] - a
        imgOutput = imgOutput[a:h, a:w]

        # h, w = imgOutput.shape[:2]
        # imgOutput = cv2.rectangle(imgOutput, (0, 0), (w, h), (255, 255, 255), 10)
        return imgOutput


    def get_paper_image(self, frame, pred, draw= None):
        """Transform image"""
        self.frame = frame
        img_warp = frame
        if pred is not None:
            biggest = self.getContours(pred*255, draw)

            if biggest is not None:
                img_warp = self.get_warp(frame, biggest)
                return True, img_warp, draw

        return False, frame, draw