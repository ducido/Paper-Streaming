import cv2
import numpy as np
import time
from ..utils.common import *
import threading


model = cv2.dnn.readNetFromONNX("UNET.050.onnx")

class PaperProcessor():
    points = []
    pred = None
    WIDTH = 128
    HEIGHT = 128

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

        array_axis0 = myPointsNew[:, 0] * (self.frame.shape[1]/128)  # Multiply only the first column

        # Multiply along axis 1 with 'b'
        array_axis1 = myPointsNew[:, 1] * (self.frame.shape[0]/128)  # Multiply only the second column

        # Create the output array
        myPointsNew = np.column_stack((array_axis0, array_axis1))
        
        return myPointsNew


    def getContours(self, img, draw):
        img = img.astype('uint8')
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        max_area = 0
        biggest = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)                   
                if area > max_area and len(approx) in [4]:
                    max_area = area
                    biggest = approx
                    self.points.append(biggest)

        if self.points:
            biggest = self.reorder(self.points[-1])
            biggest = np.round(biggest).astype('int').reshape(4,1,2)

        cv2.drawContours(draw, biggest, -1, (255,0,0), 10)
        # cv2.circle(draw, biggest[0][0], 10, (255,0,0), -1)  
        # cv2.circle(draw, biggest[0][1], 10, (255,0,0), -1)  
        # cv2.circle(draw, biggest[2][0], 10, (255,0,0), -1)  
        # cv2.circle(draw, biggest[3][0], 10, (255,0,0), -1)  
   
        return biggest

    def get_warp(self, img, biggest):
        biggest = biggest.reshape(4,2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0], [img.shape[1],0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])
        # print(pts1)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
        # imgOutput = imgOutput[150:imgOutput.shape[0],25:imgOutput.shape[1]-45]
        # imgOutput = cv2.resize(imgOutput,(img.shape[1], img.shape[0]))
        return imgOutput
    
    def detect_paper(self):
        frame = cv2.resize(self.frame, (self.WIDTH, self.HEIGHT)) / 127.5 - 1
        model.setInput(frame.reshape(1, self.HEIGHT, self.WIDTH, 3))
        self.pred = model.forward()
        self.pred = self.pred[:,:,:,1:2].reshape(self.HEIGHT, self.WIDTH, 1)

        self.pred[self.pred >= 0.05] = 1
        self.pred[self.pred < 0.05] = 0


    def get_paper_image(self, frame, draw):
        """Transform image"""
        self.frame = frame
        thread_detect_paper = threading.Thread(target= self.detect_paper)
        thread_detect_paper.start()

        # try:
        #     cv2.imshow("pred", self.pred)
        #     cv2.waitKey(1)
        # except: pass

        img_warp = frame
        if self.pred is not None:
            biggest = self.getContours(self.pred, draw)
            if biggest is not None:
                img_warp = self.get_warp(frame, biggest)
                return True, img_warp, draw

        return False, frame, draw
        
    def _remove_aruco(self, image):
        
        if self.aruco_remove_mask is not None:
            image[self.aruco_remove_mask > 0] = [255,255,255]
            
        return image
        
        
    def _update_transform_matrices(self, gray):
        """Find aruco and update transformation matrices"""
        
        # detect aruco markers in gray frame
        res_corners, res_ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters = self.parameters)
        self.res_corners = res_corners
        self.res_ids = res_ids

        # if markers were not detected
        if res_ids is None:
            return False

        # find which markers in frame match those in reference image
        idx = which(self.ref_ids, res_ids)
        
        # if # of detected points is too small => ignore the result
        if len(idx) <= 2:
            return False

        # flatten the array of corners in the frame and reference image
        these_res_corners = np.concatenate(res_corners, axis = 1)
        these_ref_corners = np.concatenate([self.ref_corners[x] for x in idx], axis = 1)

        # estimate homography matrix
        h, s = cv2.findHomography(these_ref_corners, these_res_corners, cv2.RANSAC, 10.0)

        # if we want smoothing
        if self.smooth:
            self.h_array.append(h)
            self.M = np.mean(self.h_array, axis = 0)
        else:
            self.M = h

        # transform the rectangle using the homography matrix
        new_rect = cv2.perspectiveTransform(self.rect, self.M, (gray.shape[1],gray.shape[0]))

        self.M_inv, s = cv2.findHomography(these_res_corners, these_ref_corners, cv2.RANSAC, 10.0)

        return True