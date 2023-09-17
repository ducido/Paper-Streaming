import argparse
import os
import sys
import time
import urllib
from distutils import util
from threading import Lock
import threading

import cv2
import numpy as np
# from PyQt5 import QtWidgets, uic
# from PyQt5 import QtCore, QtGui

import _thread
from libs.config import *
from libs.hand_remover.hand_remover import HandRemover
from libs.paper_processor.paper_processor import PaperProcessor
from libs.stroke_filter.stroke_filter import StrokeFilter
import filter
import onnxruntime

OUTPUT_SIMULATED_CAMERA = False
size = 144


class paper_segment:
    def __init__(self):
        self.model = onnxruntime.InferenceSession("pretrained/psp_0.98_only.onnx",providers=['CUDAExecutionProvider'])
        self.input_name = self.model.get_inputs()[0].name
        print(self.model.get_inputs()[0])

    def preprocess(self, image):
        image = cv2.resize(image, (size, size)).reshape(1,size,size,3)
        return image.astype('float32') / 255
    
    def predict(self, image):
        image = self.preprocess(image)
        result = self.model.run(None, {self.input_name: image})
        pred = result[0].reshape(size, size)
        # pred = result[0].reshape(size, size, 2)[:,:,1:2]

        # pred[pred < 0.5] == 0
        # pred[pred >= 0.5] == 1
        return pred
    
paper_processor = PaperProcessor()
hand_remover = HandRemover()
stroke_filter = StrokeFilter()
model = paper_segment()
frame = None
    
# camera reading thread
new_camera_url = "4.mp4"
cap = cv2.VideoCapture(new_camera_url)
# processing thread    
def processing_thread():
    index = 0
    while(True):
        s = time.time()
        isframe, frame = cap.read()
        if isframe == False:
            break

        image = cv2.transpose(frame)
        image = cv2.flip(image, 0)
        draw = image.copy()

        # get paper image
        pred = model.predict(image)

        cv2.imshow('pred', pred)
        cv2.waitKey(1)
        is_cropped, processed_image, draw = paper_processor.get_paper_image(image, pred, draw= draw)
        # cv2.imshow('crop', processed_image)
        # cv2.waitKey(1)
        # remove hand
        processed_image = hand_remover.process(processed_image, is_cropped=is_cropped)
        cv2.imshow('remove hand', processed_image)
        cv2.waitKey(1)
        
        # post processing
        processed_image = filter.remove_shadow(processed_image)
        cv2.imshow(' filter', processed_image)
        cv2.waitKey(1)
        e = time.time()
        # print(e-s)

processing_thread()