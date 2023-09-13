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
from PyQt5 import QtWidgets, uic
from PyQt5 import QtCore, QtGui

import _thread
from libs.config import *
from libs.hand_remover.hand_remover import HandRemover
from libs.paper_processor.paper_processor import PaperProcessor
from libs.stroke_filter.stroke_filter import StrokeFilter
import filter

OUTPUT_SIMULATED_CAMERA = False

paper_processor = PaperProcessor()
hand_remover = HandRemover()
stroke_filter = StrokeFilter()

    
frame = None
    
new_camera_url = "4.mp4"
cap = cv2.VideoCapture(new_camera_url)

def camera_reading_thread():
    global frame
    index= 0
    while True:
        isframe, frame = cap.read()
        if isframe is None:
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(10)

def processing_thread():
    global frame
    index = 0
    while(True):
        image = None
        try:
            image = frame.copy()
            image = cv2.transpose(image)
            image = cv2.flip(image, 0)
            draw = image.copy()
        except:
            pass
        if image is None:
            break

        # cv2.imwrite('test_data/hand.png', image)
        
        is_cropped, processed_image, draw = paper_processor.get_paper_image(image, draw= draw)
        #cv2.imwrite("test_data/0_warp.png", processed_image)
        folder_path = "test_data"  
        extension = ".png"  

        existing_files = [f for f in os.listdir(folder_path) if f.endswith(extension)]
        max_index = max([int(os.path.splitext(f)[0]) for f in existing_files], default=-1)

        new_filename = os.path.join(folder_path, f"{max_index + 1}{extension}")

        cv2.imwrite(new_filename, processed_image)
        processed_image = hand_remover.process(processed_image, is_cropped=is_cropped)
        # cv2.imwrite("test_data/remove_hand.png", processed_image)
        cv2.imshow('remove hand', processed_image)
        cv2.waitKey(1)
        
        processed_image = filter.thresh_image(processed_image)
        # cv2.imwrite("test_data/filter.png", processed_image)
        # cv2.imshow(' filter', processed_image)
        cv2.waitKey(1)
        
thread1 = threading.Thread(target=camera_reading_thread)
thread1.start()
time.sleep(1)
thread2 = threading.Thread(target=processing_thread)
thread2.start()