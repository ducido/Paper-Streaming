import argparse
import os
import sys
import time
import urllib
from distutils import util
from threading import Lock

import cv2
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5 import QtCore, QtGui

import _thread
from libs.config import *
from libs.hand_remover.hand_remover import HandRemover
from libs.paper_processor.paper_processor import PaperProcessor
from libs.stroke_filter.stroke_filter import StrokeFilter
from libs.utils.common import *
from libs.utils.ui_utils import *
# from libs.webcam import pyfakewebcam

paper_processor = PaperProcessor(REFERENCE_ARUCO_IMAGE_PATH, aruco_remove_mask_path=REFERENCE_ARUCO_REMOVE_IMAGE_PATH, smooth=True, debug=False, output_video_path=None)
hand_remover = HandRemover()
stroke_filter = StrokeFilter()
      
def processing_thread(image_path):
    # global frame, frame_mutex
    image = cv2.imread(image_path)

    is_cropped, processed_image = paper_processor.get_paper_image(image)
    cv2.imshow('process', processed_image)
    cv2.waitKey(0)

    
    # remove hand
    processed_image = hand_remover.process(processed_image, is_cropped=is_cropped)
    cv2.imshow('',processed_image)
    cv2.waitKey(0)
    
    # post processing
    processed_image = stroke_filter.process(processed_image)
    cv2.imshow('',processed_image)
    cv2.waitKey(0)


image_path = 'test_data/1.png'
processing_thread(image_path)

