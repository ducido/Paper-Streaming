from flask import Flask, request, render_template
from flask_socketio import SocketIO
from flask_cors import CORS
import cv2
import numpy as np
from libs.config import *
import os, sys
sys.path.append("/home/saplab/Documents/paper_stream/libs")
from libs.hand_remover.hand_remover import HandRemover
from libs.paper_processor.paper_processor import PaperProcessor
from filter import FilterImage
import filter
import base64
from engineio.payload import Payload
from PIL import Image, ImageEnhance
# from gevent import monkey
import onnxruntime
# Delete later
import time
#==============================
size= 144
class paper_segment:
    def __init__(self):
        self.model = onnxruntime.InferenceSession("pretrained/psp_0.98_only.onnx",providers=['CUDAExecutionProvider'])
        self.input_name = self.model.get_inputs()[0].name
        print(self.model.get_inputs()[0])

    def preprocess(self, image):
        image = image[:,:,::-1]
        image = cv2.resize(image, (size, size)).reshape(1,size,size,3)
        return image.astype('float32')/255
    
    def predict(self, image):
        image = self.preprocess(image)
        result = self.model.run(None, {self.input_name: image})
        # pred = result[0].reshape(size, size, 2)[:,:,1:2]
        pred = result[0].reshape(size, size)

        # pred[pred < 0.05] == 0
        pred[pred >= 0.0] == 1
        return pred
model = paper_segment()
# monkey.patch_all()

Payload.max_decode_packets = 50
app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent')
#allow all origin
CORS(app)

# Initialization
paper_processor = PaperProcessor()
hand_remover = HandRemover()
filter_image = FilterImage()

@app.route('/')
def index():
    return render_template('index.html')
@socketio.on('video_frame')
def handle_frame(data):
    print('receive')
    start = time.time()
    encoded_data = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the frame
    image = frame.copy()
    # image = cv2.transpose(image)
    image = cv2.flip(image, 0)
    image = cv2.flip(image, 1)
    draw = image.copy()
    # processed_image = image.copy()
    pred = model.predict(image)
    # cv2.imshow('pred', pred)
    # cv2.waitKey(1)
    is_cropped, processed_image, draw = paper_processor.get_paper_image(image, pred, draw=draw)
    processed_image = hand_remover.process(processed_image, is_cropped=is_cropped)
    # cv2.imshow('processed_image', processed_image)
    # cv2.waitKey(1)

    processed_image = filter.remove_shadow(processed_image)
    # processed_image = filter_image.run(processed_image, hand_area)
    # cv2.imshow('processed_image', processed_image)
    # cv2.waitKey(1)


    # Convert the processed image to base64 and send it back to the client
    ret, jpeg = cv2.imencode('.jpg', processed_image)
    processed_encoded = base64.b64encode(jpeg.tobytes()).decode('utf-8')
    socketio.emit('processed_frame', processed_encoded)
    end = time.time()
    print(end-start)

if __name__ == '__main__':
    socketio.run(app, debug=True,host='0.0.0.0')

