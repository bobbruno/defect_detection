#!/usr/bin/env python
# coding: utf-8

from dlr import DLRModel
import numpy as np
import time
import os.path
import cv2
import mxnet as mx
import random

current_milli_time = lambda: int(round(time.time() * 1000))

resolution=512
dtype = 'float32'

base = "ssd_mobilenet"
print "Model: " + base

# DLR loads model by the folder containing the model files
model = DLRModel(base, 'gpu') 

def detect_obj(x):
# set inputs
    input_data = {'data': x}
    output = model.run(input_data)
    output = output[0]
    return output[0] 

dtype = 'float32'

CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size = (len(CLASSES), 3))

def display(img, out):
    for det in out:
        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < 0.4:
            continue
        xmin, ymin, xmax, ymax = [int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
        label = CLASSES[cid]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), COLORS[cid], 2)
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        cv2.putText(img, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cid], 2)
    cv2.imshow("SageMaker Neo runtime | SSD", img)
    cv2.waitKey(1)

def norm_image_cv2(img):
    img_data = cv2.resize(img, (512, 512))
    img_data = img_data[:, :, (2, 1, 0)].astype(np.float32)
    img_data -= np.array([123, 117, 104])
    img_data = np.transpose(np.array(img_data), (2, 0, 1))
    x = np.expand_dims(img_data, axis=0)
    return x

# Iterate over images, detect objects, display images and objects boudaries
vidcap = cv2.VideoCapture(0)
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
scales = [640, 480] * 2

while True:
    success,image = vidcap.read()
    every_n_frame = 1
    cnt = 0
    t1 = current_milli_time()
    while success:
        success, img640 = vidcap.read()
        if success:
            img = cv2.resize(img640, (resolution, resolution))
            x =  norm_image_cv2(img)
            out = detect_obj(x)
            display(img640, out)
            cnt += 1
            if cnt % 10 == 0:
              t2 = current_milli_time()
              tt = (t2 - t1) / 1000.0
              t1 = t2
              fps = cnt / tt
              cnt = 0
              print "FPS: {0:.2f}".format(fps)
            



