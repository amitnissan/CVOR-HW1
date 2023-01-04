import cv2 as cv
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import bbox_visualizer as bbv
import torch
from yolov7_package import Yolov7Detector
import sys
from tools import *


def predict(image, model, labels_path='../resources/classes.names'):
    labels = load_labels(labels_path)
    try:
        classes, boxes, scores = model.detect(image)
    except:
        image = cv.imread(image)
        classes, boxes, scores = model.detect(image)
    boxes[0] = [[int(x) for x in box] for box in boxes[0]]
    classes[0] = [[labels[class_id] for class_id in label] for label in classes]

    image = bbv.draw_multiple_rectangles(image, boxes[0], bbox_color=(255, 0, 0))
    image = bbv.add_multiple_labels(image, classes[0][0], boxes[0], text_bg_color=(255, 0, 0))

    return image
