import cv2 as cv
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import bbox_visualizer as bbv
import torch
from yolov7_package import Yolov7Detector
import sys


def predict(image):
    labels = open('../resources/classes.names').read().strip().split('\n')
    model = Yolov7Detector(weights='../resources/best.pt', img_size=[416, 416],
                           classes='../resources/classes.names')
    classes, boxes, scores = model.detect(image)
    boxes[0] = [[int(x) for x in box] for box in boxes[0]]
    classes[0] = [[labels[class_id] for class_id in label] for label in classes]

    image = bbv.draw_multiple_rectangles(image, boxes[0], bbox_color=(255, 0, 0))
    image = bbv.add_multiple_labels(image, classes[0][0], boxes[0], text_bg_color=(255, 0, 0))

    return image


if __name__ == '__main__':
    print('start')
    img = predict(plt.imread('/home/student/notebooks/models/P016_balloon1_9.jpg'))
    plt.imsave('/home/student/notebooks/models/result.jpg', img)
