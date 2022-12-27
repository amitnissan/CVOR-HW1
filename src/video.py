import cv2 as cv
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import bbox_visualizer as bbv
import torch
from yolov7_package import Yolov7Detector
import sys
import argparse
import predict


def video(video_path):
    labels = open('../resources/classes.names').read().strip().split('\n')
    model = Yolov7Detector(weights='../resources/best.pt', img_size=[416, 416],
                           classes='../resources/classes.names')
    cap = cv.VideoCapture(video_path)
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # add bounding boxes
            # bbox = [xmin, ymin, xmax, ymax]
            classes, boxes, scores = model.detect(frame)
            boxes[0] = [[int(x) for x in box] for box in boxes[0]]
            classes[0] = [[labels[class_id] for class_id in label] for label in classes]

            frame = bbv.draw_multiple_rectangles(frame, boxes[0], bbox_color=(255, 0, 0))
            frame = bbv.add_multiple_labels(frame, classes[0][0], boxes[0], text_bg_color=(255, 0, 0))

            # Display the resulting frame
            cv.imshow('Frame', frame)

            # # Press Q on keyboard to  exit
            if cv.waitKey(33) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    video('/home/student/notebooks/models/P019_tissue1.wm1')
