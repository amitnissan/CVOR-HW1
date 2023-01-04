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
from predict import *
from tools import *


def video(video_path, model, output_path, labels_path='../resources/classes.names'):
    cap = cv.VideoCapture(video_path)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv.CAP_PROP_FPS))
    size = (frame_width, frame_height)
    result = cv.VideoWriter(output_path,
                            cv.VideoWriter_fourcc(*'mp4v'),
                            video_fps, size)
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            result.write(predict(frame, model, labels_path))

            # # Press Q on keyboard to  exit
            if cv.waitKey(33) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    result.release()


if __name__ == '__main__':
    model = load_model()
    video('/home/student/notebooks/models/P019_tissue1.wm1', model)
