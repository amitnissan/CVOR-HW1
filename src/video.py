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
from tqdm import tqdm
from predict import *
from tools import *


def count_frames(path, override=False):
    # grab a pointer to the video file and initialize the total
    # number of frames read
    video = cv.VideoCapture(path)
    total = 0
    # if the override flag is passed in, revert to the manual
    # method of counting frames
    if override:
        total = count_frames_manual(video)
    # otherwise, let's try the fast way first
    else:
        # let's try to determine the number of frames in a video
        # via video properties; this method can be very buggy
        # and might throw an error based on your OpenCV version
        # or may fail entirely based on your which video codecs
        # you have installed
        try:
            # check if we are using OpenCV 3
            # if is_cv3():
            #     total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            # # otherwise, we are using OpenCV 2.4
            # else:
            total = int(video.get(cv.cv.CV_CAP_PROP_FRAME_COUNT))
        # uh-oh, we got an error -- revert to counting manually
        except:
            total = count_frames_manual(video)
    # release the video file pointer
    video.release()
    # return the total number of frames in the video
    return total


def count_frames_manual(video):
    # initialize the total number of frames read
    total = 0
    # loop over the frames of the video
    while True:
        # grab the current frame
        (grabbed, frame) = video.read()

        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break
        # increment the total number of frames read
        total += 1
    # return the total number of frames in the video file
    return total


def lpf_for_object_detection(boxes, classes, total_frames, frame_ix, frame_itr, num_of_history_frames=11):
    # if there's enough frames to evaluate back and forth in time
    if num_of_history_frames - 1 <= frame_ix < total_frames:
        frame_to_check = int(
            (num_of_history_frames - 1) / 2)  # (num_of_history_frames-1)/2 of past frames, and same for future frames
        classes, boxes, changed_frame_flag = cmp2history(boxes, classes, frame_to_check, frame_ix,
                                                         num_of_history_frames)
    else:
        frame_to_check = frame_itr
        changed_frame_flag = False
    return classes, boxes, frame_to_check, changed_frame_flag


def cmp2history(boxes, classes, cur_frame_ix, frame_ix, num_of_history_frames=11):
    global future_frame_flag, cur_eq_past
    changed_frame_flag = False  # DEBUG
    # check if past and future frames are equal
    for i in range(cur_frame_ix):
        if classes[cur_frame_ix] == classes[i] or classes[cur_frame_ix] == list(reversed(classes[i])):
            cur_eq_past = True
        else:
            cur_eq_past = False
            # print(f'{frame_ix}: classes[{cur_frame_ix}]={classes[cur_frame_ix]} != classes[{i}]={classes[i]}') # DEBUG
        if (classes[cur_frame_ix] == classes[num_of_history_frames - 1 - i] or classes[cur_frame_ix] == list(
                reversed(classes[num_of_history_frames - 1 - i]))):
            cur_eq_future = True
        else:
            cur_eq_future = False

    if not (cur_eq_past) and not (cur_eq_future):
        # print(f'{frame_ix}: old classes = {classes[cur_frame_ix]}') # DEBUG
        # print(f'{frame_ix}: old boxes   = {boxes[cur_frame_ix]}') # DEBUG
        boxes.pop(cur_frame_ix)
        classes.pop(cur_frame_ix)
        # find a good example from the past
        for i in range(cur_frame_ix):
            if len(classes[num_of_history_frames - 1 - i]) == 2:
                new_frame_ix = num_of_history_frames - 1 - i
                future_frame_flag = True
                break
            else:
                future_frame_flag = False
        # if no example from the past, find one from the future
        if not (future_frame_flag):
            for i in range(cur_frame_ix):
                if len(classes[i]) == 2:
                    new_frame_ix = i
                    break
                else:  # if no good pred is around, use the current (no change)
                    new_frame_ix = cur_frame_ix

        # print(f'{frame_ix}: new classes = {classes[new_frame_ix]}') # DEBUG
        # print(f'{frame_ix}: new boxes   = {boxes[new_frame_ix]}') # DEBUG
        # insert previous close to correct predicition
        boxes.insert(cur_frame_ix, boxes[new_frame_ix])
        classes.insert(cur_frame_ix, classes[new_frame_ix])
        changed_frame_flag = True  # DEBUG
    return classes, boxes, changed_frame_flag


def video(video_path, model, output_path, labels_path='../resources/classes.names'):
    cap = cv.VideoCapture(video_path)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv.CAP_PROP_FPS))
    size = (frame_width, frame_height)
    result = cv.VideoWriter(output_path,
                            cv.VideoWriter_fourcc(*'mp4v'),
                            video_fps, size)
    total_frames = count_frames(video_path)
    labels = load_labels(labels_path)
    all_classes, all_boxes = [], []
    hist_boxes, hist_classes, hist_frames = [], [], []
    frame_ix = 0
    num_of_history_frames = 6 + 1  # MOVE TO CONF (num_of_history_frames-1)/2 of past frames, and same for future frames
    for frame_ix in tqdm(range(total_frames)):
        if cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # result.write(predict(frame, model, labels_path))
                classes, boxes, scores = model.detect(frame)

                boxes[0] = [[int(x) for x in box] for box in boxes[0]]
                # classes[0] = [[labels[class_id] for class_id in label] for label in classes]

                # keep the list with size of 'num_of_history_frames'
                if (len(hist_classes) > num_of_history_frames or
                        len(hist_boxes) > num_of_history_frames or
                        len(hist_frames) > num_of_history_frames):
                    hist_classes.pop(0)
                    hist_boxes.pop(0)
                    hist_frames.pop(0)

                hist_classes.append(classes[0])
                hist_boxes.append(boxes[0])
                hist_frames.append(frame)

                frame_itr = frame_ix % num_of_history_frames
                hist_classes, hist_boxes, cur_frame_ix, changed_frame_flag = lpf_for_object_detection(hist_boxes,
                                                                                                      hist_classes,
                                                                                                      total_frames,
                                                                                                      frame_ix,
                                                                                                      frame_itr,
                                                                                                      num_of_history_frames)

                # final_classes  = [labels[hist_classes[frame_itr][1]], labels[hist_classes[frame_itr][0]]]
                final_classes = [labels[class_id] for class_id in hist_classes[cur_frame_ix]]
                final_classes = [final_classes]

                all_classes.append(final_classes)
                all_boxes.append(hist_boxes[cur_frame_ix])

                cur_frame = hist_frames[cur_frame_ix]
                cur_bbox = hist_boxes[cur_frame_ix]
                cur_classes = final_classes[0]

                # frame = bbv.draw_multiple_rectangles(frame, boxes[0], bbox_color=(255,0,0))
                # frame = bbv.add_multiple_labels(frame, classes[0][0], boxes[0], text_bg_color=(255,0,0))
                frame = bbv.add_multiple_labels(cur_frame, cur_classes, cur_bbox, text_bg_color=(255, 255, 0))
                frame = bbv.draw_multiple_rectangles(cur_frame, cur_bbox, bbox_color=(255, 255, 0))
                result.write(frame)
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
    video('/home/student/notebooks/models/P019_tissue1.wm1', model, '../output/res.mp4')
