import cv2 as cv
import bbox_visualizer as bbv
from model_tools import *


def predict(image, model, output_path, labels_path='../resources/classes.names'):
    labels = load_labels(labels_path)
    classes, boxes, scores = model.detect(image)
    boxes[0] = [[int(x) for x in box] for box in boxes[0]]
    classes[0] = [[labels[class_id] for class_id in label] for label in classes]

    image = bbv.draw_multiple_rectangles(image, boxes[0], bbox_color=(255, 0, 0))
    image = bbv.add_multiple_labels(image, classes[0][0], boxes[0], text_bg_color=(255, 0, 0))

    cv.imwrite(output_path, image)
