from yolov7_package import Yolov7Detector


def load_labels(labels_path):
    return open(labels_path).read().strip().split('\n')


def load_model(weights_path='../resources/best.pt', classesfile_path='../resources/classes.names', img_size=[416, 416]):
    return Yolov7Detector(weights=weights_path, img_size=img_size,
                          classes=classesfile_path)
