from video import *
from tools import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Input video file path')
    parser.add_argument('--output-path', type=str, help='Output video file path')
    opt = parser.parse_args()
    print('Starting...')

    import_or_install('yolov7-package', 'yolov7_package')
    print('Loading model...')
    model = load_model()
    print('Processing video...')
    video(opt.video_path, model, opt.output_path)
