from tools import *

import_or_install('yolov7-package', 'yolov7_package')

from video import *
from predict import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--weights_path', type=str, required=False, default='../resources/models/best_iou50.pt',
                        help='Pretrained model - supported are [adam-100, adam-300, init, iou50-default]')
    parser.add_argument('--history_frames', type=int, required=False, default=6,
                        help='Number of past and future frames to be considered in the smoothing process')
    opt = parser.parse_args()
    print('Starting...')
    print('Loading model...')
    model = load_model(opt.weights_path)
    print('Processing...')

    try:
        image = cv.imread(opt.input)
        predict(image, model, opt.output)
    except:
        video(opt.input, model, opt.output, opt.history_frames)

    print(f'DONE! \n Results can be found in {opt.output}')
