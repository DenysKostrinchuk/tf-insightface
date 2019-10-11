import argparse
from worker import Worker
import numpy as np
import cv2
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='model-r50-am-lfw/model,000', help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--dataset', type=str, default='images_name_1d.npy',
                        help='path to file with dataset records')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='threshold for comparing distance')
    parser.add_argument('--step_accuracy', type=int, default=50,
                        help='How often should save accuracy')
    parser.add_argument('--save_one_in', type=int, default=1000,
                        help='How often save "trt_1d_result.json" file with accuracy')

    args = parser.parse_args()
    work = Worker(args)

    ### get list of images
    name_all_images = np.load(args.dataset)
    work.processing_dataset(name_all_images)
    
