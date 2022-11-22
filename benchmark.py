import argparse
import cv2
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import tensorflow as tf
from glob import glob
import numpy as np
import json
import pydash as _

from utils import image
from utils import metrics

parser = argparse.ArgumentParser(
    description='Semantic segmentation of IDCard in Image.')
parser.add_argument('input_dir', type=str,
                    help='Directory of input image and ground truth')
parser.add_argument('--model', type=str, default='model.h5',
                    help='Path to .h5 model file')
parser.add_argument('--threshold', type=float, default=0.9,
                    help='Threshold of MeanIOU for debugging')

args = parser.parse_args()

INPUT_DIR = args.input_dir
MODEL_FILE = args.model
THRESHOLD = args.threshold


def load_image(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    height, width = img.shape[:2]
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = img.reshape(1, 256, 256, 1)
    return img, height, width


def read_image(img, label):
    image, h, w = load_image(img)
    mask = np.zeros([h, w], dtype=np.uint8)
    quad = json.load(open(label, 'r'))['quad']

    if not _.is_empty(quad):
        coords = cv2.approxPolyDP(
            np.array([[xy] for xy in quad], dtype=np.int32), 10, True)

        try:
            cv2.drawContours(mask, [coords], -1, (255), -1)
        except:
            print(label)

    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    mask = mask / 255.0
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
    mask = mask.reshape(1, 256, 256, 1)
    return 'success', image, mask


def evaluate_model(model, image_list, mask_list, filename_list):
    print(model.metrics_names)
    mean_iou = []
    for i, (image, mask, filename) in enumerate(zip(image_list, mask_list, filename_list)):
        evaluation = model.evaluate(image, mask, verbose=0)
        # print the image filename that has mean iou below threshold for evaluate individually
        if (i > 0) & (evaluation[2] < THRESHOLD):
            print(i, '. ', filename, 'mean_iou: ', evaluation[2])
        mean_iou.append(evaluation[2])

    return np.average(mean_iou)


def main():
    if not os.path.exists(INPUT_DIR):
        print('Input directory not found ', INPUT_DIR)
    else:
        if not os.path.isfile(MODEL_FILE):
            print('Model not found ', MODEL_FILE)

        else:
            print('Load model... ', MODEL_FILE)
            #model = load_model(MODEL_FILE, custom_objects={'mean_iou': metrics.mean_iou})
            model = load_model(MODEL_FILE, custom_objects={
                               'mean_iou': tf.keras.metrics.MeanIoU(num_classes=2)})

            print('Load image... ', INPUT_DIR)
            img_dir_path = './' + INPUT_DIR + '/images/'
            gt_dir_path = './' + INPUT_DIR + '/ground_truth/'

            image_list = []
            mask_list = []
            filename_list = []
            # Load Images and Groundtruth and store as numpy array
            for images, ground_truth in zip(sorted(os.listdir(img_dir_path)), sorted(os.listdir(gt_dir_path))):
                img_list = sorted(glob(img_dir_path + images + '/*.tif'))
                lbl_list = sorted(glob(gt_dir_path + ground_truth + '/*.json'))
                for img, label in zip(img_list, lbl_list):
                    status, image, mask = read_image(img, label)
                    if status == 'success':
                        image_list.append(image)
                        mask_list.append(mask)
                        filename_list.append(img)

            print('Evaluation...')
            evaluation = evaluate_model(
                model, image_list, mask_list, filename_list)
            print('average mean IOU: ', evaluation)

            print('Done.')


if __name__ == '__main__':
    main()
