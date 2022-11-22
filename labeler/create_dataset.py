import numpy as np
import cv2
import random
import argparse
import json
import glob
from matplotlib import pyplot as plt
import ntpath
import os, sys
import pydash as _
import shutil

# RAW_IMG_DIR = "./data/ina_id/raw/ktp_indodana/*"
IMPORT_FILES = "../dataset/data/ina_id/temp/images/raw_selfie/*"
ANOTATION_DIR = "../dataset/data/ina_id/temp/ground_truth/raw_selfie/*"

IMAGE_RESULT_DIR = "../dataset/data/ina_id/images/selfie/"
ANOTATION_RESULT_DIR = "../dataset/data/ina_id/ground_truth/selfie/"

if os.path.exists(IMAGE_RESULT_DIR):
    print('Remove Temp Directory and create a new one')
    shutil.rmtree(IMAGE_RESULT_DIR, ignore_errors=True)
if os.path.exists(ANOTATION_RESULT_DIR):
    print('Remove Temp Directory and create a new one')
    shutil.rmtree(ANOTATION_RESULT_DIR, ignore_errors=True)

os.makedirs(IMAGE_RESULT_DIR, exist_ok=True)
os.makedirs(ANOTATION_RESULT_DIR, exist_ok=True)

# raw_files = glob.glob(RAW_IMG_DIR)

def read_image(img, label):
    image = cv2.imread(img)
    mask = np.zeros(image.shape, dtype=np.uint8)
    quad = json.load(open(label, 'r'))['quad']
    
    if not _.is_empty(quad):
        coords = cv2.approxPolyDP(np.array([[xy] for xy in quad], dtype=np.int32), 10, True)
        quad = [[int(point[0][0]), int(point[0][1])] for point in coords]
        try:
            cv2.drawContours(mask, [coords], -1, (255,255,255), -1)
            # cv2.fillPoly(mask, coords, color=(255, 255, 255))
        except:
            print(label)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    return image, mask, quad

def change_brightness_contrast(image, alpha, beta):
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return new_image

img_list = sorted(glob.glob(IMPORT_FILES))
label_list = sorted(glob.glob(ANOTATION_DIR))

for i, (img, label) in enumerate(zip(img_list, label_list)):
    image, mask, coords = read_image(img, label)
    filename = ntpath.basename(img).split('.')[0].split('_')[0]
    
    output_img_dir = IMAGE_RESULT_DIR + filename+'/'
    output_anotation_dir = ANOTATION_RESULT_DIR + filename+'/'
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_anotation_dir, exist_ok=True)
        
    for j in range(4):
        image_list = []
        quad = {"quad":[]}
        if j == 0:
            quad = {"quad":coords}
            image_augment = image.copy()
        elif j == 1:
            image_augment = image.copy()
            image_augment = cv2.rotate(image_augment, cv2.cv2.ROTATE_90_CLOCKWISE)
            if not _.is_empty(coords):
                mask_ = mask.copy()
                mask_ = cv2.rotate(mask_, cv2.cv2.ROTATE_90_CLOCKWISE)

                contours, hierarchy = cv2.findContours(mask_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                minRect = [None]*len(contours)
                boxes = [None]*len(contours)
                for i, c in enumerate(contours):
                    approx_c = cv2.approxPolyDP(c, 10, True)
                    minRect[i] = cv2.minAreaRect(approx_c)
                    boxes[i] = np.int0(cv2.boxPoints(minRect[i]))
                    
                quad = {"quad":[[int(point[0]), int(point[1])] for point in boxes[0]]}
        elif j == 2:
            image_augment = image.copy()
            image_augment = cv2.rotate(image_augment, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            if not _.is_empty(coords):
                mask_ = mask.copy()
                mask_ = cv2.rotate(mask_, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

                contours, hierarchy = cv2.findContours(mask_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                minRect = [None]*len(contours)
                boxes = [None]*len(contours)
                for i, c in enumerate(contours):
                    approx_c = cv2.approxPolyDP(c, 10, True)
                    minRect[i] = cv2.minAreaRect(approx_c)
                    boxes[i] = np.int0(cv2.boxPoints(minRect[i]))

                quad = {"quad":[[int(point[0]), int(point[1])] for point in boxes[0]]}
        elif j == 3:
            image_augment = image.copy()
            image_augment = cv2.flip(image_augment, 1)
            if not _.is_empty(coords):
                mask_ = mask.copy()
                mask_ = cv2.flip(mask_, 1)

                contours, hierarchy = cv2.findContours(mask_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                minRect = [None]*len(contours)
                boxes = [None]*len(contours)
                for i, c in enumerate(contours):
                    approx_c = cv2.approxPolyDP(c, 10, True)
                    minRect[i] = cv2.minAreaRect(approx_c)
                    boxes[i] = np.int0(cv2.boxPoints(minRect[i]))

                quad = {"quad":[[int(point[0]), int(point[1])] for point in boxes[0]]}

        image_list.append(image_augment)
        image_list.append(cv2.medianBlur(image_augment,9))
        image_list.append(cv2.GaussianBlur(image_augment,(9,9),0))
        image_list.append(cv2.blur(image_augment,(9,9)))
        # image_list.append(change_brightness_contrast(image, random.randrange(1.0,3.0), random.randint(20,40)))
        # image_list.append(change_brightness_contrast(image, random.randrange(1.0,3.0), random.randint(40,60)))
        # image_list.append(change_brightness_contrast(image, random.randrange(1.0,3.0), random.randint(60,80)))

        for k, img in enumerate(image_list):
            with open(output_anotation_dir + filename + '_' + str(i) + '_' + str(j) + '_' + str(k) + '.json', 'w') as outfile:
                json.dump(quad, outfile)

            cv2.imwrite(output_img_dir + filename + '_' + str(i) + '_' + str(j) + '_' + str(k) + '.tif', img)