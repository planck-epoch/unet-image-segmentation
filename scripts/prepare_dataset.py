import os
import sys
import glob
import json
import shutil
import random
import argparse

import cv2
import numpy as np
import pydash as _
from matplotlib import pyplot as plt
import ntpath

# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------

def read_image(img_path, label_path):
    """
    Reads an image and its corresponding JSON label (containing a 'quad' array).
    Returns the image, a binary mask constructed from the 'quad', and the quad itself.
    """
    image = cv2.imread(img_path)
    mask = np.zeros(image.shape, dtype=np.uint8)

    with open(label_path, 'r') as f:
        annotation_data = json.load(f)

    quad = annotation_data.get('quad', [])
    if not _.is_empty(quad):
        coords = cv2.approxPolyDP(
            np.array([[pt] for pt in quad], dtype=np.int32),
            epsilon=10,
            closed=True
        )
        try:
            cv2.drawContours(mask, [coords], contourIdx=-1, color=(255, 255, 255), thickness=-1)
        except Exception as e:
            print(f"Failed to draw contour for label: {label_path}\nError: {e}")

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

    return image, mask, quad


def rotate_and_extract_quad(mask):
    """
    Rotates the mask, then finds the contour bounding box (minAreaRect).
    Returns the 'quad' dict based on the bounding box points.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"quad": []}

    approx_c = cv2.approxPolyDP(contours[0], 10, True)
    min_rect = cv2.minAreaRect(approx_c)
    box_points = cv2.boxPoints(min_rect)
    box_points = np.int0(box_points)

    return {"quad": [[int(pt[0]), int(pt[1])] for pt in box_points]}


def change_brightness_contrast(image, alpha, beta):
    """
    Adjusts brightness and contrast of the input image.
    alpha: contrast multiplier (1.0 = no change, >1.0 = higher contrast)
    beta: brightness offset
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


# --------------------------------------------------------------------------------
# Main Script
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset by applying rotations, flips, and blurs to images."
    )
    parser.add_argument(
        "--import_files",
        default="../datasets/data/images/raw_selfie/*",
        help="Glob pattern for input images in the raw folder"
    )
    parser.add_argument(
        "--annotation_dir",
        default="../datasets/data/ground_truth/raw_selfie/*",
        help="Glob pattern for annotation JSON files in the raw folder"
    )
    parser.add_argument(
        "--image_result_dir",
        default="../datasets/data/images/selfie/",
        help="Directory to store processed (augmented) images"
    )
    parser.add_argument(
        "--annotation_result_dir",
        default="../datasets/data/ground_truth/selfie/",
        help="Directory to store processed (augmented) annotations"
    )
    args = parser.parse_args()

    IMPORT_FILES = args.import_files
    ANNOTATION_DIR = args.annotation_dir
    IMAGE_RESULT_DIR = args.image_result_dir
    ANNOTATION_RESULT_DIR = args.annotation_result_dir

    if os.path.exists(IMAGE_RESULT_DIR):
        print(f"Removing existing directory: {IMAGE_RESULT_DIR}")
        shutil.rmtree(IMAGE_RESULT_DIR, ignore_errors=True)
    if os.path.exists(ANNOTATION_RESULT_DIR):
        print(f"Removing existing directory: {ANNOTATION_RESULT_DIR}")
        shutil.rmtree(ANNOTATION_RESULT_DIR, ignore_errors=True)

    os.makedirs(IMAGE_RESULT_DIR, exist_ok=True)
    os.makedirs(ANNOTATION_RESULT_DIR, exist_ok=True)
    img_list = sorted(glob.glob(IMPORT_FILES))
    label_list = sorted(glob.glob(ANNOTATION_DIR))

    if len(img_list) != len(label_list):
        print("Warning: The number of images and annotation files differ.")
        print(f"Images found: {len(img_list)}, Annotations found: {len(label_list)}")
        # sys.exit(1)

    for i, (img_path, label_path) in enumerate(zip(img_list, label_list)):
        image, mask, coords = read_image(img_path, label_path)
        # e.g. "photo_001.jpg" -> "photo_001"
        filename = ntpath.basename(img_path).split('.')[0]

        output_img_dir = os.path.join(IMAGE_RESULT_DIR, filename)
        output_annot_dir = os.path.join(ANNOTATION_RESULT_DIR, filename)
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_annot_dir, exist_ok=True)

        # We will generate 4 transformations: none, rotate 90°, rotate -90°, flip horizontally
        for j in range(4):
            if j == 0:
                image_aug = image.copy()
                mask_aug = mask.copy()
                quad_info = {"quad": coords}
            elif j == 1:
                # Rotate 90° clockwise
                image_aug = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                mask_aug = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                quad_info = rotate_and_extract_quad(mask_aug)
            elif j == 2:
                # Rotate 90° counterclockwise
                image_aug = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                mask_aug = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                quad_info = rotate_and_extract_quad(mask_aug)
            else:
                # Horizontal flip
                image_aug = cv2.flip(image, 1)
                mask_aug = cv2.flip(mask, 1)
                quad_info = rotate_and_extract_quad(mask_aug)

            image_list = [
                image_aug,
                cv2.medianBlur(image_aug, 9),
                cv2.GaussianBlur(image_aug, (9, 9), 0),
                cv2.blur(image_aug, (9, 9)),
            ]

            for k, variant_img in enumerate(image_list):
                annot_name = f"{filename}_{i}_{j}_{k}.json"
                img_name = f"{filename}_{i}_{j}_{k}.tif"

                annot_path = os.path.join(output_annot_dir, annot_name)
                with open(annot_path, 'w') as outfile:
                    json.dump(quad_info, outfile)

                img_path_out = os.path.join(output_img_dir, img_name)
                cv2.imwrite(img_path_out, variant_img)


if __name__ == "__main__":
    main()
