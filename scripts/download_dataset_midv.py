#!/usr/bin/env python3

"""
download_dataset_midv.py
------------------------
1) Downloads MIDV-500 or MIDV-2019 zip files (if not present)
2) Unzips them into 'dataset/data/...'
3) Extracts images & ground truth to 'dataset/temp/'
4) Splits data into train/val/test
"""

import os
import re
import cv2
import json
import wget
import zipfile
import shutil
import random
import numpy as np


from glob import glob
from PIL import Image
import pydash as _

# Import your link lists
from midv_links import MIDV500_LINKS, MIDV2019_EXTRA_LINKS

TARGET_PATH = "dataset/data/"
TEMP_PATH = "dataset/temp/"
TEMP_IMAGE_PATH = os.path.join(TEMP_PATH, "image")
TEMP_MASK_PATH = os.path.join(TEMP_PATH, "mask")

DATA_PATH = "dataset/train/"
SEED = 230

# The path offsets identify the substring at which
# the meaningful file name begins, e.g. "01_alb_id.zip"
PATH_OFFSET_500 = 40      # for ftp://smartengines.com/midv-500/dataset/...
PATH_OFFSET_2019 = 56     # for ftp://smartengines.com/midv-500/extra/midv-2019/dataset/...


def read_image(img_path: str, label_path: str):
    """
    Loads an image (via OpenCV), builds a binary mask based on
    'quad' points from the label JSON, resizes them if needed,
    and returns a status plus the processed image/mask.
    """
    image = cv2.imread(img_path)
    if image is None:
        return "error", None, None

    mask = cv2.cvtColor(
        cv2.drawContours(
            np.zeros(image.shape, dtype=np.uint8),
            [
                cv2.approxPolyDP(
                    np.array([[pt] for pt in json.load(open(label_path)).get("quad", [])], dtype=np.int32),
                    10,
                    True
                )
            ] if not _.is_empty(json.load(open(label_path)).get("quad", [])) else [],
            -1,
            (255, 255, 255),
            -1
        ),
        cv2.COLOR_BGR2GRAY
    )

    # Resize to half resolution (optional)
    mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))
    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

    # Threshold
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    return "success", image, mask


def download_and_extract(
    links,
    path_offset,
    dataset_label="MIDV-500"
):
    """
    Downloads each link in `links` if not already present,
    then unzips into `TARGET_PATH`, extracts images into `TEMP_PATH`.
    `path_offset` is used to slice the ftp link to get the zip filename.
    `dataset_label` is just a friendly name used in print statements.
    """
    print(f"\n=== Processing {dataset_label} Datasets ===")

    file_idx = 1  # For naming extracted images
    os.makedirs(TEMP_IMAGE_PATH, exist_ok=True)
    os.makedirs(TEMP_MASK_PATH, exist_ok=True)

    for link in links:
        # e.g., "01_alb_id.zip"
        zip_filename = link[path_offset:]
        # e.g., "dataset/data/01_alb_id.zip"
        full_zip_path = os.path.join(TARGET_PATH, zip_filename)
        # e.g., "dataset/data/01_alb_id" (strip .zip)
        directory_name = os.path.join(TARGET_PATH, zip_filename[:-4])

        print(f"\nDataset directory: {directory_name}")
        if not os.path.exists(directory_name):
            # If .zip not found locally, download it
            if not os.path.isfile(full_zip_path):
                print("Downloading:", link)
                wget.download(link, full_zip_path)
                print()

            print("Unzipping:", full_zip_path)
            with zipfile.ZipFile(full_zip_path, "r") as zip_ref:
                zip_ref.extractall(TARGET_PATH)

        print("Preparing dataset from:", directory_name)
        img_dir_path = os.path.join(directory_name, "images")
        gt_dir_path = os.path.join(directory_name, "ground_truth")

        # Clean single tif/json if they exist with .zip name
        # e.g., remove "01_alb_id.zip.tif" if found
        stray_tif = os.path.join(img_dir_path, zip_filename + ".tif")
        stray_json = os.path.join(gt_dir_path, zip_filename + ".json")
        if os.path.isfile(stray_tif):
            os.remove(stray_tif)
        if os.path.isfile(stray_json):
            os.remove(stray_json)

        # For each subfolder in images/ and ground_truth/
        for images_sub, ground_sub in zip(
            sorted(os.listdir(img_dir_path)),
            sorted(os.listdir(gt_dir_path))
        ):
            img_list = sorted(glob(os.path.join(img_dir_path, images_sub, "*.tif")))
            label_list = sorted(glob(os.path.join(gt_dir_path, ground_sub, "*.json")))

            for img, label in zip(img_list, label_list):
                status, proc_img, proc_mask = read_image(img, label)
                if status == "success":
                    out_img = os.path.join(TEMP_IMAGE_PATH, f"image{file_idx}.png")
                    out_msk = os.path.join(TEMP_MASK_PATH, f"image{file_idx}.png")
                    cv2.imwrite(out_img, proc_img)
                    cv2.imwrite(out_msk, proc_mask)
                    file_idx += 1
        print("-" * 70)


def train_validation_split():
    """
    Splits the temp images/masks into train/val/test sets (70/20/10).
    Saves them in `dataset/train/[train_frames|train_masks|val_frames|...]`.
    """
    print("\n=== Splitting data into train/val/test sets ===")

    # Remove old directory if it exists
    if os.path.exists(DATA_PATH):
        print(f"Removing old data directory: {DATA_PATH}")
        shutil.rmtree(DATA_PATH, ignore_errors=True)

    # Create subfolders for each split
    folders = [
        "train_frames/image",
        "train_masks/image",
        "val_frames/image",
        "val_masks/image",
        "test_frames/image",
        "test_masks/image",
    ]
    for folder in folders:
        os.makedirs(os.path.join(DATA_PATH, folder), exist_ok=True)

    # Collect & sort frames/masks
    all_frames = sorted(os.listdir(TEMP_IMAGE_PATH), key=lambda x: int(re.findall(r'\d+', x)[0]))
    all_masks = sorted(os.listdir(TEMP_MASK_PATH), key=lambda x: int(re.findall(r'\d+', x)[0]))

    # Shuffle frames (and implicitly masks) with the same seed
    random.seed(SEED)
    random.shuffle(all_frames)

    # 70% train, 20% val, 10% test
    train_split = int(0.7 * len(all_frames))
    val_split = int(0.9 * len(all_frames))

    train_frames = all_frames[:train_split]
    val_frames = all_frames[train_split:val_split]
    test_frames = all_frames[val_split:]

    # Filter mask filenames to match frames
    train_masks = [m for m in all_masks if m in train_frames]
    val_masks   = [m for m in all_masks if m in val_frames]
    test_masks  = [m for m in all_masks if m in test_frames]

    def copy_frame(dst_folder, filename):
        Image.open(os.path.join(TEMP_IMAGE_PATH, filename)) \
             .save(os.path.join(DATA_PATH, dst_folder, filename))

    def copy_mask(dst_folder, filename):
        Image.open(os.path.join(TEMP_MASK_PATH, filename)) \
             .save(os.path.join(DATA_PATH, dst_folder, filename))

    # Distribute frames
    for fname in train_frames:
        copy_frame("train_frames/image", fname)
    for fname in val_frames:
        copy_frame("val_frames/image", fname)
    for fname in test_frames:
        copy_frame("test_frames/image", fname)

    # Distribute masks
    for mname in train_masks:
        copy_mask("train_masks/image", mname)
    for mname in val_masks:
        copy_mask("val_masks/image", mname)
    for mname in test_masks:
        copy_mask("test_masks/image", mname)


def main():
    """
    1) Clean up 'dataset/temp/'
    2) Download & extract MIDV-500
    3) Download & extract MIDV-2019 Extra (optional)
    4) Split data into train/val/test
    """
    if os.path.exists(TEMP_PATH):
        print(f"Removing existing temp dir: {TEMP_PATH}")
        shutil.rmtree(TEMP_PATH, ignore_errors=True)
    os.makedirs(TEMP_PATH, exist_ok=True)
    download_and_extract(MIDV500_LINKS, PATH_OFFSET_500, dataset_label="MIDV-500")
    download_and_extract(MIDV2019_EXTRA_LINKS, PATH_OFFSET_2019, dataset_label="MIDV-2019 Extra")
    train_validation_split()

if __name__ == "__main__":
    main()
