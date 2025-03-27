#!/usr/bin/env python3

"""
benchmark.py
------------
Evaluates a U-Net model (trained with 3-channel input & single-channel output)
by computing the mean IoU on a directory structure like:

dataset/data/01_alb_id/
├── images/
│   ├── CA/
│   │   ├── CA01_01.tif
│   │   ├── CA01_02.tif
│   │   ...
│   ├── TS/
│   ├── ...
└── ground_truth/
    ├── CA/
    │   ├── CA01_01.json
    │   ├── CA01_02.json
    │   ...
    ├── TS/
    ├── ...

Each .tif is loaded in color (3 channels),
each JSON has "quad" for polygons.
The final model layer is Conv2D(1, 1, activation='sigmoid'), so we produce
(256x256x1) masks. We compare the model's predictions to our ground-truth
to compute mean IoU.

Usage:
  python benchmark.py <input_dir> --model <model_file.h5> --threshold <iou_threshold>

Example:
  python benchmark.py dataset/data/01_alb_id --model model.h5 --threshold 0.9
"""

import argparse
import json
import os
import cv2
import numpy as np
import pydash as _
import tensorflow as tf
from glob import glob
from keras.models import load_model

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate mean IoU for a U-Net that uses 3-channel input & 1-channel output."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Top-level directory containing 'images/' and 'ground_truth/' subfolders."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model.h5",
        help="Path to the .h5 Keras model file."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="If MeanIoU < threshold, log the filename for further inspection."
    )
    return parser.parse_args()

def load_image_color(img_path):
    """
    Loads an image in 3-channel BGR, normalizes to [0..1],
    resizes to (256,256), and adds batch dimension.

    Returns:
        np.ndarray of shape (1, 256, 256, 3).
    """
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    # Convert from [0..255] to [0..1]
    img_norm = img_bgr / 255.0
    # Resize to (256, 256)
    img_resized = cv2.resize(img_norm, (256, 256), interpolation=cv2.INTER_AREA)
    # Add batch dim -> (1, 256, 256, 3)
    return img_resized.reshape((1, 256, 256, 3))

def build_mask_from_quad(json_path, orig_h, orig_w):
    """
    Builds a single-channel binary mask from JSON 'quad' polygon,
    resizes to (256×256), returns shape (1,256,256,1).
    """
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    with open(json_path, "r") as f:
        data = json.load(f)
    quad = data.get("quad", [])

    if not _.is_empty(quad):
        coords = cv2.approxPolyDP(
            np.array([[pt] for pt in quad], dtype=np.int32),
            epsilon=10,
            closed=True
        )
        try:
            cv2.drawContours(mask, [coords], -1, 255, thickness=-1)
        except Exception as e:
            print(f"Warning: drawContours failed for {json_path}: {e}")

    # Convert to [0..1]
    mask = mask.astype(np.float32) / 255.0
    # Resize to (256,256)
    mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
    # Add batch + channel -> (1,256,256,1)
    return mask_resized.reshape((1, 256, 256, 1))

def evaluate_model_on_pairs(model, images, masks, filenames, threshold):
    """
    Evaluates the model on each (image, mask) pair (3-channel in, 1-channel out).
    If mean_iou < threshold, logs the filename.

    Returns:
        float: The average mean IoU across all evaluated samples.
    """
    mean_ious = []
    for img_tensor, mask_tensor, fname in zip(images, masks, filenames):
        # Evaluate -> [loss, <other metrics>..., mean_iou(?)]
        results = model.evaluate(img_tensor, mask_tensor, verbose=0)
        if len(results) < 3:
            print("Warning: Model does not appear to have mean_iou at index 2.")
            continue

        mean_iou_value = results[2]
        if mean_iou_value < threshold:
            print(f"Below threshold (MeanIoU={mean_iou_value:.3f}): {fname}")
        mean_ious.append(mean_iou_value)

    return float(np.mean(mean_ious)) if mean_ious else 0.0

def main():
    args = parse_args()

    # Ensure input_dir exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory.")
        return

    images_root = os.path.join(args.input_dir, "images")
    gtruth_root = os.path.join(args.input_dir, "ground_truth")
    if not (os.path.isdir(images_root) and os.path.isdir(gtruth_root)):
        print("Error: 'images/' or 'ground_truth/' folder not found under input_dir.")
        return

    if not os.path.isfile(args.model):
        print(f"Error: Model file not found at {args.model}")
        return

    print(f"Loading model: {args.model}")
    # Make sure the model expects (256,256,3) input, with a final Conv2D(1,sigmoid)
    # compiled with mean IoU as a metric
    model = load_model(args.model, custom_objects={
        "mean_iou": tf.keras.metrics.MeanIoU(num_classes=2)
    })

    all_images, all_masks, all_filenames = [], [], []

    # For each subfolder in "images", find matching subfolder in "ground_truth"
    subdirs = sorted(os.listdir(images_root))
    for sub in subdirs:
        sub_img_dir = os.path.join(images_root, sub)
        sub_gt_dir  = os.path.join(gtruth_root, sub)

        if not (os.path.isdir(sub_img_dir) and os.path.isdir(sub_gt_dir)):
            print(f"Skipping '{sub}'—no matching subfolder in both images/ & ground_truth/")
            continue

        tif_files  = sorted(glob(os.path.join(sub_img_dir, "*.tif")))
        json_files = sorted(glob(os.path.join(sub_gt_dir, "*.json")))

        # Pair them up
        for tif_path, json_path in zip(tif_files, json_files):
            # Read raw color for original dims
            raw = cv2.imread(tif_path, cv2.IMREAD_COLOR)
            if raw is None:
                print(f"Warning: Could not read image {tif_path}, skipping.")
                continue

            h, w = raw.shape[:2]

            # Convert them to model-ready shapes
            try:
                img_tensor  = load_image_color(tif_path)
                mask_tensor = build_mask_from_quad(json_path, h, w) 
                all_images.append(img_tensor)
                all_masks.append(mask_tensor)
                all_filenames.append(os.path.basename(tif_path))
            except Exception as e:
                print(f"Error processing {tif_path} / {json_path}: {e}")
                continue

    print("Evaluating model...")
    avg_mean_iou = evaluate_model_on_pairs(
        model, all_images, all_masks, all_filenames, args.threshold
    )
    print(f"Average Mean IoU: {avg_mean_iou:.3f}")
    print("Done.")

if __name__ == "__main__":
    main()
