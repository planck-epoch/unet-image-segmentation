#!/usr/bin/env python3

"""
benchmark.py
------------
Evaluates a trained U-Net segmentation model against a dataset where
ground truth masks are derived from JSON polygon annotations.

Calculates the overall Mean Intersection over Union (MeanIoU) for the dataset
and logs images scoring below a specified IoU threshold.

Dataset Structure Expectation:
<input_dir>/
├── images/
│   ├── SUBDIR1/
│   │   ├── img1.tif
│   │   └── ...
│   ├── SUBDIR2/
│   └── ...
└── ground_truth/
    ├── SUBDIR1/
    │   ├── img1.json # Contains "quad": [[x1,y1], [x2,y2], ...]
    │   └── ...
    ├── SUBDIR2/
    └── ...

Usage:
  python scripts/benchmark.py <input_dir> --model <model_file> [options]

Example:
  python scripts/benchmark.py ./datasets/prepared_validation --model ./models/model.h5 \
    --iou_threshold 0.8 --pred_threshold 0.5 --low_score_log low_iou_files.txt
"""

import argparse
import json
import os
import sys
import time
from typing import Tuple, Optional, List, Dict

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from glob import glob

# --- Project Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Import custom objects potentially needed by the loaded model
from utils.loss import dice_loss, iou_loss, jaccard_loss
from utils.metrics import dice_coef, iou_coef
# --- End Setup ---

# Constants
IMG_HEIGHT = 256
IMG_WIDTH = 256
# Use a small epsilon for manual IoU calculation stability
SMOOTH = 1e-6

def parse_args() -> argparse.Namespace:
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser(
        description="Benchmark a U-Net segmentation model using JSON ground truth."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Top-level directory containing 'images/' and 'ground_truth/' subfolders."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/model.h5",
        help="Path to the trained Keras (.h5 or .keras) model file."
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.9,
        help="Log filenames where the sample's MeanIoU is BELOW this threshold."
    )
    parser.add_argument(
        "--pred_threshold",
        type=float,
        default=0.5,
        help="Threshold (0-1) to convert model's probability prediction to a binary mask for IoU calculation."
    )
    parser.add_argument(
        "--low_score_log",
        type=str,
        default=None,
        help="Optional file path to save the list of files scoring below the iou_threshold."
    )
    return parser.parse_args()

def load_image_for_predict(img_path: str) -> Optional[np.ndarray]:
    """ Loads image, normalizes, resizes to model input size (H, W), adds batch dim. """
    try:
        # Load in color (3 channels)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Warning: Could not read image: {img_path}. Skipping.")
            return None

        # Normalize to [0, 1]
        img_norm = img_bgr.astype(np.float32) / 255.0
        # Resize
        img_resized = cv2.resize(img_norm, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
        # Add batch dim -> (1, H, W, 3)
        return np.expand_dims(img_resized, axis=0)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

def build_mask_from_quad(json_path: str, target_height: int, target_width: int) -> Optional[np.ndarray]:
    """ Builds a binary mask (0/1) from JSON 'quad', resizes to target, adds batch/channel dims. """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        quad = data.get("quad", [])

        # Get original dimensions if needed (e.g., from associated image, assume large enough for now)
        # Or read from JSON if available. If not, create a reasonably large canvas.
        # Let's assume we need original dims later, but for resizing, only target matters.
        # We need the original dims to create the initial mask *before* resizing.
        # Hacky: Try to infer from companion image (less robust)
        img_companion_path_tif = json_path.replace("/ground_truth/", "/images/").replace(".json", ".tif")
        img_companion_path_png = json_path.replace("/ground_truth/", "/images/").replace(".json", ".png")
        img_companion_path_jpg = json_path.replace("/ground_truth/", "/images/").replace(".json", ".jpg")
        
        orig_h, orig_w = -1, -1
        for img_path in [img_companion_path_tif, img_companion_path_png, img_companion_path_jpg]:
             if os.path.exists(img_path):
                  dims = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).shape[:2]
                  if dims:
                       orig_h, orig_w = dims
                       break
        
        if orig_h <=0 or orig_w <= 0:
             print(f"Warning: Could not determine original dimensions for mask from {json_path}. Using default large canvas (2048x2048).")
             orig_h, orig_w = 2048, 2048 # Default large size if image not found

        mask = np.zeros((orig_h, orig_w), dtype=np.uint8) # Start with 0

        if quad: # Check if quad list is not empty
            # Convert points to integer format required by drawContours
            # Use raw points, approxPolyDP seemed too aggressive
            points = np.array(quad, dtype=np.int32)
            # Ensure shape is (N, 1, 2) for drawContours
            if points.ndim == 2:
                 points = points.reshape((-1, 1, 2))
            
            try:
                # Draw filled polygon (value 255)
                cv2.drawContours(mask, [points], contourIdx=-1, color=255, thickness=cv2.FILLED)
            except Exception as e:
                print(f"Warning: drawContours failed for {json_path} (Points: {points.shape}). Error: {e}. Mask might be empty.")
                # Continue with potentially empty mask

        # Resize to target size (e.g., 256x256)
        # Use INTER_NEAREST for binary masks to avoid intermediate values
        mask_resized = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        # Ensure values are 0 or 1 (resize might introduce noise if not INTER_NEAREST)
        mask_binary = (mask_resized > 128).astype(np.uint8) # Threshold just in case

        # Add batch and channel dims -> (1, H, W, 1)
        return np.expand_dims(mask_binary, axis=[0, -1])

    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_path}")
        return None
    except Exception as e:
        print(f"Error processing JSON/Mask {json_path}: {e}")
        return None

def calculate_sample_iou(y_true_sample: np.ndarray, y_pred_sample: np.ndarray, smooth: float = SMOOTH) -> float:
     """ Calculates IoU for a single sample (H, W, 1) or (H, W). """
     y_true = tf.cast(y_true_sample.squeeze(), tf.float32)
     y_pred = tf.cast(y_pred_sample.squeeze(), tf.float32)
     
     intersection = tf.reduce_sum(y_true * y_pred)
     sum_true = tf.reduce_sum(y_true)
     sum_pred = tf.reduce_sum(y_pred)
     union = sum_true + sum_pred - intersection
     
     iou = (intersection + smooth) / (union + smooth)
     return float(iou.numpy())

def main():
    args = parse_args()
    start_time = time.time()

    # --- Validate Paths ---
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found -> {args.input_dir}")
        sys.exit(1)
    images_root = os.path.join(args.input_dir, "images")
    gtruth_root = os.path.join(args.input_dir, "ground_truth")
    if not (os.path.isdir(images_root) and os.path.isdir(gtruth_root)):
        print(f"Error: '{images_root}' or '{gtruth_root}' not found.")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Error: Model file not found -> {args.model}")
        sys.exit(1)
    if not (0.0 <= args.pred_threshold <= 1.0):
         print(f"Error: Prediction threshold must be between 0.0 and 1.0 -> {args.pred_threshold}")
         sys.exit(1)
    if not (0.0 <= args.iou_threshold <= 1.0):
         print(f"Error: IoU threshold must be between 0.0 and 1.0 -> {args.iou_threshold}")
         sys.exit(1)

    # --- Load Model ---
    print(f"Loading model: {args.model} ...")
    # ** Edit this dictionary based on the model being loaded **
    required_custom_objects = {
         "dice_loss": dice_loss,
         "dice_coef": dice_coef
    }
    print(f"Using custom_objects for load_model: {list(required_custom_objects.keys())}")
    try:
        model = load_model(args.model, custom_objects=required_custom_objects, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"\n--- Error loading model ---")
        print(f"{e}")
        print("---------------------------\n")
        sys.exit(1)

    # --- Find Data Pairs ---
    print("Finding image and ground truth pairs...")
    image_files = sorted(glob(os.path.join(images_root, "**", "*.tif"), recursive=True))
    print(f"Found {len(image_files)} '.tif' images.")
    
    data_pairs: List[Dict] = []
    processed_count = 0
    skipped_count = 0

    for img_path in image_files:
        relative_path = os.path.relpath(img_path, images_root)
        base_name = os.path.splitext(relative_path)[0]
        json_path = os.path.join(gtruth_root, base_name + ".json")

        if os.path.isfile(json_path):
            data_pairs.append({"image": img_path, "json": json_path, "id": base_name})
            processed_count += 1
        else:
            print(f"Warning: No corresponding JSON found for {img_path}. Skipping.")
            skipped_count += 1
            
    if not data_pairs:
         print("Error: No valid image/JSON pairs found. Check dataset structure and file extensions.")
         sys.exit(1)
         
    print(f"Prepared {len(data_pairs)} image/JSON pairs for evaluation ({skipped_count} images skipped).")

    # --- Initialize Metrics ---
    iou_metric = tf.keras.metrics.MeanIoU(num_classes=2, name="overall_mean_iou")
    low_iou_files: List[Tuple[str, float]] = []

    # --- Evaluation Loop ---
    print(f"Evaluating model (Prediction Threshold: {args.pred_threshold:.2f})...")
    for i, pair in enumerate(data_pairs):
        img_path = pair["image"]
        json_path = pair["json"]
        file_id = pair["id"]
        
        print(f"\rProcessing [{i+1}/{len(data_pairs)}]: {file_id}", end="")

        # Load data
        img_tensor = load_image_for_predict(img_path)
        mask_true_tensor = build_mask_from_quad(json_path, IMG_HEIGHT, IMG_WIDTH)

        if img_tensor is None or mask_true_tensor is None:
            print(f"\nSkipping pair due to loading error: {file_id}")
            continue

        # Predict
        mask_pred_prob = model.predict(img_tensor, verbose=0) # Shape (1, H, W, 1)

        if mask_pred_prob is None:
             print(f"\nSkipping pair due to prediction error: {file_id}")
             continue

        # Threshold prediction to get binary mask (0 or 1)
        mask_pred_binary = (mask_pred_prob > args.pred_threshold).astype(np.uint8)

        # --- Calculate Sample IoU for Logging ---
        # Use temporary metric or manual calculation for *this sample only*
        # Using manual calculation helper function:
        sample_iou = calculate_sample_iou(mask_true_tensor[0], mask_pred_binary[0]) # Pass single masks
        
        if sample_iou < args.iou_threshold:
             low_iou_files.append((file_id, sample_iou))
             print(f"\nBelow threshold (IoU={sample_iou:.3f}): {file_id}")

        # --- Update Overall Metric ---
        # update_state expects labels (y_true) then predictions (y_pred)
        try:
             iou_metric.update_state(mask_true_tensor, mask_pred_binary)
        except Exception as e:
             print(f"\nError updating MeanIoU state for {file_id}: {e}")
             #print(mask_true_tensor.shape, mask_pred_binary.shape)


    print("\nEvaluation complete.")

    # --- Report Results ---
    final_mean_iou = iou_metric.result().numpy()
    print(f"\n{'='*30}")
    print(f"Overall Mean IoU: {final_mean_iou:.4f}")
    print(f"{'='*30}")

    if low_iou_files:
        print(f"\nFiles scoring below IoU threshold ({args.iou_threshold:.2f}):")
        # Sort by IoU score (ascending)
        low_iou_files.sort(key=lambda item: item[1])
        for file_id, score in low_iou_files:
            print(f"  - IoU: {score:.4f} | File: {file_id}")
            
        # Save low scores to file if requested
        if args.low_score_log:
             print(f"\nSaving low score list to: {args.low_score_log}")
             try:
                  log_dir = os.path.dirname(args.low_score_log)
                  if log_dir: os.makedirs(log_dir, exist_ok=True)
                  with open(args.low_score_log, 'w') as f:
                       f.write("FileID,MeanIoU_Score\n")
                       for file_id, score in low_iou_files:
                            f.write(f"{file_id},{score:.4f}\n")
             except Exception as e:
                  print(f"Error saving low score log: {e}")
    else:
        print(f"\nNo files scored below the IoU threshold ({args.iou_threshold:.2f}).")

    end_time = time.time()
    print(f"\nTotal benchmark time: {end_time - start_time:.2f} seconds.")
    print("Benchmark script finished.")


if __name__ == "__main__":
    main()