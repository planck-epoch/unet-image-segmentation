#!/usr/bin/env python3

"""
inference.py
------------
Script to run inference using a trained U-Net model for image segmentation.

It loads a Keras/TensorFlow model, reads and preprocesses an input image,
predicts a segmentation mask, optionally finds the largest contour in the mask,
crops the original image based on the contour's bounding box, and saves the
binary mask and the cropped image.

Usage:
  $ python3 scripts/inference.py /path/to/input_image.jpg \
                --output_mask /path/to/mask_out.png \
                --output_cropped /path/to/cropped_out.png \
                --model /path/to/model.h5

Example:
    # Use default model path ./models/model.h5 and save to ./outputs_test/
    $ python3 scripts/inference.py ./samples/test_images/id_card.png

    # Specify everything
    $ python3 scripts/inference.py \
        "./samples/test_images/passport.png" \
        --output_mask "./outputs_test/output_mask.png" \
        --output_cropped "./outputs_test/output_cropped.png" \
        --model "./models/unet_trained_specific.h5" \
        --threshold 0.6
"""

import argparse
import os
import sys
from typing import Tuple, Optional

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.loss import dice_loss, iou_loss, jaccard_loss
from utils.metrics import dice_coef, iou_coef

# Constants for preprocessing - should match training
IMG_HEIGHT = 256
IMG_WIDTH = 256

MIN_CONTOUR_AREA = 100 

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform segmentation and cropping using a trained U-Net model."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--output_mask",
        type=str,
        default="./outputs_test/output_mask.png",
        help="Output path for the predicted binary mask image (0 or 255).",
    )
    parser.add_argument(
        "--output_cropped",
        type=str,
        default="./outputs_test/output_cropped.png",
        help="Output path for the cropped image based on the largest mask contour.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/model.h5",
        help="Path to the trained Keras (.h5 or .keras) model file.",
    )
    parser.add_argument(
         "--threshold",
         type=float,
         default=0.5,
         help="Threshold value (0.0 to 1.0) to convert probability mask to binary mask."
    )
    parser.add_argument(
         "--min_area",
         type=float,
         default=MIN_CONTOUR_AREA,
         help=f"Minimum contour area threshold for cropping (default: {MIN_CONTOUR_AREA})."
    )


    return parser.parse_args()

def load_and_preprocess_image(input_path: str, target_height: int, target_width: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[int]]:
    """ Reads, preprocesses image. Returns tensor, original BGR, original dims. """
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Error: Could not read image from {input_path}")
        return None, None, None, None
    original_height, original_width = img_bgr.shape[:2]
    img_normalized = img_bgr.astype(np.float32) / 255.0
    resized_img = cv2.resize(
        img_normalized, (target_width, target_height), interpolation=cv2.INTER_LINEAR
    )
    input_tensor = np.expand_dims(resized_img, axis=0)
    return input_tensor, img_bgr, original_height, original_width

def predict_mask(model: tf.keras.Model, input_tensor: np.ndarray) -> Optional[np.ndarray]:
    """ Runs model prediction. Returns probability mask (H, W, 1). """
    print("Running prediction...")
    try:
        prediction = model.predict(input_tensor, verbose=0)
        if prediction is not None and prediction.ndim == 4 and prediction.shape[0] == 1:
             # prediction shape (1, H, W, 1) -> return (H, W, 1)
            return prediction[0] 
        else:
            print(f"Error: Unexpected model prediction shape: {prediction.shape if prediction is not None else 'None'}")
            return None
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

def postprocess_and_save_results(
    prob_mask_pred: np.ndarray,
    original_bgr: np.ndarray,
    orig_height: int,
    orig_width: int,
    output_mask_path: str,
    output_cropped_path: str,
    binary_threshold: float = 0.5,
    min_contour_area: float = 100.0
):
    """
    Resizes mask, thresholds, saves binary mask, finds largest contour,
    crops original image based on bounding box, and saves cropped image.
    """
    if prob_mask_pred is None or original_bgr is None:
        print("Error: Invalid input provided for postprocessing.")
        return

    # --- 1. Process and Save Mask ---
    print("Processing predicted mask...")
    # Resize probability mask back to original image size
    try:
        resized_prob_mask = cv2.resize(
            prob_mask_pred, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR
        )
    except Exception as e:
         print(f"Error resizing mask: {e}")
         return
         
    # Ensure mask is 2D (H, W) after resize
    if resized_prob_mask.ndim == 3 and resized_prob_mask.shape[2] == 1:
        resized_prob_mask = resized_prob_mask.squeeze(axis=2)
    elif resized_prob_mask.ndim != 2:
        print(f"Error: Resized mask has unexpected dimensions: {resized_prob_mask.shape}")
        return

    # Create binary mask (0 or 255) using the specified threshold
    binary_mask = (resized_prob_mask > binary_threshold).astype(np.uint8) * 255

    # Save the binary mask
    print(f"Saving binary mask to {output_mask_path} ...")
    mask_dir = os.path.dirname(output_mask_path)
    if mask_dir:
        os.makedirs(mask_dir, exist_ok=True)
    try:
        if not cv2.imwrite(output_mask_path, binary_mask):
             print(f"Warning: cv2.imwrite failed to save mask to {output_mask_path}")
    except Exception as e:
        print(f"Error saving binary mask: {e}")

    # --- 2. Find Contour and Crop Original Image ---
    print("Finding largest contour for cropping...")
    # Find contours in the *binary* mask
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > min_contour_area:
            # Get the bounding box (x, y, width, height) of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Crop the *original BGR image* using the bounding box coordinates
            # Add a small safety margin if desired (e.g., max(0, y-pad), min(H, y+h+pad))
            print(f"Largest contour area: {area:.0f} > {min_contour_area:.0f}. Cropping region: (x={x}, y={y}, w={w}, h={h})")
            cropped_bgr = original_bgr[y : y + h, x : x + w]

            # Save the cropped image
            print(f"Saving cropped image to {output_cropped_path} ...")
            crop_dir = os.path.dirname(output_cropped_path)
            if crop_dir:
                 os.makedirs(crop_dir, exist_ok=True)
            try:
                if not cv2.imwrite(output_cropped_path, cropped_bgr):
                     print(f"Warning: cv2.imwrite failed to save cropped image to {output_cropped_path}")
            except Exception as e:
                print(f"Error saving cropped image: {e}")
        else:
            print(f"Largest contour area ({area:.0f}) is below minimum threshold ({min_contour_area:.0f}). Cropped image not saved.")
    else:
        print("No contours found in the binary mask. Cropped image not saved.")


def main():
    args = parse_args()

    # --- Validate Inputs ---
    if not os.path.isfile(args.input):
        print(f"Error: Input image not found -> {args.input}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Error: Model file not found -> {args.model}")
        sys.exit(1)
    if not (0.0 < args.threshold < 1.0):
         print(f"Error: Threshold must be between 0.0 and 1.0 -> {args.threshold}")
         sys.exit(1)

    # --- Load Model ---
    print(f"Loading model from {args.model} ...")
    # Define the custom objects potentially needed by this specific model file
    required_custom_objects = {
         "dice_loss": dice_loss,
         "dice_coef": dice_coef
         # Add iou_loss or iou_coef if the specific model used them
    }
    print(f"Using custom_objects for load_model: {list(required_custom_objects.keys())}")

    try:
        # Load model with compile=False for inference
        model = load_model(args.model, custom_objects=required_custom_objects, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        # Improved error message guidance
        print(f"\n--- Error loading model ---")
        print(f"{e}")
        print("\nTroubleshooting:")
        print(f"1. Is the model path correct? -> {args.model}")
        print(f"2. Are the correct custom objects defined in 'required_custom_objects' dictionary within this script?")
        print(f"   (Must match the loss/metrics used when the model was saved)")
        print(f"   Currently defined: {list(required_custom_objects.keys())}")
        print(f"3. Do TensorFlow/Keras versions match the training environment?")
        print(f"4. Consider re-saving the model using the '.keras' format if issues persist.")
        print("---------------------------\n")
        sys.exit(1)

    # --- Preprocess Image ---
    print(f"Loading and preprocessing image: {args.input} ...")
    input_tensor, original_bgr, orig_h, orig_w = load_and_preprocess_image(
        args.input, IMG_HEIGHT, IMG_WIDTH
    )
    if input_tensor is None:
        sys.exit(1)

    # --- Predict Mask ---
    probability_mask = predict_mask(model, input_tensor)
    if probability_mask is None:
        sys.exit(1)

    # --- Postprocess and Save ---
    print("Postprocessing results...")
    postprocess_and_save_results(
        probability_mask,
        original_bgr,
        orig_h,
        orig_w,
        args.output_mask,
        args.output_cropped,
        args.threshold,
        args.min_area
    )

    print("Inference script finished.")

if __name__ == "__main__":
    main()