#!/usr/bin/env python3

"""
test.py
-------
Script to test a U-Net model for image segmentation using Keras / TensorFlow.
It loads a Keras (TensorFlow) model, reads an input image, resizes and normalizes it, 
performs inference, and finally saves the resulting segmentation mask and a warped version 
of the original image based on the predicted mask.

Usage:
  python test.py /path/to/input_image.jpg 
                --output_mask /path/to/mask_out.png (optional)
                --output_prediction /path/to/prediction_out.png (optional)
                --model /path/to/model.h5
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model

import sys  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import image

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Semantic segmentation of an ID Card in an image."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input image file (with ID card).",
    )
    parser.add_argument(
        "--output_mask",
        type=str,
        default="./model/output_mask.png",
        help="Output path for the predicted mask image.",
    )
    parser.add_argument(
        "--output_prediction",
        type=str,
        default="./model/output_pred.png",
        help="Output path for the warped image (prediction).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./model/model.h5",
        help="Path to the .h5 model file.",
    )

    return parser.parse_args()


def load_and_preprocess_image(input_path):
    """
    Reads an image (in color), normalizes pixel values,
    resizes it to (256, 256), and returns the preprocessed 
    image (ready for model prediction), plus original height/width.
    """
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read the image from {input_path}")

    # Normalize pixel values to [0, 1]
    img_normalized = img_bgr / 255.0
    original_height, original_width = img_bgr.shape[:2]

    # Resize to (256, 256) - adjust as needed
    resized_img = cv2.resize(
        img_normalized, 
        (256, 256), 
        interpolation=cv2.INTER_AREA
    )
    # Prepare batch dimension for the model: (1, 256, 256, 3)
    input_tensor = resized_img.reshape((1, 256, 256, 3))

    return input_tensor, original_height, original_width


def predict_image_segmentation(model, input_tensor):
    """
    Runs model prediction on the preprocessed tensor.
    Returns the segmentation mask for the single input image.
    """
    prediction = model.predict(input_tensor, verbose=1)
    # prediction[0] is the mask for the single input; shape might be (256,256,1) or (256,256)
    return prediction[0]


def main():
    args = parse_args()

    input_file = args.input
    output_mask_file = args.output_mask
    output_prediction_file = args.output_prediction
    model_file = args.model

    # Validate input file
    if not os.path.isfile(input_file):
        print(f"Error: Input image not found: {input_file}")
        return

    # Validate model file
    if not os.path.isfile(model_file):
        print(f"Error: Model file not found: {model_file}")
        return

    print(f"Loading model from {model_file} ...")
    # If your model has custom metrics/losses, supply custom_objects here.
    # e.g. load_model(model_file, custom_objects={"mean_io_u": tf.keras.metrics.MeanIoU(num_classes=2)})
    model = load_model(model_file)

    print(f"Loading and preprocessing image from {input_file} ...")
    preprocessed_img, orig_h, orig_w = load_and_preprocess_image(input_file)

    print("Running segmentation prediction...")
    output_image = predict_image_segmentation(model, preprocessed_img)

    # Resize the predicted mask back to the original image size
    print("Resizing predicted mask to original image dimensions...")
    mask_image = cv2.resize(output_image, (orig_w, orig_h))

    # Warping the original image (based on your custom method)
    # This presumably uses the predicted mask to transform the original image
    original_bgr = cv2.imread(input_file, cv2.IMREAD_COLOR)
    warped_image = image.convert_object(mask_image, original_bgr)

    # Save output mask (grayscale)
    print(f"Saving mask image to {output_mask_file} ...")
    plt.imsave(output_mask_file, mask_image, cmap="gray")

    # Save warped or "cut out" image
    print(f"Saving warped prediction image to {output_prediction_file} ...")
    plt.imsave(output_prediction_file, warped_image)

    print("Done.")

if __name__ == "__main__":
    main()
