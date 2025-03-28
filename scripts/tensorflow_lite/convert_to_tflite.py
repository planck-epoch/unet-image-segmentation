#!/usr/bin/env python3

"""
convert_to_tflite.py
--------------------
Converts a trained Keras model (.h5 or .keras format) into the TensorFlow Lite
(.tflite) format, optionally applying optimizations like float16 quantization.

Usage:
  python scripts/convert_to_tflite.py <input_model_path> <output_tflite_path> [options]

Example:
  # Basic conversion
  python scripts/convert_to_tflite.py ./models/model.h5 ./models/model.tflite

  # Conversion with default optimizations (might include float16)
  python scripts/convert_to_tflite.py ./models/model.h5 ./models/model_optimized.tflite --optimize

  # Explicit float16 quantization
  python scripts/convert_to_tflite.py ./models/model.h5 ./models/model_fp16.tflite --float16
"""

import argparse
import os
import sys
import tensorflow as tf
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from utils.loss import dice_loss, iou_loss, jaccard_loss
    from utils.metrics import dice_coef, iou_coef
    print("Successfully imported custom loss/metrics functions.")
except ImportError as e:
    print(f"Warning: Could not import custom objects from utils: {e}")
    print("Proceeding without them. If the model requires custom objects, loading will fail.")
    # Define placeholders if imports fail, load_model will error later if they were needed
    dice_loss = iou_loss = jaccard_loss = dice_coef = iou_coef = None

# Suppress TensorFlow INFO/WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

print(f"Using TensorFlow version: {tf.__version__}")

def parse_args() -> argparse.Namespace:
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser(
        description="Convert a Keras model to TensorFlow Lite (.tflite)."
    )
    parser.add_argument(
        "input_model",
        type=str,
        help="Path to the input Keras model file (.h5 or .keras)."
    )
    parser.add_argument(
        "output_tflite",
        type=str,
        help="Path to save the output TensorFlow Lite model (.tflite)."
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply default TFLite optimizations (may include quantization, e.g., to float16)."
    )
    parser.add_argument(
        "--float16",
        action="store_true",
        help="Explicitly enable float16 quantization."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.isfile(args.input_model):
        print(f"Error: Input model file not found -> {args.input_model}")
        sys.exit(1)

    required_custom_objects: Dict[str, Any] = {
        "dice_loss": dice_loss,
        "dice_coef": dice_coef
    }
    required_custom_objects = {k: v for k, v in required_custom_objects.items() if v is not None}
    print(f"Using custom_objects for loading Keras model: {list(required_custom_objects.keys())}")

    # --- Load Keras Model ---
    print(f"Loading Keras model from: {args.input_model}...")
    try:
        # Load model without compiling optimizer state (usually not needed for conversion)
        model = tf.keras.models.load_model(
            args.input_model,
            custom_objects=required_custom_objects,
            compile=False
        )
        print("Keras model loaded successfully.")
        model.summary()
    except Exception as e:
        print(f"\n--- Error loading Keras model ---")
        print(f"{e}")
        print("\nCheck:")
        print(f"1. If the model path is correct: {args.input_model}")
        print(f"2. If the necessary custom objects are correctly defined in 'required_custom_objects'.")
        print(f"   (Must match loss/metrics used during original training/saving)")
        print(f"   Functions available: {[k for k,v in list(globals().items()) if callable(v) and k.endswith(('_loss','_coef'))]}") # List available functions
        print(f"3. If TensorFlow/Keras versions are compatible with the saved model.")
        print("-----------------------------------\n")
        sys.exit(1)

    # --- Convert to TensorFlow Lite ---
    print("\nConverting model to TensorFlow Lite format...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Apply Optimizations / Quantization if requested
        optimizations = []
        if args.optimize:
            print("Applying default optimizations...")
            optimizations.append(tf.lite.Optimize.DEFAULT)
            converter.optimizations = optimizations
            
        # Explicit Float16 Quantization
        if args.float16:
            print("Applying float16 quantization...")
            # Ensure default optimizations aren't conflicting if both flags are set
            # (Usually DEFAULT includes float16 where possible, but this makes it explicit)
            if tf.lite.Optimize.DEFAULT not in optimizations:
                 converter.optimizations = optimizations # Apply if not already done
            converter.target_spec.supported_types = [tf.float16]

        # Add INT8 quantization logic here if needed (requires representative dataset)
        # if args.int8:
        #    print("Applying int8 quantization...")
        #    def representative_dataset():
        #        # Create a generator yielding samples from your training/validation data
        #        # E.g., for _ in range(100): yield [input_image_batch]
        #        raise NotImplementedError("Representative dataset generator required for INT8 quantization.")
        #    converter.optimizations = [tf.lite.Optimize.DEFAULT] # Usually needed for INT8
        #    converter.representative_dataset = representative_dataset
        #    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        #    converter.inference_input_type = tf.int8 # Or tf.uint8
        #    converter.inference_output_type = tf.int8 # Or tf.uint8

        # Perform the conversion
        tflite_model = converter.convert()
        print("Model converted successfully.")

    except Exception as e:
        print(f"\n--- Error during TFLite conversion ---")
        print(f"{e}")
        print("\nCheck TensorFlow Lite documentation for compatibility issues.")
        print("---------------------------------------\n")
        sys.exit(1)

    # --- Save TFLite Model ---
    print(f"Saving TFLite model to: {args.output_tflite}...")
    try:
        output_dir = os.path.dirname(args.output_tflite)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        with open(args.output_tflite, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved successfully ({os.path.getsize(args.output_tflite)/1024:.2f} KB).")

    except Exception as e:
        print(f"\n--- Error saving TFLite model ---")
        print(f"{e}")
        print("-----------------------------------\n")
        sys.exit(1)

    print("\nConversion script finished.")

if __name__ == '__main__':
    main()