#!/usr/bin/env python3

"""
add_tflite_metadata.py
----------------------
Adds metadata to a TensorFlow Lite (TFLite) image segmentation model.

This script populates a TFLite model file with metadata including model
description, version, author, license, input/output tensor descriptions,
normalization parameters, and optionally, associated label files.

Based on the TensorFlow Lite metadata examples:
https://www.tensorflow.org/lite/models/convert/metadata

Usage:
  python scripts/add_tflite_metadata.py \
    --model_file=path/to/your/model.tflite \
    --export_directory=path/to/output/directory \
    --model_name="My Segmentation Model" \
    --model_version="v1.0" \
    --input_norm_mean=0.0 \
    --input_norm_std=1.0 \
    [--label_file=path/to/labels.txt] \
    [--author="Your Name"] \
    [--license_type="Apache License 2.0"]

Example for a binary U-Net model expecting input [0,1]:
  python scripts/add_tflite_metadata.py \
    --model_file=./models/model.tflite \
    --export_directory=./models_with_metadata \
    --model_name="U-Net ID Card Segmentation" \
    --input_norm_mean=0.0 \
    --input_norm_std=1.0
"""

import argparse
import os
import sys
import tensorflow as tf
from typing import List, Optional, Sequence, Union

# Ensure tflite-support is installed: pip install tflite-support
try:
    import flatbuffers # Required by tflite-support
    from tflite_support import metadata_schema_py_generated as _metadata_fb
    from tflite_support import metadata as _metadata
    print("TensorFlow Lite Support library imported successfully.")
except ImportError:
    print("="*70)
    print("ERROR: The 'tflite-support' library is required but not found.")
    print("Please install it:")
    print("  pip install tflite-support")
    print("="*70)
    sys.exit(1)

# Suppress TensorFlow INFO/WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

print(f"Using TensorFlow version: {tf.__version__}")

# Default license text (can be overridden)
DEFAULT_LICENSE = "Apache License Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0)"

def parse_args() -> argparse.Namespace:
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser(
        description="Add metadata to a TFLite image segmentation model."
    )
    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="Path to the input TFLite model file.",
    )
    parser.add_argument(
        "--export_directory",
        type=str,
        required=True,
        help="Directory where the TFLite model with metadata will be saved.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Image Segmentation Model",
        help="Name of the model to be stored in metadata.",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="v1",
        help="Version of the model."
    )
    parser.add_argument(
        "--model_description",
        type=str,
        default="Performs pixel-wise segmentation on an input image.",
        help="Description of the model's function."
    )
    parser.add_argument(
        "--author",
        type=str,
        default="Unknown Author",
        help="Author of the model."
    )
    parser.add_argument(
        "--license_type",
        type=str,
        default=DEFAULT_LICENSE,
        help="License information for the model."
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default=None, # Optional
        help="Path to the label file (e.g., labels.txt). Required for multi-class "
             "segmentation if associating labels with output tensor."
    )
    # Normalization parameters MUST match model training and inference preprocessing
    parser.add_argument(
        "--input_norm_mean",
        type=float,
        nargs='+', # Allows multiple values, e.g., --input_norm_mean 127.5 127.5 127.5
        default=[0.0],
        help="Mean value(s) used for input normalization (e.g., [0.0] for [0,1] scaling, "
             "[127.5] or [127.5, 127.5, 127.5] for [-1,1] scaling)."
    )
    parser.add_argument(
        "--input_norm_std",
        type=float,
        nargs='+',
        default=[1.0],
        help="Standard deviation value(s) used for input normalization (e.g., [1.0] for [0,1] scaling, "
             "[127.5] or [127.5, 127.5, 127.5] for [-1,1] scaling)."
    )
    # Input tensor range (before normalization) - typically 0-255 for uint8 input
    parser.add_argument(
        "--input_min", type=int, default=0, help="Minimum expected input tensor value before normalization."
    )
    parser.add_argument(
         "--input_max", type=int, default=255, help="Maximum expected input tensor value before normalization."
    )

    return parser.parse_args()


class MetadataPopulatorForSegmentation:
    """Populates metadata for an image segmentation TFLite model."""

    def __init__(self,
                 model_file: str,
                 export_path: str,
                 model_name: str,
                 model_version: str,
                 model_description: str,
                 author: str,
                 license_type: str,
                 input_norm_mean: List[float],
                 input_norm_std: List[float],
                 input_min: int,
                 input_max: int,
                 label_file_path: Optional[str] = None):
        
        self.model_file = model_file
        self.export_path = export_path
        self.model_name = model_name
        self.model_version = model_version
        self.model_description = model_description
        self.author = author
        self.license_type = license_type
        self.input_norm_mean = input_norm_mean
        self.input_norm_std = input_norm_std
        self.input_min = input_min
        self.input_max = input_max
        self.label_file_path = label_file_path
        self.metadata_buf = None

        # --- Get tensor details from model ---
        self._get_tensor_details()

    def _get_tensor_details(self):
        """ Uses Interpreter to find input/output tensor details. """
        interpreter = tf.lite.Interpreter(model_path=self.model_file)
        interpreter.allocate_tensors() # Needed to query details

        input_details = interpreter.get_input_details()
        if not input_details:
             raise ValueError("Could not get input details from the TFLite model.")
        self.input_detail = input_details[0] # Assume single input

        output_details = interpreter.get_output_details()
        if not output_details:
             raise ValueError("Could not get output details from the TFLite model.")
        self.output_detail = output_details[0] # Assume single output

        print("\n--- Detected Tensor Details ---")
        print(f"Input Name : {self.input_detail['name']} | Shape: {self.input_detail['shape']} | Type: {self.input_detail['dtype']}")
        print(f"Output Name: {self.output_detail['name']} | Shape: {self.output_detail['shape']} | Type: {self.output_detail['dtype']}")
        print("-----------------------------\n")

        # Extract required info (e.g., dimensions) - assuming NHWC format
        try:
             self.input_height = int(self.input_detail['shape'][1])
             self.input_width = int(self.input_detail['shape'][2])
             self.num_output_classes = int(self.output_detail['shape'][-1]) # Last dimension
        except (IndexError, TypeError) as e:
             raise ValueError(f"Could not parse tensor shapes for H/W/Classes: {e}")


    def _create_metadata(self):
        """Creates the metadata structure."""

        # --- Model Info ---
        model_meta = _metadata_fb.ModelMetadataT()
        model_meta.name = self.model_name
        model_meta.description = self.model_description
        model_meta.version = self.model_version
        model_meta.author = self.author
        model_meta.license = self.license_type

        # --- Input Tensor Info ---
        input_meta = _metadata_fb.TensorMetadataT()
        input_meta.name = self.input_detail["name"] # Use detected name
        input_meta.description = (
            f"Input image for segmentation. Expected {self.input_height}x{self.input_width} RGB image. "
            f"Pixel values should be normalized using mean={self.input_norm_mean}, std={self.input_norm_std}."
        )
        input_meta.content = _metadata_fb.ContentT()
        input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
        input_meta.content.contentProperties.colorSpace = _metadata_fb.ColorSpaceType.RGB
        input_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.ImageProperties

        # Normalization Process Unit
        input_normalization = _metadata_fb.ProcessUnitT()
        input_normalization.optionsType = _metadata_fb.ProcessUnitOptions.NormalizationOptions
        input_normalization.options = _metadata_fb.NormalizationOptionsT()
        # Ensure mean/std match expected number of channels if needed (often 1 value is broadcast)
        input_normalization.options.mean = self.input_norm_mean
        input_normalization.options.std = self.input_norm_std
        input_meta.processUnits = [input_normalization]

        # Input Stats (Range before normalization)
        input_stats = _metadata_fb.StatsT()
        input_stats.max = [float(self.input_max)] # Ensure float
        input_stats.min = [float(self.input_min)] # Ensure float
        input_meta.stats = input_stats

        # --- Output Tensor Info ---
        output_meta = _metadata_fb.TensorMetadataT()
        output_meta.name = self.output_detail["name"] # Use detected name
        
        if self.num_output_classes == 1:
            output_meta.description = (
                f"Output probability mask for the foreground class (sigmoid activation). "
                f"Shape: {self.output_detail['shape']}. Values are in [0.0, 1.0]."
             )
        else:
             output_meta.description = (
                f"Output probability map for {self.num_output_classes} classes (softmax activation). "
                f"Shape: {self.output_detail['shape']}. Each channel corresponds to a class probability."
             )
             
        # Content type depends on what the output represents.
        # For segmentation masks, often ImageProperties or FeatureProperties are used.
        # Let's describe it as an Image where each pixel/channel has meaning.
        output_meta.content = _metadata_fb.ContentT()
        # If multi-class output corresponds to labels, contentProperties could be FeatureProperties.
        # If output is mask (binary/multi-channel), ImageProperties might fit better.
        # Let's use ImageProperties for a segmentation mask output.
        output_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
        output_meta.content.contentProperties.colorSpace = _metadata_fb.ColorSpaceType.GRAYSCALE if self.num_output_classes == 1 else _metadata_fb.ColorSpaceType.UNKNOWN # Or RGB if channels=3? Needs context.
        output_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.ImageProperties
        
        # Output Stats (Range after activation)
        output_stats = _metadata_fb.StatsT()
        output_stats.max = [1.0]
        output_stats.min = [0.0]
        output_meta.stats = output_stats

        # --- Associated Files (Labels - Optional) ---
        if self.label_file_path:
            if not os.path.isfile(self.label_file_path):
                 print(f"Warning: Label file not found at {self.label_file_path}. Skipping association.")
            else:
                 print(f"Associating label file: {self.label_file_path}")
                 label_file = _metadata_fb.AssociatedFileT()
                 label_file.name = os.path.basename(self.label_file_path)
                 label_file.description = f"Labels for the {self.num_output_classes} output classes."
                 # TENSOR_AXIS_LABELS is common for classification output axis,
                 # For segmentation output tensor (H, W, C), associating labels might need different type or TENSOR_VALUE_LABELS?
                 # Check metadata spec - TENSOR_AXIS_LABELS often assumes axis 1.
                 # Let's stick with TENSOR_AXIS_LABELS, assuming it applies to the channel dimension.
                 label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
                 output_meta.associatedFiles = [label_file]
                 
                 # For segmentation, maybe define label content range? Needs checking.
                 # output_meta.content.range = _metadata_fb.ValueRangeT()
                 # output_meta.content.range.min = 0
                 # output_meta.content.range.max = self.num_output_classes - 1

        # --- Subgraph Info ---
        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = [output_meta]
        model_meta.subgraphMetadata = [subgraph]

        # --- Build Flatbuffer ---
        builder = flatbuffers.Builder(0)
        builder.Finish(
            model_meta.Pack(builder),
            _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        self.metadata_buf = builder.Output()
        print("Metadata buffer created successfully.")

    def populate(self):
        """Populates the metadata and associated files to the model file."""
        if self.metadata_buf is None:
            self._create_metadata()

        print(f"Populating metadata to: {self.export_path}")
        populator = _metadata.MetadataPopulator.with_model_file(self.export_path)
        populator.load_metadata_buffer(self.metadata_buf)

        associated_files_to_load = []
        if self.label_file_path and os.path.isfile(self.label_file_path):
             associated_files_to_load.append(self.label_file_path)
             
        if associated_files_to_load:
            print(f"Loading associated files: {associated_files_to_load}")
            populator.load_associated_files(associated_files_to_load)

        populator.populate()
        print("Metadata populated successfully.")


def main():
    args = parse_args()

    # --- Validate Paths ---
    if not os.path.isfile(args.model_file):
        print(f"Error: Input TFLite model not found -> {args.model_file}")
        sys.exit(1)
    if args.label_file and not os.path.isfile(args.label_file):
         print(f"Warning: Specified label file not found -> {args.label_file}. Proceeding without it.")
         args.label_file = None # Clear it so it's not used later
         
    if not args.export_directory:
         print(f"Error: Export directory must be specified.")
         sys.exit(1)

    # Create export directory if it doesn't exist
    os.makedirs(args.export_directory, exist_ok=True)
    print(f"Ensured export directory exists: {args.export_directory}")

    # Define the final export path for the model with metadata
    model_basename = os.path.basename(args.model_file)
    export_model_path = os.path.join(args.export_directory, model_basename)

    # --- Copy Model to Export Directory ---
    # Metadata is written directly into the file, so work on a copy
    print(f"Copying original model to export location: {export_model_path}")
    try:
        tf.io.gfile.copy(args.model_file, export_model_path, overwrite=True)
    except Exception as e:
        print(f"Error copying model file: {e}")
        sys.exit(1)

    # --- Populate Metadata ---
    try:
        print("Initializing metadata populator...")
        populator = MetadataPopulatorForSegmentation(
            model_file=args.model_file, # Pass original path to read tensor details
            export_path=export_model_path, # Path where metadata will be written
            model_name=args.model_name,
            model_version=args.model_version,
            model_description=args.model_description,
            author=args.author,
            license_type=args.license_type,
            input_norm_mean=args.input_norm_mean,
            input_norm_std=args.input_norm_std,
            input_min=args.input_min,
            input_max=args.input_max,
            label_file_path=args.label_file
        )
        populator.populate() # Creates and writes metadata

    except Exception as e:
        print(f"\n--- Error during metadata population ---")
        print(f"{e}")
        # Optionally clean up copied file?
        # if os.path.exists(export_model_path): os.remove(export_model_path)
        print("---------------------------------------\n")
        sys.exit(1)

    # --- Validation (Optional but Recommended) ---
    try:
        print("\nValidating model metadata...")
        displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)

        # Display metadata summary
        print("--- Metadata Summary ---")
        print(displayer.get_metadata_json())
        print("-----------------------")

        # Display associated files
        print("Packed Associated Files:")
        print(displayer.get_packed_associated_file_list())
        print("-----------------------")

        # Save metadata JSON file
        export_json_file = os.path.join(args.export_directory,
                                        os.path.splitext(model_basename)[0] + "_metadata.json")
        metadata_json = displayer.get_metadata_json()
        print(f"Saving metadata JSON to: {export_json_file}")
        with open(export_json_file, "w") as f:
            f.write(metadata_json)

    except Exception as e:
        print(f"\nWarning: Error during metadata validation/display: {e}")
        # Don't exit, population might have succeeded even if display fails

    print("\nScript finished successfully.")
    print(f"Model with metadata saved to: {export_model_path}")


if __name__ == "__main__":
    # Note: No need for absl.app.run() if using argparse
    main()