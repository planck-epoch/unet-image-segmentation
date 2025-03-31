#!/usr/bin/env python3

"""
train.py
--------
Trains a U-Net model for binary semantic segmentation using TensorFlow/Keras.

This script sets up data generators with optional augmentation for image/mask pairs,
builds the U-Net model (defined in `model/u_net.py`), compiles it using the
AdamW optimizer and Dice loss, and trains it using standard Keras callbacks
(ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard).

Dataset Structure Expectation:
The script assumes data has been prepared (e.g., by `scripts/prepare_dataset.py`)
into the following structure relative to the project root:
./dataset/train/
├── train_frames/
│   └── image/      # Training images (e.g., imgA.jpg)
├── train_masks/
│   └── image/      # Training masks (e.g., imgA.png)
├── val_frames/
│   └── image/      # Validation images (e.g., imgB.jpg)
└── val_masks/
    └── image/      # Validation masks (e.g., imgB.png)

Usage:
  python scripts/train.py [options]

Options:
  --epochs          Number of training epochs (default: 30)
  --batch-size      Batch size for training and validation (default: 2)
  --learning-rate   Initial learning rate for AdamW (default: 0.001)
  --weight-decay    Weight decay for AdamW (default: 0.0001)
  --model-out       Path to save the best trained model (default: ./models/model.h5)

Example:
  # Train for 50 epochs with batch size 4 and specific output path
  python scripts/train.py --epochs 50 --batch-size 4 --model-out ./models/unet_custom.h5

Monitoring:
  TensorBoard logs are saved to ./logs/YYYYMMDD_HHMMSS/
  Run: tensorboard --logdir ./logs
"""

import os
import random as rn
import time
import argparse
import numpy as np
import tensorflow as tf
import sys
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import MeanIoU

from model.u_net import U_NET
from utils.loss import dice_loss
from utils.metrics import dice_coef

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
print(f"Using TensorFlow version: {tf.__version__}")

DEFAULT_EPOCHS = 30
DEFAULT_BATCHSIZE = 2
DEFAULT_LR = 2e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MODEL_OUT = "./models/model.h5"

SEED = 2301

TRAIN_FRAMES_DIR = "dataset/train/train_frames/image"
TRAIN_MASKS_DIR = "dataset/train/train_masks/image"
VAL_FRAMES_DIR = "dataset/train/val_frames/image"
VAL_MASKS_DIR = "dataset/train/val_masks/image"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3
MODEL_INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
TARGET_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

NUM_CLASSES = 1

def parse_args() -> argparse.Namespace:
    """ Parses command-line arguments for training configuration. """
    parser = argparse.ArgumentParser(
        description="Train a U-Net model for binary segmentation using AdamW."
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})."
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCHSIZE,
        help=f"Batch size (default: {DEFAULT_BATCHSIZE})."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LR,
        help=f"Initial learning rate for AdamW optimizer (default: {DEFAULT_LR})."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY,
        help=f"Weight decay for AdamW optimizer (default: {DEFAULT_WEIGHT_DECAY})."
    )
    parser.add_argument(
        "--model-out", type=str, default=DEFAULT_MODEL_OUT,
        help=f"File path to save the best trained model (default: {DEFAULT_MODEL_OUT})."
    )
    return parser.parse_args()

def fix_gpu():
    """ Enables memory growth for GPUs to prevent TensorFlow from allocating all memory at once. """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {len(gpus)} GPU(s).")
        else:
            print("No GPU detected by TensorFlow. Training will use CPU.")
    except Exception as e:
        print(f"GPU configuration warning (continuing with CPU or default GPU settings): {e}")

def combine_generators(image_gen, mask_gen):
    """ Combines image and mask generators to yield (image_batch, mask_batch) tuples. """
    while True:
        try:
            yield (next(image_gen), next(mask_gen))
        except StopIteration:
            print("Warning: Data generator exhausted unexpectedly.")
            return

def main():
    """ Main function to run the training pipeline. """
    args = parse_args()

    rn.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    MODEL_OUT = args.model_out

    # Attempt to configure GPU memory growth
    fix_gpu()
    
    print(f"\n--- Training Configuration ---")
    print(f"Epochs        : {EPOCHS}")
    print(f"Batch Size    : {BATCH_SIZE}")
    print(f"Learning Rate : {LR}")
    print(f"Weight Decay  : {WEIGHT_DECAY} (for AdamW)")
    print(f"Model Output  : {MODEL_OUT}")
    print(f"Input Shape   : {MODEL_INPUT_SHAPE}")
    print(f"Target Size   : {TARGET_SIZE}")
    print(f"Seed          : {SEED}")
    print(f"------------------------------\n")

    train_data_gen_args = dict(
        rescale=1.0/255.0,
        horizontal_flip=True,
        # Consider adding more augmentation like rotation, zoom, shift
        # if needed for your specific dataset. Ensure 'fill_mode' is
        # appropriate for masks (e.g., 'nearest' or 'constant' with cval=0).
    )
    # Validation data should not be augmented, only rescaled
    # (to avoid introducing noise into validation metrics)
    val_data_gen_args = dict(rescale=1.0/255.0)

    print("Setting up Data Generators...")
    try:
        train_img_datagen = ImageDataGenerator(**train_data_gen_args)
        train_mask_datagen = ImageDataGenerator(**train_data_gen_args)
        val_img_datagen = ImageDataGenerator(**val_data_gen_args)
        val_mask_datagen = ImageDataGenerator(**val_data_gen_args)

        train_image_generator = train_img_datagen.flow_from_directory(
            directory=os.path.dirname(TRAIN_FRAMES_DIR), classes=[os.path.basename(TRAIN_FRAMES_DIR)],
            target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode=None, color_mode="rgb", seed=SEED,
            interpolation="bilinear"
        )
        train_mask_generator = train_mask_datagen.flow_from_directory(
            directory=os.path.dirname(TRAIN_MASKS_DIR), classes=[os.path.basename(TRAIN_MASKS_DIR)],
            target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode=None, color_mode="grayscale", seed=SEED,
            interpolation="nearest"
        )
        val_image_generator = val_img_datagen.flow_from_directory(
            directory=os.path.dirname(VAL_FRAMES_DIR), classes=[os.path.basename(VAL_FRAMES_DIR)],
            target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode=None, color_mode="rgb", seed=SEED,
            interpolation="bilinear", shuffle=False
        )
        val_mask_generator = val_mask_datagen.flow_from_directory(
            directory=os.path.dirname(VAL_MASKS_DIR), classes=[os.path.basename(VAL_MASKS_DIR)],
            target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode=None, color_mode="grayscale", seed=SEED,
            interpolation="nearest", shuffle=False
        )
        print("Data Generators created successfully.")
    except Exception as e:
        print(f"\n--- Error initializing ImageDataGenerator ---")
        print(f"{e}")
        print("Please ensure dataset directories exist and follow the expected structure:")
        print(f"  Train Images: {TRAIN_FRAMES_DIR}/..")
        print(f"  Train Masks : {TRAIN_MASKS_DIR}/..")
        print(f"  Val Images  : {VAL_FRAMES_DIR}/..")
        print(f"  Val Masks   : {VAL_MASKS_DIR}/..")
        print("-------------------------------------------\n")
        sys.exit(1)

    train_generator = combine_generators(train_image_generator, train_mask_generator)
    val_generator = combine_generators(val_image_generator, val_mask_generator)

    print("Building U-Net model...")
    model = U_NET(input_size=MODEL_INPUT_SHAPE, num_classes=NUM_CLASSES)

    print("Compiling model with AdamW optimizer and Dice loss...")
    optimizer = AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)
    model.compile(
        optimizer=optimizer,
        loss=dice_loss,
        metrics=[
            MeanIoU(num_classes=2, name='mean_io_u'),
            dice_coef 
        ]
    )
    model.summary(line_length=100)

    try:
        no_of_training_images = train_image_generator.samples
        no_of_val_images = val_image_generator.samples
        print(f"Found {no_of_training_images} training samples and {no_of_val_images} validation samples.")
    except Exception as e:
        print(f"Warning: Could not read sample count from generators ({e}). Falling back to file listing.")
        try:
            no_of_training_images = len(os.listdir(TRAIN_FRAMES_DIR))
            no_of_val_images = len(os.listdir(VAL_FRAMES_DIR))
            print(f"(Fallback) Counted {no_of_training_images} train files, {no_of_val_images} val files.")
        except FileNotFoundError as fe:
            print(f"Error counting files: {fe}. Please check dataset paths.")
            sys.exit(1)

    if no_of_training_images == 0 or no_of_val_images == 0:
        print("Error: No training or validation images found/loaded. Check dataset paths and contents.")
        sys.exit(1)

    # Calculate steps, ensuring at least 1 step even if dataset < batch_size
    steps_per_epoch = max(1, no_of_training_images // BATCH_SIZE)
    validation_steps = max(1, no_of_val_images // BATCH_SIZE)
    print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")
    if no_of_training_images < BATCH_SIZE:
         print(f"Warning: Training dataset size ({no_of_training_images}) < batch size ({BATCH_SIZE}).")
    if no_of_val_images < BATCH_SIZE:
         print(f"Warning: Validation dataset size ({no_of_val_images}) < batch size ({BATCH_SIZE}).")

    monitor_metric = 'val_mean_io_u' # Common choices: 'val_mean_io_u', 'val_dice_coef', 'val_loss'
    monitor_mode = 'max' if 'loss' not in monitor_metric else 'min'
    print(f"Setting up Callbacks - Monitoring: '{monitor_metric}' (mode: {monitor_mode})")

    model_dir = os.path.dirname(MODEL_OUT)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        print(f"Ensured model save directory exists: {model_dir}")

    checkpoint = ModelCheckpoint(
        filepath=MODEL_OUT,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=10,
        mode=monitor_mode,
        restore_best_weights=True, 
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.2,
        patience=3,
        mode=monitor_mode,
        min_lr=1e-6, 
        verbose=1
    )

    log_dir = os.path.join("./logs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks_list = [checkpoint, early_stopping, reduce_lr, tensorboard]

    print(f"\n--- Starting Training ({EPOCHS} epochs) ---")
    try:
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            verbose=1 # 1 = progress bar, 2 = one line per epoch
        )
        print("\n--- Training complete ---")
        best_score = np.inf if monitor_mode == 'min' else -np.inf
        stopped_epoch = EPOCHS
        
        if early_stopping.stopped_epoch > 0:
            stopped_epoch = early_stopping.stopped_epoch + 1 # epochs are 0-indexed in callback
            print(f"Early stopping triggered at epoch {stopped_epoch}")
            best_score = early_stopping.best
        else:
             scores = history.history.get(monitor_metric)
             if scores:
                 best_score = min(scores) if monitor_mode == 'min' else max(scores)
        
        print(f"Best monitored score ({monitor_metric}): {best_score:.4f} (from epoch {stopped_epoch - early_stopping.patience if early_stopping.stopped_epoch > 0 else np.argmax(history.history[monitor_metric]) + 1 if monitor_mode == 'max' and monitor_metric in history.history else 'N/A'})")
        print(f"Best model saved to: {MODEL_OUT}")

    except Exception as e:
        print(f"\n--- Error during model training ---")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        print("-----------------------------------\n")
        sys.exit(1)
    except KeyboardInterrupt:
         print("\n--- Training interrupted by user ---")
         print(f"Model state might not be saved correctly to {MODEL_OUT} unless a checkpoint occurred.")
         sys.exit(1)


if __name__ == "__main__":
    main()