#!/usr/bin/env python3
"""
train.py
--------
Script to train a U-Net model for image segmentation using Keras / TensorFlow.
It expects:
  - Color images in train_frames/image (3 channels, color_mode="rgb")
  - Single-channel (grayscale) masks in train_masks/image (1 channel, color_mode="grayscale")

Folder structure assumption:
  dataset/train/
  ├── train_frames/image
  ├── train_masks/image
  ├── val_frames/image
  └── val_masks/image

Requirements:
  - Keras / TensorFlow
  - GPU recommended (though not required)
  - A U-Net definition that outputs (H,W,1) for binary segmentation
Usage:
  $ python scripts/train.py \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --model-out ./models/unet_model.h5
"""
import os
import random as rn
import time
import argparse
import numpy as np
import tensorflow as tf

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Keras / TF imports
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.metrics import MeanIoU
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model.u_net import U_NET  # or wherever your U-Net definition resides

# Default hyperparams (can be overridden via command-line)
DEFAULT_EPOCHS    = 30
DEFAULT_BATCHSIZE = 2
DEFAULT_LR        = 1e-3
DEFAULT_MODEL_OUT = "model.h5"

# For consistent results
SEED = 2301
rn.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Directory assumptions
TRAIN_FRAMES_DIR = "dataset/train/train_frames/image"      # color images
TRAIN_MASKS_DIR  = "dataset/train/train_masks/image"       # grayscale masks
VAL_FRAMES_DIR   = "dataset/train/val_frames/image"
VAL_MASKS_DIR    = "dataset/train/val_masks/image"

IMAGE_SIZE       = (256, 256)
NUM_CLASSES      = 2

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a U-Net model for binary segmentation."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of epochs (default: 30)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCHSIZE,
        help="Batch size (default: 2)."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate for Adam optimizer (default: 1e-3)."
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=DEFAULT_MODEL_OUT,
        help="File path to save the trained model (default: model.h5)."
    )
    return parser.parse_args()

def fix_gpu():
    """Optional GPU config: enable memory growth if GPU is available."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU config error:", e)

def combine_generators(image_gen, mask_gen):
    """
    Returns a generator that yields (image_batch, mask_batch).
    Assumes both 'flow_from_directory' calls produce matching batch sizes.
    """
    while True:
        x_batch = next(image_gen)  # shape (batch, 256, 256, 3)
        y_batch = next(mask_gen)   # shape (batch, 256, 256, 1)
        yield (x_batch, y_batch)

def main():
    args = parse_args()

    EPOCHS      = args.epochs
    BATCH_SIZE  = args.batch_size
    LR          = args.learning_rate
    MODEL_OUT   = args.model_out

    fix_gpu()

    print(f"Starting training with epochs={EPOCHS}, batch_size={BATCH_SIZE}, learning_rate={LR}")
    print(f"Model output file: {MODEL_OUT}")

    # Data augmentation & rescaling
    train_img_datagen  = ImageDataGenerator(rescale=1.0/255.0)
    train_mask_datagen = ImageDataGenerator(rescale=1.0/255.0)
    val_img_datagen    = ImageDataGenerator(rescale=1.0/255.0)
    val_mask_datagen   = ImageDataGenerator(rescale=1.0/255.0)

    # Flow for images (RGB)
    train_image_generator = train_img_datagen.flow_from_directory(
        directory=os.path.dirname(TRAIN_FRAMES_DIR),
        classes=[os.path.basename(TRAIN_FRAMES_DIR)],
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=None,
        color_mode="rgb",
        seed=SEED
    )
    val_image_generator = val_img_datagen.flow_from_directory(
        directory=os.path.dirname(VAL_FRAMES_DIR),
        classes=[os.path.basename(VAL_FRAMES_DIR)],
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=None,
        color_mode="rgb",
        seed=SEED
    )

    # Flow for masks (grayscale)
    train_mask_generator = train_mask_datagen.flow_from_directory(
        directory=os.path.dirname(TRAIN_MASKS_DIR),
        classes=[os.path.basename(TRAIN_MASKS_DIR)],
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=None,
        color_mode="grayscale",
        seed=SEED
    )
    val_mask_generator = val_mask_datagen.flow_from_directory(
        directory=os.path.dirname(VAL_MASKS_DIR),
        classes=[os.path.basename(VAL_MASKS_DIR)],
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=None,
        color_mode="grayscale",
        seed=SEED
    )

    # Combine into (X, Y) pairs
    train_generator = combine_generators(train_image_generator, train_mask_generator)
    val_generator   = combine_generators(val_image_generator, val_mask_generator)

    # Build U-Net: input=(256,256,3), output=(256,256,1)
    model = U_NET((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy", MeanIoU(num_classes=NUM_CLASSES)]
    )
    model.summary()

    # Count images for steps_per_epoch
    no_of_training_images = len(os.listdir(TRAIN_FRAMES_DIR))
    no_of_val_images      = len(os.listdir(VAL_FRAMES_DIR))

    # Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_OUT, monitor="val_mean_io_u", mode="max", save_best_only=False
    )
    early_stopping = EarlyStopping(
        monitor="val_mean_io_u", patience=10, mode="max"
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_mean_io_u", factor=0.2, patience=3, mode="max"
    )
    tensorboard = TensorBoard(
        log_dir="./logs/" + time.strftime("%Y%m%d_%H%M%S")
    )

    # Train
    model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=no_of_training_images // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=no_of_val_images // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard]
    )

    print("Training complete.")

if __name__ == "__main__":
    main()