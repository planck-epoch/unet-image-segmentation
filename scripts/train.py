#!/usr/bin/env python3
"""
train.py
--------
Script to train a U-Net model for image segmentation using Keras / TensorFlow.
Saves the best model by default to ./models/model.h5
...
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import MeanIoU 

# Import model, loss, and metrics
from model.u_net import U_NET
from utils.loss import dice_loss # Corrected import name
from utils.metrics import dice_coef # Import dice_coef metric

# Default hyperparams
DEFAULT_EPOCHS    = 30
DEFAULT_BATCHSIZE = 2
DEFAULT_LR        = 1e-3
DEFAULT_MODEL_OUT = "./models/model.h5"

# For consistent results
SEED = 2301
rn.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Directory assumptions
TRAIN_FRAMES_DIR = "dataset/train/train_frames/image"
TRAIN_MASKS_DIR  = "dataset/train/train_masks/image"
VAL_FRAMES_DIR   = "dataset/train/val_frames/image"
VAL_MASKS_DIR    = "dataset/train/val_masks/image"

# Image dimensions
IMAGE_HEIGHT     = 256
IMAGE_WIDTH      = 256
IMAGE_CHANNELS   = 3
MODEL_INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
TARGET_SIZE      = (IMAGE_HEIGHT, IMAGE_WIDTH)

NUM_CLASSES      = 1

def parse_args():
    """
    Parse command-line arguments for training configuration.
    """
    parser = argparse.ArgumentParser(
        description="Train a U-Net model for binary segmentation."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of epochs (default: {DEFAULT_EPOCHS})." # Use f-string
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCHSIZE,
        help=f"Batch size (default: {DEFAULT_BATCHSIZE})." # Use f-string
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate for Adam optimizer (default: {DEFAULT_LR})." # Use f-string
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=DEFAULT_MODEL_OUT,
        help=f"File path to save the trained model (default: {DEFAULT_MODEL_OUT})." 
    )
    return parser.parse_args()

def fix_gpu():
    """ Optional GPU config. """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {len(gpus)} GPU(s).")
        else:
            print("No GPU found or TF not built with GPU support.")
    except Exception as e:
        print("GPU config error:", e)

def combine_generators(image_gen, mask_gen):
    """ Yields (image_batch, mask_batch) tuples. """
    while True:
        yield (next(image_gen), next(mask_gen))

def main():
    args = parse_args()

    EPOCHS      = args.epochs
    BATCH_SIZE  = args.batch_size
    LR          = args.learning_rate
    MODEL_OUT   = args.model_out 

    fix_gpu()

    print(f"Config: Epochs={EPOCHS}, BatchSize={BATCH_SIZE}, LR={LR}")
    print(f"Model output path: {MODEL_OUT}") 

    # --- Data Augmentation ---
    data_gen_args = dict(
        rescale=1.0/255.0,
        horizontal_flip=True,
    )
    print("Setting up Data Generators...")
    train_img_datagen  = ImageDataGenerator(**data_gen_args)
    train_mask_datagen = ImageDataGenerator(**data_gen_args)
    val_img_datagen    = ImageDataGenerator(rescale=1.0/255.0)
    val_mask_datagen   = ImageDataGenerator(rescale=1.0/255.0)

    # --- Flow from Directory ---
    print("Flowing data from directories...")
    train_image_generator = train_img_datagen.flow_from_directory(
        directory=os.path.dirname(TRAIN_FRAMES_DIR), classes=[os.path.basename(TRAIN_FRAMES_DIR)],
        target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode=None, color_mode="rgb", seed=SEED
    )
    train_mask_generator = train_mask_datagen.flow_from_directory(
        directory=os.path.dirname(TRAIN_MASKS_DIR), classes=[os.path.basename(TRAIN_MASKS_DIR)],
        target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode=None, color_mode="grayscale", seed=SEED
    )
    val_image_generator = val_img_datagen.flow_from_directory(
        directory=os.path.dirname(VAL_FRAMES_DIR), classes=[os.path.basename(VAL_FRAMES_DIR)],
        target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode=None, color_mode="rgb", seed=SEED
    )
    val_mask_generator = val_mask_datagen.flow_from_directory(
        directory=os.path.dirname(VAL_MASKS_DIR), classes=[os.path.basename(VAL_MASKS_DIR)],
        target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode=None, color_mode="grayscale", seed=SEED
    )

    train_generator = combine_generators(train_image_generator, train_mask_generator)
    val_generator   = combine_generators(val_image_generator, val_mask_generator)

    # --- Build and Compile Model ---
    print("Building U-Net model...")
    model = U_NET(input_size=MODEL_INPUT_SHAPE, num_classes=NUM_CLASSES)

    print("Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss=dice_loss,
        metrics=[ MeanIoU(num_classes=2, name='mean_io_u'), dice_coef ]
    )
    model.summary()

    # --- Calculate Steps ---
    try:
        no_of_training_images = len(os.listdir(TRAIN_FRAMES_DIR))
        no_of_val_images      = len(os.listdir(VAL_FRAMES_DIR))
        print(f"Found {no_of_training_images} train images, {no_of_val_images} val images.")
    except FileNotFoundError as e:
        print(f"Error counting files: {e}")
        sys.exit(1)

    steps_per_epoch = no_of_training_images // BATCH_SIZE
    validation_steps = no_of_val_images // BATCH_SIZE
    
    if steps_per_epoch == 0 or validation_steps == 0:
         print("Warning: steps_per_epoch or validation_steps is zero. Check image counts and batch size.")
         if steps_per_epoch == 0: steps_per_epoch = 1 
         if validation_steps == 0: validation_steps = 1

    print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

    # --- Callbacks ---
    monitor_metric = 'val_mean_io_u' 
    print(f"Setting up Callbacks, monitoring: {monitor_metric}")
    
    model_dir = os.path.dirname(MODEL_OUT)
    if model_dir: 
        os.makedirs(model_dir, exist_ok=True)
        print(f"Ensured directory exists: {model_dir}")
    
    checkpoint = ModelCheckpoint(
        MODEL_OUT, monitor=monitor_metric, mode="max", save_best_only=True, 
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor=monitor_metric, patience=10, mode="max", restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor_metric, factor=0.2, patience=3, mode="max",
        verbose=1
    )

    log_dir = os.path.join("./logs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    tensorboard = TensorBoard(log_dir=log_dir)

    # --- Train Model ---
    print(f"Starting training for {EPOCHS} epochs...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard]
    )

    print("Training complete.")

if __name__ == "__main__":
    main()