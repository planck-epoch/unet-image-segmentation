import numpy as np
import os
import random as rn
import tensorflow as tf
import time
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


import models
from utils import metrics

NO_OF_TRAINING_IMAGES = len(os.listdir('dataset/train/train_frames/image'))
NO_OF_VAL_IMAGES = len(os.listdir('dataset/train/val_frames/image'))

NO_OF_EPOCHS = 1
BATCH_SIZE = 4

IMAGE_SIZE = (256, 256)

SEED = 230
rn.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    print('GPUS: ', gpus)

    # Saving model in Lite format
    model = load_model('./model/my_model_5.h5', custom_objects={
        'mean_iou': tf.keras.metrics.MeanIoU(num_classes=2)})
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tfmodel = converter.convert()
    open("model.tflite", "wb").write(tfmodel)


if __name__ == '__main__':
    main()
