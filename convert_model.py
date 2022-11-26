import numpy as np
import os
import random as rn
import tensorflow as tf

from keras.models import load_model

def main():    
    # Saving model in Lite format
    model = load_model('./model.h5')
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('./model.h5')
    tfmodel = converter.convert()
    open("model.tflite", "wb").write(tfmodel)

if __name__ == '__main__':
    main()
