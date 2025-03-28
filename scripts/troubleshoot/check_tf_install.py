#!/usr/bin/env python3

"""
check_tf_install.py
-------------------
A simple script to verify the TensorFlow installation and basic Keras functionality.

Checks for TensorFlow import, GPU detection/configuration, and the ability
to build, compile, and run prediction on a minimal Keras model.

Usage:
  python scripts/check_tf_install.py
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

# Suppress excessive TF logging for this simple check, can be set to '0' or '1' for more detail
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR') # Hide INFO/WARNING messages

print(f"--- TensorFlow Installation Check ---")
print(f"Using TensorFlow version: {tf.__version__}")
print(f"TensorFlow installed at: {tf.__file__}")

def check_gpu():
    """Checks for GPU availability and attempts to configure memory growth."""
    print("\nChecking for GPUs...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu.name}")
        # Attempt to set memory growth for each GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Successfully set memory growth for GPU(s).")
            return True # Indicate GPU configured
        except RuntimeError as e:
            print(f"⚠️ Could not set memory growth (this might be okay for some setups): {e}")
            return True # Indicate GPU found, but config had issue
    else:
        print("ℹ️ No GPU detected by TensorFlow. Using CPU.")
        return False # Indicate no GPU found

def check_keras_basic():
    """Tests building, compiling, and predicting with a minimal Keras model."""
    print("\nTesting basic Keras model functionality...")
    try:
        # 1. Build a minimal sequential model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,), name="InputLayer"), # Dummy input shape
            tf.keras.layers.Dense(8, activation='relu', name="Dense_1"),
            tf.keras.layers.Dense(1, activation='sigmoid', name="OutputLayer")
        ], name="SimpleTestModel")
        print("  - Model built successfully.")
        # model.summary() # Optional: Print summary

        # 2. Compile the model (tests optimizer, loss, metrics linkage)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        print("  - Model compiled successfully.")

        # 3. Create dummy data and run prediction (tests forward pass)
        batch_size = 4
        dummy_input = np.random.rand(batch_size, 10).astype(np.float32)
        start_pred = time.time()
        prediction = model.predict(dummy_input, verbose=0) # verbose=0 prevents log lines
        end_pred = time.time()
        print(f"  - Model prediction ran successfully (output shape: {prediction.shape}, time: {end_pred-start_pred:.4f}s).")
        
        # Basic check on output values (sigmoid should be 0-1)
        if np.all(prediction >= 0) and np.all(prediction <= 1):
             print("  - Prediction values are within the expected range [0, 1].")
        else:
             print("  - Warning: Prediction values outside expected [0, 1] range.")
             
        return True # Success
        
    except Exception as e:
        print(f"❌ Error during Keras model test: {e}")
        # Optional: uncomment below to print full traceback for debugging
        # import traceback
        # traceback.print_exc()
        return False # Failure

def main():
    start_time = time.time()
    
    # Check GPU
    gpu_ok = check_gpu()

    # Check basic Keras
    keras_ok = check_keras_basic()

    # --- Summary ---
    print("\n--- Check Summary ---")
    if keras_ok:
        print("✅ TensorFlow and Keras basic functionality verified.")
        if gpu_ok:
             print("✅ GPU detected and configured successfully.")
        else:
             # This is informational, not necessarily an error
             print("ℹ️ No GPU detected or configured; TensorFlow will use CPU.")
    else:
        print("❌ Error encountered during Keras test. TensorFlow/Keras installation may have issues.")
        print("   Review the error message above and check your installation steps.")

    end_time = time.time()
    print(f"\nCheck completed in {end_time - start_time:.2f} seconds.")
    print("---------------------\n")

    # Exit with status 0 if Keras check passed, 1 otherwise
    sys.exit(0 if keras_ok else 1)

if __name__ == '__main__':
    main()