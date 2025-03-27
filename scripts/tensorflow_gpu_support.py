#!/usr/bin/env python3

import os
import time
import tensorflow as tf

def main():
    """
    1) Checks GPU availability with TensorFlow.
    2) Prints CPU core count if no GPU is found.
    3) Runs a small matrix multiplication benchmark on CPU and GPU 
       (if GPU is available), then reports which is faster.
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    num_gpus = len(physical_devices)
    print(f"Num GPUs Available: {num_gpus}")

    if num_gpus == 0:
        # No GPUs -> Just print CPU core count
        cpu_cores = os.cpu_count()
        print(f"No GPU found. CPU core count: {cpu_cores}")
    else:
        # If GPU(s) are present, let's do a CPU-vs-GPU benchmark
        cpu_time = benchmark_matmul(device_name="/CPU:0", trials=10)
        gpu_time = benchmark_matmul(device_name="/GPU:0", trials=10)

        print("\nQuick Benchmark (Matrix Multiplication, 10 trials each):")
        print(f"CPU total time: {cpu_time:.4f} seconds")
        print(f"GPU total time: {gpu_time:.4f} seconds")

        speedup = cpu_time / gpu_time if gpu_time != 0 else float('inf')
        if speedup > 1:
            print(f"GPU speedup over CPU: ~{speedup:.2f}x faster")
        else:
            print(f"GPU is slower than CPU in this test (~{speedup:.2f}x)")

def benchmark_matmul(device_name="/CPU:0", trials=10):
    """
    Runs a small matrix multiplication repeatedly on the specified device.
    Returns total elapsed time in seconds for the specified number of trials.

    Note: For smaller matrix sizes or older GPUs, the CPU might be faster 
    due to overhead (data transfer, memory allocation, etc.).
    """
    import numpy as np  # local import if you like, or put at top
    start_time = time.time()
    # Force execution on a specific device
    with tf.device(device_name):
        for _ in range(trials):
            # Create random tensors and multiply
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)  
            # Force execution now (instead of lazy evaluation)
            _ = c.numpy()
    return time.time() - start_time

if __name__ == "__main__":
    main()
