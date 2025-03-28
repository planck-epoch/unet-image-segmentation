#!/usr/bin/env python3

"""
check_gpu_benchmark.py
----------------------
Checks for GPU availability via TensorFlow and runs a more intensive matrix
multiplication benchmark comparing CPU vs GPU performance averaged over multiple runs.

Helps verify if TensorFlow is detecting the GPU and provides a more stable
measure of the potential speedup for compute-bound tasks.

Usage:
  python scripts/check_gpu_benchmark.py
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from typing import Optional, List

# Suppress TensorFlow INFO/WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

print(f"--- TensorFlow GPU Benchmark (Intensive) ---")
print(f"Using TensorFlow version: {tf.__version__}")
print(f"TensorFlow installed at: {tf.__file__}")

# --- Benchmark Parameters ---
MATRIX_SIZE = 4096      # Increased Matrix Size (e.g., 4096x4096)
NUM_TRIALS_PER_RUN = 50 # Increased Trials within each run
WARMUP_TRIALS = 2       # Warm-up iterations before timing each run
NUM_BENCHMARK_RUNS = 3  # Number of times to repeat the entire CPU vs GPU comparison
# --- End Parameters ---

# Use a small epsilon for stability if needed, though less critical now
SMOOTH = 1e-9 

def benchmark_matmul(device_name: str, size: int, trials: int, warmup: int) -> Optional[float]:
    """
    Runs matrix multiplication benchmark on the specified device for one run.

    Args:
        device_name (str): TensorFlow device name (e.g., "/CPU:0", "/GPU:0").
        size (int): Dimension of the square matrices.
        trials (int): Number of timed multiplication trials in this run.
        warmup (int): Number of initial untimed trials for warm-up.

    Returns:
        Optional[float]: Total elapsed time in seconds for timed trials, or None if error.
    """
    print(f"  Benchmarking on {device_name} (Size: {size}x{size}, Trials: {trials}, Warmup: {warmup})... ", end="", flush=True)
    total_time = 0.0
    try:
        # Force execution on the specified device
        with tf.device(device_name):
            # Warm-up runs
            for _ in range(warmup):
                # Use tf.function for potential graph optimization during warmup? Maybe overkill.
                a_warmup = tf.random.normal([size, size], dtype=tf.float32)
                b_warmup = tf.random.normal([size, size], dtype=tf.float32)
                c_warmup = tf.matmul(a_warmup, b_warmup)
                _ = c_warmup.numpy() # Force execution

            # Timed runs
            start_time = time.perf_counter()
            for _ in range(trials):
                # Consider creating tensors outside the loop if creation time is significant vs matmul
                a = tf.random.normal([size, size], dtype=tf.float32)
                b = tf.random.normal([size, size], dtype=tf.float32)
                c = tf.matmul(a, b)
                _ = c.numpy() # Force execution
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
        print(f"Done ({total_time:.4f}s)")
        return total_time
        
    except tf.errors.ResourceExhaustedError as e:
         print(f"\n  ❌ Error: Resource Exhausted (likely OOM). Try smaller MATRIX_SIZE. Details: {e}")
         return None
    except Exception as e:
        print(f"\n  ❌ Error during benchmark on {device_name}: {e}")
        return None

def main():
    """
    Checks GPU, runs benchmarks multiple times, reports average performance.
    """
    start_time_main = time.perf_counter()
    
    # 1. Check GPU Availability
    print("\nChecking for GPUs...")
    physical_gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(physical_gpus)

    if not physical_gpus:
        print("ℹ️ No GPU detected by TensorFlow.")
        cpu_cores = os.cpu_count()
        print(f"CPU core count: {cpu_cores}")
        print("\nSkipping CPU vs GPU benchmark.")
        
    else:
        print(f"✅ Found {num_gpus} GPU(s):")
        for i, gpu in enumerate(physical_gpus):
             try:
                  details = tf.config.experimental.get_device_details(gpu)
                  detail_str = f" (Type: {details.get('device_name', 'N/A')})"
             except:
                  detail_str = ""
             print(f"  - GPU {i}: {gpu.name}{detail_str}")
             
        # Attempt memory growth configuration
        try:
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Successfully set memory growth for GPU(s).")
        except RuntimeError as e:
            print(f"⚠️ Could not set memory growth (continuing...): {e}")

        # --- 2. Run Multiple Benchmark Runs ---
        print(f"\nRunning {NUM_BENCHMARK_RUNS} benchmark run(s)...")
        cpu_times: List[float] = []
        gpu_times: List[float] = []

        for i in range(NUM_BENCHMARK_RUNS):
            print(f"\n--- Benchmark Run {i+1}/{NUM_BENCHMARK_RUNS} ---")
            # Run on CPU
            cpu_time = benchmark_matmul(
                device_name="/CPU:0", 
                size=MATRIX_SIZE, 
                trials=NUM_TRIALS_PER_RUN, 
                warmup=WARMUP_TRIALS
            )
            if cpu_time is not None:
                cpu_times.append(cpu_time)
            
            # Run on the first available GPU
            gpu_time = benchmark_matmul(
                device_name="/GPU:0", 
                size=MATRIX_SIZE, 
                trials=NUM_TRIALS_PER_RUN, 
                warmup=WARMUP_TRIALS
            )
            if gpu_time is not None:
                gpu_times.append(gpu_time)
                
            # Simple pause between runs if desired
            # if i < NUM_BENCHMARK_RUNS - 1: time.sleep(1)

        # --- 3. Calculate and Report Average Results ---
        print(f"\n--- Benchmark Summary ({NUM_BENCHMARK_RUNS} Run(s)) ---")
        
        avg_cpu_time = np.mean(cpu_times) if cpu_times else None
        std_cpu_time = np.std(cpu_times) if len(cpu_times) > 1 else 0.0
        
        avg_gpu_time = np.mean(gpu_times) if gpu_times else None
        std_gpu_time = np.std(gpu_times) if len(gpu_times) > 1 else 0.0

        if avg_cpu_time is not None:
            print(f"CPU Avg Time : {avg_cpu_time:.4f} seconds (StdDev: {std_cpu_time:.4f})")
        else:
            print("CPU benchmark did not complete successfully.")
            
        if avg_gpu_time is not None:
            print(f"GPU Avg Time : {avg_gpu_time:.4f} seconds (StdDev: {std_gpu_time:.4f})")
        else:
             print("GPU benchmark did not complete successfully.")

        # Calculate and report average speedup
        if avg_cpu_time is not None and avg_gpu_time is not None and avg_gpu_time > SMOOTH:
            avg_speedup = avg_cpu_time / avg_gpu_time
            if avg_speedup > 1.1: 
                print(f"✅ Average GPU speedup over CPU: ~{avg_speedup:.2f}x faster")
            elif avg_speedup < 0.9: 
                print(f"⚠️ Average GPU was slower than CPU: ~{avg_speedup:.2f}x")
            else:
                print(f"ℹ️ Average GPU and CPU performance were similar: ~{avg_speedup:.2f}x")
            print(f"(Benchmark: {MATRIX_SIZE}x{MATRIX_SIZE} matmul, {NUM_TRIALS_PER_RUN} trials/run. Real-world results vary.)")
        elif avg_gpu_time is not None and avg_gpu_time <= SMOOTH:
             print("⚠️ GPU time was near zero, speedup calculation unreliable.")
        else:
             print("Could not calculate average speedup due to benchmark errors.")
             
    end_time_main = time.perf_counter()
    print(f"\nScript finished in {end_time_main - start_time_main:.2f} seconds.")
    print("-------------------------------------------\n")

if __name__ == "__main__":
    main()