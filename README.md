# Image Semantic Segmentation (TensorFlow)

```bash
██╗   ██╗      ███╗   ██╗███████╗████████╗
██║   ██║      ████╗  ██║██╔════╝╚══██╔══╝
██║   ██║█████╗██╔██╗ ██║█████╗     ██║   
██║   ██║╚════╝██║╚██╗██║██╔══╝     ██║   
╚██████╔╝      ██║ ╚████║███████╗   ██║   
 ╚═════╝       ╚═╝  ╚═══╝╚══════╝   ╚═╝   
```

## Table of Contents
1.  [Overview](#overview)
2.  [Features](#features)
3.  [Architecture](#model-architecture)
4.  [Installation](#installation)
5.  [Data Preparation](#data-preparation)
    - [Using MIDV Data](#using-midv-id-card-data)
    - [Using Your Own Dataset](#using-your-own-dataset)
6.  [Training](#training)
7.  [Inference](#inference)
8.  [Benchmarking](#benchmarking)
9.  [Contributing](#contributing)
10. [License](#license)
11. [References](#references)


## Overview

Welcome to this **one-stop shop** for `U-Net` semantic segmentation with `TensorFlow`.

This project implements a robust pipeline inspired by the original U-Net ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)) capable of separating foreground objects from the background pixel by pixel. The primary application driving its development was the segmentation of identity documents (like `passport`) for downstream processing such as OCR and identity verification (e.g., with [Regula Forensics](https://regulaforensics.com/id-verification)).

Futhermore this implementation allows easy adaptation for diverse segmentation tasks across different domains (medical, satellite, etc.). Please consider this repository a strong starting point and learning tool; although based on concepts proven effective in production environments, it is **not** provided as a finalized solution.

### Example Result (Light Training)

To illustrate this model capability even with minimal training (e.g., just 10 epochs on a relevant dataset) here's a sample inference result showing the predicted mask and the subsequent cropping of the ID card from the original image:

<table>
  <tr>
    <td align="center"><b>1. Predicted Segmentation Mask</b></td>
    <td align="center"><b>2. Resulting Cropped Object</b></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/planck-epoch/unet-image-segmentation/main/samples/usage/portugal_id_card/output_mask.png" alt="Predicted Mask" width="256"></td>
    <td><img src="https://raw.githubusercontent.com/planck-epoch/unet-image-segmentation/main/samples/usage/portugal_id_card/output_pred.png" alt="Cropped ID Card" width="256"></td>
    </tr>
  <tr>
    <td align="center"><i>The segmentation output from the model,<br>highlighting the predicted ID card pixels.</i></td>
    <td align="center"><i>The original image cropped using the<br>bounding box of the predicted mask.</i></td>
  </tr>
</table>

This demonstrates the potential for accurately locating and isolating objects like ID cards which is a crutial step for many automated verification or data extraction workflows.

## Fetures

### Data Management
- `scripts/download_dataset.py`: Automatically download and extract standard datasets like [MIDV-500](https://arxiv.org/abs/1807.05786)/[MIDV-2019](https://arxiv.org/abs/1910.04009)
- `scripts/prepare_dataset.py`: Splits raw labeled data (`images` + `masks`) into `train`/`validation` sets formatted for Keras `ImageDataGenerator`.

### U-Net Model
- Implements the standard U-Net architecture with skip connections.
- Uses efficient separable Convolutions.
- Configurable for binary (by default) but can be extend to multi-class segmentation.

### Training Pipeline
- Configurable hyperparameters (epochs, batch size, learning rate).
- Supports custom loss functions: Includes `Dice Loss` (`utils/loss.py`) as the default and `IoU` (Jaccard) Loss. Standard `Binary Cross-Entropy` can also be used.
- Monitors performance using `MeanIoU` (Intersection over Union) and dice_coef metrics (`utils/metrics.py`).
- Integrates TensorBoard logging (`./logs`) for real-time monitoring.

### Inference Pipeline
- Performs segmentation on single images using a trained model.
- Handles model loading, including necessary custom objects (`loss/metrics`).
- Outputs a binary segmentation mask.
- Includes optional cropping of the original image based on the largest contour found in the predicted mask (`utils/image.py`).

### Evaluation
- Calculates the overall `MeanIoU` for a model on a dataset (using JSON polygon ground truth).
- Identifies and logs images performing below a specified `IoU` threshold.

### Environment Checks
- `scripts/check_tf_install.py`: Verifies basic TensorFlow/Keras installation and device detection.
- `scripts/check_gpu_benchmark.py`: Specifically tests GPU detection and runs a performance comparison against the CPU.

## Model Architecture

This project utilizes the U-Net architecture, originally proposed by [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597) which is renowned for its effectiveness in image segmentation tasks. The architecture consists of:

* **Encoder (Contracting Path):** Captures context from the input image. It uses repeated blocks, typically consisting of Convolutional layers (e.g., `Conv2D` or `SeparableConv2D`), `BatchNormalization`, and `ReLU` activation, followed by `MaxPooling` operations. This progressively reduces the spatial resolution (height/width) while increasing the number of feature channels (depth).
* **Bottleneck:** The central part of the network connecting the encoder and decoder. It operates at the lowest spatial resolution and typically has the highest number of feature channels.
* **Decoder (Expansive Path):** Gradually reconstructs the high-resolution segmentation map. It uses `Transposed Convolutions` (or Upsampling followed by Convolution) to increase the spatial resolution. **Skip Connections** are a key feature, concatenating feature maps from the corresponding encoder stage with the upsampled features from the decoder. This allows the network to combine high-level semantic information (what the object is) learned in the deeper layers with low-level spatial details (where edges are) from earlier layers, enabling precise boundary localization in the output mask.

*(Refer to the original paper or the implementation in `model/u_net.py` for the specific layer configurations, filter counts, and use of Separable Convolutions in this project.)*

### Binary vs. Multi-Class Segmentation Support

The `U_NET` function in `model/u_net.py` is built to handle both binary and multi-class segmentation tasks directly. This flexibility is controlled by the `num_classes` argument passed when creating the model instance:

* **Binary Segmentation (Default Behaviour):**
    * Instantiate the model providing `num_classes=1` (or omitting it, as 1 is the default):
        ```python
        # Example for input size 256x256x3
        model = U_NET(input_size=(256, 256, 3), num_classes=1)
        ```
    * The model's final layer automatically becomes a `Conv2D` with **1 filter** and `sigmoid` activation.
    * **Output Shape:** `(Height, Width, 1)` - A single-channel probability map (values 0.0 to 1.0) representing the likelihood of each pixel belonging to the single foreground class.
    * **Training Setup (`scripts/train.py`):** Requires a binary loss function (like the default `dice_loss` or `binary_crossentropy`), single-channel binary ground truth masks, and metrics configured for 2 categories (e.g., `MeanIoU(num_classes=2)` for background + foreground).

* **Multi-Class Segmentation:**
    * Instantiate the model providing `num_classes=N`, where `N` is the total number of distinct classes (e.g., `background + car + building = 3 classes`):
        ```python
        # Example for input size 256x256x3 and 3 classes
        model = U_NET(input_size=(256, 256, 3), num_classes=3)
        ```
    * The model final layer automatically becomes a `Conv2D` with **N filters** and `softmax` activation.
    * **Output Shape:** `(Height, Width, N)` - A multi-channel map where each channel corresponds to a class, containing the probability distribution across classes for each pixel.
    * **Training Setup (`scripts/train.py`):**
        1.  **Ground Truth Masks:** Must be formatted for multi-class (e.g., integer labels `(H, W)` or `(H, W, 1)` where values are class indices 0 to N-1).
        2.  **Loss Function:** Change the loss to a suitable multi-class loss (e.g., `tf.keras.losses.SparseCategoricalCrossentropy` if using integer labels).
        3.  **Metrics:** Update the `num_classes` parameter in `MeanIoU` to `N` (e.g., `MeanIoU(num_classes=3)`).


## Installation

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/planck-epoch/unet-image-segmentation.git
    cd unet-image-segmentation
    ```

2.  **Create Environment (Recommended):**
    Requires **Python 3.8+**.
    ```bash
    # Create environment
    python3 -m venv venv
    # Activate environment
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Installs TensorFlow, OpenCV, NumPy, **tflite-support**, etc. Ensure `pip` is updated (`pip install --upgrade pip`).
    ```bash
    pip install -r requirements.txt
    ```
    * **Note:** Ensure `tflite-support` is listed in your `requirements.txt` file if using the metadata script.
    * **macOS Users:** Refer to official TensorFlow docs for Metal GPU support.

4.  **Verify General Installation (Optional):**
    Run the basic environment check script from its new location:
    ```bash
    python utils/troubleshoot/check_tf_install.py
    ```
    This confirms basic imports, model building, and compilation work correctly.

5.  **Verify GPU Performance (Optional, if GPU expected):**
    To specifically test GPU detection and performance, run the benchmark script from its new location:
    ```bash
    python utils/troubleshoot/check_gpu_benchmark.py
    ```
    This lists GPUs runs an benchmark on CPU vs. GPU and reports average times/speedup.

## Getting Started

### Using MIDV dataset
1. **Download the dataset**:
By using the provided script:
    ```bash
    $ python scripts/download_dataset.py
    ```
    ....or manually from the [MIDV-500](https://arxiv.org/abs/1807.05786) website and place it under `datasets/data/`.
    same for [MIDV-2019](https://arxiv.org/abs/1904.05626).

### **Using your own dataset**
1. **Dataset Structure**
Make sure your datasets are placed under datasets/ following this structure:
    ```bash
    datasets/
    ├── data/
    │   ├── images/
    │   └── ground_truth/
    ```
Or update paths in your scripts accordingly.

2. **Prepare Dataset*** 
Use the script to split data into training and validation sets:

    ```bash
    $ python scripts/prepare_dataset.py
    ```

## Training
Use the `train.py` script to train the U-Net model:
```bash
$ python scripts/train.py \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --model-out models/unet_model.h5
```
By default, it uses:

- RGB input images (3‐channel).
- Grayscale masks (1‐channel).
- **Dice Coefficient** (similarity/overlap between two sets) as the loss and **Mean IoU** (mean intersection over union) as metric.

You can edit hyperparameters, callbacks, or data paths inside the script.

## Inference
Run inference on new images using a trained model:

```bash
$ python scripts/test.py /path/to/input_image.jpg \
  --output_mask /path/to/mask_out.png \
  --output_prediction /path/to/prediction_out.png \
  --model /path/to/model.h5
```
This script:

1. Loads the model (`.h5`).
2. Reads/resizes the input image to `(256,256,3)`.
3. Predicts a `(256,256,1)` mask.
4. Resizes the mask back to the original resolution.
5. Optionally warps the original image for ID cropping.

## Benchmarking
Evaluate your trained model using the provided `benchmark.py` script:
    
```bash
$ python scripts/benchmark.py \
    --dataset datasets/data \
    --model models/unet_model.h5 \
    --threshold 0.9
```
It checks whether each prediction  **IoU** is below threshold, logs it, and computes an **average IoU**.

## Contributing
Contributions are welcome! To get started:

1. [Fork the repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).

2. Create a new branch for your feature or fix:

    ```bash
    $ git checkout -b feat/my-new-feature
    ```

3. Commit your changes and push to your fork:
    ```bash
    $ git commit -m "Add amazing new feature"
    $ git push origin feat/my-new-feature
    ```

4. Open a Pull Request (PR) describing your changes.

## License
This repository is open source and released under the GNU General Public License (GPL) Version 3, 29 June 2007. You're free to use, modify, and share the code—as long as you follow the terms of the GPL. 

Have fun and happy coding!

## References
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

[TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)

[PyTorch Documentation](https://pytorch.org/docs/stable/index.html) (if you ever plan to convert)
