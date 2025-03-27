# U-Net Semantic Segmentation

```bash
██╗   ██╗      ███╗   ██╗███████╗████████╗
██║   ██║      ████╗  ██║██╔════╝╚══██╔══╝
██║   ██║█████╗██╔██╗ ██║█████╗     ██║   
██║   ██║╚════╝██║╚██╗██║██╔══╝     ██║   
╚██████╔╝      ██║ ╚████║███████╗   ██║   
 ╚═════╝       ╚═╝  ╚═══╝╚══════╝   ╚═╝   
```

**A general U-Net pipeline for semantic segmentation, originally used for ID card segmentation but easily adapted to any segmentation task.**

This repository includes:
- A labeling application (Jupyter notebook) to annotate images.
- Data preprocessing scripts.
- U-Net training and inference scripts.
- Evaluation and benchmarking utilities.

---

## Table of Contents
1.  [Overview](#overview)
2.  [Features](#features)
3.  [Architecture](#model-architecture)
4.  [Installation](#installation)
5.  [Getting Started](#getting-started)
6.  [Training](#training)
7.  [Inference](#inference)
8.  [Benchmarking](#benchmarking)
9.  [Contributing](#contributing)
10. [License](#license)
11. [References](#references)

---

## Overview

This project implements a U-Net model for semantic segmentation. While originally developed for **ID card segmentation**, it can be adapted to any dataset requiring pixel-level labeling. The repository also provides:
- Scripts to **prepare** and **split** data into train/validation sets.
- Example model weights and instructions to run inference and measure performance.

If you are new to semantic segmentation, you might want to check out [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)  for more details on the architecture.

## Features

- **Labeling Notebook**: Easily draw polygon or bounding-box annotations.  
- **Flexible Data Preprocessing**: Customize how images and masks are preprocessed.  
- **U-Net Architecture**: Standard or extended U-Net with additional layers or custom losses.  
- **Metrics**: Includes IoU, Dice, Precision/Recall, etc.  
- **Benchmarking Script**: Evaluate segmentation performance at specified thresholds.  



## **Model Architecture**


We use a classic U-Net structure (see [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)).
The model has an encoder-decoder architecture with skip connections:
- Encoder downsamples the input image to extract features.
- Decoder upsamples the feature maps to produce the segmentation mask.
- Skip connections merge low-level features with higher-level ones for more precise segmentations.

### Binary Segmentation

Here, we’re doing a binary segmentation approach (masks are black/white representing a single foreground class vs. background). If you need multiple classes, you can modify the final layer to have `Conv2D(#classes, 1, activation='softmax')` adjust your masks accordingly, and update the loss/metrics. Our use case (e.g., **MIDV** datasets) only requires **one** label class.

```bash
              ┌───Conv──Conv───┐
              ↑                ↓
      Input →→ [64] → Pool →→ [128] → Pool →→ [256] → Pool →→ [512] → Pool →→ [1024]
                       ↓           ↑                             ↓     ↑ 
                       ↓           └───────── UpSampling ────────┘     ↑
                       └────────────── Skip connections ───────────────┘
      
         ...Until the final 1-channel (sigmoid) output for binary segmentation.
```

#### Why **mean IoU** & **Binary Crossentropy**?

- **Mean IoU** (Intersection over Union) is a common metric for segmentation quality; it directly measures overlap between the predicted mask and ground truth.

- **Binary Crossentropy** is a straightforward choice for binary segmentation, treating each pixel as a binary classification. You can switch to Dice loss, focal loss, etc., if that fits better.

This approach can be easily extended to multi-class tasks by changing the final layer and using a multi-class loss like **categorical_crossentropy**.

## Installation

1. **Clone the repository**:
   ```bash
   $ git clone https://github.com/planck-epoch/unet-image-segmentation.git
   $ cd unet-image-segmentation
   ```

2. **Create and activate a Python environment (recommended)** :
    ```bash
    $ python3 -m venv venv
    $ source venv/bin/activate  
    ```

3. **Install dependencies:**
    ```bash
    $ pip install -r requirements.txt
    ```
    **Note:** If you’re on macOS, you can use `requirements.macOS.txt` to set up Tensorflow with GPU support on Mac

## Getting Started

### Using MIDV ID Card Data
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
- **Binary crossentropy** as the loss, **mean IoU** & **accuracy** as metrics.

You can edit hyperparameters, callbacks, or data paths inside the script.

### TODO:
- Add argument for specifying training/validation splits if you want.
- Check GPU support and performance optimizations.
- Possibly integrate TensorBoard logging.

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

TODO: Provide style guides, code coverage guidelines, and continuous integration details for larger contributions.

## License
This repository is open source and released under the GNU General Public License (GPL) Version 3, 29 June 2007. You're free to use, modify, and share the code—as long as you follow the terms of the GPL. 

Have fun and happy coding!


## References
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

[MIDV-500 Dataset (for ID card images)](https://arxiv.org/abs/1807.05786)

[TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)

[PyTorch Documentation](https://pytorch.org/docs/stable/index.html) (if you ever plan to convert)