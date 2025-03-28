# Segmentation Utilities (`./utils`)

This directory contains helper modules for the U-Net segmentation pipeline, including custom loss functions, evaluation metrics, and image processing tools based primarily on TensorFlow/Keras and OpenCV.

## `loss.py`

Provides custom loss functions suitable for semantic segmentation tasks, particularly useful when dealing with class imbalance where standard cross-entropy might struggle. These functions can be passed directly to `model.compile()`.

**Available Losses:**

* **`dice_loss(y_true, y_pred)`:**
    * Calculated as `1.0 - dice_coef`.
    * Directly optimizes for the Dice Coefficient, a measure of spatial overlap focusing on the harmonic mean of precision and recall (`Dice = 2 * Intersection / (Area_Predicted + Area_True)`).
    * **Rationale:** Generally robust to class imbalance compared to pixel-wise cross-entropy, making it a strong default choice for segmentation.
* **`iou_loss(y_true, y_pred)` / `jaccard_loss(y_true, y_pred)`:**
    * Calculated as `1.0 - iou_coef`.
    * Directly optimizes for the IoU (Jaccard Index), another common measure of spatial overlap (`IoU = Intersection / Union`).
    * Offers benefits similar to `dice_loss` regarding class imbalance.

*Note: Both loss functions rely on their corresponding coefficient functions in `metrics.py` and utilize a small `SMOOTH` constant for numerical stability, preventing division by zero.*

**Usage Example:**
```python
# In your training script (e.g., scripts/train.py)
from utils.loss import dice_loss
# ...
model.compile(optimizer='adam', loss=dice_loss, metrics=[...])
```

## `metrics.py`
Provides functions implementing common segmentation metrics compatible with the Keras API. These can be used during model.compile() to monitor performance during training and evaluation.

**Available Metrics:**

* **`dice_coef(y_true, y_pred)`:**
    * Calculates the Dice Coefficient (averaged over the batch).
    + This is the value that `dice_loss` aims to maximize (closer to 1.0 is better).
* ***`iou_coef(y_true, y_pred)`:**
    * Calculates the IoU Coefficient / Jaccard Index (averaged over the batch).
    * This is the value that `iou_loss` aims to maximize (closer to 1.0 is better). It measures the same underlying quantity as the built-in `tf.keras.metrics.MeanIoU`.

**Note:** Both metric functions use a small `SMOOTH` constant for numerical stability.

**Usage Example:**

```python
# In your training script (e.g., scripts/train.py)
from utils.metrics import dice_coef
from tensorflow.keras.metrics import MeanIoU
# ...
model.compile(optimizer='adam', 
              loss=..., 
              metrics=[MeanIoU(num_classes=2, name='mean_io_u'), dice_coef]) 
# Note: Adding iou_coef alongside MeanIoU is generally redundant.
```

## `image.py`

Contains utility functions primarily for post-processing segmentation masks using OpenCV.

**Key Functions:**

* **`extract_object_from_mask(...)`**: Takes an image and a corresponding mask. It attempts to find contours in the mask (often filtering for quadrilateral shapes) and then applies a perspective transformation to the identified region in the original image. This is useful for "straightening" or "dewarping" detected objects like documents.

* **`four_point_transform(...)`:** A helper function that performs the actual perspective warp given four corner points.

* **`order_points(...)`:** A helper function to consistently order four corner points (e.g., top-left, top-right, bottom-right, bottom-left).

**Note:** The main `scripts/inference.py` script currently defaults to performing bounding box cropping based on the largest contour found in the mask, which does not use the perspective warping from this `image.py` module. However, this module provides the necessary functions if perspective correction is desired.


