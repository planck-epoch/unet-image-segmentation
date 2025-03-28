import tensorflow as tf
from tensorflow.keras import backend as K
from typing import Callable
from .metrics import dice_coef

# small epsilon value for smoothing to avoid division by zero
SMOOTH = K.epsilon()


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the Dice loss (1 - Dice coefficient).

    Relies on a `dice_coef` function (imported from .metrics) which should ideally
    handle its own smoothing and axis summation appropriately for the task.

    Args:
        y_true (tf.Tensor): Ground truth tensor (e.g., shape = (batch, H, W, C)).
                            Values typically binary {0, 1} or probabilities [0, 1].
        y_pred (tf.Tensor): Predicted tensor (e.g., shape = (batch, H, W, C)).
                            Values typically probabilities [0, 1].

    Returns:
        tf.Tensor: The calculated Dice loss (scalar).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate Dice coefficient using the imported function
    # Ensure dice_coef handles reduction correctly (e.g., returns mean over batch)
    dice_coefficient = dice_coef(y_true, y_pred) # Assumes it uses appropriate smoothing

    # Dice loss is 1 minus the coefficient
    loss = 1.0 - dice_coefficient
    return loss

def iou_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = SMOOTH) -> tf.Tensor:
    """
    Computes the IoU loss (1 - IoU coefficient), also known as Jaccard loss.

    Args:
        y_true (tf.Tensor): Ground truth tensor (e.g., shape = (batch, H, W, C)).
        y_pred (tf.Tensor): Predicted tensor (e.g., shape = (batch, H, W, C)).
        smooth (float): Smoothing factor passed to `iou_coef`.

    Returns:
        tf.Tensor: The calculated IoU loss (scalar).
    """
    # Calculate IoU coefficient
    iou_coefficient = iou_coef(y_true, y_pred, smooth=smooth)

    # IoU loss is 1 minus the coefficient
    loss = 1.0 - iou_coefficient
    return loss

# Alias for Jaccard Loss
jaccard_loss = iou_loss
