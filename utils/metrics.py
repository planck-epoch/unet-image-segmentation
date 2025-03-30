from tensorflow.keras import backend as K
import tensorflow as tf

SMOOTH = K.epsilon()

def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = SMOOTH) -> tf.Tensor:
    """
    Computes the Dice coefficient, a common metric for segmentation tasks.

    Calculates the Dice coefficient averaged over the batch and channels. 
    Assumes channels_last format (batch, H, W, C).
    Dice = (2 * |Intersection|) / (|A| + |B|)
         = (2 * sum(y_true * y_pred)) / (sum(y_true) + sum(y_pred))

    Args:
        y_true (tf.Tensor): Ground truth tensor (e.g., shape = (batch, H, W, C)).
                            Values typically binary {0, 1} or probabilities [0, 1].
        y_pred (tf.Tensor): Predicted tensor (e.g., shape = (batch, H, W, C)).
                            Values typically probabilities [0, 1].
        smooth (float): Smoothing factor added to numerator and denominator
                        to avoid division by zero.

    Returns:
        tf.Tensor: The mean Dice coefficient (scalar tensor).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Sum over spatial dimensions (Height, Width) -> (batch, channels)
    # Assuming channels_last format (batch, H, W, C)
    axis_to_sum = [1, 2]
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis_to_sum)
    sum_true = tf.reduce_sum(y_true, axis=axis_to_sum)
    sum_pred = tf.reduce_sum(y_pred, axis=axis_to_sum)
    # Dice_Score = (2 * Intersection + Smooth) / (Sum(True) + Sum(Pred) + Smooth)
    numerator = 2. * intersection + smooth
    denominator = sum_true + sum_pred + smooth
    dice_score = numerator / denominator
    mean_dice_score = tf.reduce_mean(dice_score)
    return mean_dice_score

def iou_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = SMOOTH) -> tf.Tensor:
    """
    Computes the Intersection over Union (IoU) coefficient, also known as the Jaccard Index.

    Calculates IoU averaged over the batch. Assumes channels_last format.

    Args:
        y_true (tf.Tensor): Ground truth tensor (e.g., shape = (batch, H, W, C)).
        y_pred (tf.Tensor): Predicted tensor (e.g., shape = (batch, H, W, C)).
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        tf.Tensor: The mean IoU coefficient (scalar).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    sum_true = tf.reduce_sum(y_true, axis=[1, 2])
    sum_pred = tf.reduce_sum(y_pred, axis=[1, 2])
    union = sum_true + sum_pred - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)
