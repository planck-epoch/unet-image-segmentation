from tensorflow.keras import backend as K
import tensorflow as tf

# small epsilon value for smoothing to avoid division by zero
SMOOTH = K.epsilon()

def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = SMOOTH) -> tf.Tensor:
    """
    Computes the Dice coefficient, a common metric for segmentation tasks.

    Calculates the Dice coefficient averaged over the batch. Assumes channels_last format.
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
        tf.Tensor: The mean Dice coefficient (scalar).
    """
    # Ensure inputs are float type
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Sum over spatial dimensions (Height, Width) -> (batch, channels)
    # Assuming channels_last format (batch, H, W, C)
    axis_to_sum = [1, 2] 

    # Intersection: Sum of element-wise multiplication over spatial axes
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis_to_sum)

    # Calculate sums of ground truth and predictions over spatial axes
    sum_true = tf.reduce_sum(y_true, axis=axis_to_sum)
    sum_pred = tf.reduce_sum(y_pred, axis=axis_to_sum)

    # Calculate Dice coefficient score with smoothing
    # Dice = (2 * Intersection + Smooth) / (Sum(True) + Sum(Pred) + Smooth)
    # Adding smooth to numerator and denominator prevents 0/0 issues
    numerator = 2. * intersection + smooth
    denominator = sum_true + sum_pred + smooth
    dice_score = numerator / denominator

    # Return the mean Dice score across the batch and channels
    return tf.reduce_mean(dice_score)

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

    # Intersection: Sum over spatial dimensions (H, W) for each sample/channel
    # Keep batch and channel dims
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])

    # Sums for Union: Sum over spatial dimensions (H, W)
    sum_true = tf.reduce_sum(y_true, axis=[1, 2])
    sum_pred = tf.reduce_sum(y_pred, axis=[1, 2])

    # Union = Sum(y_true) + Sum(y_pred) - Intersection
    union = sum_true + sum_pred - intersection

    # IoU = (Intersection + Smooth) / (Union + Smooth) per sample/channel
    iou = (intersection + smooth) / (union + smooth)

    # Return the mean IoU score across the batch and channels
    return tf.reduce_mean(iou)
