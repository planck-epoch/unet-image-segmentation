import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import tensorflow.keras as keras

def conv_block(input_tensor, num_filters, kernel_size=3, use_batch_norm=True):
    """Creates a convolutional block with SeparableConv2D, Batch Normalization, and ReLU."""
    x = layers.SeparableConv2D(num_filters, kernel_size, padding="same", use_bias=not use_batch_norm)(input_tensor)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def U_NET(input_size, num_classes=1, dropout_rate=0.2, use_batch_norm=True):
    """
    Builds an improved U-Net model for segmentation using separable convolutions,
    standard skip connections, and optional dropout.

    Args:
        input_size (tuple): Dimensions of the input image (height, width, channels).
        num_classes (int): Number of segmentation classes.
        dropout_rate (float): Dropout rate for regularization (0.0 means no dropout).
        use_batch_norm (bool): Whether to use Batch Normalization layers.

    Returns:
        tf.keras.Model: An improved U-Net model.

    Architecture:
        - Encoder: Downsampling path with convolutional blocks and max pooling.
                   Skip connections are saved before pooling.
        - Bottleneck: Connects encoder and decoder. Optional dropout.
        - Decoder: Upsampling path using Conv2DTranspose and convolutional blocks.
                   Concatenates skip connections from the encoder. Optional dropout.
        - Output Layer: 1x1 Convolution with sigmoid (binary) or softmax (multi-class) activation.
    """
    inputs = Input(shape=input_size)
    filters = [64, 128, 256, 512]
    skip_connections = []

    # --- Encoder ---
    x = inputs
    for i, f in enumerate(filters):
        # Two convolutional blocks per level
        x = conv_block(x, f, use_batch_norm=use_batch_norm)
        x = conv_block(x, f, use_batch_norm=use_batch_norm)
        skip_connections.append(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # --- Bottleneck ---
    bottle_neck_filters = filters[-1] * 2
    x = conv_block(x, bottle_neck_filters, use_batch_norm=use_batch_norm)
    x = conv_block(x, bottle_neck_filters, use_batch_norm=use_batch_norm)
    if dropout_rate > 0.0:
         x = layers.Dropout(dropout_rate)(x)

    # --- Decoder ---
    filters.reverse()
    skip_connections.reverse()

    for i, f in enumerate(filters):
        # Upsample using Conv2DTranspose
        x = layers.Conv2DTranspose(f, kernel_size=2, strides=2, padding='same')(x)

        # Concatenate with the corresponding skip connection
        # Ensure skip connection shape matches if needed (usually handled by padding='same')
        skip = skip_connections[i]
        x = layers.Concatenate()([x, skip])

        # Optional dropout
        # Apply dropout except last decoder block
        if dropout_rate > 0.0 and i < len(filters) -1 :
             x = layers.Dropout(dropout_rate)(x)

        # Two convolutional blocks per level
        x = conv_block(x, f, use_batch_norm=use_batch_norm)
        x = conv_block(x, f, use_batch_norm=use_batch_norm)


    # --- Output Layer ---
    # Use 1x1 conv for final classification
    final_activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Conv2D(num_classes, kernel_size=1, padding="same", activation=final_activation)(x)

    model = Model(inputs, outputs, name="U-NET-planck")
    return model