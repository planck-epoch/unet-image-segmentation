import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from typing import Tuple

def conv_block(input_tensor: tf.Tensor, 
               num_filters: int, 
               kernel_size: int = 3, 
               use_batch_norm: bool = True, 
               name_prefix: str = "conv_block") -> tf.Tensor:
    """
    Creates a convolutional block with SeparableConv2D, Batch Normalization, and ReLU.
    Includes layer naming based on name_prefix.
    """
    x = layers.SeparableConv2D(
        num_filters, 
        kernel_size, 
        padding="same", 
        use_bias=not use_batch_norm,
        name=f"{name_prefix}_sepconv" 
    )(input_tensor)
    
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
        
    x = layers.Activation("relu", name=f"{name_prefix}_relu")(x)
    return x

def U_NET(input_size: Tuple[int, int, int], 
          num_classes: int = 1, 
          dropout_rate: float = 0.2, 
          use_batch_norm: bool = True) -> tf.keras.Model:
    """
    Builds a U-Net model for segmentation using separable convolutions,
    standard skip connections, and optional dropout. (Polished with layer names)

    Args:
        input_size (tuple): Dimensions (height, width, channels) e.g., (256, 256, 3).
        num_classes (int): Number of output segmentation classes (1 for binary).
        dropout_rate (float): Dropout rate for regularization (0.0 means no dropout).
        use_batch_norm (bool): Whether to use Batch Normalization layers.

    Returns:
        tf.keras.Model: A U-Net model.

    Architecture:
        - Encoder (4 stages): Downsampling path with two conv_blocks and max pooling per stage.
        - Bottleneck: Connects encoder and decoder.
        - Decoder (4 stages): Upsampling path using Conv2DTranspose, concatenation
          with skip connections, and two conv_blocks per stage.
        - Output Layer: 1x1 Convolution with sigmoid (binary) or softmax (multi-class).
    """
    if len(input_size) != 3:
        raise ValueError("input_size must be a tuple of (height, width, channels)")

    inputs = Input(shape=input_size, name="input_image")
    
    filters = [64, 128, 256, 512] 
    skip_connections = []
    x = inputs

    # --- Encoder ---
    print("Building Encoder...")
    for i, f in enumerate(filters):
        stage = i + 1
        print(f"  Encoder Stage {stage}, Filters: {f}")
        x = conv_block(x, f, use_batch_norm=use_batch_norm, name_prefix=f"enc{stage}_block1")
        x = conv_block(x, f, use_batch_norm=use_batch_norm, name_prefix=f"enc{stage}_block2")
        skip_connections.append(x) # Save features before pooling
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"enc{stage}_pool")(x)

    # --- Bottleneck ---
    print("Building Bottleneck...")
    bottle_neck_filters = filters[-1] * 2 # 512 * 2 = 1024 filters
    print(f"  Bottleneck Filters: {bottle_neck_filters}")
    x = conv_block(x, bottle_neck_filters, use_batch_norm=use_batch_norm, name_prefix="bneck_block1")
    x = conv_block(x, bottle_neck_filters, use_batch_norm=use_batch_norm, name_prefix="bneck_block2")
    if dropout_rate > 0.0:
         x = layers.Dropout(dropout_rate, name="bneck_dropout")(x)

    # --- Decoder ---
    print("Building Decoder...")
    filters.reverse() 
    skip_connections.reverse()

    for i, f in enumerate(filters):
        stage = len(filters) - i
        print(f"  Decoder Stage {stage}, Filters: {f}")
        x = layers.Conv2DTranspose(
            f, 
            kernel_size=2, 
            strides=2, 
            padding='same', 
            name=f"dec{stage}_upsample"
        )(x)
        skip = skip_connections[i]
        x = layers.Concatenate(name=f"dec{stage}_concat")([x, skip])
        if dropout_rate > 0.0 and i < len(filters) - 1 : 
             x = layers.Dropout(dropout_rate, name=f"dec{stage}_dropout")(x)

        x = conv_block(x, f, use_batch_norm=use_batch_norm, name_prefix=f"dec{stage}_block1")
        x = conv_block(x, f, use_batch_norm=use_batch_norm, name_prefix=f"dec{stage}_block2")

    # --- Output Layer ---
    print("Building Output Layer...")
    final_activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Conv2D(
        num_classes, 
        kernel_size=1, 
        padding="same", 
        activation=final_activation,
        name="output_mask"
    )(x)

    model = Model(inputs, outputs, name="U-NET-Segmentation")
    print("U-Net model built successfully.")
    return model