"""The 28,353-parameter architecture recorded in notebook cell 16."""
import tensorflow as tf


def build_model(size=256):
    if size <= 0 or size % 4:
        raise ValueError('Image size must be a positive multiple of four')
    layers = tf.keras.layers
    inputs = layers.Input((size, size, 1))
    x = inputs
    for _ in range(2):
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2, padding='same')(x)
    for _ in range(2):
        x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    outputs = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    model = tf.keras.Model(inputs, outputs, name='mammography_cdae')
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
