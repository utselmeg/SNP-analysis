
from tensorflow.keras import layers, regularizers
import tensorflow as tf

def build_model(output_units: int, input_shape: tuple):
    dropout = 0.3

    # Input layer 
    inputs = layers.Input(shape=input_shape)
    
    # Conv Block 1
    conv1 = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    pool1 = layers.MaxPooling2D()(conv1)
    drop1 = layers.Dropout(dropout)(pool1)
    
    # Conv Block 2 
    conv3 = layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(drop1)
    pool2 = layers.MaxPooling2D()(conv3)
    drop2 = layers.Dropout(dropout)(pool2)

    # Conv Block 3 
    conv5 = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(drop2)
    pool3 = layers.MaxPooling2D()(conv5)
    drop3 = layers.Dropout(dropout)(pool3)

    # Dense layers
    flat = layers.Flatten()(drop3) 
    dense1 = layers.Dense(4096, activation='relu')(flat)
    dense2 = layers.Dense(1024, activation='relu')(dense1)
    dense3 = layers.Dense(128, activation='relu')(dense2)
    dense4 = layers.Dense(16, activation='relu')(dense3)
    outputs = layers.Dense(output_units, activation='softmax')(dense4)
    model = tf.keras.Model(inputs, outputs)

    return model
