import os
import sys
import random
from math import ceil

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from options import TrainingOptions
from dataset import CellColonyTestSequence

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K

random.seed(42)

NUM_LAYERS = int(sys.argv[1])

base_model = InceptionV3(include_top=False, input_shape=(512,512,3), pooling='avg', classes=2)
x = base_model.output
x = Dense(32, activation='relu')(x)
x = Dense(8, activation='relu')(x)
out = Dense(2, activation='softmax')(x)

model = Model(outputs=out,inputs=base_model.input)

model = load_model("/home/nitish/Desktop/ninad/kras/code/kras-keras-old/output/s512_20190710-144144/ic_model.h5")

model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
classifier = model
layer_outputs = [layer.output for layer in classifier.layers[:NUM_LAYERS]] 
# Extracts the outputs of the top 12 layers
activation_model = Model(inputs=classifier.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

img_path = random.choice(os.listdir("/home/nitish/Desktop/ninad/kras/data/data4/test/kras"))
img_path = os.path.join("/home/nitish/Desktop/ninad/kras/data/data4/test/kras",img_path)

img = image.load_img(img_path, target_size=(512,512,3))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
images_per_row = 16

layer_names = []
for layer in classifier.layers[:NUM_LAYERS]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

activations = activation_model.predict(img_tensor)

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    plt.savefig(f"{layer_name}.png")
    plt.close()



