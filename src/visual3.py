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

model = load_model("/home/nitish/Desktop/ninad/kras/code/kras-keras-old/output/s512_20190710-210507/model.h5")

model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

complete_model = model

layer_outputs = outputs = [layer.output for layer in model.layers][1:]

img_path = random.choice(os.listdir("/home/nitish/Desktop/ninad/kras/data/data5/test/kras"))
test_image = os.path.join("/home/nitish/Desktop/ninad/kras/data/data5/test/kras",img_path)

print(f"Name: {test_image}")

img = image.load_img(test_image, target_size=(512,512,3))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

activation_model = Model(inputs=complete_model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

print("computed activations")

layer_names = ['conv2d_1', 'activation_1', 'conv2d_4', 'activation_4', 'conv2d_9', 'activation_9']
activ_list = [activations[1], activations[3], activations[11], activations[13], activations[18], activations[20]]

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activ_list):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='plasma')
    plt.savefig(layer_name+"_grid.jpg", bbox_inches='tight')
