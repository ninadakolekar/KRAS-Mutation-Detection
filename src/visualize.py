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
from dataset import CellColonySequence

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K

random.seed(42)

base_model = InceptionV3(include_top=False, input_shape=(512,512,3), pooling='avg', classes=2)
x = base_model.output
x = Dense(32, activation='relu')(x)
x = Dense(8, activation='relu')(x)
out = Dense(2, activation='softmax')(x)

model = Model(outputs=out,inputs=base_model.input)

model = load_model("/home/nitish/Desktop/ninad/kras/code/kras-keras-old/output/s512_20190710-144144/ic_model.h5")

model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

print(f"Metrics: {model.metrics_names}")

# print("=== TRAIN ===")
# train_gen = CellColonySequence("/home/nitish/Desktop/ninad/kras/data/data4/train",512,1,augmentations=None)
# print(model.evaluate_generator(train_gen))

# predictions = model.predict_generator(train_gen).argmax(axis=1)
# unique, counts = np.unique(predictions, return_counts=True)
# print(dict(zip(unique, counts)))

# del train_gen

# print("=== VALIDATION ===")
# val_gen = CellColonySequence("/home/nitish/Desktop/ninad/kras/data/data4/valid",512,1,augmentations=None)
# print(model.evaluate_generator(val_gen))

# predictions = model.predict_generator(val_gen).argmax(axis=1)
# unique, counts = np.unique(predictions, return_counts=True)
# print(dict(zip(unique, counts)))

# del val_gen

# print("=== TEST ===")
# test_gen = CellColonySequence("/home/nitish/Desktop/ninad/kras/data/data4/test",512,1,augmentations=None)
# print(model.evaluate_generator(test_gen))

# predictions = model.predict_generator(test_gen).argmax(axis=1)
# unique, counts = np.unique(predictions, return_counts=True)
# print(dict(zip(unique, counts)))

# del test_gen

complete_model = model

layer_outputs = [layer.output for layer in complete_model.layers[:50]]
test_image = random.choice([file for file in os.listdir('/home/nitish/Desktop/ninad/kras/data/data4/test/kras/') if file.endswith('.JPG')])
test_image = os.path.join("/home/nitish/Desktop/ninad/kras/data/data4/test/kras/",test_image)

img = image.load_img(test_image, target_size=(512,512))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

def deprocess_image(x):
    
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150):
    
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(80):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)

layer_name = 'conv2d_4'
size = 512
margin = 5
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
        
plt.figure(figsize=(20, 20))
plt.savefig(results)



