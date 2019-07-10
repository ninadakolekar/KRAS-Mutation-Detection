import os
import sys
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

base_model = InceptionV3(include_top=False, input_shape=(512,512,3), pooling='avg', classes=2)
x = base_model.output
x = Dense(32, activation='relu')(x)
x = Dense(8, activation='relu')(x)
out = Dense(2, activation='softmax')(x)

model = Model(outputs=out,inputs=base_model.input)

model = load_model(sys.argv[1])

model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

print(f"Metrics: {model.metrics_names}")

print("=== TRAIN ===")
# train_gen = CellColonySequence("/home/nitish/Desktop/ninad/kras/data/data4/train",512,1,augmentations=None)
# print(model.evaluate_generator(train_gen))

predictions = model.predict_generator(train_gen)
print(predictions)
exit(0)
unique, counts = np.unique(predictions, return_counts=True)
print(dict(zip(unique, counts)))

del train_gen

print("=== VALIDATION ===")
# val_gen = CellColonySequence("/home/nitish/Desktop/ninad/kras/data/data4/valid",512,1,augmentations=None)
# print(model.evaluate_generator(val_gen))

predictions = model.predict_generator(val_gen)
unique, counts = np.unique(predictions, return_counts=True)
print(dict(zip(unique, counts)))

del val_gen

print("=== TEST ===")
# test_gen = CellColonySequence("/home/nitish/Desktop/ninad/kras/data/data4/test",512,1,augmentations=None)
# print(model.evaluate_generator(test_gen))

predictions = model.predict_generator(test_gen)
unique, counts = np.unique(predictions, return_counts=True)
print(dict(zip(unique, counts)))

del test_gen