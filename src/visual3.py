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

print("=== TEST ===")
test_gen = CellColonySequence("/home/nitish/Desktop/ninad/kras_newdata",512,1,augmentations=None)
print(model.evaluate_generator(test_gen))

predictions = model.predict_generator(test_gen).argmax(axis=1)
unique, counts = np.unique(predictions, return_counts=True)
print(dict(zip(unique, counts)))

del test_gen