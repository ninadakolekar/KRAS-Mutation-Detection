# -*- coding: utf-8 -*-
"""Computes and saves activations of global average pooling layer and the of the InceptionV3 on test and validation sets

A TSV file containing activations and predictions will be saved in the same directory as the script
"""

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

val_gen = CellColonySequence("/home/nitish/Desktop/ninad/kras/data/data4/valid",512,1,augmentations=None)

test_gen = CellColonySequence("/home/nitish/Desktop/ninad/kras/data/data4/test",512,1,augmentations=None)

complete_model = model

layer_name = 'global_average_pooling2d'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict_generator(test_gen)
out = np.argmax(model.predict_generator(test_gen),axis=1)

np.savetxt("testgen_int.tsv",intermediate_output,delimiter='\t')
np.savetxt("testgen_pred.tsv",out,delimiter='\t')

layer_name = 'global_average_pooling2d'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict_generator(val_gen)
out = np.argmax(model.predict_generator(val_gen),axis=1)

np.savetxt("val_gen_int.tsv",intermediate_output,delimiter='\t')
np.savetxt("val_gen_pred.tsv",out,delimiter='\t')