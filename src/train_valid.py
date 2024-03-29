import os
from math import ceil

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from options import ValidationOptions
from dataset import CellColonySequence

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.utils import multi_gpu_model
from sklearn.metrics import roc_auc_score

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def get_base_model(input_size):
    return InceptionV3(include_top=False, input_shape=(input_size,input_size,3), pooling='avg', classes=2,weights=None)

if __name__ == "__main__":
    
    args = TrainingOptions().parse()

    train_gen = CellColonySequence(os.path.join(args.dataset_path,"train"),args.input_size,args.batch_size,augmentations='train')
    val_gen = CellColonySequence(os.path.join(args.dataset_path,"valid"),args.input_size,args.batch_size,augmentations=None)

    callbacks = [TensorBoard(log_dir=args.outdir),ModelCheckpoint(filepath=os.path.join(args.outdir,"models","weights.{epoch:02d}-{val_loss:.2f}.hdf5"),verbose=0,save_best_only=True),ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.001)]

    base_model = get_base_model(args.input_size)
    x = base_model.output
    x = Dropout(0.4)(x)
    predictions = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    # model = multi_gpu_model(Model(inputs=base_model.input, outputs=predictions),gpus=2)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

    history = model.fit_generator(train_gen,steps_per_epoch=100*ceil(train_gen.__len__()/args.batch_size),epochs=args.epochs,callbacks=callbacks,validation_data=val_gen,validation_steps=10*ceil(val_gen.__len__()/args.batch_size),workers=1,use_multiprocessing=False,shuffle=True)

    model.save(os.path.join(args.outdir,"model.h5"))

    # Accuracy Curve

    plt.plot(history.history['acc'],label='train')
    plt.plot(history.history['val_acc'],label='val')
    plt.title('InceptionV3 Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(args.outdir,"acc.png"))

    # Loss Curve

    plt.plot(history.history['loss'],label='train')
    plt.plot(history.history['val_loss'],label='val')
    plt.title('InceptionV3 Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(args.outdir,"loss.png"))
