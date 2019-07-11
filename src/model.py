import os
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
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50

from sklearn.metrics import roc_auc_score

def get_base_model(input_size):
    return InceptionV3(include_top=False, input_shape=(input_size,input_size,3), pooling='avg', classes=2)

if __name__ == "__main__":
    
    # args = TrainingOptions().parse()

    # train_gen = CellColonySequence(os.path.join(args.dataset_path,"train"),args.input_size,args.batch_size,augmentations='train')
    # val_gen = CellColonySequence(os.path.join(args.dataset_path),args.input_size,args.batch_size,augmentations=None,mode='valtest')

    # callbacks = [TensorBoard(log_dir=args.outdir),ModelCheckpoint(filepath=os.path.join(args.outdir,"models","weights.{epoch:02d}-{val_loss:.2f}.hdf5"),verbose=0,save_best_only=False,period=1),ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.001)]

    # # model = get_base_model(args.input_size)

    # # model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

    model = load_model("/home/nitish/Desktop/ninad/kras/code/kras-keras-old/output/s512_20190710-210507/model.h5")

    # model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

    # history = model.fit_generator(train_gen,steps_per_epoch=10*ceil(train_gen.__len__()),epochs=args.epochs,callbacks=callbacks,validation_data=val_gen,validation_steps=ceil(val_gen.__len__()),workers=1,use_multiprocessing=False,shuffle=True)

    # model.save(os.path.join(args.outdir,"model.h5"))

    # # Accuracy Curve

    # plt.plot(history.history['acc'],label='train')
    # plt.plot(history.history['val_acc'],label='val')
    # plt.title('InceptionV3 Accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(loc='upper left')
    # plt.savefig(os.path.join(args.outdir,"acc.png"))

    # plt.close()

    # # Loss Curve

    # plt.plot(history.history['loss'],label='train')
    # plt.plot(history.history['val_loss'],label='val')
    # plt.title('InceptionV3 Loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(loc='upper left')
    # plt.savefig(os.path.join(args.outdir,"loss.png"))

    print("=== VALIDATION ===")
    val_gen = CellColonySequence("/home/nitish/Desktop/ninad/kras/data/data4/valid",512,1,augmentations=None,mode='valtest')
    print(model.evaluate_generator(val_gen))

    print("=== TEST ===")
    test_gen = CellColonySequence("/home/nitish/Desktop/ninad/kras/data/data4/test",512,1,augmentations=None,mode='valtest')
    print(model.evaluate_generator(test_gen))
