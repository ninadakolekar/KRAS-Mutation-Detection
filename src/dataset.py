import glob
import random

import numpy as np

import cv2
from PIL import Image

from tensorflow.python.keras.utils.data_utils import Sequence
from keras.utils import np_utils

from albumentations import (
    Compose, HorizontalFlip, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, RandomRotate90, Resize
)

LABELS = ["kras",'others']

class CellColonySequence(Sequence):

    def __init__(self, path, input_size, batch_size, augmentations,mode='train'):

        random.seed(42)

        def filenames(index):
            if mode == 'valtest':
                filenames = list(glob.glob(path + '/valid/' + LABELS[index] + '/*.JPG'))+list(glob.glob(path + '/test/' + LABELS[index] + '/*.JPG'))
            elif mode=='train':
                filenames = list(glob.glob(path + '/' + LABELS[index] + '/*.JPG'))
            
            return filenames

        
        labels = {name: index for index in range(len(LABELS)) for name in filenames(index)}
        l = list(labels.items())
        labels = dict(l)

        self.path = path
        self.names = list(labels.keys())
        self.labels = list(labels.values())
        self.input_size = input_size
        self.batch_size = batch_size

        AUGMENTATIONS_TRAIN = Compose([
            HorizontalFlip(p=0.5),
            RandomContrast(limit=0.2, p=0.5),
            RandomGamma(gamma_limit=(80, 120), p=0.5),
            RandomBrightness(limit=0.2, p=0.5),
            HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                            val_shift_limit=10, p=.9),
            RandomRotate90(),
            Resize(input_size,input_size),
            ToFloat(max_value=255)
        ])

        AUGMENTATIONS_TEST = Compose([
            Resize(input_size,input_size),
            ToFloat(max_value=255)
        ])

        self.augment = AUGMENTATIONS_TRAIN if augmentations == 'train' else AUGMENTATIONS_TEST

    def __len__(self):

        return int(np.ceil(len(self.names) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = self.names[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = np_utils.to_categorical(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size],num_classes=2)
                
        return (np.stack([self.augment(image=np.array(Image.open(name)))["image"] for name in batch_x], axis=0), np.array(batch_y))

class CellColonyTestSequence(Sequence):

    def __init__(self, path, input_size, batch_size, augmentations,mode='train'):

        random.seed(42)

        
        labels = {name: index for index in range(1) for name in glob.glob(path + '/*.JPG')}
        l = list(labels.items())
        labels = dict(l)

        self.path = path
        self.names = list(labels.keys())
        self.labels = list(labels.values())
        self.input_size = input_size
        self.batch_size = batch_size

        AUGMENTATIONS_TRAIN = Compose([
            HorizontalFlip(p=0.5),
            RandomContrast(limit=0.2, p=0.5),
            RandomGamma(gamma_limit=(80, 120), p=0.5),
            RandomBrightness(limit=0.2, p=0.5),
            HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                            val_shift_limit=10, p=.9),
            RandomRotate90(),
            Resize(input_size,input_size),
            ToFloat(max_value=255)
        ])

        AUGMENTATIONS_TEST = Compose([
            Resize(input_size,input_size),
            ToFloat(max_value=255)
        ])

        self.augment = AUGMENTATIONS_TRAIN if augmentations == 'train' else AUGMENTATIONS_TEST

    def __len__(self):

        return int(np.ceil(len(self.names) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = self.names[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = np_utils.to_categorical(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size],num_classes=2)
                
        return (np.stack([self.augment(image=np.array(Image.open(name)))["image"] for name in batch_x], axis=0), np.array(batch_y))
