# -*- coding: utf-8 -*-
"""Defines data sequence objects to load dataset using Keras generators
"""

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
    ToFloat, RandomRotate90, RandomSizedCrop, Resize
)

LABELS = ["kras",'others']

class CellColonySequence(Sequence):
    '''
    Training sequence model class for the cell-colony datatset. An object of this class loads cell-colony data in batches of specified batch-size.

    An object of this class loads and saves model weights for PW and IW network.

    Attributes:
        path (str): Path to the dataset directory
        names (list): List of paths to cell-colony images
        labels (list): List of labels (KRAS/EGFR) corresponding to the `names` attribute
        input_size (int): Height/Width of the input image
        batch_size (int): Batch size to be used by the generator
        augment (`albumentations.Object`): Specifies the augmentations to be applied to the datatset
    '''

    def __init__(self, path, input_size, batch_size, augmentations,mode='train'):
        '''
        Initialises the attributes of the class

        Args:
            path (str): Path to the dataset directory
            input_size (int): Height/Width of the input image
            batch_size (int): Batch size to be used
            augmentations (str): If set to "train", image augmentations are applied
        '''

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
            RandomSizedCrop(min_max_height=((512,1024)),height=input_size,width=input_size,p=0.5),
            Resize(input_size,input_size),
            ToFloat(max_value=255)
        ])

        AUGMENTATIONS_TEST = Compose([
            Resize(input_size,input_size),
            ToFloat(max_value=255)
        ])

        self.augment = AUGMENTATIONS_TRAIN if augmentations == 'train' else AUGMENTATIONS_TEST

    def __len__(self):
        '''
        Computes the number of batches in the dataset generator

        Args:
            None
        
        Returns:
            Number of batches in the sequence generator (int)
        '''

        return int(np.ceil(len(self.names) / float(self.batch_size)))

    def __getitem__(self, idx):
        '''
        Fetches the batch of specified index

        Args:
            idx (int): Index of the batch
        
        Returns:
            Tuple containing `batch_size` number of images and the corresponding labels (tuple(np.array,np.array))

        '''

        batch_x = self.names[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = np_utils.to_categorical(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size],num_classes=2)
                
        return (np.stack([self.augment(image=np.array(Image.open(name)))["image"] for name in batch_x], axis=0), np.array(batch_y))

class CellColonyTestSequence(Sequence):
    '''
    Test sequence model class for the cell-colony datatset. An object of this class loads cell-colony data in batches of specified batch-size.

    An object of this class loads and saves model weights for PW and IW network.

    Attributes:
        path (str): Path to the dataset directory
        names (list): List of paths to cell-colony images
        labels (list): List of labels (KRAS/EGFR) corresponding to the `names` attribute
        input_size (int): Height/Width of the input image
        batch_size (int): Batch size to be used by the generator
        augment (`albumentations.Object`): Specifies the augmentations to be applied to the datatset
    '''

    def __init__(self, path, input_size, batch_size, augmentations,mode='train'):
        '''
        Initialises the attributes of the class

        Args:
            path (str): Path to the dataset directory
            input_size (int): Height/Width of the input image
            batch_size (int): Batch size to be used
            augmentations (str): If set to "train", image augmentations are applied
        
        Returns:
            None
        '''

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
        '''
        Computes the number of batches in the dataset generator

        Args:
            None
        
        Returns:
            Number of batches in the sequence generator (int)
        '''

        return int(np.ceil(len(self.names) / float(self.batch_size)))

    def __getitem__(self, idx):
        '''
        Fetches the batch of specified index

        Args:
            idx (int): Index of the batch
        
        Returns:
            Tuple containing `batch_size` number of images and the corresponding labels (tuple(np.array,np.array))
        '''

        batch_x = self.names[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = np_utils.to_categorical(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size],num_classes=2)
                
        return (np.stack([self.augment(image=np.array(Image.open(name)))["image"] for name in batch_x], axis=0), np.array(batch_y))
