import glob
import random

import numpy as np

import cv2
from PIL import Image

from tensorflow.python.keras.utils.data_utils import Sequence

from albumentations import (
    Compose, HorizontalFlip, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, RandomRotate90, Resize
)

LABELS = ["kras",'others']

class CellColonySequence(Sequence):

    def __init__(self, path, input_size, batch_size, augmentations):

        random.seed(42)
        
        labels = {name: index for index in range(len(LABELS)) for name in glob.glob(path + '/' + LABELS[index] + '/*.JPG')}
        l = list(labels.items())
        random.shuffle(l)
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
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        return (np.stack([self.augment(image=np.array(Image.open(name)))["image"] for name in batch_x], axis=0), np.array(batch_y))