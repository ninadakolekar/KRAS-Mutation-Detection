# -*- coding: utf-8 -*-
"""Defines training and validation command-line options
"""

from __future__ import print_function
import os
import argparse
from datetime import datetime


class TrainingOptions:
    '''
    Defines command-line arguments/options for training and functions to parse them

    An object of this class constructs an argument parser with appropriate options for training of Patch-wise and Image-wise network

    Attributes:
        _parser: An object of the ArgumentParser class of the argparse module of the Python3 standard library
    '''

    def __init__(self):
        ''' Initialises the argument parser with appropriate options

        Args:
            None
        '''
        parser = argparse.ArgumentParser(description='Classification of morphology in cancer cell-lines')
        parser.add_argument('--dataset-path', type=str, default='./dataset',  help='dataset path (default: ./dataset)')
        parser.add_argument('--checkpoints-path', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--val-path', type=str, default=None, help='Custom validation set')
        parser.add_argument('--outdir', type=str, default=None, help='Output val directory')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
        parser.add_argument('--input-size', type=int, default=256, metavar='N', help='How far the centers of two consecutive patches are in the image (default: 256)')
        parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 30)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.01)')
        parser.add_argument('--beta1', type=float, default=0.9, metavar='M', help='Adam beta1 (default: 0.9)')
        parser.add_argument('--beta2', type=float, default=0.999, metavar='M', help='Adam beta2 (default: 0.999)')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
        parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self._parser = parser

    def parse(self):
        ''' Parses the arguments from the CLI

        Note:
            Also sets the GPU settings as mentioned in the CLI arguments

        Args:
            None

        Returns:
            dict: A dictionary containing the selected options for the training
        '''
        opt = self._parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

        assert(os.path.exists(opt.outdir))
        assert(os.path.isdir(opt.outdir))
        
        os.mkdir(os.path.join(opt.outdir,"s"+str(opt.input_size)+"_"+datetime.now().strftime("%Y%m%d-%H%M%S")))
        opt.outdir = os.path.join(opt.outdir,"s"+str(opt.input_size)+"_"+datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.mkdir(os.path.join(opt.outdir,"models"))

        args = vars(opt)
        print('\n------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------\n')

        return opt

class ValidationOptions:
    '''
    Defines command-line arguments/options for training and functions to parse them

    An object of this class constructs an argument parser with appropriate options for training of Patch-wise and Image-wise network

    Attributes:
        _parser: An object of the ArgumentParser class of the argparse module of the Python3 standard library
    '''
    def __init__(self):
        ''' Initialises the argument parser with appropriate options

        Args:
            None
        '''
        parser = argparse.ArgumentParser(description='Classification of morphology in cancer cell-lines')
        parser.add_argument('--checkpoints-path', type=str, default='./checkpoints', help='Saved model file')
        parser.add_argument('--val-path', type=str, default=None, help='Custom validation set')
        parser.add_argument('--outdir', type=str, default=None, help='Output val directory')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
        parser.add_argument('--input-size', type=int, default=256, metavar='N', help='How far the centers of two consecutive patches are in the image (default: 256)')
        parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self._parser = parser

    def parse(self):
        ''' Parses the arguments from the CLI

        Note:
            Also sets the GPU settings as mentioned in the CLI arguments

        Args:
            None

        Returns:
            dict: A dictionary containing the selected options for the training
        '''
        opt = self._parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

        assert(os.path.exists(opt.outdir))
        assert(os.path.isdir(opt.outdir))
        
        os.mkdir(os.path.join(opt.outdir,"s"+str(opt.input_size)+"_"+datetime.now().strftime("%Y%m%d-%H%M%S")))
        opt.outdir = os.path.join(opt.outdir,"s"+str(opt.input_size)+"_"+datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.mkdir(os.path.join(opt.outdir,"models"))

        args = vars(opt)
        print('\n------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------\n')

        return opt