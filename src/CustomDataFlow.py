#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bsds500.py

import os
import glob
import numpy as np
import cv2
from random import randint
from functools import reduce

from tensorpack.utils.fs import download, get_dataset_path
from tensorpack.dataflow.base import RNGDataFlow

IMG_W, IMG_H = 512, 512
IMG_SUFFIX = '_clean.png'
GT_SUFFIX = '_texturemap.bin'

class CustomDataFlow(RNGDataFlow):
    def __init__(self, data_dir, name, shuffle=True):
        """
        Args:
            name (str): 'train', 'test', 'val'
            data_dir (str): a directory containing the "data" folders, which has folders with the names.
        """
        self.data_root = os.path.join(data_dir, 'data')
        print(self.data_root)
        assert os.path.isdir(self.data_root)

        self.shuffle = shuffle
        assert name in ['train', 'test', 'val']
        self._load(name)

    def _load(self, name):
        image_glob = os.path.join(self.data_root, 'images', name, '*' + IMG_SUFFIX)
        image_files = glob.glob(image_glob)
        gt_dir = os.path.join(self.data_root, 'groundTruth', name)
        self.data = np.zeros((len(image_files), IMG_H, IMG_W), dtype='float32') # NHW
        self.label = np.zeros((len(image_files), IMG_H, IMG_W), dtype='float32') # NHW

        for idx, f in enumerate(image_files):
            im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            assert im is not None
            orig_im_shape = im.shape
            if im.shape != (IMG_H, IMG_W):
                assert im.shape[0] >= IMG_H and im.shape[1] >= IMG_W, "{} < {}".format(im.shape, (IMG_H, IMG_W))
                hi = randint(0, im.shape[0] - IMG_H)
                hf = hi + IMG_H
                wi = randint(0, im.shape[1] - IMG_W)
                wf = wi + IMG_W
                im = im[hi:hf, wi:wf]
            im = im.astype('float32')

            imgid = os.path.basename(f).split('.')[0]
            gt_file = os.path.join(gt_dir, imgid + '' + GT_SUFFIX)
            gt = np.fromfile(gt_file, dtype='uint32')
            print(max(max(gt)))
            assert gt is not None
            gt = gt.astype('float32')
            assert gt.shape[0] == reduce(lambda x, y: x*y, orig_im_shape), "Different number of elements: {} != {}".format(gt.shape, orig_im_shape)
            gt = np.reshape(gt, orig_im_shape)
            if gt.shape != (IMG_H, IMG_W):
                gt = gt[hi:hf, wi:wf]

            self.data[idx] = im
            self.label[idx] = gt

    def size(self):
        return self.data.shape[0]

    def get_data(self):
        idxs = np.arange(self.data.shape[0])
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield [self.data[k], self.label[k]]
