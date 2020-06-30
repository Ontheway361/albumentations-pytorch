#!/usr/bin/env python3
#-*- coding:utf-8 -*-
"""
Created on 2020/06/30
author: relu
"""
import os
import cv2
import time
import numpy as np
import albumentations as alt
from albumentations.pytorch import ToTensorV2 as ToTensor
# from matplotlib import pyplot as plt
from IPython import embed


def augment_and_show(aug, image, savename):
    image = aug(image=image)['image']
    image = image.permute(1, 2, 0).numpy()
    cv2.imwrite(savename, image)


def faceaug():
    ''' choose the augmentation for face-recognition '''
    aug = alt.Compose([
              alt.HorizontalFlip(p=0.5),
              alt.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2),
              alt.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),
              alt.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
              alt.ToGray(p=0.01),
              alt.MotionBlur(blur_limit=7, p=0.2),    # default=11
              alt.GaussianBlur(blur_limit=7, p=0.2),  # default=11
              alt.GaussNoise(var_limit=(5.0, 20.0), mean=0, p=0.1), # default var_limit=(10.0, 30.0)
              alt.ISONoise(p=0.2),
              # alt.Normalize(),
              ToTensor()])
    return aug

if __name__ == "__main__":

    root_dir = '/Users/relu/data/passdoor'
    src_dir  = os.path.join(root_dir, 'align_check')
    tar_dir  = os.path.join(root_dir, 'align_aug_2.0')

    if not os.path.exists(tar_dir):
        os.mkdir(tar_dir)

    aug  = faceaug()
    start_time = time.time()
    idx = 0
    for file in os.listdir(src_dir):

        src_file = os.path.join(src_dir, file)
        tar_file = os.path.join(tar_dir, file)
        img = cv2.imread(src_file)
        augment_and_show(aug, img, tar_file)
        if (idx + 1) % 5000 == 0:
            print('already processed %3d files, total %3d files' % (idx+1, len(os.listdir(src_dir))))
        idx += 1
    finish_time = time.time()
    total_time = (finish_time - start_time) / 60
    print('augmentation costs %.4f mins' % total_time)
