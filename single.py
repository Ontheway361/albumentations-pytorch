#!/usr/bin/env python3
#-*- coding:utf-8 -*-
"""
Created on 2020/06/30
author: relu
"""

import cv2
import numpy as np
import albumentations as alt
# from matplotlib import pyplot as plt
from IPython import embed


def augment_and_show(aug, image, savename):
    image = aug(image=image)['image']
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # plt.axis('off')
    # plt.savefig(savename)
    cv2.imwrite(savename, image)


if __name__ == "__main__":

    img  = cv2.imread('imgs/parrot.jpg')
    # aug  = alt.HorizontalFlip(p=1)
    n_times = 5
    for i in range(1, n_times + 1):
        aug  = alt.CLAHE(p=1)
        # aug  = alt.HueSaturationValue(p=1)
        file = 'examples/CLAHE_%d.jpg' % i
        augment_and_show(aug, img, file)
