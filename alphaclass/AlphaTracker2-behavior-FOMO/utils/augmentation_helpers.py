import numpy as np
import os
import json
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2


def main_augmentation(img, norm_mode='imagenet'):

    if norm_mode == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif norm_mode == 'norm':
        mean = (0, 0, 0)
        std = (1, 1, 1)


    train_transform = A.Compose([
        A.Flip(),
        #A.ColorJitter(p=0.2),
        #A.OneOf([
        #    A.MotionBlur(),
        #    A.MedianBlur(blur_limit=3),
        #    A.Blur(blur_limit=3),
        #], p=0.1),
        A.HueSaturationValue(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.35, rotate_limit=45, p=0.8),
        #A.RandomSizedCrop(min_max_height=[int(min(img.shape[0], img.shape[1])/1.5),
        #                                  int(max(img.shape[0], img.shape[1])/1.5)],
        #                  height=img.shape[0], width=img.shape[1], p=0.25),
        ], p=1)
        #A.Normalize(mean=mean, std=std)], p=1)


    val_transform = A.Compose([A.Normalize(mean=mean, std=std)], p=1)

    output_transforms = {'train': train_transform, 'val': val_transform}

    return output_transforms
