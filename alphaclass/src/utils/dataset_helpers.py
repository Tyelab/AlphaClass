import numpy as np
import os
import json
import cv2

from . import heatmap_helpers as hm_helpers


def read_labels(labels, string):
    return labels[labels.image == string]


def pd_reader(df, idx, resW = 80, resH = 64):
    behavior = df['behavior']
    x = df['x']
    y = df['y']
    image = df['image']
    width = df['width']
    height = df['height']

    nClasses = len(idx)


    blanks = np.zeros((nClasses, resH, resW))
    if len(behavior) > 0:

        for i_count, i in enumerate(image):
            blanks[idx[behavior[i_count]]] = hm_helpers.drawGaussian(blanks[idx[behavior[i_count]]],
                                                                     (x[i_count], y[i_count]),
                                                                     sigma=3)

    return blanks


def pd_reader_inference(df, idx, resW = 80, resH = 64):
    behavior = df['behavior'].to_numpy().tolist()
    x = df['x'].to_numpy().tolist()
    y = df['y'].to_numpy().tolist()
    image = df['image'].to_numpy().tolist()
    width = df['width']
    height = df['height']

    nClasses = len(idx)
    behavior_channel = [idx[behavior[i]] for i in range(len(behavior))]

    return (x, y, behavior_channel)

    
def read_image(full_path):
    return cv2.imread(full_path)
