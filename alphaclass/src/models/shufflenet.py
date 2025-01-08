import torch
import torchvision
from torch import nn

import numpy as np
import os
import json
import yaml
from copy import deepcopy


class UpsampleBlock(nn.Module):
    def __init__(self, in_, out_):
        super(UpsampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_, out_, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor = 2)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.upsample(x)
        return x


class ShuffleNet_Upsample(nn.Module):
    def __init__(self, backend, nClasses):
        super().__init__()

        self.backend = backend
        self.nClasses = nClasses
        self.factor = 8

        self.upsample1 = UpsampleBlock(48, int(512/self.factor))
        self.upsample2 = UpsampleBlock(int(512/self.factor), int(256/self.factor))
        self.out_conv = nn.Conv2d(int(256/self.factor), nClasses, 1, 1)
        self.shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backend(x)
        x = self.shuffle(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.out_conv(x)
        x = self.sigmoid(x)
        return x
