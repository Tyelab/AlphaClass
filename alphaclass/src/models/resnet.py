import torch
import torchvision
from torch import nn

import numpy as np
import os
import json
import yaml
from copy import deepcopy

from models.layers_helper import DUC

#combine ResNet (Residual Network) with DUC (Dense Upsampling Convolution)
class ResNet_DUC(nn.Module):
    def __init__(self, backend, nClasses):
        super(ResNet_DUC, self).__init__()

        self.backend = backend
        self.nClasses = nClasses
        self.factor = 1

        self.shuffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, int(1024/self.factor), upscale_factor=2)
        self.duc2 = DUC(int(256/self.factor), int(512/self.factor), upscale_factor=2)
        self.conv_out = nn.Conv2d(int(128/self.factor), self.nClasses, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.backend(x)
        out = self.shuffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)
        out = self.conv_out(out)
        out = self.sigmoid(out)
        return out
