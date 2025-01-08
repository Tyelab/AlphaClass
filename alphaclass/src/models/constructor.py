import torch
import torchvision
from torch import nn

import numpy as np
import os
import json
import yaml
from copy import deepcopy

from models.unet import ConvBlock, DualConvBlock, FullUnetConstructor
from models.mobilenet import MobileNet_DUC
from models.shufflenet import ShuffleNet_Upsample
from models.resnet import ResNet_DUC

unet_param_size = {'small': 16, 'medium': 32, 'large': 64}
mobilenet_upsample_param_size = {'small': 4, 'medium': 2, 'large': 1}
mobilenet_downsample_param_size = {'large': [[1, 16, 1, 1],
                                             [6, 24, 2, 2],
                                             [6, 32, 3, 2],
                                             [6, 64, 4, 2],
                                             [6, 96, 3, 1],
                                             [6, 160, 3, 2],
                                             [6, 320, 1, 1],],

                                  'medium': [[1, 8, 1, 1],
                                             [6, 12, 2, 2],
                                             [6, 16, 3, 2],
                                             [6, 32, 4, 2],
                                             [6, 48, 3, 1],
                                             [6, 80, 3, 2],
                                             [6, 160, 1, 1],],

                                   'small': [[1, 4, 1, 1],
                                             [3, 6, 2, 2],
                                             [3, 8, 3, 2],
                                             [3, 16, 4, 2],
                                             [3, 24, 3, 1],
                                             [3, 40, 3, 2],
                                             [3, 80, 1, 1]]}

def return_model(configs, nClasses):

    if type(configs) == str:
        with open(configs, 'r') as f:
            configs = json.load(f)

    if configs['model_type'] == 'unet_double':
        model = FullUnetConstructor(unet_param_size[configs['downsample_parameter_size']], nClasses, DualConvBlock)

    elif configs['model_type'] == 'unet_single':
        model = FullUnetConstructor(unet_param_size[configs['downsample_parameter_size']], nClasses, ConvBlock)

    elif configs['model_type'] == 'mobilenet':
        irs = mobilenet_downsample_param_size[configs['downsample_parameter_size']]
        #m = torchvision.models.mobilenetv2.MobileNetV2(inverted_residual_setting=irs).features
        m = torchvision.models.mobilenet_v2(pretrained=True).features
        model = MobileNet_DUC(m, nClasses, mobilenet_upsample_param_size, configs['upsample_parameter_size'])

    elif configs['model_type'] == 'shufflenet':
        shufflenet = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
        modules = list(shufflenet.children())[:-2]
        shuffle_backend = nn.Sequential(*modules)
        model = ShuffleNet_Upsample(shuffle_backend, nClasses)

    elif configs['model_type'] == 'resnet':
        resnet = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        resnet_backend = nn.Sequential(*modules)
        model = ResNet_DUC(resnet_backend, nClasses)


    return model
