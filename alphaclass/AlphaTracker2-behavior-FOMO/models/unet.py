import torch
import torchvision
from torch import nn

import numpy as np
import os
import json
import yaml
from copy import deepcopy


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.activation(self.bn(self.conv1(x)))
        return out

    
class DualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.activation(self.bn(self.conv1(x)))
        out = self.activation(self.bn(self.conv2(out)))
        return out


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block):
        super(DownSampleBlock, self).__init__()
        
        self.maxpool = nn.MaxPool2d(2)
        self.conv_block = block(in_channels, out_channels)

    def forward(self, x):
        out = self.conv_block(self.maxpool(x))
        return out


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block):
        super(UpSampleBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = block(in_channels, out_channels)
       
    def forward(self, x1, x2):
        x1 = self.up(x1)
        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)
        return out


class FullUnetConstructor(nn.Module):
    def __init__(self, in_channels, nClasses, block):
        super(FullUnetConstructor, self).__init__()
        
        self.in_channels = in_channels
        self.nClasses = nClasses
        self.block = block
        factor = 2

        self.first = self.block(3, self.in_channels)
        self.d1 = DownSampleBlock(self.in_channels, self.in_channels*2, self.block)
        self.d2 = DownSampleBlock(self.in_channels*2, self.in_channels*4, self.block)
        self.d3 = DownSampleBlock(self.in_channels*4, self.in_channels*8, self.block)
        self.d4 = DownSampleBlock(self.in_channels*8, self.in_channels*16 // factor, self.block)
        
        self.u1 = UpSampleBlock(self.in_channels*16, self.in_channels*8 // factor, self.block)
        self.u2 = UpSampleBlock(self.in_channels*8, self.in_channels*4 // factor, self.block)
        self.u3 = UpSampleBlock(self.in_channels*4, self.in_channels*2 // factor, self.block)
        self.u4 = UpSampleBlock(self.in_channels*2, self.in_channels, self.block)
        
        self.output_conv = nn.Conv2d(self.in_channels, self.nClasses, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        
        out = self.u1(x5, x4)
        out = self.u2(out, x3)
        out = self.u3(out, x2)
        out = self.u4(out, x1)
            
        confidences = self.output_conv(out)
        #confidences = self.sigmoid(out)
        return confidences
