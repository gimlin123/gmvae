#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
        Project: GM Deep Learning Aesthetic Attribute Prediction
        Date: June 9, 2018
        Authors: Dr. Alex Burnap, Pangaea Research / MIT Postdoc
        Email: aburnap@pangaearesearch.com
        
        Description: 
        Code for training the deep learning model specified in config.
        Deep learning model for predicting aesthetic attribute
        ratings in range 1-5.  Models include baselines and deep learning
        models including best model: PGGAN-512 features + pretrained.
        
        License: 
        Copyright (C) General Motors Corporation - All Rights Reserved
        Unauthorized copying of this file, via any medium is strictly prohibited
        Proprietary and confidential

        GM Reference Numer: 4200283729
        GIC Number: 2017-551

        OSS Disclosure:
        Signed July, 31 2017

        IP Agreement:
        Date: July 31, 2017
        Andrew Norton, Executive Director, General Motors Global Market Research
        Alex Burnap, Pangaea Research, LLC
        
        Contact:
        Jeff L. Hartley, General Motors Corportation
        <jeff.l.hartley@gm.com>
        Joyce Salisbury, General Motors Corportation
        <joyce.a.salisbury@gm.com>
'''
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_, calculate_gain
import numpy as np
import sys
if sys.version_info.major == 3:
    from functools import reduce

DEBUG = False


class PixelNormLayer(nn.Module):
    """
    Pixelwise feature vector normalization.
    """

    def __init__(self, eps=1e-8):
        super(PixelNormLayer, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)


class WScaleLayer(nn.Module):
    """
    Applies equalized learning rate to the preceding layer.
    """

    def __init__(self, incoming):
        super(WScaleLayer, self).__init__()
        self.incoming = incoming
        self.scale = (torch.mean(self.incoming.weight.data**2))**0.5
        self.incoming.weight.data.copy_(self.incoming.weight.data / self.scale)
        self.bias = None
        if self.incoming.bias is not None:
            self.bias = self.incoming.bias
            self.incoming.bias = None

    def forward(self, x):
        #print(x.type())
        if x.is_cuda:
            self.scale = self.scale.cuda()
        x = self.scale * x
        if self.bias is not None:
            x += self.bias.view(1, self.bias.size()[0], 1, 1)
        return x

    def __repr__(self):
        param_str = '(incoming = %s)' % (self.incoming.__class__.__name__)
        return self.__class__.__name__ + param_str


def mean(tensor, axis, **kwargs):
    if isinstance(axis, int):
        axis = [axis]
    for ax in axis:
        tensor = torch.mean(tensor, axis=ax, **kwargs)
    return tensor


class LayerNormLayer(nn.Module):
    """
    Layer normalization. Custom reimplementation based on the paper: https://arxiv.org/abs/1607.06450
    """

    def __init__(self, incoming, eps=1e-4):
        super(LayerNormLayer, self).__init__()
        self.incoming = incoming
        self.eps = eps
        self.gain = Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.bias = None

        if self.incoming.bias is not None:
            self.bias = self.incoming.bias
            self.incoming.bias = None

    def forward(self, x):
        x = x - mean(x, axis=range(1, len(x.size())))
        x = x * 1.0 / (torch.sqrt(
            mean(x**2, axis=range(1, len(x.size())), keepdim=True) + self.eps))
        x = x * self.gain
        if self.bias is not None:
            x += self.bias
        return x

    def __repr__(self):
        param_str = '(incoming = %s, eps = %s)' % (
            self.incoming.__class__.__name__, self.eps)
        return self.__class__.__name__ + param_str


def he_init(layer, nonlinearity='conv2d', param=None):
    nonlinearity = nonlinearity.lower()
    if nonlinearity not in [
            'linear', 'conv1d', 'conv2d', 'conv3d', 'relu', 'leaky_relu',
            'sigmoid', 'tanh'
    ]:
        if not hasattr(layer, 'gain') or layer.gain is None:
            gain = 0  # default
        else:
            gain = layer.gain
    elif nonlinearity == 'leaky_relu':
        assert param is not None, 'Negative_slope(param) should be given.'
        gain = calculate_gain(nonlinearity, param)
    else:
        gain = calculate_gain(nonlinearity)
    kaiming_normal_(layer.weight, a=gain)
