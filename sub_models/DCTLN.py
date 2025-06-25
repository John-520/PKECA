# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:26:45 2022

@author: 29792
"""

from torch import nn

class DCTLN(nn.Module):
    def __init__(self, pretrained=False):
        super(DCTLN, self).__init__()
        self.conv1 = nn.Sequential(  # input: 64,1,2048--64-16
            nn.Conv1d(1, 16, kernel_size=64, stride=1, padding=32),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(  # input: 64,16,62
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(  # input: 64,32,31
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(150 * 64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.__in_features = 1024

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(-1, 150 * 64)
        x_fc1 = self.fc1(x)
        return x_fc1

    def output_num(self):
        return self.__in_features
    
    
import torch
from torch.autograd import Function
from typing import Any, Optional, Tuple

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None



    
class DomainClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.BatchNorm1d(out_dim),
            # nn.ReLU(),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x):
        x = ReverseLayerF.apply(x, 1.0)
        # x = self.grl(x)
        output = self.fc1(x)
        return output
    
    
    
