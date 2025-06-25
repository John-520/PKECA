# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:08:07 2021

@author: 29792
"""
#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings

# ----------------------------inputsize >=28-------------------------------------------------------------------------
class DNCNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(DNCNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.feature_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=1,padding=1),  # 16, 26 ,26
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2,padding=1),
            
            nn.Conv1d(16, 32, kernel_size=15, stride=1,padding=1),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
          #  nn.Dropout(),
            nn.MaxPool1d(kernel_size=2, stride=2,padding=1)
            )
        
        
        self.__in_features = 9344
    def forward(self, x):
        x = self.feature_layers(x)
        return x


    def output_num(self):
        return self.__in_features


