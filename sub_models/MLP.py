#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings

# ----------------------------inputsize >=28-------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(MLP, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.feature_layers = nn.Sequential(
        
            nn.Linear(1200, 1200),
            nn.ReLU(inplace=True),
            )
        
        
        

        self.FC = nn.Linear(1200, 600)   
        
        
        self.__in_features = 600
        
    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.FC(x)
        return x


    def output_num(self):
        return self.__in_features



