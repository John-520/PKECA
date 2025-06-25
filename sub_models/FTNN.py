#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings



# ----------------------------inputsize >=28-------------------------------------------------------------------------
class FTNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(FTNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")
            
        
        
        self.feature_layers1 = nn.Sequential(
            ########   81 = 5+4*19   ;11+10*7;    21+20*3
            nn.Conv1d(1, 20, kernel_size=5),  # 16, 26 ,26
            nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            )
        self.feature_layers2 = nn.Sequential(
            nn.Conv1d(20, 20, kernel_size=5),  # 32, 24, 24
            nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            )
    
        self.pooling_layers = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2)
            # nn.AdaptiveMaxPool1d(4)  # 128, 4,4
            )
        self.__in_features = 5940
        # self.CBAMnet = CBAMLayer(self.__in_features)
        
    def forward(self, x):
        x = self.feature_layers1(x)
        # fea = x.shape[1]
        # CBAMnet = CBAMLayer(fea)
        # x = CBAMnet(x)
        
        x = self.feature_layers2(x)
        # fea = x.shape[1]
        # CBAMnet = CBAMLayer(fea)
        # x = CBAMnet(x)
        # x = self.pooling_layers(x)

        
        return x


    def output_num(self):
        return self.__in_features
    
    
    
# model = FTNN()
# print(model)    
# from DNN_printer import DNN_printer
# DNN_printer(model, (1, 1200),64) 
    
    
    
    
    
    
    
    

import torch
import torch.nn as nn
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv1d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
    
    