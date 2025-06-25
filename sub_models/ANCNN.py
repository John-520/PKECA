#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings

# ----------------------------inputsize >=28-------------------------------------------------------------------------
class ANCNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(ANCNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.feature_layers = nn.Sequential(

            ########   81 = 5+4*19   ;11+10*7;    21+20*3
            nn.Conv1d(1, 16, kernel_size=25, stride=4,padding=1,dilation=1),  # 16, 26 ,26
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
           # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2,padding=1),
            
            nn.Conv1d(16, 32, kernel_size=17, stride=2,padding=1),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
          #  nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=8, stride=1,padding=1),  # 32, 24, 24
         #   nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
           # nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            )

        self.FC = nn.Linear(2176, 448)   
        
        
        self.__in_features = 448
        
    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.FC(x)
        return x


    def output_num(self):
        return self.__in_features


