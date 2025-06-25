# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:26:45 2022

@author: 29792
"""

from torch import nn

class DASAN(nn.Module):
    def __init__(self, pretrained=False):
        super(DASAN, self).__init__()
        self.conv1 = nn.Sequential(  # input: 64,1,2048--64-16
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(  # input: 64,16,62
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(  # input: 64,32,31
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(37*32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.__in_features = 1024

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(-1, 37*32)
        x_fc1 = self.fc1(x)
        return x_fc1

    def output_num(self):
        return self.__in_features
    
    


