# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:41:56 2023

@author: luzy1
"""
from torch import nn
import torch.nn.functional as F


class ConvAutoencoder_encoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_encoder, self).__init__()

        #编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=81, stride=8, padding=1, dilation=1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )



    
    def forward(self, x):
        x = self.encoder(x)
        return x
    
    
    
    
    
    
    
    
    
    
    
    
    
    

class ConvAutoencoder_decoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_decoder, self).__init__()


        #解码器          特征相加的解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            nn.ConvTranspose1d(16, 1, kernel_size=80, stride=8, padding=0, output_padding=0),
            # nn.Sigmoid()  # Assuming you want the output in the range [0, 1]
        )





    def forward(self, x):
        x = self.decoder(x)
        return x
    
    

