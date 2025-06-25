# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:17:25 2023

@author: luzy1
"""

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
            nn.Linear(39, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 384)
        )


    
    def forward(self, x):
        x = self.encoder(x)
        return x
    
    
    

class ConvAutoencoder_decoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_decoder, self).__init__()


        #解码器      
        self.decoder = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 39)
        )


        # self.decoder = nn.Sequential(
        #     nn.Linear(384*2, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 32)
        # )
        
        
    def forward(self, x):
        x = self.decoder(x)
        return x
    
    
    
