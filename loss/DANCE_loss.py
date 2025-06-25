# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:23:29 2023

@author: luzy1
"""

import torch
import torch.nn.functional as F

def entropy(p):
    p = F.softmax(p)
    return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))

def entropy_margin(p, value, margin=0.2, weight=None):
    p = F.softmax(p)
    return -torch.mean(hinge(torch.abs(-torch.sum(p * torch.log(p+1e-5), 1)-value), margin))

def hinge(input, margin=0.2):
    return torch.clamp(input, min=margin)

