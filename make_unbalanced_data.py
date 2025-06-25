# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:37:11 2021

@author: 29792
"""
from functools import partial
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import torch.utils.data as data
import numpy as np
import torch
from torchsampler import ImbalancedDatasetSampler

class My_unbalanced_dataset(data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)
    def get_labels(self): 
        return self.y

def Make_data(X_train, Y_train,batch_size,shuffle=True):
    datasets = My_unbalanced_dataset(X_train, Y_train)  # 初始化

    dataloader = torch.utils.data.DataLoader(datasets,sampler=ImbalancedDatasetSampler(datasets),
                                           batch_size=batch_size)
    return dataloader
