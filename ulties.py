# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 22:31:50 2022

@author: 29792
"""
import torch
import torch.nn.functional as F

# from LapSF import *
################################################################
'数据生成'
###############################################################
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from scipy import fftpack
from scipy import signal
from Fea_Extra import *

import numpy as np
import pandas as pd
from scipy.io import loadmat
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

from sklearn import preprocessing

############外圈  内圈  滚动体##################
deta_CWRU = [3.58, 5.42, 4.71]
deta_IMS = [7.09, 8.91, 4.197]
deta_BJTU = [7.21, 9.79, 3.21]
deta_Cana = [3.57, 5.43, 4.31]
deta_XJTU = [3.05, 4.95, 1.99]

# deta_HUST = [3.588, 5.412, 4.733]   #6205
deta_HUST = [3.620, 5.380, 4.915]  # 6206


def data_make(file_gou):
    '输入原始数据和标签'
    # data_raw
    # label

    '找到对应的掩码'
    feas = []
    data = data_raw
    for i in range(len(data)):
        Fea_Extra_model = Fea_Extra(data[i], Fs=fs)
        fea = Fea_Extra_model.Both_Fea()
        feas.append(fea)
    feas = np.array(feas)

    # 假设你的numpy数据矩阵和标签数组如下：
    matrix = feas
    labels = label

    'PIE'
    # 遍历矩阵的每一列（即每个特征）   特征编码
    for i in range(matrix.shape[1]):
        # 取出特定的特征值
        feature_values = matrix[:, i]
        # 用特征值和对应的标签创建一个pandas数据框
        feature_df = pd.DataFrame({'values': feature_values, 'labels': labels})
        # 按照标签对值进行分组，并对每个组进行求和
        sum_df = feature_df.groupby('labels').sum()
        # 找出和最大的标签
        max_label = sum_df['values'].idxmax()
        # 创建一个掩码，标记出哪些样本的标签是和最大的标签
        # mask = (labels == max_label)
        mask = (labels != max_label)
        # 将和最大的标签不对应的所有特征值置为0
        matrix[mask, i] = 0

        mask_mask = (labels == max_label)
        # 将和最大的标签对应的所有特征值置为1
        matrix[mask_mask, i] = 1

    # data_bearing_canshu = np.ones((4000,1500))
    data_bearing_canshu = np.ones((num_c * len(ran), num))

    A = {"hello": data_raw}
    B = {"hello": matrix}
    C = {"hello": matrix}
    D = {"hello": matrix}
    E = {"hello": data_bearing_canshu * deta_CWRU[1]}
    F = {"hello": data_bearing_canshu * deta_CWRU[2]}

    data = [A, B, C, D, E, F]

    return data, label


# 按照标签来划分数据集

def data_split(data_input, label_input, num_split):
    '''
    data_input:(a,b)
    label_input:(a,)
    num_split:每种标签下提取的样本数量
    return:
    '''
    zhonglei = int(label_input.max()) + 1
    num_split = int(num_split / zhonglei)
    data_all = []
    label_all = []
    data_all_r = []
    label_all_r = []
    for n_label in range(zhonglei):
        ip = np.where(label_input == n_label)
        data = data_input[ip]
        data_f = data[0:num_split, :]  # 按照标签提取前num个数据
        data_r = data[num_split:, :]  # 按照标签提取num以后的数据

        label = label_input[ip]
        label_f = label[0:num_split]
        label_r = label[num_split:]

        data_all.append(data_f)
        label_all.append(label_f)
        data_all_r.append(data_r)
        label_all_r.append(label_r)

    data_all = np.array(data_all)
    label_all = np.array(label_all)
    data_all = data_all.reshape(data_all.shape[0] * data_all.shape[1], data_all.shape[2])
    label_all = label_all.reshape(label_all.shape[0] * label_all.shape[1])

    data_all_r = np.array(data_all_r)
    label_all_r = np.array(label_all_r)
    data_all_r = data_all_r.reshape(data_all_r.shape[0] * data_all_r.shape[1], data_all_r.shape[2])
    label_all_r = label_all_r.reshape(label_all_r.shape[0] * label_all_r.shape[1])

    # return data_all, label_all, data_all_r, label_all_r
    return data_all, data_all_r, label_all, label_all_r


# 按照标签来划分数据集      对于多数据输入的情况很有用。  比方说：振动信号、转速信号、轴承参数值等等

def data_split_muti(data_input, label_input, num_split):
    '''
    data_input:(a,b)   字典数据
    label_input:(a,)
    num_split:每种标签下提取的样本数量
    return:
    '''

    data_data = []
    data_data_r = []

    for index in range(len(data_input)):
        data_all, data_all_r, label_all, label_all_r = data_split(data_input[index]["hello"], label_input, num_split)

        '将数据转为tensor向量，为了不出错'
        data_all = torch.tensor(data_all, dtype=torch.float32)
        data_all_r = torch.tensor(data_all_r, dtype=torch.float32)
        data_all = data_all.unsqueeze(1)
        data_all_r = data_all_r.unsqueeze(1)

        label_all = torch.tensor(label_all, dtype=torch.long)
        label_all_r = torch.tensor(label_all_r, dtype=torch.long)

        data_data_all = {"hello": data_all}
        data_data.append(data_data_all)

        data_data_all_r = {"hello": data_all_r}
        data_data_r.append(data_data_all_r)

    return data_data, data_data_r, label_all, label_all_r



