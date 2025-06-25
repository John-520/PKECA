# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 17:24:13 2022

@author: 29792
"""

''' ============== 特征提取的类 =====================
时域特征 ：11类
频域特征 : 13类
总共提取特征 ： 24类

参考文献 英文文献 016_C_(Q1 时域和频域共24种特征参数 )  Fault diagnosis of rotating machinery based on multiple ANFIS combination with GAs

'''
import math
import numpy as np
import scipy.stats

class Fea_Extra():
    def __init__(self, Signal, Fs = 25600):
        self.signal = Signal
        self.Fs = Fs

    def Time_fea(self, signal_):
        """
        提取时域特征 11 类
        """
        N = len(signal_)
        y = signal_
        t_mean_1 = np.mean(y)                                    # 1_均值（平均幅值）

        t_std_2  = np.std(y, ddof=1)                             # 2_标准差

        t_fgf_3  = ((np.mean(np.sqrt(np.abs(y)))))**2           # 3_方根幅值

        t_rms_4  = np.sqrt((np.mean(y**2)))                      # 4_RMS均方根

        t_pp_5   = 0.5*(np.max(y)-np.min(y))                     # 5_峰峰值  (参考周宏锑师姐 博士毕业论文)

        #t_skew_6   = np.sum((t_mean_1)**3)/((N-1)*(t_std_3)**3)
        t_skew_6   = scipy.stats.skew(y)                         # 6_偏度 skewness

        #t_kur_7   = np.sum((y-t_mean_1)**4)/((N-1)*(t_std_3)**4)
        t_kur_7 = scipy.stats.kurtosis(y)                        # 7_峭度 Kurtosis

        t_cres_8  = np.max(np.abs(y))/t_rms_4                    # 8_峰值因子 Crest Factor

        t_clear_9  = np.max(np.abs(y))/t_fgf_3                   # 9_裕度因子  Clearance Factor

        t_shape_10 = (N * t_rms_4)/(np.sum(np.abs(y)))           # 10_波形因子 Shape fator

        t_imp_11  = ( np.max(np.abs(y)))/(np.mean(np.abs(y)))  # 11_脉冲指数 Impulse Fator


        t_fea = np.array([t_mean_1, t_std_2, t_fgf_3, t_rms_4, t_pp_5,
                          t_skew_6,   t_kur_7,  t_cres_8,  t_clear_9, t_shape_10, t_imp_11 ])

        return t_fea

    def Fre_fea(self, signal_):
        """
        提取频域特征 13类
        :param signal_:
        :return:
        """
        L = len(signal_)
        PL = abs(np.fft.fft(signal_ / L))[: int(L / 2)]
        PL[0] = 0
        f = np.fft.fftfreq(L, 1 / self.Fs)[: int(L / 2)]
        x = f
        y = PL
        K = len(y)

        f_12 = np.mean(y)

        f_13 = np.var(y)

        f_14 = (np.sum((y - f_12)**3))/(K * ((np.sqrt(f_13))**3))

        f_15 = (np.sum((y - f_12)**4))/(K * ((f_13)**2))

        f_16 = (np.sum(x * y))/(np.sum(y))

        f_17 = np.sqrt((np.mean(((x- f_16)**2)*(y))))

        f_18 = np.sqrt((np.sum((x**2)*y))/(np.sum(y)))

        f_19 = np.sqrt((np.sum((x**4)*y))/(np.sum((x**2)*y)))

        f_20 = (np.sum((x**2)*y))/(np.sqrt((np.sum(y))*(np.sum((x**4)*y))))

        f_21 = f_17/f_16

        f_22 = (np.sum(((x - f_16)**3)*y))/(K * (f_17**3))

        f_23 = (np.sum(((x - f_16)**4)*y))/(K * (f_17**4))

        f_fea = np.array([f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23])

        return f_fea

    def Both_Fea(self):
        """
        :return: 时域、频域特征 array
        """
        t_fea = self.Time_fea(self.signal)
        f_fea = self.Fre_fea(self.signal)
        tf_fea = get_wavelet_packet_feature(self.signal)
        

        fea = np.concatenate((t_fea, f_fea,  tf_fea))

        return fea




import pywt
import numpy as np


def get_wavelet_packet_feature(data, wavelet='db2', mode='symmetric', maxlevel=4):  # db3
    """
    提取 小波包特征
    
    @param data: shape 为 (n, ) 的 1D array 数据，其中，n 为样本（信号）长度
    @return: 最后一层 子频带 的 能量百分比
    """
    wp = pywt.WaveletPacket(data, wavelet=wavelet, mode=mode, maxlevel=maxlevel)
    
    nodes = [node.path for node in wp.get_level(maxlevel, 'freq')]  # 获得最后一层的节点路径   freq    natural
    
    e_i_list = []  # 节点能量
    for node in nodes:
        e_i = np.linalg.norm(wp[node].data, ord=None) ** 2  # 求 2范数，再开平方，得到 频段的能量（能量=信号的平方和）
        e_i_list.append(e_i)

    e_total = np.sum(e_i_list)  # 总能量
    features = []
    for e_i in e_i_list:
        features.append(e_i / e_total * 100)  # 能量百分比
    
    return np.array(features)
