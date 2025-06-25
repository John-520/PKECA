# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 19:12:18 2021

@author: 29792
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
from loss.DAN import DAN
from loss.MMD import mmd_rbf_noaccelerate, mmd_rbf_accelerate
from loss.KMMD import kmmd_loss
from loss.JAN import JAN
from loss.MMD_loss import MMD_loss
from loss.CORAL import CORAL
import sys 
sys.path.append("D:\北京交通大学博士\论文【小】\论文【第四章】\code") 
from MMSD_main.MMSD import MMSD

from loss.lmmd import LMMD_loss
from loss.contrastive_center_loss import ContrastiveCenterLoss
from loss.SupervisedContrastiveLoss import SupervisedContrastiveLoss
from loss.ContrastiveLoss import ContrastiveLoss

import sub_models
from loss.adv import *
import CKButils

from loss.DANCE_loss import *



'行列正则化'
def l2row_torch(X):
	N = torch.sqrt((X**2).sum(axis=1)+1e-8)
	Y = (X.T/N).T
	return Y,N

BCEWithLogitsLoss = nn.BCEWithLogitsLoss()


class models(nn.Module):

    def __init__(self, args):
        super(models, self).__init__()
       # self.feature_layers = CNN_1d(True)
        self.args = args
        
        self.num_classes= args.num_classes
        self.adversarial_loss = self.args.adversarial_loss
        #这'句话很重要'

        self.feature_layers = getattr(sub_models, args.model_name)(args,  args.pretrained)
        self.feature_layers_yuyi = getattr(sub_models, 'TICNN_5')(args, args.pretrained)
        self.ad_layers = getattr(sub_models, 'DomainClassifier')(100,1)

        self.bottle = nn.Sequential(nn.Linear(384, 150),  # 192    384
                                    nn.GELU(),
                                    nn.Dropout()
                                    ) #100   512   64     512 , nn.ReLU(), nn.Dropout()
        self.bottle_taskA = nn.Sequential(nn.Linear(self.feature_layers.output_num(), 256),
                                    nn.GELU(), 
                                    # nn.Sigmoid(), 
                                    nn.Dropout()
                                    ) #100   512   64     512 , nn.ReLU(), nn.Dropout()
        self.drop = nn.Dropout(0)
        if self.args.model_name == 'RFFDN':
            self.cls_fc = nn.Linear(self.feature_layers.output_num(), self.num_classes)
        else:
            self.cls_fc = nn.Linear(150, self.num_classes) #
            
            self.cls_fc_taskA = nn.Linear(256, 3)     # 多任务下的分类器  ；或者是辅助任务下的分类器
            self.cls_fc_taskB = nn.Linear(256, 2) 
            self.cls_fc_taskC = nn.Linear(256, 2) 
            
            
            self.cls_fc1 = nn.Linear(self.feature_layers.output_num(), 256) #
            self.cls_fc2 = nn.Linear(256, self.num_classes) #
            
            self.cls_fc_pseudo = nn.Linear(1, self.num_classes) #
            # self.cls_fc3 = nn.Linear(100, self.num_classes) #
        #定义对比损失函数的输入
        self.prelu_ip1 = nn.PReLU()
        self.contrast = nn.Linear(self.feature_layers.output_num(), 2)  #

        self.Sigmoid = nn.Sigmoid()
        self.Dropout = nn.Dropout()   # 0.5
        

        self.fc_mu = nn.Linear(384, 384 * 2)

        
        self.bn=nn.BatchNorm1d(384)
        # self.head = nn.Linear(384, 64)
        self.head = nn.Sequential(
                nn.Linear(384, 100),
                nn.ReLU(inplace=True),
                nn.Linear(100, 30)
            )
        
        self.PEC_bottle = nn.Sequential(
                nn.Linear(384, 10),
            )
        

        
        self.ConvAutoencoder_encoder = getattr(sub_models, 'ConvAutoencoder_encoder')()
        self.ConvAutoencoder_decoder = getattr(sub_models, 'ConvAutoencoder_decoder')()
        
        self.ConvAutoencoderVAE_encoder = getattr(sub_models, 'ConvAutoencoderVAE_encoder')()
        self.ConvAutoencoderVAE_decoder = getattr(sub_models, 'ConvAutoencoderVAE_decoder')()
        
        self.ConvAutoencoder_encoder_PCE = getattr(sub_models, 'ConvAutoencoder_encoder_PCE')()
        self.ConvAutoencoder_decoder_PCE = getattr(sub_models, 'ConvAutoencoder_decoder_PCE')()



        self.sigmoid = nn.Sigmoid()
        self.feature_dim = 384

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
        

    def l2_zheng(self,x):
        x = torch.sqrt(x**2 + 1e-8)
        x = l2row_torch(x)[0]
        x = l2row_torch(x.T)[0].T        
        return x



        
    def forward(self, source,target,label_source,label_target, index_target, epoch,mu_value,task, ndata): #(64,1,1200)
        loss = 0
        adv_loss=0


        source_source = source[0]["hello"].cuda()
        target_target = target[0]["hello"].cuda()
        source_PCE = source[3]["hello"].squeeze().cuda()
        
        
        '源域输入' ############################################################################################################

        s_pred_A = 0

        ############################################################################################################融合的自编码
        s_z = self.ConvAutoencoderVAE_encoder(source_source)   #
        source_z_PCE = self.ConvAutoencoder_encoder_PCE(source_PCE) 
        source_z = s_z
        source_z = source_z.view(source_z.size(0), -1)
        source_PCE = source_PCE.view(source_PCE.size(0), -1)

        f1=torch.cat([source_z,source_z_PCE],dim=1)
        f2=torch.cat([source_z_PCE,source_z],dim=1)
        f1 = f1.reshape(s_z.shape[0],s_z.shape[1],s_z.shape[2] * 2)
        
        
        
        source_out = self.ConvAutoencoderVAE_decoder(f1) 
        loss_R1=F.mse_loss(source_source,source_out)
        
         
        source_out_PCE = self.ConvAutoencoder_decoder_PCE(f2)
        loss_R2=F.mse_loss(source_PCE,source_out_PCE)
        


        '均方误差'
        loss_PCE = F.mse_loss(source_z,source_z_PCE)

      
        '分类功能' ############################################################################################################
        
        source_z = self.l2_zheng(source_z)
        source = self.bottle(source_z)
        source = self.l2_zheng(source)
        s_pred = self.cls_fc(source)
        
        
        PCE_pred = self.PEC_bottle(source_z_PCE)
        loss_cls = F.nll_loss(F.log_softmax(PCE_pred, dim=1), label_source)
        

            
        cep_loss_sou = self.cep_loss(source, label_source)
        if self.training == True:    
            
            if epoch< self.args.middle_epoch:
                loss_cls = 0
                loss_PCE = 0
                loss_PCEC = 0


            loss = loss_R1 + loss_R2 + loss_PCE + cep_loss_sou#


        '目标域输入'############################################################################################################
        target = self.ConvAutoencoder_encoder(target_target)
        target = target.view(target.size(0), -1)

        target = self.bottle(source_z)
        t_pred = self.cls_fc(target)


        return source_out, source_source, s_pred, s_pred_A, t_pred,   loss,  0 * adv_loss 


    '上述损失函数的软化'
    def cep_loss(self, target, label_target):
        # 定义损失函数值
        intra_class_loss = 0.0
        inter_class_loss = 0.0
        # 计算质心
        centroids_list = []
        labels = label_target.unique()
        # labels =  torch.arange(self.num_classes)
        for label in labels:
            target_class = target[label_target == label]
            centroid = target_class.mean(dim=0)
            centroids_list.append(centroid)
        centroids = torch.stack(centroids_list)
        # 计算类内分布和损失
        for label in labels:
            target_class = target[label_target == label]
            
            try:
                distance_to_centroid = torch.norm(target_class - centroids[label], p=2, dim=1)
            except IndexError as e:
                print("An IndexError occurred:", e)
                distance_to_centroid = torch.zeros_like(target_class)
            intra_prob = F.softmax(-distance_to_centroid, dim=0)
            intra_class_loss += intra_prob.mean()
        # 计算类间分布和损失
        for i, centroid_i in enumerate(centroids):
            for j, centroid_j in enumerate(centroids):
                if i != j:
                    distance_between_centroids = torch.norm(centroid_i - centroid_j, p=2)
                    # 使用softmax函数来软化类间距离
                    inter_prob = F.softmax(distance_between_centroids.unsqueeze(0), dim=0)
                    inter_class_loss += inter_prob.mean()

        loss = intra_class_loss - inter_class_loss
        return loss




    def predict(self, x):
        target_target = x[0]["hello"].cuda()
        x = self.ConvAutoencoderVAE_encoder(target_target)
        '###########################################'
        x = x.view(x.size(0), -1)
        x = self.l2_zheng(x)
        x = self.bottle(x)
        x = self.l2_zheng(x)

        return self.cls_fc(x)