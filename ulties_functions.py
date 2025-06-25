# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:06:33 2022

@author: 29792
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import argparse
import numpy as np
import os

#import data_loader
import Shuffle as Shuffle
import make_data
import make_unbalanced_data
import mmd
from models import models

import datasets
import sys
import numpy as np
from scipy.io import loadmat
import warnings
from ulties import *


if torch.cuda.is_available():
    device = torch.device("cuda")
    # device = torch.device("cpu")
else:
    warnings.warn("gpu is not available")
    device = torch.device("cpu")
                    


from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score

def test_score(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    prob_all = []
    lable_all = []
    labellabel = []
    probprob = []
    
    target_num = torch.zeros((1, args.num_classes)) # n_classes为分类任务类别数量
    predict_num = torch.zeros((1, args.num_classes))
    acc_num = torch.zeros((1, args.num_classes))
    labels = [x for x in range(0,args.num_classes)]

    with torch.no_grad():
        for data, target,_ in dataloader:
            # data, target = data.cuda(), target.cuda()
            target = target.cuda()

            pred = model.predict(data)
            
            ########################计算F1和AUC#############
            _, predicted = pred.max(1)
            
            pre_mask = torch.zeros(pred.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
            tar_mask = torch.zeros(pred.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)  # 得到数据中每类的数量
            acc_mask = pre_mask * tar_mask 
            acc_num += acc_mask.sum(0) # 得到各类别分类正确的样本数量


            ########################计算ACC#############
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            ########################计算AUC#############  .cpu().numpy()
            probprob.extend(pred.cpu().data.numpy())
            labellabel.extend(target.cpu().data.numpy())
            ytest = label_binarize(target.cpu().data.numpy(), classes=labels)
            ypreds = label_binarize(pred.cpu().data.numpy(), classes=labels)
            
            prob_all.extend(ypreds) #prob[:,1]返回每一行第二列的数，根据该函数的参数可知，y_score表示的较大标签类的分数，因此就是最大索引对应的那个值，而不是最大索引值
            lable_all.extend(ytest)
            
        try:   
            AUC = roc_auc_score(lable_all,prob_all,average='macro',multi_class='ovo')  #,average='macro',multi_class='ovo'
        except ValueError:
            AUC = 0
        print("AUC:{:.4f}".format(AUC))

        recall = acc_num / target_num
        precision = acc_num / predict_num
        F1 = 2 * recall * precision / (recall + precision)
        f1 = f1_score(labellabel, probprob, average='weighted')
        accuracy = 100. * acc_num.sum(1) / target_num.sum(1)

        print('Test Acc {}, recal {}, precision {}, F1-score {}'.format(accuracy, recall, precision, F1))
        print(f1)

        test_loss /= len(dataloader)
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)')
    
    return f1, AUC




# Entropy Loss           CKB中的条件熵最小化
def Entropy(prob):
    num_sam = prob.shape[0]
    Entropy = -(prob.mul(prob.log()+1e-4)).sum()
    return Entropy/num_sam


# compute entropy loss           RFAN中的条件熵最小化
def get_entropy_loss(p_softmax):
    mask = p_softmax.ge(0.00000001)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    
    return entropy / float(p_softmax.size(0))



def load_data(X_train, Y_train, X_test,Y_test,X_test_t,Y_test_t
              ,batch_size,shuffle=True):
    loader_src = make_data.Make_data(X_train,Y_train,batch_size,shuffle=shuffle)
    loader_tar = make_data.Make_data(X_test,Y_test,batch_size,shuffle=shuffle)
    loader_tar_test = make_data.Make_data(X_test_t,Y_test_t,batch_size,shuffle=shuffle)
    return loader_src, loader_tar, loader_tar_test










from timm.loss import LabelSmoothingCrossEntropy

def train_epoch(labels, label_cha, epoch_epoch, epoch, model, dataloaders, optimizer,task):
    
    train_loss_list,train_acc_list= [],[]
    acc_sum =0.0
    n = 0
    dist_loss_shang = 0

    
    model.train()
    source_loader, target_train_loader, _ = dataloaders
    ndata = len(target_train_loader.dataset)
    
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)

    num_iter = len(source_loader)
    num_iter_target = len(target_train_loader)
    for i in range(1, num_iter):
        data_source, label_source, index_source = next(iter_source)  #一个个取值
        data_target, label_target, index_target = next(iter_target)

        label_source =  label_source.to(device)
        label_target =  label_target.to(device)
        index_target =  index_target.to(device)
        
        if i % num_iter_target == 0:
            iter_target = iter(target_train_loader)
        if i % num_iter == 0:
            iter_source = iter(source_loader)
            

        '训练过程'
        optimizer.zero_grad()

        if i == num_iter-1:
            mu_value = 2
        else:
            mu_value = 0

        source_out, source_source, label_source_pred, label_source_pred_A, label_target_pred, loss_mmd, adversarial_loss = model(data_source, data_target,label_source,label_target,index_target, epoch,mu_value,task, ndata)

        
        '更改数据值'# 定义映射关系
        mapping = {0: 0, 1: 1, 2: 1}
        # 使用索引操作进行映射
        label_source_mapped_tensor = label_source.clone()  # 克隆输入张量
        for k, v in mapping.items():
            label_source_mapped_tensor[label_source == k] = v
        
        
        loss_cls = LabelSmoothingCrossEntropy(smoothing=0.3)(label_source_pred, label_source) + \
                    0.0 * LabelSmoothingCrossEntropy(smoothing=0.3)(label_source_pred, label_source_mapped_tensor)

        if epoch > args.middle_epoch + 10 and epoch > epoch_epoch: #   
            
            pseudo_model = torch.load('pseudo_model.pkl')
            pred = pseudo_model.predict(data_target)
            pred = pred.data.max(1)[1]
            pseudo_loss_cls = F.nll_loss(F.log_softmax(label_target_pred, dim=1), label_target,ignore_index=-1) #label_target  
            

        else:
            pseudo_loss_cls = 0


        loss =  loss_cls  +  1 *  loss_mmd

        
        loss.backward()
        optimizer.step()


        #导出loss和acc
        acc_sum += (label_source_pred.argmax(dim=1)== label_source).sum()
        n +=label_source.shape[0]
        acc = acc_sum/n
        train_loss_list.append(loss)
        train_acc_list.append(acc)
    return train_loss_list, train_acc_list






def test(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in dataloader:
            # data = data[0]["hello"]
            # data, target = data.cuda(), target.cuda()
            target = target.cuda()
            pred = model.predict(data)
            # sum up batch loss
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(dataloader)
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)')
    return correct





def t_SNE_fig(i, epoch,model,S_train_data,T_train_data,S_train_label,T_train_label):
    #保存结果
    if epoch % 250 == 0:
    # if epoch == 20:
                    
        from sklearn import datasets
        from openTSNE import TSNE
        import utils
        import matplotlib
        import matplotlib.pyplot as plt
        font2 = {'family' : 'Times New Roman',
                'weight' : 'normal',
                'size' : 25,
                }
        colors = ['k','r','gold','y','g','c','cyan','b','m','pink']
        # fig = plt.figure(figsize=(10,8))            #figsize=(10,10)
        fig,ax = plt.subplots(figsize=(10,8))            #figsize=(10,10)
        #cl =  [0,1,2,3,4,5,6,7,8,9]
  
        
        ###############################################
        '计算每种标签的准确度'
        # model = torch.load('model.pkl')
        # model.eval()
        
        outputs_1 = model.predict_fea(S_train_data[0:2000].cuda())    #.detach().numpy() predict predict_fea
        a = F.softmax(outputs_1, dim=1)
        predicted_1 = a.cpu().detach().numpy()
        
        outputs_2 = model.predict_fea(T_train_data[0:2000].cuda())    #.detach().numpy()  predict
        a = F.softmax(outputs_2, dim=1)
        predicted_2 = a.cpu().detach().numpy()
        ###############################################
        # ax3 = fig.add_subplot(133)
        x, y = predicted_1[:,:], S_train_label[0:2000].numpy() 
        X = TSNE().fit(x)
        
        x_t, y_t = predicted_2[:,:], T_train_label[0:2000].numpy() 
        X_t = TSNE().fit(x_t)
        for cl in range(4):
            indices = np.where(y==cl)
            indices = indices[0]
            yi = np.ones((1,len(indices)))  
            plt.scatter(X[indices, 0], X[indices, 1],  label=cl, color =colors[cl], marker='o',s=50)  


            indices = np.where(y_t==cl)
            indices = indices[0]
            yi = np.ones((1,len(indices)))  
            plt.scatter(X_t[indices, 0], X_t[indices, 1],  label=cl, color =colors[cl], marker='+',s=50) 
            # plt.yticks(fontsize=20,color='#000000','family' : 'Times New Roman',)
            # plt.xticks(fontsize=20,color='#000000') #不显示x轴刻度值    
            # plt.xlabel('CNN',font2)
            plt.tick_params(labelsize=23)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
        
        
        
        # ax2.legend(bbox_to_anchor=(0,1.02,1,0.2),ncol=5,borderaxespad = 0)
        # plt.savefig("D:/figures/temp{}.png".format(i))
        name_a= '%d_%d.png'%(i,epoch)
        plt.savefig(name_a, dpi=500, bbox_inches='tight') # 解决图片不清晰，不完整的问题
        #############################################################################################
        ##############################################################################################





def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default='/data/zhuyc/OFFICE31/')
    parser.add_argument('--src', type=str,
                        help='Source domain', default='设备A')
    parser.add_argument('--tar', type=str, 
                        help='Target domain', default='设备B')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    
    
    
    parser.add_argument('--model_name', type=str, default='TICNN', help='the name of the model')  #DASAN DCTLN   resnet_features_1d   resnet_features_1d
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    #显示距离域损失
    parser.add_argument('--distance_metric', type=bool, default=False, help='whether use distance metric')
    parser.add_argument('--distance_loss', type=str, choices=['MK-MMD','MMD', 'JMMD', 'CORAL','LMMD','CKB+MMD'], 
                        default=False, help='which distance loss you use')    #CKB+MMD
    
    #类间和类内间距
    parser.add_argument('--class_distance_loss', type=str,
                        default=False, help=['cep_loss'])

    #对比损失
    parser.add_argument('--constract_loss_metric', type=bool, default=False, help='whether use constract_loss metric')
    parser.add_argument('--constract_loss', type=str, choices=['constract_center_loss','SupervisedContrastiveLoss',
                                                               'constract_loss', 'CORAL','LMMD'], 
                        default='constract_loss', help='which distance loss you use')
    #隐式距离---对抗域损失
    parser.add_argument('--domain_adversarial', type=bool, default=False, help='whether use domain_adversarial')
    parser.add_argument('--adversarial_loss', type=str, choices=['DA', 'CDA', 'CDA+E'], 
                        default='b', help='which adversarial loss you use')
    
    #ADNN
    parser.add_argument('--ADNN', type=bool,
                        default=True, help='which distance loss you use')
    
    
    parser.add_argument('--hidden_size', type=int, default=1024, help='whether using the last batch')
    parser.add_argument('--trade_off_adversarial', type=str, default='Step', help='')
    parser.add_argument('--lam_adversarial', type=float, default=1, help='this is used for Cons')
    parser.add_argument('--transfer_loss_weight', type=float, default=0.5) #10
    
    parser.add_argument('--num_classes', type=int,
                        help='Number of classes', default=4)
    parser.add_argument('--batch_size', type=float,
                        help='batch size', default=64)    #64    128
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=500)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.002)
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=185)  #35  85
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2024)
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=0.5)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-1)   #5e-3   5e-1
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True) 
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=30)
    parser.add_argument('--middle_epoch', type=int, default=0, help='max number of epoch')
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    # model and data parameters     
    parser.add_argument('--data_name', type=str,choices=['CWRU_BJTU'], 
                        default='CWRU_BJTU', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default='D:\北京交通大学博士\实验数据\西储大学轴承数据中心网站', help='the directory of the data')
    parser.add_argument('--transfer_task', type=list, default=[[3], [1]], help='transfer learning tasks')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')
    parser.add_argument('--last_batch', type=bool, default=True, help='whether using the last batch')


    parser.add_argument('--gama1', type=int, help='xishu', default=1)
    parser.add_argument('--gama2', type=int, help='xishu', default=1)
    parser.add_argument('--gama3', type=int, help='xishu', default=0)
    
    parser.add_argument('--arf', type=int, help='xishu', default=2)
    parser.add_argument('--bata', type=int, help='xishu', default=1)
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    return args


args = get_args()