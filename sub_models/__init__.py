#!/usr/bin/python
# -*- coding:utf-8 -*-
from sub_models.cnn_1d import cnn_features as cnn_features_1d
from sub_models.TICNN_5 import TICNN_5 as TICNN_5
from sub_models.TICNN_10 import TICNN_10 as TICNN_10
from sub_models.TICNN_PGDDTN import TICNN_PGDDTN as TICNN_PGDDTN

from sub_models.MIXCNN import MIXCNN as MIXCNN
from sub_models.DCTLN import DCTLN as DCTLN
from sub_models.DASAN import DASAN as DASAN

from sub_models.DTN_JDA import DTN_JDA as DTN_JDA

from sub_models.DCTLN import DomainClassifier as DomainClassifier

from sub_models.MLP import MLP as MLP
from sub_models.FTNN import FTNN as FTNN
from sub_models.ANCNN import ANCNN as ANCNN
from sub_models.TICNN import TICNN as TICNN
from sub_models.DNCNN import DNCNN as DNCNN
from sub_models.RFFDN import RFFDN as RFFDN
from sub_models.CNN1d import CNN1d as CNN1d
from sub_models.AdversarialNet import AdversarialNet
from sub_models.resnet18_1d import resnet18_features as resnet_features_1d
from sub_models.Resnet1d import resnet18 as resnet_1d

from sub_models.TPTLN import TPTLN as TPTLN

from sub_models.CNNAE import ConvAutoencoder_encoder as ConvAutoencoder_encoder
from sub_models.CNNAE import ConvAutoencoder_decoder as ConvAutoencoder_decoder

from sub_models.CNNVAE import ConvAutoencoder_encoder as ConvAutoencoderVAE_encoder
from sub_models.CNNVAE import ConvAutoencoder_decoder as ConvAutoencoderVAE_decoder

from sub_models.CNNAE_PCE import ConvAutoencoder_encoder as ConvAutoencoder_encoder_PCE
from sub_models.CNNAE_PCE import ConvAutoencoder_decoder as ConvAutoencoder_decoder_PCE


from sub_models.CNNAE_1 import ConvAutoencoder_encoder as ConvAutoencoder_encoder_1
from sub_models.CNNAE_1 import ConvAutoencoder_decoder as ConvAutoencoder_decoder_1