# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 17:52:09 2021

@author: 29792
"""
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)








import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
    
    
class BasicBlock(nn.Module):
    """
    basic building block for ResNet-18, ResNet-34
    """
    message = "basic"

    def __init__(self, in_channels, out_channels, strides, is_se=False):
        super(BasicBlock, self).__init__()
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)  # same padding
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False)
        if self.is_se:
            self.se = SE(out_channels, 16)

        # fit input with residual output
        self.short_cut = nn.Sequential()
        if strides is not 1:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=strides, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + self.short_cut(x)
        return F.relu(out)



class BottleNeck(nn.Module):
    """
    BottleNeck block for RestNet-50, ResNet-101, ResNet-152
    """
    message = "bottleneck"

    def __init__(self, in_channels, out_channels, strides, is_se=False):
        super(BottleNeck, self).__init__()
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)  # same padding
        self.conv2 = BN_Conv2d(out_channels, out_channels, 3, stride=strides, padding=1, bias=False)
        self.conv3 = BN_Conv2d(out_channels, out_channels * 4, 1, stride=1, padding=0, bias=False, activation=False)
        if self.is_se:
            self.se = SE(out_channels * 4, 16)

        # fit input with residual output
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1, stride=strides, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + self.shortcut(x)
        return F.relu(out)
    
    
class ResNet(nn.Module):
    """
    building ResNet_34
    """

    def __init__(self, block: object, groups: object, num_classes=1000) -> object:
        super(ResNet, self).__init__()
        self.channels = 64  # out channels from the first convolutional layer
        self.block = block

        self.conv1 = nn.Conv2d(3, self.channels, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.channels)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2_x = self._make_conv_x(channels=64, blocks=groups[0], strides=1, index=2)
        self.conv3_x = self._make_conv_x(channels=128, blocks=groups[1], strides=2, index=3)
        self.conv4_x = self._make_conv_x(channels=256, blocks=groups[2], strides=2, index=4)
        self.conv5_x = self._make_conv_x(channels=512, blocks=groups[3], strides=2, index=5)
        self.pool2 = nn.AvgPool2d(1)
        patches = 512 if self.block.message == "basic" else 512 * 4
        self.fc = nn.Linear(patches, num_classes)  # for 224 * 224 input size

    def _make_conv_x(self, channels, blocks, strides, index):
        """
        making convolutional group
        :param channels: output channels of the conv-group
        :param blocks: number of blocks in the conv-group
        :param strides: strides
        :return: conv-group
        """
        list_strides = [strides] + [1] * (blocks - 1)  # In conv_x groups, the first strides is 2, the others are ones.
        conv_x = nn.Sequential()
        for i in range(len(list_strides)):
            layer_name = str("block_%d_%d" % (index, i))  # when use add_module, the name should be difference.
            conv_x.add_module(layer_name, self.block(self.channels, channels, list_strides[i]))
            self.channels = channels if self.block.message == "basic" else channels * 4
        return conv_x

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn(out))
        out = self.pool1(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        print(out.shape)
        out = self.pool2(out)
        print(out.shape)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out))
        return out
    
    
def ResNet_18(num_classes=3):
    return ResNet(block=BasicBlock, groups=[2, 2, 2, 2], num_classes=num_classes)

def ResNet_34(num_classes=1000):
    return ResNet(block=BasicBlock, groups=[3, 4, 6, 3], num_classes=num_classes)

def ResNet_50(num_classes=1000):
    return ResNet(block=BottleNeck, groups=[3, 4, 6, 3], num_classes=num_classes)

def ResNet_101(num_classes=1000):
    return ResNet(block=BottleNeck, groups=[3, 4, 23, 3], num_classes=num_classes)

def ResNet_152(num_classes=1000):
    return ResNet(block=BottleNeck, groups=[3, 8, 36, 3], num_classes=num_classes)



def test():
    # net = ResNet_18()
    # net = ResNet_34()
    # net = ResNet_50()
    # net = ResNet_101()
    net = ResNet_18()
    summary(net, (3, 32, 32))
test()
