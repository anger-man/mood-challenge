#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:53:13 2022

@author: c
"""

#%%

#load packages

import torch
import torchvision
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader, Dataset
import gc, os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm as tq
from scipy.ndimage import rotate
import time
from skimage.measure import block_reduce

#%%
"""
define a 3d Unet with 3 downsampling steps, skip connection and depthwise 
separable convolutions
"""

def NORM(out_f,normalization):
    normlayer = nn.ModuleDict([
        ['instance',nn.InstanceNorm3d(out_f)],
        ['batch', nn.BatchNorm3d(out_f)]
        ])
    return normlayer[normalization]

class Conv3dsep(nn.Module):
    def __init__(self, in_f, out_f, stride, padding):
        super(Conv3dsep,self).__init__()
        
        # self.layer  = nn.Sequential(
        #     nn.Conv3d(in_f,  in_f, kernel_size=3,stride=stride,padding=padding,groups=in_f),
        #     nn.Conv3d(in_f,  out_f,kernel_size=1,stride=1))
        self.layer = nn.Conv3d(in_f,out_f,kernel_size=3, stride=stride, padding=padding)
    
    def forward(self, x):
        return (self.layer(x))
    

class double_conv(nn.Module):
    def __init__(self,in_f,out_f,normalization = 'instance'):
        super(double_conv, self).__init__()
        
        self.conv1  = nn.Sequential(
            Conv3dsep(in_f, out_f,stride=1,padding='same'),           
            NORM(out_f, normalization),
            nn.ReLU())
        
        self.conv2  = nn.Sequential(
            Conv3dsep(out_f, out_f,stride=1,padding='same'),   
            NORM(out_f, normalization),
            nn.ReLU())
                 
    def forward(self, x):
        return self.conv2(self.conv1(x))
    
    
class downsampling(nn.Module):
    def __init__(self,in_f,out_f,normalization = 'instance'):
        super(downsampling, self).__init__()
        
        self.down  = nn.Sequential(
            Conv3dsep(in_f, out_f,stride=2,padding=(1,1,1)),   
            NORM(out_f,normalization),
            nn.ReLU())
                 
    def forward(self, x):
        return(self.down(x))
    
    
class upsampling(nn.Module):
    def __init__(self,in_f,out_f,normalization='instance',nearest=True):
        super(upsampling, self).__init__()
        
        if nearest:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode = 'nearest'),
                Conv3dsep(in_f,out_f,stride=1,padding='same'),
                NORM(out_f, normalization),
                nn.LeakyReLU(0.2))
        else:
            self.up = nn.ConvTranspose3d(in_f,out_f,kernel_size=4,stride=2,padding=(1,1,1))
            
        self.conv1  = nn.Sequential(
            Conv3dsep(in_f, out_f,stride=1,padding='same'),   
            NORM(out_f, normalization),
            nn.LeakyReLU(0.2))
        
        self.conv2  = nn.Sequential(
            Conv3dsep(out_f, out_f,stride=1,padding='same'),   
            NORM(out_f, normalization),
            nn.LeakyReLU(0.2))
            
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x,skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class unet(nn.Module):
    def __init__(self, n_channels, f_size, normalization='instance',
                 out_acti = 'sigmoid'):
        super(unet, self).__init__()
        self.dc0 = double_conv(n_channels, f_size, normalization)
        self.ds0 = downsampling(f_size, f_size, normalization)
        self.dc1 = double_conv(f_size, 2*f_size, normalization)
        self.ds1 = downsampling(2*f_size, 2*f_size, normalization)
        self.dc2 = double_conv(2*f_size, 4*f_size, normalization)
        self.ds2 = downsampling(4*f_size, 4*f_size, normalization)
        self.dc3 = double_conv(4*f_size, 8*f_size, normalization)
        self.ds3 = downsampling(8*f_size, 8*f_size, normalization)
        self.dc4 = double_conv(8*f_size, 16*f_size, normalization)
        
        self.up3 = upsampling(16*f_size, 8*f_size, normalization)
        self.up2 = upsampling(8*f_size, 4*f_size, normalization)
        self.up1 = upsampling(4*f_size, 2*f_size, normalization)
        self.up_out = upsampling(2*f_size, f_size, normalization)
        self.up_unc = upsampling(2*f_size, f_size, normalization)
        self.out = nn.Sequential(
            Conv3dsep(f_size, 1,stride=1,padding='same'),   
            nn.ModuleDict([['relu',nn.ReLU()],['sigmoid',nn.Sigmoid()]]
                          )[out_acti])
        self.unc = nn.Sequential(
            Conv3dsep(f_size, 1,stride=1,padding='same'),   
            nn.Softplus())
        
        
    def forward(self, x):
        x = x.float()
      
        skip0 = self.dc0(x)
        down1 = self.ds0(skip0)
        skip1 = self.dc1(down1)
        down2 = self.ds1(skip1)
        skip2 = self.dc2(down2)
        down3 = self.ds2(skip2)
        skip3 = self.dc3(down3)
        # down4 = self.ds3(skip3)
        # skip4 = self.dc4(down4)
        
        # upsa3 = self.up3(skip4,skip3)
        upsa2 = self.up2(skip3,skip2)
        upsa1 = self.up1(upsa2,skip1)
        
        obranch1 = self.up_out(upsa1,skip0)
        img_out = self.out(obranch1)
        
        obranch2 = self.up_unc(upsa1,skip0)
        unc_out = self.unc(obranch2)
        
        return([img_out,unc_out])
    
class random_net(nn.Module):
    def __init__(self, n_channels, f_size, normalization='instance',
                 out_acti = 'sigmoid'):
        super(random_net, self).__init__()
        self.dc0 = double_conv(n_channels, f_size, normalization)
        self.dc1 = double_conv(f_size, f_size, normalization)
        self.out = nn.Sequential(
            Conv3dsep(f_size, 1,stride=1,padding='same'),   
            nn.ModuleDict([['relu',nn.ReLU()],['sigmoid',nn.Sigmoid()]]
                          )[out_acti])
       
        
    def forward(self, x):
        x = x.float()
      
        lay = self.dc0(x)
        lay = self.dc1(lay)
        img_out = self.out(lay)
                
        return(img_out)

