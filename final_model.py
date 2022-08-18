#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:26:02 2022

@author: c
"""

#%%

# define the task [brain,abdom] and the corresponding data directory

task = 'brain'
data_path = 'data/%s'%task

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
from datetime import datetime
from data_functions import generate_random_mask, add_gaussian_noise, random_intensity
from data_functions import DiceLoss, MedicalDataset, BCE
from models import unet

#%%

train_on_gpu = torch.cuda.is_available()

#%%


#also evaluate on the provided toy data

input_path = os.path.join(data_path,'final')
input_ids = np.sort([os.path.join(input_path,f) for f in os.listdir(input_path)])

target_path = os.path.join(data_path,'final')
target_ids = np.sort([os.path.join(target_path,f) for f in os.listdir(target_path)])


#%%

#evaluate the global model for downsampled version of spatial size 64x64x64 

criterion = DiceLoss(evaluation_mode = True)

model_global = unet(n_channels = 1, f_size=32)
if train_on_gpu:
    model_global.cuda()
model_global.load_state_dict(torch.load('weights/weights_%s_global.pt'%task))

model_local = unet(n_channels = 1, f_size=32)
if train_on_gpu:
    model_local.cuda()
model_local.load_state_dict(torch.load('weights/weights_%s_local.pt'%task))

#define a index for the plots and saved model parameters
dt = datetime.now()
index = '%d%d%d_%d%d_final'%(dt.year,dt.month,dt.day,dt.hour,dt.minute)



        
################################################
# validate the model on data with random masks #
################################################
model_global.eval() 
model_local.eval()
valid_loss = 0.0; save_data = 0; count=0

for k in range(len(input_ids)):
    nifti = nib.load(input_ids[k])
    data = nifti.get_fdata()
    data = np.expand_dims(np.expand_dims(data,0),0)
    data = torch.from_numpy(data)
    pred = model_global(data)[0]
    
    nifti = nib.load(target_ids[k])
    target = nifti.get_fdata()
    target = np.expand_dims(np.expand_dims(target,0),0)
    target = torch.from_numpy(target)
    pred = model_global(target)[0]
    loss = criterion(pred[0], target)
    valid_loss += loss.item()*data.size(0)
    
    #TO BE CONTINUED
    
    
    

    
