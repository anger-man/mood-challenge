#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:26:02 2022

@author: c
"""

#%%


import argparse
import torch
import torchvision
import torch.nn as nn
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

#torch.cuda.empty_cache()
#gc.collect()
#train_on_gpu = torch.cuda.is_available()
train_on_gpu = False

#%%

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, type=str)
parser.add_argument("-o", "--output", required=True, type=str)
parser.add_argument('-t', '--task', required=True, type=str)
parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)

args = parser.parse_args()

input_dir = args.input
output_dir = args.output
task = args.task

test_ids = os.listdir(input_dir)
test_dataset = MedicalDataset(
    datatype = 'test',
    path = input_dir,
    task = task,
    img_ids = test_ids,
    disturb_input = False,
    affine_matrix = True)

#%%

criterion = DiceLoss(evaluation_mode = True)
model = unet(n_channels = 1, f_size=32)
if train_on_gpu:
    model.cuda()
model.load_state_dict(torch.load('/weights/weights_%s_global.pt'%task, map_location='cpu'))

#define a index for the plots and saved model parameters
dt = datetime.now()
index = '%d%d%d_%d%d'%(dt.year,dt.month,dt.day,dt.hour,dt.minute)


test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    num_workers = 6)

#%%

model.eval(); count=0
with torch.no_grad():
    bar = tq(test_loader)
    for data, target, aff_mat in bar:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)

        inp   = data.cpu().detach().numpy()[0,0]
        preds = output[0].cpu().detach().numpy()[0,0]
        afm   = aff_mat.cpu().detach().numpy()[0]
            
        nif_pred = nib.Nifti1Image(np.round(preds),affine=afm)
        nib.save(nif_pred,  os.path.join(output_dir,test_ids[count]))
        count += 1

    
    
    

    
