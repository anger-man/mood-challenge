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

#only use samples that have been unseen during training
vali_ids = np.sort(os.listdir(os.path.join(data_path,'%s_train'%task)))[::6]
vali_dataset = MedicalDataset(
    datatype = 'train',
    path = data_path,
    task = task,
    img_ids = vali_ids,
    affine_matrix = True)

#also evaluate on the provided toy data
test_ids = os.listdir(os.path.join(data_path,'toy'))
test_dataset = MedicalDataset(
    datatype = 'test',
    path = data_path,
    task = task,
    img_ids = test_ids,
    disturb_input = False,
    affine_matrix = True)

#%%

#evaluate the global model for downsampled version of spatial size 64x64x64 

criterion = DiceLoss(evaluation_mode = True)
model = unet(n_channels = 1, f_size=32)
if train_on_gpu:
    model.cuda()
model.load_state_dict(torch.load('weights/weights_%s_global.pt'%task))

#define a index for the plots and saved model parameters
dt = datetime.now()
index = '%d%d%d_%d%d'%(dt.year,dt.month,dt.day,dt.hour,dt.minute)

valid_loader = DataLoader(
    vali_dataset, batch_size=4, shuffle=True,
    num_workers = 6)
test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    num_workers = 6)


        
################################################
# validate the model on data with random masks #
################################################
model.eval() #to tell layers you are in test mode (batchnorm, dropout,....)
valid_loss = 0.0; save_data = 0; count=0; ll=[]
os.mkdir(os.path.join('testing',index))
os.mkdir(os.path.join('testing',index,'final'))
os.mkdir(os.path.join('testing',index,'final_label'))
with torch.no_grad(): #deactivates the autograd engine
    bar = tq(valid_loader, postfix={"valid_loss":0.0})
    for data, target,aff_mat in bar:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output[0], target)
        # update average validation loss 
        valid_loss += loss.item()*target.size(0)
        # dice_cof = dice_no_threshold(output.cpu(), target.cpu()).item()
        # dice_score +=  dice_cof * data.size(0)
        bar.set_postfix(ordered_dict={"valid_loss":loss.item()})
        
        if save_data>-10:
            inp   = data.cpu().detach().numpy()[:,0]
            preds = output[0].cpu().detach().numpy()[:,0]
            uncer = output[1].cpu().detach().numpy()[:,0]
            tar   = target.cpu().detach().numpy()[:,0]
            afm   = aff_mat.cpu().detach().numpy()
            save_data=-1
            
            # save predictions as nifti
            for ns in range(preds.shape[0]):
                nif_pred = nib.Nifti1Image(np.round(preds[ns]),affine=afm[ns])
                nib.save(nif_pred, os.path.join('testing',index,'%d_pred.nii.gz'%count))
                nif_inp = nib.Nifti1Image(inp[ns],affine=afm[ns])
                nib.save(nif_inp, os.path.join('testing',index,'final/%d.nii.gz'%count))
                nif_tar = nib.Nifti1Image(tar[ns],affine=afm[ns])
                ll.append(np.max(tar[ns]))
                nib.save(nif_tar, os.path.join('testing',index,'final_label/%d.nii.gz'%count))
                count+=1
            
print('Validation Dice: %.3f'%(valid_loss/len(valid_loader.dataset)))
            


    
#%%

################################################
# validate the model on the challenge toy data
################################################
save_data = 0
with torch.no_grad(): #deactivates the autograd engine
    bar = tq(test_loader)
    for data, target,aff_mat in bar:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        if save_data==0:
            inp   = data.cpu().detach().numpy()[:,0]
            preds = output[0].cpu().detach().numpy()[:,0]
            uncer = output[1].cpu().detach().numpy()[:,0]
            tar   = target.cpu().detach().numpy()[:,0]
            afm   = aff_mat.cpu().detach().numpy()
            save_data=-1
            
# save predictions as nifti
os.mkdir(os.path.join('testing',index,'toy'))
for ns in range(preds.shape[0]):
    nif_pred = nib.Nifti1Image(np.round(preds[ns]),affine=afm[ns])
    nib.save(nif_pred, os.path.join('testing',index,'toy','%d_pred.nii.gz'%ns))
    nif_inp = nib.Nifti1Image(inp[ns],affine=afm[ns])
    nib.save(nif_inp, os.path.join('testing',index,'toy','%d_inp.nii.gz'%ns))

    
    
    

    
