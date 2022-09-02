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
from torchmetrics.functional import average_precision

#%%

train_on_gpu = torch.cuda.is_available()


#%%

#generate evaluation data

#only use samples that have been unseen during training
# vali_ids = np.sort(os.listdir(os.path.join(data_path,'%s_train'%task)))[::6]
# vali_dataset = MedicalDataset(
#     datatype = 'train',
#     path = data_path,
#     task = task,
#     img_ids = vali_ids,
#     affine_matrix = True,
#     reduce_dim = False)

# valid_loader = DataLoader(
#     vali_dataset, batch_size=4, shuffle=True,
#     num_workers = 6)

# #define a index for the plots and saved model parameters
# dt = datetime.now()
# index = '%d%d%d_%d%d'%(dt.year,dt.month,dt.day,dt.hour,dt.minute)


# os.mkdir(os.path.join('testing',index))
# os.mkdir(os.path.join('testing',index,'final'))
# os.mkdir(os.path.join('testing',index,'final_label'))
# count = 0; ll=[]
# with torch.no_grad(): #deactivates the autograd engine
#     bar = tq(valid_loader, postfix={"valid_loss":0.0})
#     for data, target,aff_mat in bar:
#         # move tensors to GPU if CUDA is available
    
#         inp   = data.cpu().detach().numpy()[:,0].astype(np.float32)
#         tar   = target.cpu().detach().numpy()[:,0]
#         afm   = aff_mat.cpu().detach().numpy()
        
#         # save predictions as nifti
#         for ns in range(inp.shape[0]):
#             nif_inp = nib.Nifti1Image(inp[ns],affine=afm[ns])
#             nib.save(nif_inp, os.path.join('testing',index,'final/%d.nii.gz'%count))
#             nif_tar = nib.Nifti1Image(tar[ns],affine=afm[ns])
#             ll.append(np.max(tar[ns]))
#             nib.save(nif_tar, os.path.join('testing',index,'final_label/%d.nii.gz'%count))
#             count+=1
#         print(ll)

#%%


#also evaluate on the provided toy data

input_path = os.path.join(data_path,'final')
input_ids = np.sort([os.path.join(input_path,f) for f in os.listdir(input_path)])

target_path = os.path.join(data_path,'final_label')
target_ids = np.sort([os.path.join(target_path,f) for f in os.listdir(target_path)])


#%%

#evaluate the global model for downsampled version of spatial size 64x64x64 

criterion = DiceLoss(evaluation_mode = True)

model_global = unet(n_channels = 1, f_size=32)
if train_on_gpu:
    model_global.cuda()
model_global.load_state_dict(torch.load('weights/weights_%s_global.pt'%task))


model_local_0 = unet(n_channels = 2, f_size=32)
model_local_1 = unet(n_channels = 2, f_size=32)

if train_on_gpu:
    model_local_0.cuda()
    model_local_1.cuda()
model_local_0.load_state_dict(torch.load('weights/weights_%s_local_0.pt'%task))
model_local_1.load_state_dict(torch.load('weights/weights_%s_local_1.pt'%task))


#define a index for the plots and saved model parameters
dt = datetime.now()
index = '%d%d%d_%d%d_final'%(dt.year,dt.month,dt.day,dt.hour,dt.minute)




################################################
# validate the model on data with random masks #
################################################

def evaluate(model_global, model_local_0,model_local_1):
    model_global.eval() 
    model_local_0.eval()
    model_local_1.eval()
    GLOBAL = 0.0; FINAL = 0.0; save_data = 0
    sg = []; sf = []
    
    if task == 'abdom':
        subs = 1
    else:
        subs = 1
       
    for itera in range(len(input_ids[:50])):
        #load target
        nifti = nib.load(target_ids[itera])
        target = nifti.get_fdata()[::subs,::subs,::subs].astype(np.uint8)
        ta = target.copy()
        sg.append(target.max())
        target = torch.from_numpy(target)
        target = target.unsqueeze(0).unsqueeze(0)
        
        #load input
        nifti = nib.load(input_ids[itera])
        data = nifti.get_fdata()[::subs,::subs,::subs]
        data /= np.quantile(data,.98)
        dim = data.shape[0]
        aff_mat = nifti.affine
        fac = int(dim/64)
        img_down = torch.from_numpy(data[::fac,::fac,::fac])
        img_down = img_down.unsqueeze(0).unsqueeze(0)
        data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        
        #switch to gpu if cuda available
        if train_on_gpu:
            img_down = img_down.cuda()
            target = target.cuda()
            data = data.cuda()
        
        #evaluate global model
        with torch.no_grad():
            pred_global = model_global(img_down)[0]
        up = nn.Upsample(scale_factor = fac)
        pred_global = up(pred_global)
        pg = pred_global.cpu().detach().numpy()[0,0]
        loss_global = criterion(pred_global, target).item()
        # ap_global = average_precision(pred_global.reshape(-1),target.reshape(-1)).item()
        
        #evaluate local model
        ii = np.arange(0,240,48); count = 0
        final_pred = torch.zeros(pred_global.shape).cuda()
        hidden_mask = torch.zeros(pred_global.shape).cuda()
        # if train_on_gpu:
        #     final_pred = final_pred.cuda()
        #     hidden_mask = hidden_mask.cuda()
        if torch.sum(pred_global).item()<.95*8**3 or torch.max(pred_global).item()<.95:
            final_pred = torch.zeros(pred_global.shape).cuda()
        else:
            for i in ii:
                for j in ii:
                    for k in ii:
                        patch = pred_global[:,:,i:i+64,j:j+64,k:k+64]
                        if torch.sum(patch).item()<.95*3**3:
                            continue;
                        else:
                            inp_data = torch.cat((data[:,:,i:i+64,j:j+64,k:k+64],patch),1)
                            with torch.no_grad():
                                tmp0 = model_local_0(inp_data)[0]
                                tmp1 = model_local_1(inp_data)[0]
                                tmp = .4*tmp0+.6*tmp1
                            final_pred[:,:,i:i+64,j:j+64,k:k+64] += tmp
                            hidden_mask[:,:,i:i+64,j:j+64,k:k+64] += 1
                            count += 1
            final_pred = final_pred / torch.clip(hidden_mask,1,1e7)
        
        loss_final = criterion(final_pred, target).item()
        # ap_final = average_precision(final_pred.reshape(-1),target.reshape(-1)).item()
        GLOBAL += loss_global; FINAL += loss_final
        print('%.3f  %.3f  %03d' %(loss_global,loss_final, count))
        
        result = final_pred[0,0].cpu().detach().numpy()
        sf.append(np.clip(.5*np.sum(result)/(12**3),0,1))

    return GLOBAL,FINAL,np.array(sg),np.array(sf)
                    
g,f,sg,sf = evaluate(model_global,model_local_0,model_local_1)
print(np.sum(2*sg*sf)/(np.sum(sg)+np.sum(sf)))

#0.13 (50 samples, brain, 48 overlapping), sample acc. 0.97
#using .95 threshold: 0.09, sample ac: 0.97
#final brain 0.07, sample ac: 0.97


#0.15 (50 samples, abdom, 48 overlapping), sample acc. .99
#0.16 (50 samples, abdom, 64 overlapping), sample acc. .99
#using .95 threshold: 0.13, sample ac: 0.98
#final abdom: 0.099, sample ac: 1.0


