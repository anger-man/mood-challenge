#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 16:39:46 2022

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
from tqdm import tqdm as tq
from scipy.ndimage import rotate
import time
from skimage.measure import block_reduce
from datetime import datetime
from models import unet


#%%

def evaluate(model_global, model_local,source_file):
    model_global.eval() 
    model_local.eval()
   
    for itera in range(1):
        #load input
        nifti = nib.load(source_file)
        data = nifti.get_fdata()
        dim = data.shape[0]
        if dim>256:
            data = data[::2,::2,::2]
        aff_mat = nifti.affine
        fac = int(dim/64)
        img_down = torch.from_numpy(data[::fac,::fac,::fac])
        img_down = img_down.unsqueeze(0).unsqueeze(0)
        data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        
        #switch to gpu if cuda available
        if train_on_gpu:
            img_down = img_down.cuda()
            data = data.cuda()
        
        #evaluate global model
        with torch.no_grad():
            pred_global = model_global(img_down)[0]
        up = nn.Upsample(scale_factor = fac)
        pred_global = up(pred_global)
        
        #evaluate local model
        ii = np.arange(0,240,48); count = 0
        final_pred = torch.zeros(pred_global.shape).cuda()
        hidden_mask = torch.zeros(pred_global.shape).cuda()
        if torch.max(pred_global).item()<.5:
            final_pred = torch.zeros(pred_global.shape).cuda()
        else:
            for i in ii:
                for j in ii:
                    for k in ii:
                        patch = pred_global[:,:,i:i+64,j:j+64,k:k+64]
                        if torch.sum(patch).item()<.5*2**3:
                            continue;
                        else:
                            inp_data = torch.cat((data[:,:,i:i+64,j:j+64,k:k+64],patch),1)
                            with torch.no_grad():
                                tmp = model_local(inp_data)[0]
                            final_pred[:,:,i:i+64,j:j+64,k:k+64] += tmp
                            hidden_mask[:,:,i:i+64,j:j+64,k:k+64] += 1
                            count += 1
            final_pred = final_pred / torch.clip(hidden_mask,1,1e7)
                        
        result = final_pred[0,0].cpu().detach().numpy()
        return(result.astype(np.float32),aff_mat)
                        
                        


def predict_folder_pixel_abs(input_folder, target_folder, task):
    for f in os.listdir(input_folder):

        source_file = os.path.join(input_folder, f)
        target_file = os.path.join(target_folder, f)

        result, aff_mat = evaluate(model_global,model_local,source_file)

        if task == 'abdom':
            result = result.repeat(2,axis=0).repeat(2,axis=1).repeat(2,axis=2)
        final_nimg = nib.Nifti1Image(result, affine=aff_mat)
        nib.save(final_nimg, target_file)
        print(target_file)


def predict_folder_sample_abs(input_folder, target_folder):
    for f in os.listdir(input_folder):
        source_file = os.path.join(input_folder, f)

        result, aff_mat = evaluate(model_global,model_local,source_file)
        if -np.sort(-result.reshape(-1))[9**3]>.5: #higher dice than 8**3
            score = 1
        else:
            score = 0
            
        with open(os.path.join(target_folder, f + ".txt"), "w") as write_file:
            write_file.write(str(score))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-t", "--task", required=True, type=str)
    parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    task = args.task
    mode = args.mode
    
    train_on_gpu = torch.cuda.is_available()
    print('train on gpu: ',train_on_gpu)
    
    model_global = unet(n_channels = 1, f_size=32)
    if train_on_gpu:
        model_global.cuda()
        model_global.load_state_dict(torch.load('/weights/weights_%s_global.pt'%task))
    else:
        model_global.load_state_dict(torch.load('/weights/weights_%s_global.pt'%task, map_location='cpu'))
    

    model_local = unet(n_channels = 2, f_size=32)
    if train_on_gpu:
        model_local.cuda()
        model_local.load_state_dict(torch.load('/weights/weights_%s_local.pt'%task))
    else:
        model_local.load_state_dict(torch.load('/weights/weights_%s_local.pt'%task, map_location='cpu'))

    if mode == "pixel":
        predict_folder_pixel_abs(input_dir, output_dir, task)
    elif mode == "sample":
        predict_folder_sample_abs(input_dir, output_dir)
    else:
        print("Mode not correctly defined. Either choose 'pixel' oder 'sample'")

    # predict_folder_sample_abs("/home/david/data/datasets_slow/mood_brain/toy", "/home/david/data/datasets_slow/mood_brain/target_sample")