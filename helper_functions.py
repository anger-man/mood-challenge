#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:30:26 2022

@author: c
"""
#%%

#load packages

import torch
torch.cuda.get_device_name(0)
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

def generate_random_mask(
    dim: int=64, 
    prop: float=np.random.uniform(0,.05),
    diameter: float=.25,
    shape: str='elliptical'):

    """
    dim: defines the spatial dimension of the mask
    prop: specifies the relative amount of voxels occupied by the mask objects
    diameter: defines the maximal diameter of a connected component relative
                to the mask dimension
    """

    f = np.zeros([dim, dim, dim])
    a = dim*diameter
    x = np.linspace(-dim/2, dim/2-1, dim)
    X,Y,Z = np.meshgrid(x,x,x)
    
    if shape=='elliptical':
        while True:
            aa = a*np.random.rand(1)
            bb = a*np.random.rand(1)
            cc = a*np.random.rand(1)
        
            X_shift = X - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1)
            Y_shift = Y - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1)
            Z_shift = Z - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1)
        
            idx = (X_shift**2/aa**2 + Y_shift**2/bb**2 + Z_shift**2/cc**2) <= 1
            # angle = np.random.randint(1, 90+1)
            # idx = ndimage.rotate(idx, angle, reshape = False)
            idx = np.rot90(idx,k=np.random.choice([1,2,3]))
            f[idx] = 1
            fflat = f.flatten()
            
            if np.count_nonzero(fflat)/np.prod(f.shape) >= prop:
                break
            
        return f.astype(np.uint8)
    
    elif shape=='cuboid':
        while True:
            aa = int(.5*a*np.random.rand(1))
            bb = int(.5*a*np.random.rand(1))
            cc = int(.5*a*np.random.rand(1))
        
            x_mid = int(dim/2 - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1))
            y_mid = int(dim/2 - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1))
            z_mid = int(dim/2 - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1))
        
            f[x_mid-aa:x_mid+aa,y_mid-bb:y_mid+bb,z_mid-cc:z_mid+cc] = 1
            fflat = f.flatten()
            
            if np.count_nonzero(fflat)/np.prod(f.shape) >= prop:
                break
            
        return f.astype(np.uint8)
        
    else:
        print('No valid shape'); pass;
        
    
#%%

def random_intensity(data, rand_mask):
    """
    Set voxel values of data to the same random intensity wrt to 
    the random mask
    """
    rand_int = np.random.choice(np.linspace(data.min(),data.max(),1000))
    per_data = np.where(rand_mask!=1,data,rand_int)
    return per_data

###############################################################################

def add_gaussian_noise(data, rand_mask, scale=.05):
    """
    Disturb the voxel values of data with Gaussian noise wrt to 
    the random mask
    """
    noisy_sample = data + np.random.normal(loc=0, scale=scale,
                                           size=data.shape)
    per_data = np.where(rand_mask!=1,data,noisy_sample)
    return per_data
