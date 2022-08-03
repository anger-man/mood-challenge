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
from skimage.filters import sobel


    
#%%

# define the DiceLoss Function

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, eps=1e-6):
                
        inputs = inputs[0].view(-1) #equal to np.reshape(-1)
        targets = targets.view(-1)  #equal to np.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + eps)/(inputs.sum() + targets.sum() + eps)  
        
        return 1 - dice



#%%

#types of perturbations

def random_intensity(data, rand_mask):
    """
    Set voxel values of data to the same random intensity wrt to 
    the random mask
    """
    rand_int = np.random.choice(np.linspace(data.min(),data.max(),1000))
    per_data = np.where(rand_mask!=1,data,rand_int)
    return per_data

###############################################################################

def add_gaussian_noise(data, rand_mask):
    """
    Disturb the voxel values of data with Gaussian noise wrt to 
    the random mask
    """
    scale=np.random.uniform(.07,.14)
    noisy_sample = data + np.random.normal(loc=0, scale=scale,
                                           size=data.shape)
    per_data = np.where(rand_mask!=1,data,noisy_sample)
    return per_data

###############################################################################

def shift_intensity(data, rand_mask):
    """
    Shift the intensity of voxel values up to +-0.5 wrt to 
    the random mask
    """
    shift = np.random.choice([-.5,-.4,-.3,.3,.4,.5])
    shifted = shift+data
    per_data = np.where(rand_mask!=1,data,shifted)
    return per_data

###############################################################################

def apply_sobel_filter(data, rand_mask):
    """
    Apply the sobel filter to the regions indicated by  
    the random mask
    """
    sobel_data = sobel(data)
    per_data = np.where(rand_mask!=1,data,sobel_data)
    return per_data

#%%

def generate_random_mask(
    dim: int=64, 
    propval: float=.1,
    shapes: list=['elliptical','cuboid'],
    shape_prop: list=[.5,.5]):

    """
    dim: defines the spatial dimension of the mask
    propval: specifies the maximum relative amount of voxels occupied by the mask objects
    diameter: defines the maximal diameter of a connected component relative
                to the mask dimension
    """

    prop=np.random.uniform(0,propval),
    f = np.zeros([dim, dim, dim])
    diameter = 48/dim
    a = dim*diameter
    x = np.linspace(-dim/2, dim/2-1, dim)
    X,Y,Z = np.meshgrid(x,x,x)
    
    generate_components = np.random.rand()
    if generate_components>.2:
        
        for dummy in range(4):
            
            s_prop = np.random.uniform(0,np.sum(shape_prop))
            if s_prop < shape_prop[0]:
                shape = shapes[0]
            else:
                shape = shapes[1]
                
            if shape=='elliptical':
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
                
            elif shape=='cuboid':
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
                
            else:
                print('No valid shape'); pass;
                
    return f.astype(np.uint8)
        
#%%

#data loader

class MedicalDataset(Dataset):
    
    """
    define a dataset class 
    -this class reads a clean sample from the provided data,
    -uses average pooling to resize the scan to [64,64,64]
    -generates a random volumetric mask, where the connected components of the mask
     consist of either elliptical or cuboid objects
    -considering the random masks, different perturbations are added to the medical
     input data:
         -setting area to random intensity
         -adding Gaussian noise to the area
         -tbc
     the type of perturbation is chosen uniformly at random
    """
    
    def __init__(
            self,
            datatype: str = 'train',
            task: str = 'brain',
            path: str = None,
            img_ids: np.array = None,
            reduce_dim: bool = True,
            disturb_input = True,
            affine_matrix = False
        ):
        #######################################################################
            if datatype == 'train':
                self.img_folder  = f"{path}/%s_train"%task
            else:
                self.img_folder  = f"{path}/toy"
            self.img_ids = img_ids
            self.reduce_dim = reduce_dim
            self.disturb_input = disturb_input
            self.affine_matrix = affine_matrix
            
         ######################################################################   
            
            
            
    def __getitem__(self,idx):
        
        #generate the image path
        image_name = self.img_ids[idx]
        image_path = os.path.join(self.img_folder , image_name)
        
        #load the nifti file (affine matrix is omitted here)
        nifti = nib.load(image_path)
        data = nifti.get_fdata()#[::2,::2,::2]
        aff_mat = nifti.affine
        
        #define the random mask
        rand_mask = generate_random_mask(dim=data.shape[0],
                                    shape_prop=[.5,.5])
        # mask[data==0]=0 
        #according to challenge page no guarantee that components lie within the
        # investigated object
        
        # devide the entire scan by its 98% quantile
        data /= np.quantile(data,.98)
        
        type_of_per = np.random.choice(['rand_int','gauss_noise','shift_int','sobel'])
        if type_of_per == 'rand_int':
            per_data = random_intensity(data,rand_mask)
        elif type_of_per == 'gauss_noise':
            per_data = add_gaussian_noise(data,rand_mask)
        elif type_of_per == 'sobel':
            rand_mask[data==0] = 0
            per_data = apply_sobel_filter(data,rand_mask)
        elif type_of_per == 'shift_int':
            per_data = shift_intensity(data,rand_mask)
        else:
            print('Problem with data perturbations'); pass;
        
        #when evaluating on provided toy data, no perturbations are needed for 
        #the input instances, i.e. disturb_input is set to FALSE during testing
        if self.disturb_input:
            
            #scale the data to [64,64,64]
            if self.reduce_dim:
                fac = int(per_data.shape[0]/64)
                per_data = block_reduce(per_data, (fac,fac,fac), np.mean)
                rand_mask = block_reduce(rand_mask, (fac,fac,fac), np.median).round()
                
            if self.affine_matrix:
                return np.expand_dims(per_data,0), np.expand_dims(rand_mask,0),aff_mat
            else:
                return np.expand_dims(per_data,0), np.expand_dims(rand_mask,0)
                
        else:
            if self.reduce_dim:
                fac = int(data.shape[0]/64)
                data = block_reduce(data, (fac,fac,fac), np.mean)    
                
            if self.affine_matrix:
                data = np.expand_dims(data,0)
                return data, data, aff_mat
            else:
                data = np.expand_dims(data,0)
                return data, data

    def __len__(self):
        return(len(self.img_ids))
    

    
