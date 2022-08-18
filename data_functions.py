#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:30:26 2022

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
from skimage.filters import sobel
from models import unet

    
#%%

# define the DiceLoss Function

class DiceLoss(nn.Module):
    def __init__(self, smooth: float=1e-6, evaluation_mode: bool=False):
        self.smooth = smooth
        self.evaluation_mode = evaluation_mode
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        preds = preds.view(-1) #equal to np.reshape(-1)
        targets = targets.view(-1)  #equal to np.reshape(-1)
        if self.evaluation_mode:
            preds = torch.round(preds)
            intersection = (preds * targets).sum()                            
            dice = (2.*intersection + self.smooth)/(preds.sum() + targets.sum() + self.smooth)  
            return 1 - dice
        else:
            intersection = (preds * targets).sum()                            
            dice = (2.*intersection + self.smooth)/(preds.sum() + targets.sum() + self.smooth)  
            check_tar = torch.clip(targets.sum(),0,1) 
            return( check_tar*(1-dice) + (1-check_tar)*torch.mean(torch.abs(targets-preds)) )
        
    
class BCE(nn.Module):
    def __init__(self, eps = 1e-6):
        self.eps = eps
        super(BCE, self).__init__()

    def forward(self, preds, targets):
        preds = preds.view(-1) #equal to np.reshape(-1)
        targets = targets.view(-1)  #equal to np.reshape(-1)
        
        tmp = targets*torch.log(preds+self.eps)+(1.-targets)*torch.log(1.-preds+self.eps)
        return -torch.mean(tmp)




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

def scale_intensity(data, rand_mask):
    """
    double or half the intensity of voxel values wrt to the random mask
    """
    scale = np.random.choice([.5,2.])
    scaled = scale*data
    per_data = np.where(rand_mask!=1,data,scaled)
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

###############################################################################

per_model = unet(n_channels = 1, f_size=2)
per_model.eval()


def weight_reset(m):
    if isinstance(m, nn.Conv3d):
        m.reset_parameters()

def random_convolution(data, rand_mask, per_model = per_model):
    """
    Select a region wrt to the random mask and propagate it through a vanilla
    Unet with random initialization
    """
    per_model.apply(weight_reset)
    # m = np.mean(data[data!=0]); sd = np.std(data[data!=0])
    m = np.mean(data); sd = np.std(data)
    foo = np.expand_dims(np.expand_dims(data,0),0)
    foo = per_model(torch.from_numpy(foo))
    foo = foo[0].cpu().detach().numpy()[0,0]
    # foo = np.clip((foo-np.mean(foo[data!=0]))/np.std(foo[data!=0]) * sd + m,0,1e7)
    foo = np.clip((foo-np.mean(foo))/np.std(foo) * sd + m,0,1e7)
    per_data = np.where(rand_mask!=1,data,foo)
    return per_data


#%%

def generate_random_mask(
    dim: int=64, 
    propval: float=.1,
    shapes: list=['elliptical','cuboid','non-convex'],
    shape_prop: list=[.5,.5]):

    """
    dim: defines the spatial dimension of the mask
    propval: specifies the maximum relative amount of voxels occupied by the mask objects
    shapes: list of available shapes; default: ['elliptical','cuboid','non-convex']
    shape_prop: list defining the likelihood of each shape, sum should equal 1
    concave describes the non-convex objects by simon
    """
    
    #define the maximal diameter of a connected component relative to the mask dimension
    diameter = 48/dim
    a = dim*diameter

    prop=np.random.uniform(0,propval),
    f = np.zeros([dim, dim, dim])
    x = np.linspace(-dim/2, dim/2-1, dim)
    X,Y,Z = np.meshgrid(x,x,x)
    
    generate_components = np.random.rand()
    if generate_components>.2:
        
        for dummy in range(4):
            
            shape = np.random.choice(['elliptical','cuboid','non-convex'])
                
            if shape=='elliptical':
                aa = a*np.random.uniform(.1,1)
                bb = a*np.random.uniform(.1,1)
                cc = a*np.random.uniform(.1,1)
            
                X_shift = X - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1)
                Y_shift = Y - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1)
                Z_shift = Z - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1)
            
                idx = (X_shift**2/aa**2 + Y_shift**2/bb**2 + Z_shift**2/cc**2) <= 1
                idx = np.rot90(idx,k=np.random.choice([1,2,3]))
                f[idx] = 1
                fflat = f.flatten()
                
                if np.count_nonzero(fflat)/np.prod(f.shape) >= prop:
                    break
                
            elif shape=='cuboid':
                aa = int(.5*a*np.random.uniform(.1,1))
                bb = int(.5*a*np.random.uniform(.1,1))
                cc = int(.5*a*np.random.uniform(.1,1))
            
                x_mid = int(dim/2 - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1))
                y_mid = int(dim/2 - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1))
                z_mid = int(dim/2 - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1))
            
                f[x_mid-aa:x_mid+aa,y_mid-bb:y_mid+bb,z_mid-cc:z_mid+cc] = 1
                fflat = f.flatten()
                
                if np.count_nonzero(fflat)/np.prod(f.shape) >= prop:
                    break
                
            elif shape=='non-convex':
                files = os.listdir('arrays')
                mask = np.load(os.path.join('arrays',np.random.choice(files)))
            
                f[mask==1] = 1
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
            disturb_input: bool = True,
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
        
        type_of_per = np.random.choice(['rand_int','gauss_noise',
                                        'scale_int','sobel','rand_conv'])
        
        if type_of_per == 'rand_int':
            per_data = random_intensity(data,rand_mask)
        elif type_of_per == 'gauss_noise':
            per_data = add_gaussian_noise(data,rand_mask)
        elif type_of_per == 'rand_conv':
            rand_mask[data==0] = 0
            per_data = random_convolution(data,rand_mask)
        elif type_of_per == 'scale_int':
            rand_mask[data==0] = 0
            per_data = scale_intensity(data,rand_mask)
        elif type_of_per == 'sobel':
            rand_mask[data==0] = 0
            per_data = apply_sobel_filter(data,rand_mask)
        else:
            print('Problem with data perturbations'); pass;
        
        #when evaluating on provided toy data, no perturbations are needed for 
        #the input instances, i.e. disturb_input is set to FALSE during testing
        if self.disturb_input:
            per_data /= np.quantile(per_data,.98)
            #scale the data to [64,64,64]
            if self.reduce_dim:
                fac = int(per_data.shape[0]/64)
                per_data = block_reduce(per_data, (fac,fac,fac), np.mean)
                rand_mask = block_reduce(rand_mask, (fac,fac,fac), np.max)
                
            if self.affine_matrix:
                return np.expand_dims(per_data,0), np.expand_dims(rand_mask,0),aff_mat
            else:
                return np.expand_dims(per_data,0), np.expand_dims(rand_mask,0)
                
        else:
            data /= np.quantile(data,.98)
            if self.reduce_dim:
                fac = int(data.shape[0]/64)
                data = block_reduce(data, (fac,fac,fac), np.mean)   
                
            data = np.expand_dims(data,0)
            if self.affine_matrix:
                return data, data, aff_mat
            else:
                return data, data

    def __len__(self):
        return(len(self.img_ids))
    

    
