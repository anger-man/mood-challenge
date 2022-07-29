#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:30:11 2022

@author: c
"""

#%%

# define the task [brain,abdom] and the corresponding data directory

task = 'brain'
data_path = 'data/%s'%task

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
from datetime import datetime
from helper_functions import generate_random_mask, add_gaussian_noise, random_intensity

#%%

#empty cuda cache and check, if GPU is available

torch.cuda.empty_cache()
gc.collect()
train_on_gpu = torch.cuda.is_available()
# from numba import cuda
# cuda.select_device(0)
# cuda.close()

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
        
        self.layer  = nn.Sequential(
            nn.Conv3d(in_f,  in_f, kernel_size=3,stride=stride,padding=padding,groups=in_f),
            nn.Conv3d(in_f,  out_f,kernel_size=1,stride=1))
        # self.layer = nn.Conv3d(in_f,out_f,kernel_size=3, stride=stride, padding=padding)
    
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
    def __init__(self,in_f,out_f,normalization='instance',nearest=False):
        super(upsampling, self).__init__()
        
        if nearest:
            self.up = nn.Upsample(scale_factor=2, mode = 'nearest')
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
      
#%%

# define the DiceLoss Function

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, eps=1e-7):
                
        #flatten label and prediction tensors
        inputs = inputs[0].view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + eps)/(inputs.sum() + targets.sum() + eps)  
        
        return 1 - dice


        
#%%




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
         -adding impulsive perturbations to the area
         -tbc
     the type of perturbation is chosen uniformly at random
    """
    
    def __init__(
            self,
            datatype: str = 'train',
            path: str = data_path,
            img_ids: np.array = None,
            disturb_input = True
        ):
        #######################################################################
            if datatype == 'train':
                self.img_folder  = f"{path}/%s_train"%task
            else:
                self.img_folder  = f"{path}/toy"
            self.img_ids = img_ids
            self.disturb_input = disturb_input
            
         ######################################################################   
            
            
            
    def __getitem__(self,idx):
        
        #generate the image path
        image_name = self.img_ids[idx]
        image_path = os.path.join(self.img_folder , image_name)
        
        #load the nifti file (affine matrix is omitted here)
        nifti = nib.load(image_path)
        data = nifti.get_fdata().copy()
        
        #scale the data to [64,64,64]
        fac = int(data.shape[0]/64)
        data = block_reduce(data, (fac,fac,fac), np.mean)
        
        #define the random mask
        shape = np.random.choice(['cuboid','elliptical'])
        rand_mask = generate_random_mask(dim=data.shape[0],
                                    shape=shape)
        # mask[data==0]=0 
        #according to challenge page no guarantee that components lie within the
        # investigated object
        
        # devide the entire scan by its 98% quantile
        data /= np.quantile(data,.98)
        
        type_of_per = np.random.choice(['rand_int','gauss_noise'])
        if type_of_per == 'rand_int':
            per_data = random_intensity(data,rand_mask)
        elif type_of_per == 'gauss_noise':
            per_data = add_gaussian_noise(data,rand_mask)
        else:
            print('Problem with data perturbations'); pass;
        
        #when evaluating on provided toy data, no perturbations are needed for 
        #the input instances, i.e. disturb_input equals FALSE during test
        if self.disturb_input:
            return np.expand_dims(per_data,0), np.expand_dims(rand_mask,0)
        else:
            data = np.expand_dims(data,0)
            return data, data

    def __len__(self):
        return(len(self.img_ids))
    
    
vali_ids = os.listdir(os.path.join(data_path,'%s_train'%task))[::6]
train_ids = [f for f in os.listdir(os.path.join(data_path,'%s_train'%task)) if f not in vali_ids] 
test_ids = os.listdir(os.path.join(data_path,'toy'))

train_dataset = MedicalDataset(
    datatype = 'train',
    img_ids = train_ids)
vali_dataset = MedicalDataset(
    datatype = 'train',
    img_ids = vali_ids)
test_dataset = MedicalDataset(
    datatype = 'test',
    img_ids = test_ids,
    disturb_input = False)

#%%

criterion = DiceLoss()
model = unet(n_channels =1, f_size=32)
if train_on_gpu:
    model.cuda()
summary(model, (1,64,64,64))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, cooldown=2)
dt = datetime.now()
index = '%d%d%d_%d%d'%(dt.year,dt.month,dt.day,dt.hour,dt.minute)

#%%

torch.cuda.empty_cache()
gc.collect()
batch_size = 4

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers = 6)
valid_loader = DataLoader(
    vali_dataset, batch_size=4, shuffle=True,
    num_workers = 6)
test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=True,
    num_workers = 6)


n_epochs = 50
train_loss_list = []
valid_loss_list = []
dice_score_list =  []
lr_rate_list =  []
valid_loss_min = 1e7

for epoch in range(1, n_epochs+1):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    dice_score = 0.0
    ###################
    # train the model #
    ###################
    model.train()
    bar = tq(train_loader, postfix={"train_loss":0.0})
    for data, target in bar:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.mean().backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        bar.set_postfix(ordered_dict={"train_loss":loss.item()})
        bar.update(n=1)
        
    ######################    
    # validate the model #
    ######################
    model.eval() #to tell layers you are in test mode (batchnorm, dropout,....)
    del data, target
    save_data = 1
    with torch.no_grad(): #deactivates the autograd engine
        bar = tq(valid_loader, postfix={"valid_loss":0.0})
        for data, target in bar:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
            # dice_cof = dice_no_threshold(output.cpu(), target.cpu()).item()
            # dice_score +=  dice_cof * data.size(0)
            bar.set_postfix(ordered_dict={"valid_loss":loss.item()})
            if save_data:
                preds = output[0].cpu().detach().numpy()[:,0]
                uncer = output[1].cpu().detach().numpy()[:,0]
                gts   = target.cpu().detach().numpy()[:,0]
                save_data = 0
                
    
    fig, ax = plt.subplots(4,4,figsize=(12,10)); 
    for j in range(preds.shape[0]):
        im = ax[j,0].imshow(np.sum(gts[j],0), cmap='Greys_r')
        ax[j,0].axis('off')
        plt.colorbar(im,ax=ax[j,0])
        im = ax[j,1].imshow(np.sum(preds[j],0), cmap='Greys_r')
        ax[j,1].axis('off')
        plt.colorbar(im,ax=ax[j,1])
        im = ax[j,2].imshow(np.abs(np.sum(gts[j]-preds[j],0)), cmap='gist_rainbow')
        ax[j,2].axis('off')
        plt.colorbar(im,ax=ax[j,2])
        im = ax[j,3].imshow(np.sum(uncer[j],0), cmap='gist_rainbow')
        ax[j,3].axis('off')
        plt.colorbar(im,ax=ax[j,3])
        
    fig.tight_layout(pad=.1)
    plt.savefig('plots/val/%s_ep%d.pdf'%(index,epoch))
        
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    dice_score = dice_score/len(valid_loader.dataset)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups])
    
    # print training/validation statistics 
    print('Epoch: {}  Training Loss: {:.6f}  Validation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_%s.pt'%index)
        valid_loss_min = valid_loss
    
    scheduler.step(valid_loss)
    
    ######################    
    # test the model #
    ######################
    del data, target
    preds = []; gts = []; uncers=[]
    count = 0
    for data, target in tq(test_loader):
        if train_on_gpu:
            data = data.cuda()
        target = target.cpu().detach().numpy()
        pred,uncer = model(data)
        pred = pred.cpu().detach().numpy()
        uncer = uncer.cpu().detach().numpy()
        preds.append(pred[0]); gts.append(target[0]); uncers.append(uncer[0])
    preds = np.concatenate(preds,0)
    gts = np.concatenate(gts,0)
    uncers = np.concatenate(uncers,0)

    fig, ax = plt.subplots(4,4,figsize=(12,10)); 
    for j in [0,1,2,3]:
        im = ax[j,0].imshow(np.sum(gts[j],0), cmap='Greys_r')
        ax[j,0].axis('off')
        plt.colorbar(im,ax=ax[j,0])
        im = ax[j,1].imshow(np.sum(preds[j],0), cmap='Greys_r')
        ax[j,1].axis('off')
        plt.colorbar(im,ax=ax[j,1])
        im = ax[j,2].imshow(np.abs(np.sum(gts[j]-preds[j],0)), cmap='gist_rainbow')
        ax[j,2].axis('off')
        plt.colorbar(im,ax=ax[j,2])
        im = ax[j,3].imshow(np.sum(uncers[j],0), cmap='gist_rainbow')
        ax[j,3].axis('off')
        plt.colorbar(im,ax=ax[j,3])
        
    fig.tight_layout(pad=.1)
    plt.savefig('plots/test/%s_ep%d.pdf'%(index,epoch))

#%%

#analyze weights

w = []
for param in model.parameters():
    w.append(param.cpu().detach().numpy())

#%%

#Plotting Metrics

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(121)
ax.plot([i[0] for i in lr_rate_list])
plt.ylabel('learing rate during training', fontsize=22)

ax = fig.add_subplot(122)
ax.plot(train_loss_list,  marker='x', label="Training Loss")
ax.plot(valid_loss_list,  marker='x', label="Validation Loss")
plt.ylabel('loss', fontsize=22)
plt.legend()
plt.savefig('loss/%s.pdf'%index)


# preds = []; gts = []; uncers=[]
# count = 0
# for data, target in tq(test_loader):
#     if train_on_gpu:
#         data = data.cuda()
#     target = target.cpu().detach().numpy()
#     pred,uncer = model(data)
#     pred = pred.cpu().detach().numpy()
#     uncer = uncer.cpu().detach().numpy()
#     preds.append(pred[0]); gts.append(target[0]); uncers.append(uncer[0])
# preds = np.concatenate(preds,0)
# gts = np.concatenate(gts,0)
# uncers = np.concatenate(uncers,0)

# fig, ax = plt.subplots(4,4,figsize=(12,10)); 
# for j in [0,1,2,3]:
    
#     im = ax[j,0].imshow(gts[j,0], cmap='Greys_r')
#     ax[j,0].axis('off')
#     plt.colorbar(im,ax=ax[j,0])
    
#     im = ax[j,1].imshow(preds[j,0], cmap='Greys_r')
#     ax[j,1].axis('off')
#     plt.colorbar(im,ax=ax[j,1])
    
#     im = ax[j,2].imshow(np.abs(gts[j,0]-preds[j,0]), cmap='gist_rainbow')
#     ax[j,2].axis('off')
#     plt.colorbar(im,ax=ax[j,2])
    
#     im = ax[j,3].imshow(uncers[j,0], cmap='gist_rainbow')
#     ax[j,3].axis('off')
#     plt.colorbar(im,ax=ax[j,3])
    
# fig.tight_layout(pad=.1)
# plt.savefig('plots/%s.pdf'%index)




    

    
