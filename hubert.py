#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 08:47:16 2022

@author: c
"""


#%%

import torch
torch.cuda.get_device_name(0)
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import gc, os
import numpy as np
import nibabel as nib
from tqdm import tqdm as tq


torch.cuda.empty_cache()
gc.collect()
train_on_gpu = torch.cuda.is_available()


#%%

data_path = 'path_to_data'
from scipy.ndimage import rotate

def fancy_preprocess_function(data):
    rand_angle = np.random.choice(range(180),2, replace=False)
    res = []
    for ra in rand_angle:
        rot = rotate(data, angle=ra, axes=(0,2), reshape=False)
        rot = np.sum(rot,axis=0)
        res.append(rot)
    return np.stack(res,axis=0)

class MedDataset(Dataset):
    def __init__(
            self,
            datatype: str = 'train',
            path: str = data_path,
            img_ids: np.array = None
        ):
            # self.df = df
            if datatype == 'train':
                self.img_folder  = f"{path}/train"
            else:
                self.img_folder  = f"{path}/test"
            self.img_ids = img_ids
            
    def __getitem__(self,idx):
        image_name = self.img_ids[idx]
        image_path = os.path.join(self.img_folder , image_name)
        nifti = nib.load(image_path) #or np.load(...)
        data = nifti.get_fdata()
        data = fancy_preprocess_function(data)
        data /= np.quantile(data, .98) #normalize
        data = np.expand_dims(data,1)
        
        #generate perturbed input data
        per_data = data+np.random.normal(loc = 0, scale = 0.05*np.max(data),
                                         size = data.shape)
        # matrix = nifti.affine
        return per_data, data
    
    def __len__(self):
        return(len(self.img_ids))
    
vali_ids = os.listdir(os.path.join(data_path,'%train'))[::6]
train_ids = [f for f in os.listdir(os.path.join(data_path,'train')) if f not in vali_ids] 
test_ids = os.listdir(os.path.join(data_path,'test'))

train_dataset = MedDataset(
    datatype = 'train',
    img_ids = train_ids)
vali_dataset = MedDataset(
    datatype = 'train',
    img_ids = vali_ids)
test_dataset = MedDataset(
    datatype = 'test',
    img_ids = test_ids)


batch_size = 8
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers = 6)
valid_loader = DataLoader(
    vali_dataset, batch_size=batch_size, shuffle=False,
    num_workers = 6)
test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=True,
    num_workers = 6)


#%%

def NORM(out_f,normalization):
    normlayer = nn.ModuleDict([
        ['instance',nn.InstanceNorm2d(out_f)],
        ['batch', nn.BatchNorm2d(out_f)]
        ])
    return normlayer[normalization]


class double_conv(nn.Module):
    def __init__(self,in_f,out_f,normalization = 'instance'):
        super(double_conv, self).__init__()
        
        self.conv1  = nn.Sequential(
            nn.Conv2d(in_f,  out_f,kernel_size=3,stride=1,padding='same'),
            NORM(out_f, normalization),
            nn.ReLU())
        
        self.conv2  = nn.Sequential(
            nn.Conv2d(out_f,  out_f,kernel_size=3,stride=1,padding='same'),
            NORM(out_f, normalization),
            nn.ReLU())
                 
    def forward(self, x):
        return self.conv2(self.conv1(x))
    
    
class downsampling(nn.Module):
    def __init__(self,in_f,out_f,normalization = 'instance'):
        super(downsampling, self).__init__()
        
        self.down  = nn.Sequential(
            nn.Conv2d(in_f,  out_f,kernel_size=3,stride=2,padding=(1,1)),
            NORM(out_f,normalization),
            nn.ReLU())
                 
    def forward(self, x):
        return(self.down(x))
    
    
class upsampling(nn.Module):
    def __init__(self,in_f,out_f,normalization='instance',bilinear=False):
        super(upsampling, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode = 'bilinear',align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_f,out_f,kernel_size=4,stride=2,padding=(1,1))
            
        self.conv1  = nn.Sequential(
            nn.Conv2d(in_f, out_f,kernel_size=3,stride=1,padding='same'),
            NORM(out_f, normalization),
            nn.LeakyReLU(0.2))
        
        self.conv2  = nn.Sequential(
            nn.Conv2d(out_f,  out_f,kernel_size=3,stride=1,padding='same'),
            NORM(out_f, normalization),
            nn.LeakyReLU(0.2))
            
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x,skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class unet(nn.Module):
    def __init__(self, n_channels, f_size, normalization='instance',out_acti = 'relu'):
        super(unet, self).__init__()
        self.dc0 = double_conv(n_channels, f_size, normalization)
        self.ds0 = downsampling(f_size, f_size, normalization)
        self.dc1 = double_conv(f_size, 2*f_size, normalization)
        self.ds1 = downsampling(2*f_size, 2*f_size, normalization)
        self.dc2 = double_conv(2*f_size, 4*f_size, normalization)
        self.ds2 = downsampling(4*f_size, 4*f_size, normalization)
        self.dc3 = double_conv(4*f_size, 8*f_size, normalization)
        
        self.up2 = upsampling(8*f_size, 4*f_size, normalization)
        self.up1 = upsampling(4*f_size, 2*f_size, normalization)
        self.up_out = upsampling(2*f_size, f_size, normalization)
        self.up_unc = upsampling(2*f_size, f_size, normalization)
        self.out = nn.Sequential(
            nn.Conv2d(f_size, 1, kernel_size=3, stride=1, padding='same'),
            nn.ModuleDict([['relu',nn.ReLU()],['sigmoid',nn.Sigmoid()]]
                          )[out_acti])
        self.unc = nn.Sequential(
            nn.Conv2d(f_size, 1, kernel_size=3, stride=1, padding='same'),
            nn.Softplus())
        
        
    def forward(self, x):
        x = x.float()
        d = x.shape[0]
        x = torch.cat([x[:,0],x[:,1]],dim = 0);
        # print(x.shape)
        skip0 = self.dc0(x)
        down1 = self.ds0(skip0)
        skip1 = self.dc1(down1)
        down2 = self.ds1(skip1)
        skip2 = self.dc2(down2)
        down3 = self.ds2(skip2)
        skip3 = self.dc3(down3)
        
        upsa1 = self.up2(skip3,skip2)
        upsa1 = self.up1(skip2,skip1)
        
        obranch1 = self.up_out(upsa1,skip0)
        img_out = self.out(obranch1)
        img_out = torch.stack(torch.split(img_out,[d,d],dim=0),dim=1)
        
        obranch2 = self.up_unc(upsa1,skip0)
        unc_out = self.unc(obranch2)
        unc_out = torch.stack(torch.split(unc_out,[d,d],dim=0),dim=1)
        
        return([img_out,unc_out])
      

#%%

model = unet(n_channels =1, f_size=32)
criterion = some_distance_function()
if train_on_gpu:
    model.cuda()
# summary(model, (2,1,512,512))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


n_epochs = 5
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
    model.eval()
    del data, target
    with torch.no_grad():
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
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss
    


