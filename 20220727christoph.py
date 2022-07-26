#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:02:58 2022

@author: c
"""

#%%

task = 'brain'

#%%

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


torch.cuda.empty_cache()
gc.collect()
train_on_gpu = torch.cuda.is_available()

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
    def __init__(self, n_channels, f_size, normalization='instance',
                 out_acti = 'relu'):
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
        down4 = self.ds3(skip3)
        skip4 = self.dc4(down4)
        
        upsa3 = self.up3(skip4,skip3)
        upsa2 = self.up2(upsa3,skip2)
        upsa1 = self.up1(upsa2,skip1)
        
        obranch1 = self.up_out(upsa1,skip0)
        img_out = self.out(obranch1)
        img_out = torch.stack(torch.split(img_out,[d,d],dim=0),dim=1)
        
        obranch2 = self.up_unc(upsa1,skip0)
        unc_out = self.unc(obranch2)
        unc_out = torch.stack(torch.split(unc_out,[d,d],dim=0),dim=1)
        
        return([img_out,unc_out])
      
    
class l1dist(nn.Module):
    __name__ = 'dice_loss'
    
    def __init__(self,uncertainty=True,eps=1e-6):
        self.uncertainty = uncertainty
        self.eps = eps
        super().__init__()
        
    def forward(self,pred, gt):
        p = pred[0]; 
        if self.uncertainty:
            u = torch.clip(pred[1],self.eps,1e8)
            t1 = torch.abs(p-gt)/u
            t1 = torch.where(t1<=2,t1,2+torch.log(torch.clip(t1-2+1,self.eps,1e8)))
            return torch.mean(t1+torch.log(1.+2.*u))
        else:
            return torch.mean(torch.abs(p-gt))

        
#%%


data_path = 'data/%s'%task


def intensity_perturbation_2d(x):
    m = x>0; 
    i,j = np.random.choice(range(x.shape[0]), size=2)
    while(m[i,j]==0):
        i,j = np.random.choice(range(x.shape[0]), size=2)
    size = int(np.random.choice(np.linspace(.1,.5,100))*np.min([i,j]))
    x[i-size:i+size,j-size:j+size] *= .5
    return x

def inpaint_3d(x):
    s = x.shape[0]
    m = x>0.02; 
    i,j,k = np.random.choice(range(s), size=3)
    while(m[i,j,k]==0):
        i,j,k = np.random.choice(range(s), size=3)
    size = int(np.random.choice(np.linspace(12,np.max(
        [12,np.min([i,s-i,j,s-j,k,s-k])]),100)))
    x[i-size:i+size,j-size:j+size,k-size:k+size] = np.random.choice(np.linspace(
        np.min(x),np.max(x),100))
    return x
    

    
    
def projection(data):
    rand_angle = np.random.choice(range(180),2, replace=False)
    res_per = []; res_ori = []
    for ra in rand_angle:
        rot = rotate(data, angle=ra, axes=(0,2), reshape=False)
        ori = rot.copy(); per = rot.copy()
        
        ori = np.sum(ori,axis=0)
        ori /= np.quantile(ori,.98); ori[ori>1]=1
        res_ori.append(np.expand_dims(ori,0))
        
        per = inpaint_3d(per)
        per = np.sum(per,axis=0)
        per /= np.quantile(per,.98); per[per>1]=1
        res_per.append(np.expand_dims(per,0))

    return [np.stack(res_per,axis=0), np.stack(res_ori,axis=0)]

class MedDataset(Dataset):
    def __init__(
            self,
            datatype: str = 'train',
            path: str = data_path,
            img_ids: np.array = None,
            disturb_input = True
        ):
            # self.df = df
            if datatype == 'train':
                self.img_folder  = f"{path}/%s_train"%task
            else:
                self.img_folder  = f"{path}/toy"
            self.img_ids = img_ids
            self.disturb_input = disturb_input
            
    def __getitem__(self,idx):
        image_name = self.img_ids[idx]
        image_path = os.path.join(self.img_folder , image_name)
        nifti = nib.load(image_path)
        data = nifti.get_fdata()
        per_data, ori_data = projection(data)
        # data = np.expand_dims(data,1)
        # per_data = generate_artifacts(data)
        # matrix = nifti.affine
        if self.disturb_input:
            return per_data, ori_data
        else:
            return ori_data, ori_data
    
    def __len__(self):
        return(len(self.img_ids))
    
vali_ids = os.listdir(os.path.join(data_path,'%s_train'%task))[::6]
train_ids = [f for f in os.listdir(os.path.join(data_path,'%s_train'%task)) if f not in vali_ids] 
test_ids = os.listdir(os.path.join(data_path,'toy'))

train_dataset = MedDataset(
    datatype = 'train',
    img_ids = train_ids)
vali_dataset = MedDataset(
    datatype = 'train',
    img_ids = vali_ids)
test_dataset = MedDataset(
    datatype = 'test',
    img_ids = test_ids,
    disturb_input = False)

#%%

criterion = l1dist(uncertainty = True)
model = unet(n_channels =1, f_size=32)
if train_on_gpu:
    model.cuda()
summary(model, (2,1,256,256))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, cooldown=2)

#%%

batch_size = 4

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers = 6)
valid_loader = DataLoader(
    vali_dataset, batch_size=batch_size, shuffle=True,
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
    preds = output[0].cpu().detach().numpy()[:,0]
    uncer = output[1].cpu().detach().numpy()[:,0]
    inp   = data.cpu().detach().numpy()[:,0]
    
    
    fig, ax = plt.subplots(2,4,figsize=(12,5)); 
    for j in [0,1]:
        im = ax[j,0].imshow(inp[j,0], cmap='Greys_r')
        ax[j,0].axis('off')
        plt.colorbar(im,ax=ax[j,0])
        im = ax[j,1].imshow(preds[j,0], cmap='Greys_r')
        ax[j,1].axis('off')
        plt.colorbar(im,ax=ax[j,1])
        im = ax[j,2].imshow(np.abs(inp[j,0]-preds[j,0]), cmap='gist_rainbow')
        ax[j,2].axis('off')
        plt.colorbar(im,ax=ax[j,2])
        im = ax[j,3].imshow(uncer[j,0], cmap='gist_rainbow')
        ax[j,3].axis('off')
        plt.colorbar(im,ax=ax[j,3])
        
    fig.tight_layout(pad=.1)
    plt.savefig('plots/val/%d.pdf'%time.time())
        
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
    
    scheduler.step(valid_loss)
    
    ######################    
    # test the model #
    ######################
    model.eval()
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
        
        im = ax[j,0].imshow(gts[j,0], cmap='Greys_r')
        ax[j,0].axis('off')
        plt.colorbar(im,ax=ax[j,0])
        
        im = ax[j,1].imshow(preds[j,0], cmap='Greys_r')
        ax[j,1].axis('off')
        plt.colorbar(im,ax=ax[j,1])
        
        im = ax[j,2].imshow(np.abs(gts[j,0]-preds[j,0]), cmap='gist_rainbow')
        ax[j,2].axis('off')
        plt.colorbar(im,ax=ax[j,2])
        
        im = ax[j,3].imshow(uncers[j,0], cmap='gist_rainbow')
        ax[j,3].axis('off')
        plt.colorbar(im,ax=ax[j,3])
        
    fig.tight_layout(pad=.1)
    plt.savefig('plots/test/%d.pdf'%time.time())

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
ax.plot(train_loss_list,  marker='o', label="Training Loss")
ax.plot(valid_loss_list,  marker='o', label="Validation Loss")
plt.ylabel('loss', fontsize=22)
plt.legend()
plt.savefig('loss/%d'%time.time())


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
    
    im = ax[j,0].imshow(gts[j,0], cmap='Greys_r')
    ax[j,0].axis('off')
    plt.colorbar(im,ax=ax[j,0])
    
    im = ax[j,1].imshow(preds[j,0], cmap='Greys_r')
    ax[j,1].axis('off')
    plt.colorbar(im,ax=ax[j,1])
    
    im = ax[j,2].imshow(np.abs(gts[j,0]-preds[j,0]), cmap='gist_rainbow')
    ax[j,2].axis('off')
    plt.colorbar(im,ax=ax[j,2])
    
    im = ax[j,3].imshow(uncers[j,0], cmap='gist_rainbow')
    ax[j,3].axis('off')
    plt.colorbar(im,ax=ax[j,3])
    
fig.tight_layout(pad=.1)
plt.savefig('plots/%d'%time.time())




    

    
