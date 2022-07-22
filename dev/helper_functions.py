#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 11:14:28 2022

@author: c
"""
#Tutorial

#https://www.kaggle.com/dhananjay3/image-segmentation-from-scratch-in-pytorch


#%%

import os
import gc
import cv2
import time
import tqdm
import random
import collections
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm as tq
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchsummary import summary

# ablumentations for easy image augmentation for input as well as output
import albumentations as albu
# from albumentations import torch as AT
plt.style.use('bmh')

torch.cuda.empty_cache()
gc.collect()


#%%

#seeding function for reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True


#Dataset class
class CloudDataset(Dataset):
    def __init__(
            self,
            # df: pd.DataFrame = None,
            datatype: str = 'train',
            img_ids: np.array = None,
            transforms=albu.Compose([albu.HorizontalFlip()]),
        ):
            # self.df = df
            if datatype != 'test':
                self.img_folder  = f"{img_paths}/train_images"
                self.mask_folder = f"{img_paths}/train_masks"
            else:
                self.img_folder  = f"{img_paths}/test_images"
                self.mask_folder = f"{img_paths}/test_masks"
            self.img_ids = img_ids
            self.transforms = transforms
            
    def __getitem__(self,idx):
        image_name = self.img_ids[idx]
        image_path = os.path.join(self.img_folder , image_name+'.jpg')
        mask_path  = os.path.join(self.mask_folder, image_name+'.npy')
        img = cv2.imread(image_path)
        mask = np.load(mask_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        augmented = self.transforms(image=img, mask=mask)
        img = np.transpose(augmented['image'],[2,0,1])
        mask = np.transpose(augmented['mask'],[2,0,1])
        return img, mask
    
    def __len__(self):
        return(len(self.img_ids))
    
#%%

path = "input/pairs"
img_paths = "input/pairs"
train_on_gpu = torch.cuda.is_available()
SEED = 42
MODEL_NO = 0 # in K-fold
N_FOLDS = 10 # in K-fold
seed_everything(SEED)
os.listdir(path)

train_ids = [os.path.join(f[:len(f)-4]) for f in os.listdir(f"{path}/train_images")][::4]
# train = pd.read_csv(f"{path}/train.csv")

# train["label"] = train["Image_Label"].apply(lambda x: x.split("_")[1])
# train["im_id"] = train["Image_Label"].apply(lambda x: x.split("_")[0])

# sub = pd.read_csv(f"{path}/sample_submission.csv")
# sub["label"] = sub["Image_Label"].apply(lambda x: x.split("_")[1])
# sub["im_id"] = sub["Image_Label"].apply(lambda x: x.split("_")[0])

# # split data
# id_mask_count = (
#     train.loc[train["EncodedPixels"].isnull() == False, "Image_Label"]
#     .apply(lambda x: x.split("_")[0])
#     .value_counts()
#     .sort_index()
#     .reset_index()
#     .rename(columns={"index": "img_id", "Image_Label": "count"})
# )
# ids = id_mask_count["img_id"].values
# li = [
#     [train_index, test_index]
#     for train_index, test_index in StratifiedKFold(
#         n_splits=N_FOLDS, random_state=None
#     ).split(ids, id_mask_count["count"])
# ]
# train_ids, valid_ids = ids[li[MODEL_NO][0]], ids[li[MODEL_NO][1]]
# test_ids = sub["Image_Label"].apply(lambda x: x.split("_")[0]).drop_duplicates().values

# skf = StratifiedKFold(n_splits=5)
# skf.split(y=train_ids)

# print(f"training set   {train_ids[:5]}.. with length {len(train_ids)}")
# print(f"validation set {valid_ids[:5]}.. with length {len(valid_ids)}")
# print(f"testing set    {test_ids[:5]}.. with length {len(test_ids)}")

#%%

#define dataset and dataloader

valid_ids = train_ids[::5]

num_workers = 8 #workers to feed data into RAM, can also be larger than number of cores
bs = 8
train_dataset = CloudDataset(
    datatype='train',
    img_ids=train_ids,
    )

valid_dataset = CloudDataset(
    datatype='valid',
    img_ids=valid_ids,
    )

train_loader = DataLoader(
    train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers
)
valid_loader = DataLoader(
    valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers
)

#%%

#Model definition

class double_conv(nn.Module):
    
    
    def __init__(self,in_ch,out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,3,padding='same'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch,out_ch,3,padding='same'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU())
        
    def forward(self, x):
        return self.conv(x)
    
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv,self).__init__()
        self.conv = double_conv(in_ch,out_ch)
    
    def forward(self,x):
        return self.conv(x)
    
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))
        
    def forward(self,x):
        return self.mpconv(x)
        
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear = True):
        super(up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode = 'bilinear',align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2,in_ch//2,2,stride=2)
        
        self.conv = double_conv(in_ch, out_ch)
        
    def forward(self,x1,x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, (diffX//2,diffX-diffX//2, diffY//2, diffY - diffY//2))
        
        x = torch.cat([x2,x1], dim=1)
        return self.conv(x)
    
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, f_size):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, f_size)
        self.down1 = down(f_size, 2*f_size)
        self.down2 = down(2*f_size, 4*f_size)
        self.down3 = down(4*f_size, 8*f_size)
        self.down4 = down(8*f_size, 8*f_size)
        self.up1 = up(16*f_size, 4*f_size, False)
        self.up2 = up(8*f_size, 2*f_size, False)
        self.up3 = up(4*f_size, f_size, False)
        self.up4 = up(2*f_size, f_size, False)
        self.outc = outconv(f_size, n_classes)

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)
    
model = UNet(n_channels=3, n_classes=1, f_size=32)
if train_on_gpu:
    model.cuda()
summary(model,(3,512,512))


    

#%%

class DiceLoss(nn.Module):
    __name__ = 'dice_loss'
    
    def __init__(self, eps, activation = 'none'):
        super().__init__()
        self.activation = activation
        self.eps = eps
        
    def forward(self,pred, gt):
        if self.activation is None or self.activation == "none":
            activation_fn = lambda x: x
        elif self.activation == "sigmoid":
            activation_fn = torch.nn.Sigmoid()
        elif self.activation == "softmax2d":
            activation_fn = torch.nn.Softmax2d()
        else:
            raise NotImplementedError(
                "Activation implemented for sigmoid and softmax2d"
            )
        
        pred = activation_fn(pred)
        score = 1. - 2.*gt*pred/(2.*gt*pred + gt*(1.-pred)+pred*(1.-gt) + self.eps)
        return torch.mean(score)

criterion = DiceLoss(eps=1e-7)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, cooldown=2)

#%%

n_epochs = 32
train_loss_list = []
valid_loss_list = []
dice_score_list =  []
lr_rate_list =  []
valid_loss_min = np.Inf


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
        #print(loss)
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
    dice_score_list.append(dice_score)
    lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups])
    
    # print training/validation statistics 
    print('Epoch: {}  Training Loss: {:.6f}  Validation Loss: {:.6f} Dice Score: {:.6f}'.format(
        epoch, train_loss, valid_loss, dice_score))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss
    
    scheduler.step(valid_loss)
    
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
plt.show()


valid_masks = []
count = 0
tr = min(len(valid_ids)*4, 2000)
probabilities = np.zeros((tr, *data.cpu().shape[2:]), dtype = np.float32)
for data, target in tq(valid_loader):
    if train_on_gpu:
        data = data.cuda()
    target = target.cpu().detach().numpy()
    outpu = model(data).cpu().detach().numpy()
    for p in range(data.shape[0]):
        output, mask = outpu[p], target[p]
        for m in mask:
            valid_masks.append((m))
        for probability in output:
            probabilities[count, :, :] = (probability)
            count += 1
        if count >= tr - 1:
            break
    if count >= tr - 1:
        break



weights = list(model.parameters())
weights[-1]

for k in range(3):
    torch.cuda.empty_cache()
    gc.collect()