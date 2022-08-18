# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:43:49 2022

@author: A0067477
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
import torchio as tio
import gc, os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm as tq
from scipy.ndimage import rotate
import time
from skimage.measure import block_reduce
from datetime import datetime
import time
from data_functions import generate_random_mask, add_gaussian_noise, random_intensity
from data_functions import DiceLoss, MedicalDataset
from models import unet

#%%

#empty cuda cache and check, if GPU is available

torch.cuda.empty_cache()
gc.collect()
train_on_gpu = torch.cuda.is_available()
#train_on_gpu = False
# from numba import cuda
# cuda.select_device(0)
# cuda.close()


      
#%%
#use every sixth scan for validation data
vali_ids = np.sort(os.listdir(os.path.join(data_path,'%s_train'%task)))[::6]
vali_dataset = MedicalDataset(
    datatype = 'train',
    path = data_path,
    task = task,
    img_ids = vali_ids,
    reduce_dim = False)

#use the remainings scans as train data
train_ids = [f for f in os.listdir(os.path.join(data_path,'%s_train'%task)) if f not in vali_ids]
train_dataset = MedicalDataset(
    datatype = 'train',
    path = data_path,
    task = task,
    img_ids = train_ids,
    reduce_dim = False)

#also evaluate on the provided toy data
test_ids = os.listdir(os.path.join(data_path,'toy'))
test_dataset = MedicalDataset(
    datatype = 'test',
    path = data_path,
    task = task,
    img_ids = test_ids,
    disturb_input = False,
    reduce_dim = False)

#%%

#the considered loss function is the DiceLoss
criterion = DiceLoss()


#option to use the prediction of the global network as second channel.
use_global = True
n_channels = 1
if use_global:
    model_global = unet(n_channels=1, f_size=32)
    n_channels = 2
    if train_on_gpu:
        model_global.cuda()
    model_global.load_state_dict(torch.load('weights/weights_%s_global.pt'%task))
    model_global.eval()
    
    
    
#the output channel size of the first convolution equals 32
model = unet(n_channels=n_channels, f_size=32)


if train_on_gpu:
    model = model.to('cuda:0')
    #model.cuda()
summary(model, (n_channels,64,64,64))
#we use Adam algorithm, the lr is already fine-tuned
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, cooldown=2)

#define a index for the plots and saved model parameters
dt = datetime.now()
index = '%d%d%d_%d%d_local'%(dt.year,dt.month,dt.day,dt.hour,dt.minute)

def get_subject_globpred(data, target):
    img = data.squeeze()
    fac = int(img.shape[0]/64)
    img = block_reduce(img, (fac, fac, fac), np.mean)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        pred_global = model_global(img)[0]
    up = nn.Upsample(scale_factor = fac)
    pred_global_up = up(pred_global)
    subject = tio.Subject({
        'image': tio.ScalarImage(tensor = data),
        'labels': tio.LabelMap(tensor = target),
        'pred_global': tio.ScalarImage(tensor = pred_global_up.squeeze(0))
        })
    return subject

#%%

#training pipeline


torch.cuda.empty_cache()
gc.collect()
batch_size = 4
patch_bs = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
valid_loader = DataLoader(vali_dataset, batch_size=4, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=6)


n_epochs = 50
train_loss_list = []
valid_loss_list = []
dice_score_list =  []
lr_rate_list =  []
valid_loss_min = 1e7

#%%

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
    #train_loader contains all whole training images 
    for data, target in bar:
        subject_list = []
        for i in range(data.size()[0]):
            
            if use_global:
                subject = get_subject_globpred(data[i], target[i])        
            else:
                subject = tio.Subject({
                    'image': tio.ScalarImage(tensor = data[i]),
                    'labels': tio.LabelMap(tensor = target[i])
                    })
                
            subject_list.append(subject)

        subjects_dataset = tio.SubjectsDataset(subject_list)
        sampler = tio.data.LabelSampler(patch_size = 64, label_probabilities = {0:2, 1:8})
        patches_queue = tio.data.Queue(subjects_dataset=subjects_dataset, max_length = 48, 
                                       samples_per_volume= 12, sampler = sampler)
        patches_loader = DataLoader(patches_queue, batch_size = patch_bs, num_workers=0)
        
        for patch_subject in patches_loader:
            patch_data, patch_target = patch_subject['image'][tio.DATA], patch_subject['labels'][tio.DATA]
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                patch_data, patch_target = patch_data.cuda(), patch_target.cuda()
            if use_global:
                patch_global = patch_subject['pred_global'][tio.DATA]
                patch_data = torch.cat((patch_data, patch_global),1)
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(patch_data)
            # calculate the batch loss
            loss = criterion(output[0], patch_target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.mean().backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*patch_data.size(0)

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
            subject_eval_list = []
            for i in range(data.size()[0]):
                if use_global:
                    subject_eval = get_subject_globpred(data[i], target[i])        
                
                else:
                    subject_eval = tio.Subject({
                        'image': tio.ScalarImage(tensor = data[i]),
                        'labels': tio.LabelMap(tensor = target[i])
                        })
                subject_eval_list.append(subject_eval)
            subjects_eval_dataset = tio.SubjectsDataset(subject_eval_list)
            sampler = tio.data.LabelSampler(patch_size = 64, label_probabilities = {0:3, 1:7})
            patches_eval_queue = tio.data.Queue(subjects_dataset=subjects_eval_dataset,
                                                max_length=48, samples_per_volume=12, sampler=sampler)
            
            patches_loader = DataLoader(patches_eval_queue, batch_size = 4)
            
            for patch_subject in patches_loader:
                patch_data, patch_target = patch_subject['image'][tio.DATA], patch_subject['labels'][tio.DATA]
                
                if use_global:
                    patch_global = patch_subject['pred_global'][tio.DATA]
                    patch_data = torch.cat((patch_data, patch_global), 1)

                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    patch_data, patch_target = patch_data.to('cuda:0'), patch_target.to('cuda:0')
                    # forward pass: compute predicted outputs by passing inputs to the model
                    output = model(patch_data)
                    # calculate the batch loss
                    loss = criterion(output[0], patch_target)
                    # update average validation loss 
                    valid_loss += loss.item()*data.size(0)
                    # dice_cof = dice_no_threshold(output.cpu(), target.cpu()).item()
                    # dice_score +=  dice_cof * data.size(0)
        bar.set_postfix(ordered_dict={"valid_loss":loss.item()})
        if save_data:
            preds = output[0].cpu().detach().numpy()[:,0]
            uncer = output[1].cpu().detach().numpy()[:,0]
            gts   = patch_target.cpu().detach().numpy()[:,0]
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
    train_loss = train_loss/len(train_loader.dataset)/patch_bs
    valid_loss = valid_loss/len(valid_loader.dataset)/patch_bs
    dice_score = dice_score/len(valid_loader.dataset)/patch_bs
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups])
    
    # print training/validation statistics 
    print('Epoch: {}  Training Loss: {:.6f}  Validation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss), flush=True)
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss), flush=True)
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
        if use_global:
            subject = get_subject_globpred(data[0], target[0])        
                
        else:
            subject = tio.Subject({
                'image': tio.ScalarImage(tensor = data[0]),
                'labels': tio.LabelMap(tensor = target[0])
                })
            

        sampler = tio.data.LabelSampler(patch_size = 64, label_probabilities = {0:3, 1:7})
        generator = sampler(subject, num_patches = 8)

        

        
        for patch_subject in generator:
            patch_data, patch_target = patch_subject['image'].data, patch_subject['labels'].data
            if use_global:
                patch_global =  patch_subject['pred_global'].data
                patch_data = torch.cat((patch_data, patch_global), 1)
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                if len(patch_data.size())==4:
                    patch_data = patch_data.unsqueeze(0)
                patch_data = patch_data.cuda()
                
            patch_target = patch_target.cpu().detach().numpy()
            pred,uncer = model(patch_data)
            pred = pred.cpu().detach().numpy()
            uncer = uncer.cpu().detach().numpy()
            preds.append(pred[0]); gts.append(patch_target); uncers.append(uncer[0])
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

#plot loss and lr scheduler

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



    

    
