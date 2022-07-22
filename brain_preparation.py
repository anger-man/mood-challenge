import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import time
import shutil
from matplotlib.pyplot import imsave

task = 'brain'
#%%

files = [os.path.join(task,task+'_train',f) for f in os.listdir(os.path.join(task,task+'_train'))]
print(files[:10])
f_name=files[0]

def standardize_external(x):
    mean = np.mean(x[x!=0])
    std  = np.std(x[x!=0])
    return (x-mean)/std

def load_itk(f_name, inp,tar):
    itkimage = sitk.ReadImage(f_name, sitk.sitkFloat32)
    tmp = sitk.GetArrayFromImage(itkimage)[:,]
    tmp = np.rot90(tmp, k=2,axes=(0,2))
    print(np.max(tmp)); print(np.min(tmp))
    name = f_name[len(f_name)-12:]
    np.save(os.path.join('brain','npy_files',name+'.npy'), tmp.astype(np.float16))
  
for f_name in files:
    load_itk(f_name,inp=0,tar=0)



#%%
