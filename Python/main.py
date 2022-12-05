import torch.nn.functional as f
import torch
import math
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import timeit
from utility import read_image, pad_image
from visualisation import deform_to_hsv
from AOTV import AOTV

levels     = [4,2,1]
tol1       = 1e-3
tol2       = 1e-10
taylor     = 6
max_iter   = 500
lmbda      = 0.2
solver     = 'dct'
method     ='2ndO-TV'
device     ='cuda:0'


source = read_image('./data/minicooper1.png', rescale=True)
target = read_image('./data/minicooper2.png', rescale=True)
source = source.astype(np.float32)
target = target.astype(np.float32)
target = torch.from_numpy(target).unsqueeze(0).unsqueeze(1).to(device)
source = torch.from_numpy(source).unsqueeze(0).unsqueeze(1).to(device)
# print(source.shape)

# image = read_image('./data/sa_crop.nii.gz', device=device) 
# target = torch.from_numpy(image[...,0]).unsqueeze(0).permute(3, 0, 1, 2).to(device)
# source = torch.from_numpy(image[...,10]).unsqueeze(0).permute(3, 0, 1, 2).to(device)
# print(target.shape)
start = timeit.default_timer() 
disp = AOTV(source, target, levels, lmbda=lmbda, taylor=taylor, max_iter=max_iter, 
            tol1=tol1, tol2=tol2, method=method, solver=solver, device=device)
end = timeit.default_timer()
print(' ** full time {}\n'.format(end-start));