import numpy as np
import torch
import os
import math
import torch
# from tqdm import tqdm
from torch.utils.data import ConcatDataset
#import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
from train_model import Wrapper


import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
sys.path.append('../pre_data/')
from data_path import datapath
import h5py

def channel_normalization(H):
    P_current = np.sum(np.abs(H) ** 2)
    alpha = np.sqrt((H.shape[1] * H.shape[2]) / P_current)

    # 缩放矩阵，总功率变为 N*M
    H_normalized = H * alpha  
    return H_normalized  


class generate_Dataset(Dataset):
    def __init__(self,input_fmap, input_tdim, input_fdim):
        
        self.len = 1000
        self.input_fmap = input_fmap
        self.input_tdim = input_tdim
        self.input_fdim = input_fdim
        self.task = "train"
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.task == "train":
            data_list = np.load("../pre_data/train.npy")

        while (1):
            random_number = random.randint(0, len(data_list))
            with h5py.File(datapath[0]+f"/3_{random_number}.h5py", 'r') as f:
                channel_real = f["channel_real"] [:]
                channel_imag = f["channel_imag"][:]
                channel = channel_real+ 1j* channel_imag
                UElocation = f["UElocation"][:][:2]                
            if np.sum(np.abs(channel) ** 2) >0:
                break

        channel_norm = channel_normalization(channel)
        # print(channel_norm.shape,UElocation.shape)

        data = np.zeros([self.input_fmap, channel_norm.shape[1], channel_norm.shape[2]])
        data[0,:,:] = np.abs(channel_norm).astype(np.float32)
        data[1,:,:] = np.angle(channel_norm).astype(np.float32)

        return  data, UElocation #, mask[0,:].float()


