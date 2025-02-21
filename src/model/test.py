    
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
from dataload import *


import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint


#1-488699
EnvPara = {}  
EnvPara["epochs"]  = 100
EnvPara["data_len"] = 10000#100000
EnvPara["input_tdim"]  = 32
EnvPara["input_fdim"]  = 64
EnvPara["input_fmap"]  = 2
EnvPara["fshape"]  = 4
EnvPara["tshape"]  = 4
EnvPara["fstride"]  = 4
EnvPara["tstride"]  = 4
EnvPara["model_size"]  = 'tiny'
EnvPara["task"]  = "inference_SingleBSLoc" # "woFT_SingleBSLoc" # pretrain_mpg
EnvPara["model_path"]  = '../../pretrained_model/FT_SingleBSLoc/10000' #'../../pretrained_model/pretrain_mpg'
EnvPara["load_pretrained_mdl_path"] =  '../../pretrained_model/FT_SingleBSLoc/10000/testepoch=939.ckpt' 
EnvPara["pretrain_stage"]  = True
EnvPara["device"]  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':   

    print(EnvPara)

    FMmodel = Wrapper(EnvPara)
    FMmodel= Wrapper.load_from_checkpoint(EnvPara["load_pretrained_mdl_path"], EnvPara=EnvPara, strict=False)
    FMmodel.to(EnvPara["device"])

    # sd = torch.load(EnvPara["load_pretrained_mdl_path"])
    # print(EnvPara["load_pretrained_mdl_path"])
    # FMmodel.load_state_dict(sd, strict=True)
    # FMmodel.to(EnvPara["device"])
    test_dataset = generate_Dataset_test(EnvPara)
    test_dataloader = DataLoader(test_dataset, batch_size = 32, num_workers = 8, shuffle = False, drop_last = False, pin_memory=True)
    result = np.zeros((500000,2))
    label = np.zeros((500000,2))
    mse = np.zeros((500000,1))
    cnt = 0
    for channel, position in test_dataloader:
        FMmodel.eval()
        channel = channel.to(EnvPara["device"]).float()
        position = position.to(EnvPara["device"]).float()
        pred,loss = FMmodel.channel_fdmdl(channel, position, task = EnvPara["task"])
        result[cnt : cnt+len(channel)] = pred.cpu().detach().numpy()
        label[cnt : cnt+len(channel)] = position.cpu().detach().numpy()
        mse[cnt : cnt+len(channel),0] = loss.cpu().detach().numpy()
        cnt = cnt+len(channel) 
    valid_indices = np.where(np.sum(np.abs(label),1) != 0)[0]
    print(valid_indices.shape, valid_indices)
    valid_result = result[valid_indices,:]
    valid_label = label[valid_indices,:]
    valid_mes= mse[valid_indices,:]

    distances = np.linalg.norm(valid_result - valid_label, axis=1)
    print(len(distances))


    # Compute RMSE
    rmse = np.sqrt(np.mean(distances ** 2))


    print(f"RMSE: {rmse}, mse:{np.mean(valid_mes)}")