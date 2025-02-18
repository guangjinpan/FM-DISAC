    
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
EnvPara["model_path"]  = '../../pretrained_model/woFT_SingleBSLoc/10000' #'../../pretrained_model/pretrain_mpg'
EnvPara["load_pretrained_mdl_path"] =  '../../pretrained_model/woFT_SingleBSLoc/10000/test_latest.ckpt' 
EnvPara["pretrain_stage"]  = True
EnvPara["device"]  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':   



    FMmodel = Wrapper(EnvPara)

    test_dataset = generate_Dataset_test(EnvPara)
    test_dataloader = DataLoader(test_dataset, batch_size = 32, num_workers = 8, shuffle = False, drop_last = False, pin_memory=True)
    result = np.zeros((500000,2))
    label = np.zeros((500000,2))
    cnt = 0
    for channel, position in test_dataloader:
        channel = channel.to(EnvPara["device"]).float()
        pred = FMmodel.channel_fdmdl(channel, position, task = EnvPara["task"])
        result[cnt : cnt+len(channel)] = pred.cpu().detach().numpy()
        result[cnt : cnt+len(channel)] = position.detach().numpy()
        cnt = cnt+len(channel) 
    print(sum(result),sum(label))