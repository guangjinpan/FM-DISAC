    
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

EnvPara = {}  
EnvPara["epochs"]  = 3


    
if __name__ == '__main__':   
    input_tdim = 1024
    input_fdim = 256
    input_fmap = 2
    FMmodel = Wrapper()

    train_dataset = generate_Dataset(input_fmap, input_tdim, input_fdim)
    train_dataloader = DataLoader(train_dataset, batch_size = 64, num_workers = 8, shuffle = True, drop_last = False, pin_memory=True)
    val_dataset = generate_Dataset(input_fmap, input_tdim, input_fdim)
    val_dataloader = DataLoader(val_dataset, batch_size = 64, num_workers = 8, shuffle = True, drop_last = False, pin_memory=True)
                
    model_path="test"
    model_checkpoint = ModelCheckpoint(
        dirpath="../../pretrained_model/model/",
        filename=model_path+'{epoch:02d}',
        save_top_k=1,  # 仅保存最好的模型
        monitor="val/loss",  # 根据验证集损失选择最好的模型
        mode="min",  # 'min' 表示损失越小越好，'max' 用于指标越高越好（如准确率）
    )

    latest_model_checkpoint = ModelCheckpoint(
        dirpath="../../pretrained_model/model/",
        filename=model_path+'_latest',
        save_top_k=1,  # 只保留最新的模型
        save_last=True,  # 总是保存最新的模型
        every_n_epochs=5,  # 每个 epoch 保存一次
    )

    trainer = Trainer(
        log_every_n_steps=0,  # 不记录 step 级日志
        enable_progress_bar=False,
        devices=1,
        accelerator='gpu',
        # logger=logger,
        max_epochs=EnvPara["epochs"],
        callbacks=[model_checkpoint, latest_model_checkpoint]
    )
    trainer.fit(FMmodel, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)