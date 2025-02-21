    
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


def set_seed(seed=42):
    """设置所有可能影响随机性的随机种子"""
    random.seed(seed)  # Python 内置随机数
    np.random.seed(seed)  # NumPy 随机数
    torch.manual_seed(seed)  # PyTorch CPU 随机数
    torch.cuda.manual_seed(seed)  # PyTorch GPU 随机数
    torch.cuda.manual_seed_all(seed)  # 多 GPU 训练时，确保所有 GPU 一致

#1-488699
EnvPara = {}  
EnvPara["epochs"]  = 1000
EnvPara["data_len"] = 10000#100000
EnvPara["input_tdim"]  = 32
EnvPara["input_fdim"]  = 64
EnvPara["input_fmap"]  = 2
EnvPara["fshape"]  = 4
EnvPara["tshape"]  = 4
EnvPara["fstride"]  = 4
EnvPara["tstride"]  = 4
EnvPara["model_size"]  = 'tiny'
EnvPara["task"]  = "pretrain_antenna" # "woFT_SingleBSLoc" # pretrain_mpg #FT_SingleBSLoc #pretrain_antenna
EnvPara["model_path"]  = '../../pretrained_model/pretrain_antenna' #'../../pretrained_model/woFT_SingleBSLoc/10000' #'../../pretrained_model/pretrain_mpg'
EnvPara["load_pretrained_mdl_path"] =   '../../pretrained_model/pretrain_mpg/testepoch=50.ckpt'  #'../../pretrained_model/pretrain_mpg/testepoch=99.ckpt' 
EnvPara["pretrain_stage"]  = True
EnvPara["device"]  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" 
if __name__ == '__main__':   
    print(EnvPara)
    set_seed(42)

    FMmodel= Wrapper(EnvPara=EnvPara)

    if EnvPara["task"] == "FT_SingleBSLoc":
        FMmodel= Wrapper.load_from_checkpoint(EnvPara["load_pretrained_mdl_path"], EnvPara=EnvPara, strict=False)
    
    FMmodel.to(EnvPara["device"])


    train_dataset = generate_Dataset(EnvPara)
    train_dataloader = DataLoader(train_dataset, batch_size = 32, num_workers = 8, shuffle = True, drop_last = False, pin_memory=True)
    val_dataset = generate_Dataset(EnvPara)
    val_dataloader = DataLoader(val_dataset, batch_size = 32, num_workers = 8, shuffle = True, drop_last = False, pin_memory=True)
                
    model_path="test"
    model_checkpoint = ModelCheckpoint(
        dirpath=EnvPara["model_path"],
        filename=model_path+'{epoch:02d}',
        save_top_k=1,  # 仅保存最好的模型
        monitor="val/loss",  # 根据验证集损失选择最好的模型
        mode="min",  # 'min' 表示损失越小越好，'max' 用于指标越高越好（如准确率）
    )

    latest_model_checkpoint = ModelCheckpoint(
        dirpath=EnvPara["model_path"],
        filename=model_path+'_latest',
        save_top_k=1,  # 只保留最新的模型
        save_last=True,  # 总是保存最新的模型
        every_n_epochs=5,  # 每个 epoch 保存一次
    )

    logger = pl.loggers.TensorBoardLogger('./logs', name=EnvPara["task"])
    trainer = Trainer(
        log_every_n_steps=0,  # 不记录 step 级日志
        enable_progress_bar=False,
        devices=1,
        accelerator='gpu',
        logger=logger,
        max_epochs=EnvPara["epochs"],
        callbacks=[model_checkpoint, latest_model_checkpoint],
        # gradient_clip_val=1.0,
        # precision=32,
    )
    trainer.fit(FMmodel, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)