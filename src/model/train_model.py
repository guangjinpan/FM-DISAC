import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

import argparse
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, ConcatDataset, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
import math
import torch
import math
import os
import numpy as np
from scipy.spatial.distance import cdist
import random
from fm_models import *


class Wrapper(pl.LightningModule):
    def __init__(self, EnvPara):
        super().__init__()

        self.channel_fdmdl = ASTModel(
                 fshape = EnvPara["fshape"], tshape = EnvPara["tshape"], fstride = EnvPara["fstride"], tstride = EnvPara["tstride"],
                 input_fdim = EnvPara["input_fdim"], input_tdim = EnvPara["input_tdim"], input_fmap = EnvPara["input_fmap"], model_size = EnvPara["model_size"],
                 pretrain_stage = EnvPara["pretrain_stage"], device = EnvPara["device"])
        self.task = EnvPara["task"]
        self.train_epoch_loss = [] 
        self.valepoch_loss = []
        self.EnvPara = EnvPara


    def forward(self, x, y):

        if self.task == "no_pretrain":
            loss = self.channel_fdmdl(x, y, task='no_pretrain')
        elif self.task == "pretrain_mpg":
            loss = self.channel_fdmdl(x, y, task='pretrain_mpg', mask_patch=13)
        elif  self.task == "pretrain_antenna":
            loss = self.channel_fdmdl(x, y, task='pretrain_antenna', mask_antenna_number=8)
        elif self.task == "woFT_SingleBSLoc":
            loss = self.channel_fdmdl(x, y, task = self.task)
        elif self.task == "FT_SingleBSLoc":
            loss = self.channel_fdmdl(x, y, task = self.task)
        elif self.task == "inference_SingleBSLoc":
            res = self.channel_fdmdl(x, y, task = self.task)
        

        return loss
    

    def training_step(self, batch, batch_idx):
        x, labels = batch
        x = x.float()

        loss = self.forward(x, labels)


        # for name, param in self.channel_fdmdl.named_parameters():
        #     if "pos_embed" in name or "cls_token" in name:
        #         print(f"{name} 均值: {param.data.mean().item():.4f}, 梯度均值: {param.grad.mean().item() if param.grad is not None else 0:.4f}")
        #         if torch.isnan(param).any():
        #             print(f"NaN 出现在 {name}!")
                    
        self.train_epoch_loss.append(loss.detach())
        self.log('train/loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return {'loss': loss}

    def on_train_epoch_end(self):
        # 计算平均 loss
        avg_loss = torch.stack(self.train_epoch_loss).mean()
        print(f"Epoch {self.current_epoch} - Average Training Loss: {avg_loss.item()}")
        # 清空 loss 列表，为下一轮训练准备
        self.train_epoch_loss.clear()
    
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        x = x.float()

        loss = self.forward(x, labels)

        self.valepoch_loss.append(loss.detach())
        self.log('val/loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)


    def on_validation_epoch_end(self):
        # 计算平均 loss
        avg_loss = torch.stack(self.valepoch_loss).mean()
        print(f"Epoch {self.current_epoch} - Average Validation Loss: {avg_loss.item()}")
        # 清空 loss 列表，为下一轮训练准备
        self.valepoch_loss.clear()

    def configure_optimizers(self):
        self.optim = torch.optim.Adam(self.parameters(), lr=3e-5, weight_decay=1e-4)
        # self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=3000, eta_min=0)
        self.schedule = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=1e-3, total_steps=15625000, pct_start=0.1, final_div_factor=1e2)
        # self.schedule = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=1e-3, total_steps=782*1000, pct_start=0.4, final_div_factor=1e1)

        return {
            'optimizer': self.optim, 
            # 'lr_scheduler': {'scheduler': self.schedule, 'interval': 'step'}
        }