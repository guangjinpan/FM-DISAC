import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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


class Newgenerate_Dataset(Dataset):
    def __init__(self):
        
        self.compress_dim = EnvPara["compress_dim"]
        self.len=10000
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        test_input = torch.zeros([64, 64, 4])    

        label = torch.tensor(self.original_observation_all[idx,:], dtype=torch.float32)  
        data = torch.tensor(self.original_observation_all[idx,:], dtype=torch.float32) 
        # latent_dim = random.randint(1, self.compress_dim+1) 
        # mask = torch.zeros((1, self.compress_dim))
        # mask[0, 0:latent_dim] = 1
        return  data, label #, mask[0,:].float()