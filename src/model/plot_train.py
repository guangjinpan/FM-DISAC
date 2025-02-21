import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
sys.path.append('../pre_data/')
from data_path import datapath

data_list = np.load("../pre_data/train.npy")
UElocation = np.zeros((len(data_list),2))
for i in range(len(data_list)):
    random_number = data_list[i]
    with h5py.File(datapath[0]+f"/3_{random_number}.h5py", 'r', swmr=True) as f:
        UElocation[i,:] = f["UElocation"][:][:2]

np.save("./UElocation",UElocation)