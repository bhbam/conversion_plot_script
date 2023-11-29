
import numpy as np
import os, glob
import time
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"]=str(0)


class ParquetDataset(Dataset):
    def __init__(self, filename, label):
        self.parquet = pq.ParquetFile(filename)
        #self.cols = None # read all columns
        #self.cols = ['X_jet.list.item.list.item.list.item','am','apt','iphi','ieta']
        self.cols = ['X_jet.list.item.list.item.list.item','y']
        self.label = label
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()

        data['y'] = np.float32(data['y'])
        data['label'] = self.label

        return dict(data)
    def __len__(self):
        return self.parquet.num_row_groups



train_decays = glob.glob('/storage/local/data1/gpuscratch/bbbam/classification/new/GGH_TauTau_train.parquet')
dset_train = ConcatDataset([ParquetDataset('%s'%d,i) for i,d in enumerate(train_decays)])

val_decays = glob.glob('//storage/local/data1/gpuscratch/bbbam/classification/new/TTT*_valid.parquet')
dset_val = ConcatDataset([ParquetDataset('%s'%d, i) for i,d in enumerate(val_decays)])

ntrain = len(dset_train)
nvalid   = len(dset_val)

print("Tatal training sample --", ntrain)
print("Tatal valid sample --", nvalid)
