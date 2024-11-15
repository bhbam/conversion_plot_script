import numpy as np
import glob, os
import pyarrow.parquet as pq
import pyarrow as pa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import *
import pandas as pd

print(torch.__version__)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
mass_range = 'm3p6To18'
decay = "IMG_aToTauTau_Hadronic_m3p6To18_pt30T0300"
pq_dir="/storage/local/data1/gpuscratch/bbbam/Run_3_IMG_unbiased/IMG_aToTauTau_Hadronic_m3p6To18_pt30T0300_unbiased"
out_dir_plots = "plots_unbiased/%s"%decay

class ParquetDataset(Dataset):
    def __init__(self, filename, label):
        self.parquet = pq.ParquetFile(filename)
        self.cols = ['am','apt']
        self.label = label
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()
        data['am'] = np.float32(data['am'])
        data['apt'] = np.float32(data['apt'])
        data['label'] = self.label
        return dict(data)
    def __len__(self):
        return self.parquet.num_row_groups

os.environ["CUDA_VISIBLE_DEVICES"]=str(0)

BATCH_SIZE = 256

# train_decays = glob.glob('/storage/local/data1/gpuscratch/bbbam/merged_IMG_aToTauTau_datase_1_reunbaied_mannul_v2_8_tracker_layer/*.parquet*')
train_decays = glob.glob(f'{pq_dir}/*0008.parquet')
dset_train = ConcatDataset([ParquetDataset('%s'%d,i) for i,d in enumerate(train_decays)])
n_train = len(dset_train)//BATCH_SIZE * BATCH_SIZE
print("Number of train data  ",n_train)
# idxs_train = np.random.permutation(len(dset_train))
# train_sampler = RandomSampler(dset_train, replacement=True, num_samples=n_train)
# train_loader  = DataLoader(dataset=dset_train, batch_size=BATCH_SIZE, num_workers=16, pin_memory=True, sampler=train_sampler)
# torch.cuda.empty_cache()
mass, pt = [],[]
for i, data in enumerate(dset_train):
    mass_, pt_ = data['am'], data['apt']
    mass.append(mass_.tolist())
    pt.append(pt_.tolist())
mass = np.concatenate(mass)
pt = np.concatenate(pt)

if not os.path.isdir(out_dir_plots):
        os.makedirs(out_dir_plots)

mass_bins = {"m1p2To3p6": np.arange(1.2, 3.7, .4), "m3p6To18": np.arange(3.6, 18.1, .4)}.get(mass_range, None)
pt_bins = np.arange(30,301,5)

plt.hist2d(np.squeeze(mass), np.squeeze(pt), bins=[mass_bins, pt_bins],cmap='bwr')
plt.title("Mass and pT distribution unbiased")
plt.xlabel(r'$\mathrm{mass}$ [GeV]')
plt.ylabel(r'$\mathrm{pT}$ [GeV]')
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='b', linestyle='--', linewidth=.2)
plt.savefig("%s/mass_VS_pt_unbiased.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()


print("Plots are saved to %s  directory"%out_dir_plots)
