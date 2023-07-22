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

decay = "IMG_aToTauTau_Hadronic_tauDR0p4_m3p6To14_dataset_1_ntuples_unbiased"
pq_dir="/eos/uscms/store/group/lpcml/bbbam/IMG/%s"%decay
out_dir_plots = "plots_original_unbiased/%s"%decay

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
train_decays = glob.glob('%s/IMG_aToTauTau_Hadronic_tauDR0p4_m3p6To14_dataset_1_ntuples_unbiased_*.parquet'%pq_dir)
dset_train = ConcatDataset([ParquetDataset('%s'%d,i) for i,d in enumerate(train_decays)])
n_train = len(dset_train)//BATCH_SIZE * BATCH_SIZE
print("Number of train data  ",n_train)
# idxs_train = np.random.permutation(len(dset_train))
train_sampler = RandomSampler(dset_train, replacement=True, num_samples=n_train)
train_loader  = DataLoader(dataset=dset_train, batch_size=BATCH_SIZE, num_workers=16, pin_memory=True, sampler=train_sampler)
torch.cuda.empty_cache()
mass, pt = [],[]
for i, data in enumerate(train_loader):
    mass_, pt_ = data['am'].cuda(), data['apt'].cuda()
    mass.append(mass_.tolist())
    pt.append(pt_.tolist())
mass = np.concatenate(mass)
pt = np.concatenate(pt)

if not os.path.isdir(out_dir_plots):
        os.makedirs(out_dir_plots)

mass_bins =np.arange(3.6, 14.4, .4)
pt_bins = np.arange(30,185,5)

plt.hist2d(np.squeeze(mass), np.squeeze(pt), bins=[np.arange(3.6, 14.4, .4), np.arange(30,185,5)],cmap='bwr')
plt.xticks(np.arange(3.6, 14.4, .4),size=4)
plt.yticks(np.arange(30,185,5),size=5)
plt.title("Mass and pT distribution unbiased")
plt.xlabel(r'$\mathrm{mass}$ [GeV]', size=10)
plt.ylabel(r'$\mathrm{pT}$ [GeV]', size=10)
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='b', linestyle='--', linewidth=.2)
plt.savefig("%s/mass_VS_pt_unbiased.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()



plt.hist(np.squeeze(mass),bins=mass_bins)
plt.xticks(np.arange(3.6, 14.4, .4),size=4)
plt.title("Mass distribution unbiased")
plt.xlabel(r'$\mathrm{mass}$ [GeV]', size=10)
plt.ylabel("Events/ 0.4 GeV")
plt.savefig("%s/mass_distribution_unbiased.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()



plt.hist(np.squeeze(pt),bins=pt_bins)
plt.xticks(np.arange(30,185,5),size=5)
plt.title("pT distribution unbiased")
plt.xlabel(r'$\mathrm{pT}$ [GeV]', size=10)
plt.ylabel("Events/ 5 GeV")
plt.savefig("%s/pt_distribution_unbiased.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()

print("Plots are saved to %s  directory"%out_dir_plots)
