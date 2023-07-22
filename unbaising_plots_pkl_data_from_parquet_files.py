import numpy as np
import glob, os
import pyarrow.parquet as pq
import pyarrow as pa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import mplhep as hep
plt.style.use([hep.style.ROOT, hep.style.firamath])
import torch
from torch.utils.data import *
import pickle

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
# Define the CMS color scheme
cms_colors = [
    (0.00, '#FFFFFF'),  # White
    (0.33, '#005EB8'),  # Blue
    (0.66, '#FFDD00'),  # Yellow
    (1.00, '#FF0000')   # red
]

# Create the CMS colormap
cms_cmap = LinearSegmentedColormap.from_list('CMS', cms_colors)

out_dir_plots = "plots_plk_unbaised_data"

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

BATCH_SIZE = 160*20

train_decays = glob.glob('/storage/local/data1/gpuscratch/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To14_dataset_2_unbaised_with_unphysical/*.parquet*')
dset_train = ConcatDataset([ParquetDataset('%s'%d,i) for i,d in enumerate(train_decays)])
n_train = len(dset_train)//BATCH_SIZE * BATCH_SIZE
# n_train = 10000//BATCH_SIZE * BATCH_SIZE
print("Device used  >>> ",device)
print("Number of total event  ",len(dset_train))
print("Number of total used  ",n_train)
train_sampler = RandomSampler(dset_train, replacement=True, num_samples=n_train)
train_loader  = DataLoader(dataset=dset_train, batch_size=BATCH_SIZE, num_workers=16, pin_memory=True, sampler=train_sampler)
mass, pt = [],[]
for i, data in enumerate(train_loader):
    if i%100==0: print("Processing  event .......  %i"%i)
    mass_, pt_ = data['am'].to(device), data['apt'].to(device)
    mass.append(mass_.tolist())
    pt.append(pt_.tolist())
mass = np.concatenate(mass)
pt = np.concatenate(pt)

output_dict = {}
output_dict["mass"] = mass
output_dict["pt"] = pt
if not os.path.isdir(out_dir_plots):
        os.makedirs(out_dir_plots)

with open("%s/unbaising_data_from_parquet.pkl"%out_dir_plots, "wb") as outfile:
  pickle.dump(output_dict, outfile, protocol=2) #protocol=2 for compatibility
print("Data is saved to >>>> %s  directory"%out_dir_plots)



mass_bins =np.arange(1.2, 14.4, .4)
pt_bins = np.arange(30,185,5)


# mass pt 2D hist
fig, ax = plt.subplots(figsize=(20,15))
norm = mcolors.TwoSlopeNorm(vmin=4000, vmax = 6000, vcenter=5000)
plt.hist2d(np.squeeze(mass), np.squeeze(pt), bins=[mass_bins, pt_bins],cmap=cms_cmap,norm = norm)
plt.xticks(mass_bins,size=12)
plt.yticks(pt_bins,size=15)
plt.xlabel(r'$\mathrm{mass}$ [GeV]')
plt.ylabel(r'$\mathrm{pT}$ [GeV]')
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='r', linestyle='--', linewidth=.2)
hep.cms.label(llabel="Simulation Preliminary", rlabel="13 TeV", loc=0, ax=ax)
plt.savefig("%s/mass_VS_pt_unbiased_2D_hist.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()

 # mass distribution
fig, ax = plt.subplots(figsize=(20,20))
plt.hist(np.squeeze(mass), bins=mass_bins, histtype='step',linewidth=2, linestyle='-', color='black')
plt.xticks(mass_bins,size=15)
plt.grid(color='r', linestyle='--', linewidth=.1)
plt.xlabel('Mass [GeV]')
plt.ylabel('Events/ 0.4 [GeV]')
hep.cms.label(llabel="Simulation Preliminary", rlabel="13 TeV", loc=0, ax=ax)
plt.savefig("%s/mass_distribution_unbiased.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()


# pt distribution
fig, ax = plt.subplots(figsize=(20,20))
plt.hist(np.squeeze(pt), bins=pt_bins, histtype='step',linestyle='-', color='black', linewidth=2)
plt.xticks(pt_bins,size=15)
plt.xlabel(r'$\mathrm{pT}$ [GeV]')
plt.ylabel("Events/ 5 GeV")
hep.cms.label(llabel="Simulation Preliminary", rlabel="13 TeV", loc=0, ax=ax)
plt.savefig("%s/pt_distribution_unbiased.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()

print("Plots are saved to >>>>> %s  directory"%out_dir_plots)
