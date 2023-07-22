import os, glob, time, torch, pickle, random
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset, ConcatDataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
matplotlib.use('Agg')
import mplhep as hep
plt.style.use([hep.style.ROOT, hep.style.firamath])
from matplotlib.colors import LinearSegmentedColormap
import pickle
# Define the CMS color scheme
cms_colors = [
    (0.00, '#FFFFFF'),  # White
    (0.33, '#005EB8'),  # Blue
    (0.66, '#FFDD00'),  # Yellow
    (1.00, '#FF0000')   # red
]

# Create the CMS colormap
cms_cmap = LinearSegmentedColormap.from_list('CMS', cms_colors)

import argparse
parser = argparse.ArgumentParser(description='Process dataset 0-9')
parser.add_argument('-n', '--dataset',     default=0,    type=int, help='number of dataset used  [0-9]')
args = parser.parse_args()
n = args.dataset
mass_bins =np.arange(14, 25, .4)
pt_bins = np.arange(30,185,5)
subsets= ['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009']
event_per_bins =[648,634,645,576,615,627,611,636,610,600] # number of events you wabt to mass pt bins.

subset = subsets[n]
print("processing dataset --->  ", subset)

local="/eos/uscms/store/group/lpcml/bbbam/IMG/IMG_aToTauTau_Hadronic_tauDR0p4_mass_above_17p2_Gev_dataset_2_unbaised_unphysical/IMG_aToTauTau_Hadronic_tauDR0p4_mass_above_17p2_Gev_dataset_2_unbaised_%s_train_unphysical.parquet"%subset
out_dir_plots = "plots_from_mass_above_17p2_Gev_unbiased_unphysical_from_pq/%s"%(subset)

fs = glob.glob(local)
assert len(fs) > 0
print(" >> %d files found"%len(fs))

class ParquetDatasetTable(Dataset):
    def __init__(self, filename, cols=None):
        self.parquet = pq.ParquetFile(filename)
        self.cols = cols # read all columns
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols)
        return data
    def __len__(self):
        return self.parquet.num_row_groups

dset = ConcatDataset([ParquetDatasetTable(f) for f in fs])
nevts_in = len(dset)
print('>> Input events:',nevts_in)
idxs = np.random.permutation(nevts_in)

start, stop = 0, nevts_in
# start, stop = 0, 1000

if not os.path.isdir(out_dir_plots):
        os.makedirs(out_dir_plots)

m_unphy = []
pt_unphy = []

print("Reading data for plots ...")
for i, idx in enumerate(idxs[start:stop]):
    if i%10000 == 0:
        print(' >> Reading event:',i)
    t = dset.__getitem__(idx)
    m_unphy.append(t["am"])
    pt_unphy.append(t["apt"])

fig, ax = plt.subplots(figsize=(20,15))
norm = mcolors.TwoSlopeNorm(vmin=500, vmax = 700, vcenter=600)
hist = plt.hist2d(np.squeeze(m_unphy), np.squeeze(pt_unphy), bins=[mass_bins, pt_bins],cmap=cms_cmap)
plt.xticks(mass_bins,size=12)
plt.yticks(pt_bins,size=15)
plt.xlabel(r'$\mathrm{mass}$ [GeV]')
plt.ylabel(r'$\mathrm{pT}$ [GeV]')
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='r', linestyle='--', linewidth=.2)
hep.cms.label(llabel="Simulation Preliminary", rlabel="13 TeV", loc=0, ax=ax)
plt.savefig('%s/mass_pt_2D_hist_original_unbaised_subset_%s.png'%(out_dir_plots, subset),  bbox_inches='tight', dpi=300, facecolor = "w")

output_dict_unphy = {}
output_dict_unphy["m_unphy"] = m_unphy
output_dict_unphy["pt_unphy"] = pt_unphy

with open("%s/dataset_data_for_unphysical_plots_subset_%s.pkl"%(out_dir_plots, subset), "wb") as outfile2:
    pickle.dump(output_dict_unphy, outfile2, protocol=2) #protocol=2 for compatibility
