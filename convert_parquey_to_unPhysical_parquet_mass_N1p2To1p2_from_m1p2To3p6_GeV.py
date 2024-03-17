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

def unphysical_mass(m_old,old_mass_bins, new_mass_bins):
    list_idx = np.searchsorted(old_mass_bins,m_old, side="right")-1
    m_new = random.uniform(new_mass_bins[list_idx], new_mass_bins[list_idx+1])
    factor = m_new/m_old
    return [m_new, factor]

import argparse
parser = argparse.ArgumentParser(description='Process dataset 0-9')
parser.add_argument('-n', '--dataset',     default=0,    type=int, help='number of dataset used')
args = parser.parse_args()
n = args.dataset
mass_bins_original = np.arange(1.2, 3.7, 0.4)
mass_bins =np.arange(-1.2, 1.3, 0.4)
pt_bins = np.arange(30, 185, 5)
subsets= ['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009']
# event_per_bins =[648,634,645,576,615,627,611,636,610,600] # number of events you wabt to mass pt bins.

subset = subsets[n]
# event_per_bin = event_per_bins[n]
print("processing dataset --->  ", subset)
# print("Number of evebt expected in each mass and pT bins %d"%event_per_bin)

decay = "IMG_aToTauTau_Hadronic_tauDR0p4_mass_N1p2To1p2_Gev_dataset_2_unbaised_unphysical_v2"
# local="/eos/uscms/store/user/bbbam/IMG_v2/IMG_aToTauTau_Hadronic_tauDR0p4_m3p6To14p8_dataset_2_unbaised_v2/IMG_aToTauTau_Hadronic_tauDR0p4_m3p6To14p8_dataset_2_unbaised_v2_%s_train.parquet"%subset
local="/storage/local/data1/gpuscratch/bbbam/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To3p6_dataset_2_unbaised_unphysical/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To3p6_dataset_2_unbaised_unphysical_%s_train.parquet"%subset
out_dir="/storage/local/data1/gpuscratch/bbbam/%s"%decay
out_dir_plots = "plots_from_mass_mN1p2To1p2_Gev_unbiased_unphysical_pq_to_pq/%s"%(subset)

class ParquetDatasetTable(Dataset):
    def __init__(self, filename, cols=None):
        self.parquet = pq.ParquetFile(filename)
        self.cols = cols # read all columns
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols)
        return data
    def __len__(self):
        return self.parquet.num_row_groups

# files_ = "IMG_aToTauTau_Hadronic_tauDR0p4_m14To17p2_dataset_2_unbaised_0006_train.parquet"
fs = glob.glob(local)
assert len(fs) > 0
print(" >> %d files found"%len(fs))


dset = ConcatDataset([ParquetDatasetTable(f) for f in fs])
nevts_in = len(dset)
print('>> Input events:',nevts_in)
idxs = np.random.permutation(nevts_in)

start, stop = 0, nevts_in
# start, stop = 0, 1000


if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

if not os.path.isdir(out_dir_plots):
        os.makedirs(out_dir_plots)

file_str = "%s/IMG_aToTauTau_Hadronic_tauDR0p4_mass_N1p2To1p2_Gev_dataset_2_unbaised_%s_train_unphysical_v2.parquet"%(out_dir,subset)
print('>> Doing sample:',file_str)
print('>> Output events more than : %d [ %d, %d ]'%((stop-start), start, stop))


m_original = []
pt_original = []
m_unphy = []
pt_unphy = []




# Write out the actual data
print('>> Writing data...')
table = True

for i, idx in enumerate(idxs[start:stop]):
    if i%10000 == 0:
        print(' >> Processed event:',i)
    t = dset.__getitem__(idx)
    mass_ = float(np.squeeze(t["am"]))
    pt_ = float(np.squeeze(t["apt"]))
    m_original.append(mass_)
    pt_original.append(pt_)
    if (mass_ > 3.6 or mass_ < 1.2): continue
    unpymass = unphysical_mass(mass_, mass_bins_original, mass_bins)
    data = {}
    data['X_jet'] = np.squeeze(t["X_jet"])
    data['am']    = unpymass[0]
    data['apt']   = float(np.squeeze(t["apt"]))
    data['ieta']  = float(np.squeeze(t["ieta"]))
    data['iphi']  = float(np.squeeze(t["iphi"]))
    data['y']     = float(np.squeeze(t["y"]))
    pqdata = [pa.array([d]) if (np.isscalar(d) or type(d) == list) else pa.array([d.tolist()]) for d in data.values()]
    t = pa.Table.from_arrays(pqdata, list(data.keys()))
    if table:
        # print("t.schema",t.schema)
        writer = pq.ParquetWriter(file_str, t.schema, compression='snappy')
        table = False
    writer.write_table(t)
    m_unphy.append(data['am'])
    pt_unphy.append(data['apt'])
writer.close()




output_dict_unphy = {}
output_dict_unphy["m_original"] = m_original
output_dict_unphy["pt_original"] = pt_original
output_dict_unphy["m_unphy"] = m_unphy
output_dict_unphy["pt_unphy"] = pt_unphy

with open("%s/dataset_data_for_unphysical_plots_subset_%s.pkl"%(out_dir_plots, subset), "wb") as outfile2:
    pickle.dump(output_dict_unphy, outfile2, protocol=2) #protocol=2 for compatibility


fig, ax = plt.subplots(figsize=(20,15))
norm = mcolors.TwoSlopeNorm(vmin=500, vmax = 700, vcenter=600)
plt.hist2d(np.squeeze(m_original), np.squeeze(pt_original), bins=[mass_bins, pt_bins],cmap=cms_cmap, norm=norm)
plt.xticks(mass_bins_original,size=12)
plt.yticks(pt_bins,size=15)
plt.xlabel(r'$\mathrm{mass}$ [GeV]')
plt.ylabel(r'$\mathrm{pT}$ [GeV]')
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='r', linestyle='--', linewidth=.2)
hep.cms.label(llabel="Simulation Preliminary", rlabel="13 TeV", loc=0, ax=ax)
plt.savefig('%s/mass_pt_2D_hist_original_unbaised_subset_%s.png'%(out_dir_plots, subset),  bbox_inches='tight', dpi=300, facecolor = "w")


fig, ax = plt.subplots(figsize=(20,15))
norm = mcolors.TwoSlopeNorm(vmin=500, vmax = 700, vcenter=600)
plt.hist2d(np.squeeze(m_unphy), np.squeeze(pt_unphy), bins=[mass_bins, pt_bins],cmap=cms_cmap, norm=norm)
plt.xticks(mass_bins,size=12)
plt.yticks(pt_bins,size=15)
plt.xlabel(r'$\mathrm{mass}$ [GeV]')
plt.ylabel(r'$\mathrm{pT}$ [GeV]')
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='r', linestyle='--', linewidth=.2)
hep.cms.label(llabel="Simulation Preliminary", rlabel="13 TeV", loc=0, ax=ax)
plt.savefig('%s/mass_pt_2D_hist_unbaised_unphysical_subset_%s.png'%(out_dir_plots,subset),  bbox_inches='tight', dpi=300, facecolor = "w")
