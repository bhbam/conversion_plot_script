import h5py, random
import math
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os, glob
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.ROOT, hep.style.firamath])
import pickle
from matplotlib.colors import LinearSegmentedColormap


# Define the CMS color scheme
cms_colors = [
    (0.00, '#FFFFFF'),  # White
    (0.33, '#005EB8'),  # Blue
    (0.66, '#FFDD00'),  # Yellow
    (1.00, '#FF0000')   # red
]

# Create the CMS colormap
cms_cmap = LinearSegmentedColormap.from_list('CMS', cms_colors)

file =glob.glob('/eos/uscms/store/user/bhbam/Run_3_IMG_mass_reg_unphy_m0To3p6/IMG_AToTau_Hadronic_mass_reg_m0To3p6_pt30To300_normalized_combined_unbiased/*.h5')
file_ = file[0]
data = h5py.File(f'{file_}', 'r')
num_images = len(data["all_jet"])
num_images_select = num_images
out_dir = 'unphysical_massreg_plots'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
print("Total number----", num_images)

batch_size =640
am = []
apt = []




for start_idx in tqdm(range(0, num_images_select, batch_size)):
    end_idx = min(start_idx + batch_size, num_images)
    # print(sorted_indices)
    # images_batch = data["all_jet"][start_idx:end_idx, :, :, :]
    am_batch = data["am"][start_idx:end_idx, :]
    # ieta_batch = data["ieta"][start_idx:end_idx, :]
    # iphi_batch = data["iphi"][start_idx:end_idx, :]
    # m0_batch = data["m0"][start_idx:end_idx, :]
    apt_batch = data["apt"][start_idx:end_idx, :]
    # jetpt_batch = data["jetpt"][start_idx:end_idx, :]
    # taudR_batch = data["taudR"][start_idx:end_idx, :]
    am.append(am_batch)
    apt.append(apt_batch)
    # taudR.append(taudR_batch)
am = np.concatenate(am)
apt = np.concatenate(apt)
# taudR = np.concatenate(taudR)
print("Reading data Done")


mass_bins = np.arange(-1.6,18.5,.4)
pt_bins = np.arange(25,306,5)
fig, ax = plt.subplots(figsize=(20,15))
# norm = mcolors.TwoSlopeNorm(vmin=5000, vmax = 7000, vcenter=5500)
plt.hist2d(np.squeeze(am), np.squeeze(apt), bins=[mass_bins, pt_bins],cmap=cms_cmap)
plt.xlabel(r'$\mathrm{A_{mass}}$ [GeV]')
plt.ylabel(r'$\mathrm{A_{pT}}$ [GeV]')
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='r', linestyle='--', linewidth=.2)
# hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{out_dir}/mass_pt_plot.png", dpi=300, bbox_inches='tight')
plt.close()

plt.hist(np.squeeze(am), bins=mass_bins)
plt.xlabel(r'$\mathrm{A_{mass}}$ [GeV]')
# hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{out_dir}/mass_plot.png", dpi=300, bbox_inches='tight')
plt.close()

plt.hist(np.squeeze(apt), bins=pt_bins)
plt.xlabel(r'$\mathrm{A_{pT}}$ [GeV]')

# hep.cms.label(llabel="Simulation Preliminary", rlabel="13.6 TeV", loc=0, ax=ax)
plt.savefig(f"{out_dir}/pt_plot.png", dpi=300, bbox_inches='tight')
plt.close()

print("Plotting Done")
