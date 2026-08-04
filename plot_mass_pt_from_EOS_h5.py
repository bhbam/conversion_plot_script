import h5py
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os, glob, pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
plt.style.use([hep.style.ROOT, hep.style.firamath])
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
out_dir='../analysis_run3/AN_Note_Plot/unbising_plots_from_miniAOD_Jul_10_2026'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
save = True
dpi_ =300

infile = "/eos/uscms/store/group/lpcml/bbbam/Run_3_H5_ATo2Tau_from_miniAOD_combined_final_Jul_10_2026/IMG_ATo2Tau_from_miniAOD_m0To18_combined_train_Jul_10_2026.h5"
data = h5py.File(infile, 'r')
# print("keys--", data.keys())


A_mass = data["am"][:].flatten()
A_pt = data["apt"][:].flatten()




fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi_)
norm = mcolors.TwoSlopeNorm(vmin=0, vmax = 200, vcenter=100)
counts_test, xedges, yedges, _ = plt.hist2d(A_mass, A_pt, bins=[np.arange(0,18.1, .4), np.arange(30,301,5)], rasterized = True)
plt.xlabel(r'${A_{mass}}$ [GeV]')
plt.ylabel(r'$A_{Pt}$ [GeV]')
plt.colorbar().set_label(label='' )
hep.cms.label(llabel=f"Simulation", rlabel="13.6 TeV", loc=0, ax=ax)
if save: plt.savefig(f'{out_dir}/a_mass_pt_plot_m0To18_train.pdf', bbox_inches='tight',dpi=300, facecolor = "w")
# plt.show()

fig, ax = plt.subplots(dpi=dpi_)
plt.hist(A_mass,bins=np.arange(0,18.1, .4), rasterized = True)
plt.xlabel(r'${A_{mass}}$ [GeV]')
plt.ylabel(f'Events/ 0.4 [GeV]')
hep.cms.label(llabel=f"Simulation", rlabel="13.6 TeV", loc=0, ax=ax)
if save:plt.savefig(f'{out_dir}/a_mass_plot_m0To18_train.pdf', bbox_inches='tight',dpi=300, facecolor = "w")
# plt.show()


fig, ax = plt.subplots(dpi=dpi_)
plt.hist(A_pt,bins=np.arange(30,301, 5), rasterized = True)
plt.xlabel(r'${A_{Pt}}$ [GeV]')
plt.ylabel(f'Events/ 5 [GeV]')
hep.cms.label(llabel=f"Simulation", rlabel="13.6 TeV", loc=0, ax=ax)
if save: plt.savefig(f'{out_dir}/a_pt_plot_m0To18_train.pdf', bbox_inches='tight',dpi=300, facecolor = "w")
# plt.show()

print("min----max------mean of counts")
print(np.min(counts_test), np.max(counts_test), np.mean(counts_test))
