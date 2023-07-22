import numpy as np
import os, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
plt.style.use([hep.style.ROOT, hep.style.firamath])
import pickle
import gzip

old_mass_bins =np.arange(3.6,6.1, .4)
mass_bins =np.arange(1.2, 3.7, .4)
pt_bins = np.arange(30,185,5)
m_bins = np.arange(1.2,6.1, .4)


with open('/Users/bhimbam/Desktop/test_plots/plots_from_root_pq_unbiased_unphysical/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To3p6_dataset_2_unbaised/0000/dataset_0000_data_for_unphysical_plots.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    data = u.load()

# data_file = "/Users/bhimbam/Desktop/test_plots/plots_from_root_pq_unbiased_unphysical/IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To3p6_dataset_2_unbaised/0000/dataset_0000_data_for_unphysical_plots.pkl"
# infile2= open(f"data_file", "rb")
# data = pickle.load(infile)
# infile.close()

m_or = data["m_original"]
m_un = data["m_unphy"]
pt_or = data["pt_original"]
pt_un = data["pt_unphy"]
m = m_or+m_un
pt = pt_or+pt_un

out_dir = "plots_for_unphysical_from_pkl/dataset_0000"
if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

fig, ax = plt.subplots(figsize=(35,25), dpi=70)
plt.hist2d(m_or, pt_or, bins=[old_mass_bins, pt_bins],cmap='bwr')
plt.xticks(np.arange(3.6,14.4, .4),size=20)
plt.yticks(np.arange(30,185,5),size=20)
plt.title("Mass and  pT distribution original")
plt.xlabel(r'$\mathrm{mass}$ [GeV]', size=30)
plt.ylabel(r'$\mathrm{pT}$ [GeV]', size=30)
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='r', linestyle='--', linewidth=.2)
hep.cms.label(llabel="Simulation Preliminary", rlabel="13 TeV", loc=0, ax=ax)
plt.savefig('%s/m_original_pt_2D_hist.png'%(out_dir),  bbox_inches='tight', dpi=300, facecolor = "w")
plt.close()

fig, ax = plt.subplots(figsize=(35,25), dpi=70)
norm = mcolors.TwoSlopeNorm(vmin=500, vmax = 700, vcenter=600)
plt.hist2d(m_un, pt_un, bins=[mass_bins, pt_bins],cmap='bwr',norm = norm)
plt.xticks(np.arange(1.2, 14.4, .4),size=18)
plt.yticks(np.arange(30,185,5),size=20)
plt.title("Mass and  pT distribution unphysical")
plt.xlabel(r'$\mathrm{mass}$ [GeV]', size=30)
plt.ylabel(r'$\mathrm{pT}$ [GeV]', size=30)
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='r', linestyle='--', linewidth=.2)
hep.cms.label(llabel="Simulation Preliminary", rlabel="13 TeV", loc=0, ax=ax)
plt.savefig('%s/m_unphysical_pt_2D_hist.png'%(out_dir),  bbox_inches='tight', dpi=300, facecolor = "w")
plt.close()

fig, ax = plt.subplots(figsize=(35,25), dpi=70)
plt.hist2d(m, pt, bins=[m_bins, pt_bins],cmap='bwr')
plt.xticks(np.arange(1.2, 14.4, .4),size=18)
plt.yticks(np.arange(30,185,5),size=20)
plt.title("Mass and  pT distribution unphysical")
plt.xlabel(r'$\mathrm{mass}$ [GeV]', size=30)
plt.ylabel(r'$\mathrm{pT}$ [GeV]', size=30)
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='r', linestyle='--', linewidth=.2)
hep.cms.label(llabel="Simulation Preliminary", rlabel="13 TeV", loc=0, ax=ax)
plt.savefig('%s/combined_m_pt_2D_hist.png'%(out_dir),  bbox_inches='tight', dpi=300, facecolor = "w")
plt.close()

fig, ax = plt.subplots(figsize=(35,25), dpi=150)
plt.hist(m_or, bins=np.arange(3.6, 6.1, .4), histtype='step',linewidth=2, label='$\mathrm{m_{true}}$', linestyle='-', color='blue')
plt.xticks(np.arange(3.6,6.1, .4))
plt.grid(color='r', linestyle='--', linewidth=.1)
plt.xlabel('Mass [GeV]')
plt.ylabel('Events/ 0.4 [GeV]')
plt.title(r'$\mathrm{m_{true}}$ masses distribution')
plt.legend(loc='upper right')
hep.cms.label(llabel="Simulation Preliminary", rlabel="13 TeV", loc=0, ax=ax)
plt.savefig('%s/m_original_distribution_hist.png'%(out_dir), bbox_inches='tight',dpi=300, facecolor = "w")
plt.close()

fig, ax = plt.subplots(figsize=(35,25), dpi=150)
plt.hist(m_un, bins=np.arange(1.2, 3.7, .4), histtype='step',linewidth=2, label='$\mathrm{m_{unphy}}$', linestyle='-', color='red')
plt.xticks(np.arange(1.2,3.7, .4))
plt.grid(color='r', linestyle='--', linewidth=.1)
plt.xlabel('Mass [GeV]')
plt.ylabel('Events/ 0.4 [GeV]')
plt.title(r'$\mathrm{m_{unphy}}$ mass distribution')
plt.legend(loc='upper right')
hep.cms.label(llabel="Simulation Preliminary", rlabel="13 TeV", loc=0, ax=ax)
plt.savefig('%s/m_unphysical_distribution_hist.png'%(out_dir), bbox_inches='tight',dpi=300, facecolor = "w")
plt.close()

fig, ax = plt.subplots(figsize=(35,25), dpi=150)
hist, bins, _ = plt.hist(m, bins=np.arange(1.2, 6.1, .4), histtype='step',linewidth=2, linestyle='-', color='blue')
plt.xticks(np.arange(1.2,6.1,.4))
fill_bins = np.arange(1.2,3.3,.4)
for i in range(len(fill_bins)):
    if i==0:
        plt.fill_betweenx([0, hist[i]], bins[i], bins[i+1], color="red", alpha=0.1,linewidth=.1, label="unphysical region")
    else:
        plt.fill_betweenx([0, hist[i]], bins[i], bins[i+1], color="red", alpha=0.1, linewidth=.1)
plt.grid(color='r', linestyle='--', linewidth=.1)
plt.xlabel('Mass [GeV]')
plt.ylabel('Events/ 0.4 [GeV]')
plt.title(r' mass distribution')
hep.cms.label(llabel="Simulation Preliminary", rlabel="13 TeV", loc=0, ax=ax)
plt.legend(loc='upper left',fontsize='large')
plt.savefig('%s/combined_mass_distribution_hist.png'%(out_dir), bbox_inches='tight',dpi=300, facecolor = "w")
plt.close()

fig, ax = plt.subplots(figsize=(35,25), dpi=150)
plt.hist(pt_or, bins=np.arange(30,185,5), label=r'$\mathrm{pt_{true}}$', linestyle='-', color='blue',alpha=0.1)
plt.hist(pt_un, bins=np.arange(30,185,5), histtype='step',linewidth=3, label=r'$\mathrm{pt_{unphy}}$', linestyle='-', color='red')
plt.xticks(np.arange(30,185,5))
plt.xlabel(r'$\mathrm{pT}$ [GeV]')
plt.ylabel("Events/ 5 GeV")
plt.title(r'$\mathrm{pt_{true}}$, $\mathrm{pt_{unphy}}$')
plt.legend(loc='upper right')
hep.cms.label(llabel="Simulation Preliminary", rlabel="13 TeV", loc=0, ax=ax)
plt.savefig("%s/pt_original_unphysical_hist.png"%(out_dir), dpi=300, facecolor='w')
plt.close()

print(">>>>>>>>>>>>>>>>>>All plots are saved to %s"%(out_dir))
