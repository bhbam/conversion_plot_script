import pyarrow.parquet as pq
import pyarrow as pa # pip install pyarrow==0.7.1
import ROOT
import numpy as np
import glob, os
from skimage.measure import block_reduce # pip install scikit-image
from numpy.lib.stride_tricks import as_strided
import multiprocessing

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
decay = "IMG_aToTauTau_Hadronic_tauDR0p4_m3p6To14_dataset_1_ntuples"
local="/eos/uscms/store/group/lpcml/bbbam/Ntuples/aToTauTau_Hadronic_tauDR0p4_m3p6To16_pT30To180_ctau0To3_eta0To1p4_pythia8_unbiased4ML_dataset_1/aToTauTau_Hadronic_tauDR0p4_m3p6To14_dataset_1_unbaise_false_ntuples/230416_184752/0000/"
out_dir_plots = "plots_from_root_unbiased/%s"%decay



rhFileList_ = '%s/output*.root'%(local)
rhFileList = glob.glob(rhFileList_)
assert len(rhFileList) > 0
total_files = len(rhFileList)
print " >> %d files found"%total_files
rhTree = ROOT.TChain("fevt/RHTree")
nEvts = 0
for filename in rhFileList:
  rhTree.Add(filename)

nEvts =  rhTree.GetEntries()
assert nEvts > 0
print(" >> Total nEvts:",nEvts)

##### MAIN #####

# Event range to process
iEvtStart = 0
# iEvtEnd   = 1001
iEvtEnd   = nEvts
assert iEvtEnd <= nEvts
print(" >> Processing entries: [",iEvtStart,"->",iEvtEnd,")")
mass, pt = [], []
nJets = 0
data = {} # Arrays to be written to parquet should be saved to data dict
try:
    for iEvt in range(iEvtStart,iEvtEnd):

        # Initialize event
        rhTree.GetEntry(iEvt)

        if iEvt % 1000 == 0:
            print(" .. Processing entry",iEvt)
        ams    = rhTree.a_m
        apts   = rhTree.a_pt
        for j in range(len(ams)):
            mass.append(ams[j])
            pt.append(apts[j])
            nJets += 1
except:
    print("Only %d event are plotted"%iEvt)
print("Number of jets  %d"%nJets)
#
if not os.path.isdir(out_dir_plots):
        os.makedirs(out_dir_plots)

mass_bins =np.arange(3.6, 14.4, .4)
pt_bins = np.arange(30,185,5)

plt.hist2d(mass, pt, bins=[np.arange(3.6, 14.4, .4), np.arange(30,185,5)],cmap='bwr')
plt.xticks(np.arange(3.6, 14.4, .4),size=4)
plt.yticks(np.arange(30,185,5),size=5)
plt.title("Mass and pT distribution original")
plt.xlabel(r'$\mathrm{mass}$ [GeV]', size=10)
plt.ylabel(r'$\mathrm{pT}$ [GeV]', size=10)
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='b', linestyle='--', linewidth=.2)
plt.savefig("%s/mass_VS_pt_originla.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()



plt.hist(mass,bins=mass_bins)
plt.xticks(np.arange(3.6, 14.4, .4),size=4)
plt.title("Mass distribution original")
plt.xlabel(r'$\mathrm{mass}$ [GeV]', size=10)
plt.ylabel("Events/ 0.4 GeV")
plt.savefig("%s/mass_distribution_original.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()



plt.hist(pt,bins=pt_bins)
plt.xticks(np.arange(30,185,5),size=5)
plt.title("pT distribution original")
plt.xlabel(r'$\mathrm{pT}$ [GeV]', size=10)
plt.ylabel("Events/ 5 GeV")
plt.savefig("%s/pt_distribution_original.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()

print("Plots are saved to %s  directory"%out_dir_plots)
