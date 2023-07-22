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
import pickle
import random

import argparse
parser = argparse.ArgumentParser(description='Process dataset 0-9')
parser.add_argument('-n', '--dataset',     default=0,    type=int, help='number of dataset used')
args = parser.parse_args()
n = args.dataset

subsets= ['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009']
event_per_bins =[648,634,645,576,615,627,611,636,610,600] # number of events you wabt to mass pt bins.

subset = subsets[n]
event_per_bin = event_per_bins[n]
print("processing dataset --->  ", subset)
print("Number of evebt expected in each mass and pT bins %d"%event_per_bin)
decay = "IMG_aToTauTau_Hadronic_tauDR0p4_m1p2To3p6_dataset_2_unbaised"
local="/eos/uscms/store/group/lpcml/bbbam/Ntuples/aToTauTau_Hadronic_tauDR0p4_m3p6To16_pT30To180_ctau0To3_eta0To1p4_pythia8_unbiased4ML_dataset_2/aToTauTau_Hadronic_tauDR0p4_m3p6To14_dataset_2_unbaied_as_previous_dataset/230506_061638/%s"%subset
out_dir="/eos/uscms/store/group/lpcml/bbbam/IMG/%s"%decay
out_dir_plots = "plots_from_root_pq_unbiased_unphysical/%s/%s"%(decay, subset)

def upsample_array(x, b0, b1):

    r, c = x.shape                                    # number of rows/columns
    rs, cs = x.strides                                # row/column strides
    x = as_strided(x, (r, b0, c, b1), (rs, 0, cs, 0)) # view as a larger 4D array

    return x.reshape(r*b0, c*b1)/(b0*b1)              # create new 2D array with same total occupancy

def resample_EE(imgECAL, factor=2):

    # EE-
    imgEEm = imgECAL[:140-85] # EE- in the first 55 rows
    imgEEm = np.pad(imgEEm, ((1,0),(0,0)), 'constant', constant_values=0) # for even downsampling, zero pad 55 -> 56
    imgEEm_dn = block_reduce(imgEEm, block_size=(factor, factor), func=np.sum) # downsample by summing over [factor, factor] window
    imgEEm_dn_up = upsample_array(imgEEm_dn, factor, factor)/(factor*factor) # upsample will use same values so need to correct scale by factor**2
    imgECAL[:140-85] = imgEEm_dn_up[1:] ## replace the old EE- rows

    # EE+
    imgEEp = imgECAL[140+85:] # EE+ in the last 55 rows
    imgEEp = np.pad(imgEEp, ((0,1),(0,0)), 'constant', constant_values=0) # for even downsampling, zero pad 55 -> 56
    imgEEp_dn = block_reduce(imgEEp, block_size=(factor, factor), func=np.sum) # downsample by summing over [factor, factor] window
    imgEEp_dn_up = upsample_array(imgEEp_dn, factor, factor)/(factor*factor) # upsample will use same values so need to correct scale by factor*factor
    imgECAL[140+85:] = imgEEp_dn_up[:-1] # replace the old EE+ rows

    return imgECAL

def crop_jet(imgECAL, iphi, ieta, jet_shape=125):

    # NOTE: jet_shape here should correspond to the one used in RHAnalyzer
    off = jet_shape//2
    iphi = int(iphi*5 + 2) # 5 EB xtals per HB tower
    ieta = int(ieta*5 + 2) # 5 EB xtals per HB tower

    # Wrap-around on left side
    if iphi < off:
        diff = off-iphi
        img_crop = np.concatenate((imgECAL[:,ieta-off:ieta+off+1,-diff:],
                                   imgECAL[:,ieta-off:ieta+off+1,:iphi+off+1]), axis=-1)
    # Wrap-around on right side
    elif 360-iphi < off:
        diff = off - (360-iphi)
        img_crop = np.concatenate((imgECAL[:,ieta-off:ieta+off+1,iphi-off:],
                                   imgECAL[:,ieta-off:ieta+off+1,:diff+1]), axis=-1)
    # Nominal case
    else:
        img_crop = imgECAL[:,ieta-off:ieta+off+1,iphi-off:iphi+off+1]

    return img_crop

def unphysical_mass(m_old,old_mass_bins, new_mass_bins):
    list_idx = np.searchsorted(old_mass_bins,m_old, side="right")-1
    m_new = random.uniform(new_mass_bins[list_idx], new_mass_bins[list_idx]+0.1)
    factor = m_new/m_old
    return [m_new, factor]


old_mass_bins =np.arange(3.6,6.1, 0.1)
new_mass_bins =np.arange(1.2, 3.7, 0.1)
mass_bins_ =np.arange(3.6,14.4, .4)
mass_bins =np.arange(1.2, 3.7, .4)
pt_bins = np.arange(30,185,5)

hist_biased, xedges_, yedges_ = np.histogram2d([], [], bins=[mass_bins_, pt_bins])
hist_unbiased, xedges, yedges = np.histogram2d([], [], bins=[mass_bins, pt_bins])

hist_biased_m, m_edges_ = np.histogram([],  bins=mass_bins_)
hist_unbiased_m, m_edges = np.histogram([],  bins=mass_bins)

hist_biased_pt, pt_edges_ = np.histogram([],  bins=pt_bins)
hist_unbiased_pt, pt_edges = np.histogram([],  bins=pt_bins)

rhFileList = '%s/output*.root'%(local)
rhFileList = glob.glob(rhFileList)
assert len(rhFileList) > 0
total_files = len(rhFileList)
print " >> %d files found"%total_files


rhTree = ROOT.TChain("fevt/RHTree")
nEvts = 0
for filename in rhFileList:
  rhTree.Add(filename)

nEvts =  rhTree.GetEntries()
assert nEvts > 0
print(" >> nEvts:",nEvts)
if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
outStr ='%s/%s_%s_train_unphysical.parquet'%(out_dir, decay, subset)
print(" >> Output file:",outStr)


iEvtStart = 0
# iEvtEnd   = 10
iEvtEnd   = nEvts
assert iEvtEnd <= nEvts
print(" >> Processing entries: [",iEvtStart,"->",iEvtEnd,")")

nJets = 0
data = {} # Arrays to be written to parquet should be saved to data dict
m_original_, pt_original_, m_unphy_, pt_unphy_ = [], [], [], []


sw = ROOT.TStopwatch()
sw.Start()
for iEvt in range(iEvtStart,iEvtEnd):

    # Initialize event
    rhTree.GetEntry(iEvt)

    if iEvt % 1000 == 0:
        print(" .. Processing entry",iEvt)
    ams    = rhTree.a_m
    apts   = rhTree.a_pt
    th_mass = [6]*len(ams)
    if any(np.array(ams) >= th_mass): continue
    for j in range(len(ams)):
        m_original_.append(ams[j])
        pt_original_.append(apts[j])
        # print("m_original_",m_original_)
        hist_index_mass_ = np.searchsorted(mass_bins_, ams[j], side="right")-1
        hist_index_pt = np.searchsorted(pt_bins, apts[j], side="right")-1
        hist_biased[hist_index_mass_, hist_index_pt] += 1

        hist_biased_mi, edges_i = np.histogram(ams[j], bins=m_edges_)
        hist_biased_m += hist_biased_mi

        hist_biased_pti, edges_i1 = np.histogram(apts[j], bins=pt_edges)
        hist_biased_pt += hist_biased_pti

    events = hist_biased[hist_index_mass_, hist_index_pt]
    if events > event_per_bin: continue

    ECAL_energy = np.array(rhTree.ECAL_energy).reshape(280,360)
    ECAL_energy = resample_EE(ECAL_energy)
    HBHE_energy = np.array(rhTree.HBHE_energy).reshape(56,72)
    HBHE_energy = upsample_array(HBHE_energy, 5, 5) # (280, 360)
    TracksAtECAL_pt    = np.array(rhTree.ECAL_tracksPt_atECALfixIP).reshape(280,360)
    TracksAtECAL_dZSig = np.array(rhTree.ECAL_tracksDzSig_atECALfixIP).reshape(280,360)
    TracksAtECAL_d0Sig = np.array(rhTree.ECAL_tracksD0Sig_atECALfixIP).reshape(280,360)
    PixAtEcal_1        = np.array(rhTree.BPIX_layer1_ECAL_atPV).reshape(280,360)
    PixAtEcal_2        = np.array(rhTree.BPIX_layer2_ECAL_atPV).reshape(280,360)
    PixAtEcal_3        = np.array(rhTree.BPIX_layer3_ECAL_atPV).reshape(280,360)
    PixAtEcal_4        = np.array(rhTree.BPIX_layer4_ECAL_atPV).reshape(280,360)
    TibAtEcal_1        = np.array(rhTree.TIB_layer1_ECAL_atPV).reshape(280,360)
    TibAtEcal_2        = np.array(rhTree.TIB_layer2_ECAL_atPV).reshape(280,360)
    TobAtEcal_1        = np.array(rhTree.TOB_layer1_ECAL_atPV).reshape(280,360)
    TobAtEcal_2        = np.array(rhTree.TOB_layer2_ECAL_atPV).reshape(280,360)
    # X_CMSII            = np.stack([TracksAtECAL_pt, TracksAtECAL_dZSig, TracksAtECAL_d0Sig, ECAL_energy, HBHE_energy], axis=0) # (5, 280, 360)
    # X_CMSII            = np.stack([TracksAtECAL_pt], axis=0) # (5, 280, 360)
    X_CMSII            = np.stack([TracksAtECAL_pt, TracksAtECAL_dZSig, TracksAtECAL_d0Sig, ECAL_energy, HBHE_energy, PixAtEcal_1, PixAtEcal_2, PixAtEcal_3, PixAtEcal_4, TibAtEcal_1, TibAtEcal_2, TobAtEcal_1, TobAtEcal_2], axis=0) # (13, 280, 360)
    #data['X_CMSII'] = np.stack([TracksAtECAL_pt, ECAL_energy, HBHE_energy], axis=0) # (3, 280, 360)
    #data['X_CMSII'] = np.stack([TracksAtECAL_pt, TracksAtECAL_dz, TracksAtECAL_d0, ECAL_energy], axis=0) # (4, 280, 360)
    #data['X_CMSII'] = np.stack([TracksAtECAL_pt, TracksAtECAL_dz, TracksAtECAL_d0, ECAL_energy, HBHE_energy, PixAtEcal_1, PixAtEcal_2, PixAtEcal_3, PixAtEcal_4], axis=0) # (9, 280, 360)

    # Jet attributes
    ys     = rhTree.jetIsDiTau
    # ams    = rhTree.a_m
    # apts   = rhTree.a_pt
    #dRs    = rhTree.jetadR
    #pts    = rhTree.jetPt
    #m0s    = rhTree.jetM
    iphis  = rhTree.jetSeed_iphi
    ietas  = rhTree.jetSeed_ieta
    #pdgIds = rhTree.jetPdgIds
    njets  = len(ys)

    for i in range(njets):
        unphy_m = unphysical_mass(ams[i],old_mass_bins, new_mass_bins)
        data['am']    = unphy_m[0]
        data['apt']   = apts[i]
        # print("apts[i]", apts[i])
        # print("unphy_m[0]", unphy_m[0])
        hist_index_mass_ = np.searchsorted(mass_bins, data['am'], side="right")-1
        hist_index_pt_ = np.searchsorted(pt_bins, data['apt'], side="right")-1
        hist_unbiased[hist_index_mass_, hist_index_pt_] += 1

        hist_unbiased_mj, edges_j = np.histogram(data['am'], bins=m_edges)
        hist_unbiased_m += hist_unbiased_mj

        hist_unbiased_ptj, edges_j1 = np.histogram(data['apt'], bins=pt_edges)
        hist_unbiased_pt += hist_unbiased_ptj


        data['y']     = ys[i]
        #data['dR']    = dRs[i]
        #data['pt']    = pts[i]
        #data['m0']    = m0s[i]
        data['iphi']  = iphis[i]
        data['ieta']  = ietas[i]
        #data['pdgId'] = pdgIds[i]
        data['X_jet'] = crop_jet(X_CMSII*unphy_m[1], data['iphi'], data['ieta']) # (7, 125, 125)
        m_unphy_.append(unphy_m[0])
        pt_unphy_.append(apts[i])
        # print("m_unbaised_",m_unbaised_)
        # Create pyarrow.Table

        pqdata = [pa.array([d]) if (np.isscalar(d) or type(d) == list) else pa.array([d.tolist()]) for d in data.values()]

        table = pa.Table.from_arrays(pqdata, data.keys())

        if nJets == 0:
            writer = pq.ParquetWriter(outStr, table.schema, compression='snappy')

        writer.write_table(table)

        nJets += 1

writer.close()


if not os.path.isdir(out_dir_plots):
        os.makedirs(out_dir_plots)


output_dict_unphy = {}
output_dict_unphy["m_original"] = m_original_
output_dict_unphy["pt_original"] = pt_original_
output_dict_unphy["m_unphy"] = m_unphy_
output_dict_unphy["pt_unphy"] = pt_unphy_

del m_original_
del pt_original_
del m_unphy_
del pt_unphy_


with open("%s/dataset_%s_data_for_unphysical_plots.pkl"%(out_dir_plots, subset), "wb") as outfile2:
    pickle.dump(output_dict_unphy, outfile2, protocol=2) #protocol=2 for compatibility

plt.imshow(hist_biased.T, origin='lower', extent=[xedges_[0], xedges_[-1], yedges[0], yedges[-1]], aspect='auto', cmap='bwr')
plt.xticks(np.arange(3.6, 14.4, .4),size=4)
plt.yticks(np.arange(30,185,5),size=5)
plt.title("Mass and  pT distribution original")
plt.xlabel(r'$\mathrm{mass}$ [GeV]', size=10)
plt.ylabel(r'$\mathrm{pT}$ [GeV]', size=10)
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='r', linestyle='--', linewidth=.2)
plt.savefig("%s/mass_VS_pt_original.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()

plt.imshow(hist_unbiased.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='bwr')
plt.xticks(np.arange(1.2, 14.4, .4),size=4)
plt.yticks(np.arange(30,185,5),size=5)
plt.title("Mass and pT distribution unphysical")
plt.xlabel(r'$\mathrm{mass}$ [GeV]', size=10)
plt.ylabel(r'$\mathrm{pT}$ [GeV]', size=10)
plt.colorbar().set_label(label='Events/ (0.4,5) GeV')
plt.grid(color='b', linestyle='--', linewidth=.2)
plt.savefig("%s/mass_VS_pt_unphysical.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()

plt.bar(m_edges_[:-1], hist_biased_m, width=np.diff(m_edges_), align='edge',edgecolor='red', linewidth=1, alpha=0.7)
plt.xticks(np.arange(3.6, 14.4, .4),size=4)
plt.title("Mass distribution original")
plt.xlabel(r'$\mathrm{mass}$ [GeV]', size=10)
plt.ylabel("Events/ 0.4 GeV", size=10)
plt.savefig("%s/mass_distribution_original.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()

plt.bar(m_edges[:-1], hist_unbiased_m, width=np.diff(m_edges), align='edge',edgecolor='red', linewidth=1, alpha=0.7)
plt.xticks(np.arange(1.2, 14.4, .4),size=4)
plt.title("Mass distribution unphysical")
plt.xlabel(r'$\mathrm{mass}$ [GeV]', size=10)
plt.ylabel("Events/ 0.4 GeV", size=10)
plt.savefig("%s/mass_distribution_unphysical.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()

plt.bar(pt_edges[:-1], hist_biased_pt, width=np.diff(pt_edges), align='edge',edgecolor='red', linewidth=1, alpha=0.7)
plt.xticks(np.arange(30,185,5),size=5)
plt.title("pT distribution original")
plt.xlabel(r'$\mathrm{pT}$ [GeV]', size=10)
plt.ylabel("Events/ 5 GeV",size=10)
plt.savefig("%s/pt_distribution_original.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()

plt.bar(pt_edges[:-1], hist_unbiased_pt, width=np.diff(pt_edges), align='edge', edgecolor='red', linewidth=1, alpha=0.7)
plt.xticks(np.arange(30,185,5),size=5)
plt.title("pT distribution unbiased")
plt.xlabel(r'$\mathrm{pT}$ [GeV]', size=10)
plt.ylabel("Events/ 5 GeV",size=10)
plt.savefig("%s/pt_distribution_unbiased.png"%out_dir_plots, dpi=300, facecolor='w')
plt.close()

print(" >> nJets:",nJets)
print(" >> Real time:",sw.RealTime()/60.,"minutes")
print(" >> CPU time: ",sw.CpuTime() /60.,"minutes")
print("========================================================")

# Verify output file
pqIn = pq.ParquetFile(outStr)
print(pqIn.metadata)
print(pqIn.schema)
X = pqIn.read_row_group(0, columns=['y','am','iphi','ieta']).to_pydict()
print(X)
