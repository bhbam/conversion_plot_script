### in el9: source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh
import pyarrow.parquet as pq
import pyarrow as pa # pip install pyarrow==0.7.1
import ROOT
import numpy as np
import glob, os
from numpy.lib.stride_tricks import as_strided
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


import argparse
parser = argparse.ArgumentParser(description='Process dataset 0-9')
parser.add_argument('-n', '--dataset',     default=0,    type=int, help='number of dataset used[0-9]')
args = parser.parse_args()
n = args.dataset

subsets= ['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009','0010']
event_per_bin =200 # number of events you want to mass pt bins for dataset m3p6To14_dataset_2.

subset = subsets[n]
print("processing dataset --->  ", subset)
print("Number of evebt expected in each mass and pT bins %d"%event_per_bin)
decay = "IMG_aToTauTau_Hadronic_m1p2To3p6_pt30T0300_unphysical"
local='/eos/uscms/store/group/lpcml/bbbam/Ntuples_run3/GEN_SIM_ATo2Tau_m1p2To3p6_pt30To300_v4/RHAnalyzer_ATo4Tau_Hadronic_m1p2To3p6/241108_195542/%s'%subset
out_dir="/eos/uscms/store/user/bbbam/Run_3_IMG/%s"%decay
out_dir_plots = "plots_from_root_to_pq_unbiased_m1p2To3p6_run3/%s"%subset

def upsample_array(x, b0, b1):

    r, c = x.shape                                    # number of rows/columns
    rs, cs = x.strides                                # row/column strides
    x = as_strided(x, (r, b0, c, b1), (rs, 0, cs, 0)) # view as a larger 4D array

    return x.reshape(r*b0, c*b1)/(b0*b1)              # create new 2D array with same total occupancy

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

mass_bins =np.arange(1.2, 3.7, .4)
pt_bins = np.arange(30,301,5)

hist_biased, xedges, yedges = np.histogram2d([], [], bins=[mass_bins, pt_bins])
hist_unbiased, xedges, yedges = np.histogram2d([], [], bins=[mass_bins, pt_bins])

hist_biased_m, m_edges = np.histogram([],  bins=mass_bins)
hist_unbiased_m, m_edges = np.histogram([],  bins=mass_bins)

hist_biased_pt, pt_edges = np.histogram([],  bins=pt_bins)
hist_unbiased_pt, pt_edges = np.histogram([],  bins=pt_bins)

rhFileList = '%s/output*.root'%(local)
rhFileList = glob.glob(rhFileList)
assert len(rhFileList) > 0
total_files = len(rhFileList)
print(" >> %d files found"%total_files)


rhTree = ROOT.TChain("fevt/RHTree")
nEvts = 0
for filename in rhFileList:
  rhTree.Add(filename)

nEvts =  rhTree.GetEntries()
assert nEvts > 0
print(" >> nEvts:",nEvts)
if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
outStr ='%s/%s_%s.parquet'%(out_dir, decay, subset)
print(" >> Output file:",outStr)


iEvtStart = 0
# iEvtEnd   = 100
iEvtEnd   = nEvts
assert iEvtEnd <= nEvts
print(" >> Processing entries: [",iEvtStart,"->",iEvtEnd,")")

nJets = 0
data = {} # Arrays to be written to parquet should be saved to data dict
m_original_, pt_original_, m_unbaised_, pt_unbaised_ = [], [], [], []


sw = ROOT.TStopwatch()
sw.Start()

for iEvt in range(iEvtStart,iEvtEnd):

    # Initialize event
    rhTree.GetEntry(iEvt)

    if iEvt % 10000 == 0:
        print(" .. Processing entry",iEvt)
    ams    = rhTree.a_m
    apts   = rhTree.a_pt
    for j in range(len(ams)):

        hist_index_mass = np.searchsorted(mass_bins, ams[j], side="right")-1
        hist_index_pt = np.searchsorted(pt_bins, apts[j], side="right")-1
        hist_biased[hist_index_mass, hist_index_pt] += 1

        hist_biased_mi, edges_i = np.histogram(ams[j], bins=m_edges)
        hist_biased_m += hist_biased_mi

        hist_biased_pti, edges_i1 = np.histogram(apts[j], bins=pt_edges)
        hist_biased_pt += hist_biased_pti
        m_original_.append(ams[j])
        pt_original_.append(apts[j])
    events = hist_biased[hist_index_mass, hist_index_pt]
    if events > event_per_bin: continue

    ECAL_energy = np.array(rhTree.ECAL_energy).reshape(280,360)
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
    X_CMSII            = np.stack([TracksAtECAL_pt, TracksAtECAL_dZSig, TracksAtECAL_d0Sig, ECAL_energy, HBHE_energy, PixAtEcal_1, PixAtEcal_2, PixAtEcal_3, PixAtEcal_4, TibAtEcal_1, TibAtEcal_2, TobAtEcal_1, TobAtEcal_2], axis=0) # (13, 280, 360)


    # Jet attributes
    ys     = rhTree.jetIsDiTau
    jetpts = rhTree.jetPt
    m0s    = rhTree.jetM
    iphis  = rhTree.jetSeed_iphi
    ietas  = rhTree.jetSeed_ieta
    njets  = len(ys)

    for i in range(njets):

        data['am']    = ams[i]
        data['apt']   = apts[i]

        hist_index_mass_ = np.searchsorted(mass_bins, data['am'], side="right")-1
        hist_index_pt_ = np.searchsorted(pt_bins, data['apt'], side="right")-1
        hist_unbiased[hist_index_mass_, hist_index_pt_] += 1

        hist_unbiased_mj, edges_j = np.histogram(data['am'], bins=m_edges)
        hist_unbiased_m += hist_unbiased_mj

        hist_unbiased_ptj, edges_j1 = np.histogram(data['apt'], bins=pt_edges)
        hist_unbiased_pt += hist_unbiased_ptj
        m_unbaised_.append(ams[i])
        pt_unbaised_.append(apts[i])

        # data['y']     = ys[i]
        #data['dR']    = dRs[i]
        data['jetpt']    = jetpts[i]
        data['m0']    = m0s[i]
        data['iphi']  = iphis[i]
        data['ieta']  = ietas[i]
        data['X_jet'] = crop_jet(X_CMSII, data['iphi'], data['ieta']) # (13, 125, 125)

        # Create pyarrow.Table

        pqdata = [pa.array([d]) if (np.isscalar(d) or type(d) == list) else pa.array([d.tolist()]) for d in data.values()]

        table = pa.Table.from_arrays(pqdata, list(data.keys()))

        if nJets == 0:
            writer = pq.ParquetWriter(outStr, table.schema, compression='snappy')

        writer.write_table(table)

        nJets += 1

writer.close()


if not os.path.isdir(out_dir_plots):
        os.makedirs(out_dir_plots)

output_dict = {}
output_dict["m_original"] = m_original_
output_dict["pt_original"] = pt_original_
output_dict["m_unbaised"] = m_unbaised_
output_dict["pt_unbaised"] = pt_unbaised_



with open("%s/data_for_unbaising_plots_dataset_aToTauTau_m1p2To3p6_pt30To300_%s.pkl"%(out_dir_plots, subset), "wb") as outfile:
    pickle.dump(output_dict, outfile, protocol=2) #protocol=2 for compatibility


print(" >> nJets:",nJets)
print(" >> Real time:",sw.RealTime()/60.,"minutes")
print(" >> CPU time: ",sw.CpuTime() /60.,"minutes")
print("========================================================")

# # Verify output file
# pqIn = pq.ParquetFile(outStr)
# print(pqIn.metadata)
# print(pqIn.schema)
# X = pqIn.read_row_group(0, columns=['y','am','iphi','ieta']).to_pydict()
# #X = pqIn.read_row_group(0, columns=['y','am','dR','pt','m0','iphi','ieta','pdgId']).to_pydict()
# print(X)
# #X = pqIn.read_row_group(0, columns=['X_jet.list.item.list.item.list.item']).to_pydict()['X_jet'] # read row-by-row
# #X = pqIn.read(['X_jet.list.item.list.item.list.item', 'y']).to_pydict()['X_jet'] # read entire column(s)
# #X = np.float32(X)
