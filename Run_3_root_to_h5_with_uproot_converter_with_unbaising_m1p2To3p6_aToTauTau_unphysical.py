### in el9: source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh
import h5py
import uproot
import awkward as ak
import numpy as np
import glob, os, time
from numpy.lib.stride_tricks import as_strided
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


import argparse
parser = argparse.ArgumentParser(description='Process dataset 0-9')
parser.add_argument('-n', '--dataset',     default=0,    type=int, help='number of dataset used[0-9]')
args = parser.parse_args()
n = args.dataset
start_time = time.time()
subsets= ['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009','0010']
event_per_bin =200 # number of events you want to mass pt bins for dataset m3p6To14_dataset_2.

subset = subsets[n]
print("processing dataset --->  ", subset)
print("Number of evebt expected in each mass and pT bins %d"%event_per_bin)
decay = "IMG_aToTauTau_Hadronic_m1p2To3p6_pt30T0300_unphysical"
# local='/eos/uscms/store/group/lpcml/bbbam/Ntuples_run3/GEN_SIM_ATo2Tau_m1p2To3p6_pt30To300_v4/RHAnalyzer_ATo4Tau_Hadronic_m1p2To3p6/241108_195542/%s'%subset
local='/storage/local/data1/gpuscratch/bbbam/241108_195542/%s'%subset
# out_dir="/eos/uscms/store/user/bbbam/Run_3_IMG/%s"%decay
out_dir="/storage/local/data1/gpuscratch/bbbam/%s"%decay

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

rhFileList = '%s/final*.root'%(local)
rhFileList = glob.glob(rhFileList)
assert len(rhFileList) > 0
total_files = len(rhFileList)
print(" >> %d files found"%total_files)




if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
outStr ='%s/%s_%s.h5'%(out_dir, decay, subset)
print(" >> Output file:",outStr)





m_original_, pt_original_, m_unbaised_, pt_unbaised_ = [], [], [], []


with h5py.File(f'{outStr}', 'w') as proper_data:
        dataset_names = ['all_jet', 'am', 'ieta', 'iphi', 'm0','apt', 'jetpt']
        datasets = {
            name: proper_data.create_dataset(
                name,
                shape= (0,13, 125, 125) if 'jet' in name else (0,1),
                maxshape=(None, 13, 125, 125) if 'jet' in name else (None, 1),
                dtype='float32',  # Specify an appropriate data type
                compression='lzf',
                # chunks=(chunk_size, 13, 125, 125) if 'jet' in name else (chunk_size, 1),
            ) for name in dataset_names
        }
        end_idx = 0

        RHFile = uproot.open(rhFileList[0])
        print(" .. Processing file",rhFileList[0])
        RHTree = RHFile["fevt/RHTree"]
        branches = RHTree.arrays()
        print(f"Events in this file : {len(branches['ECAL_energy'])}")

        for i in range(len(branches["ECAL_energy"])):
            if i % 10000 == 0:
                print(" .. Processing entry",i)

            ams = ak.Array(branches["a_m"])[i]
            apts = ak.Array(branches["a_pt"])[i]

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

            ECAL_energy = np.array(branches["ECAL_energy"][i]).reshape(280,360)
            HBHE_energy_ = np.array(branches["HBHE_energy"][i]).reshape(56,72)
            HBHE_energy = upsample_array(HBHE_energy_, 5, 5) # (280, 360)
            TracksAtECAL_pt = np.array(branches["ECAL_tracksPt_atECALfixIP"][i]).reshape(280,360)
            TracksAtECAL_dZSig = np.array(branches["ECAL_tracksDzSig_atECALfixIP"][i]).reshape(280,360)
            TracksAtECAL_d0Sig = np.array(branches["ECAL_tracksD0Sig_atECALfixIP"][i]).reshape(280,360)
            PixAtEcal_1        = np.array(branches["BPIX_layer1_ECAL_atPV"][i]).reshape(280,360)
            PixAtEcal_2        = np.array(branches["BPIX_layer2_ECAL_atPV"][i]).reshape(280,360)
            PixAtEcal_3        = np.array(branches["BPIX_layer3_ECAL_atPV"][i]).reshape(280,360)
            PixAtEcal_4        = np.array(branches["BPIX_layer4_ECAL_atPV"][i]).reshape(280,360)
            TibAtEcal_1        = np.array(branches["TIB_layer1_ECAL_atPV"][i]).reshape(280,360)
            TibAtEcal_2        = np.array(branches["TIB_layer2_ECAL_atPV"][i]).reshape(280,360)
            TibAtEcal_3        = np.array(branches["TIB_layer3_ECAL_atPV"][i]).reshape(280,360)
            TibAtEcal_4        = np.array(branches["TIB_layer4_ECAL_atPV"][i]).reshape(280,360)
            TobAtEcal_1        = np.array(branches["TOB_layer1_ECAL_atPV"][i]).reshape(280,360)
            TobAtEcal_2        = np.array(branches["TOB_layer2_ECAL_atPV"][i]).reshape(280,360)
            TobAtEcal_3        = np.array(branches["TOB_layer3_ECAL_atPV"][i]).reshape(280,360)
            TobAtEcal_4        = np.array(branches["TOB_layer4_ECAL_atPV"][i]).reshape(280,360)
            TobAtEcal_5        = np.array(branches["TOB_layer5_ECAL_atPV"][i]).reshape(280,360)
            TobAtEcal_6        = np.array(branches["TOB_layer6_ECAL_atPV"][i]).reshape(280,360)
            TecAtEcal_1        = np.array(branches["TEC_layer1_ECAL_atPV"][i]).reshape(280,360)
            TecAtEcal_2        = np.array(branches["TEC_layer2_ECAL_atPV"][i]).reshape(280,360)
            TecAtEcal_3        = np.array(branches["TEC_layer3_ECAL_atPV"][i]).reshape(280,360)
            TidAtEcal_1        = np.array(branches["TID_layer1_ECAL_atPV"][i]).reshape(280,360)
            X_CMSII            = np.stack([TracksAtECAL_pt, TracksAtECAL_dZSig, TracksAtECAL_d0Sig, ECAL_energy, HBHE_energy, PixAtEcal_1, PixAtEcal_2, PixAtEcal_3, PixAtEcal_4, TibAtEcal_1, TibAtEcal_2, TobAtEcal_1, TobAtEcal_2], axis=0) # (13, 280, 360)


            # Jet attributes
            ys = ak.Array(branches["jetIsDiTau"])[i]
            iphis  = ak.Array(branches["jetSeed_iphi"])[i]
            ietas  = ak.Array(branches["jetSeed_ieta"])[i]
            m0s    = ak.Array(branches["jetM"])[i]
            jetpts = ak.Array(branches["jetPt"])[i]
            end_idx = end_idx + len(ys)

            for name, dataset in datasets.items():
                dataset.resize((end_idx,13, 125, 125) if 'jet' in name else (end_idx,1))

            for ii in range(len(ys)):
                hist_index_mass_ = np.searchsorted(mass_bins, ams[ii], side="right")-1
                hist_index_pt_ = np.searchsorted(pt_bins, apts[ii], side="right")-1
                hist_unbiased[hist_index_mass_, hist_index_pt_] += 1

                hist_unbiased_mj, edges_j = np.histogram(ams[ii], bins=m_edges)
                hist_unbiased_m += hist_unbiased_mj

                hist_unbiased_ptj, edges_j1 = np.histogram(apts[ii], bins=pt_edges)
                hist_unbiased_pt += hist_unbiased_ptj
                m_unbaised_.append(ams[ii])
                pt_unbaised_.append(apts[ii])


                proper_data['all_jet'][end_idx - len(ys) + ii, :, :, :] = crop_jet(X_CMSII, iphis[ii], ietas[ii], jet_shape=125)
                proper_data['am'][end_idx - len(ys) + ii, :] = ams[ii]
                proper_data['ieta'][end_idx - len(ys) + ii, :] = ietas[ii]
                proper_data['iphi'][end_idx - len(ys) + ii, :] = iphis[ii]
                proper_data['m0'][end_idx - len(ys) + ii, :] = m0s[ii]
                proper_data['apt'][end_idx - len(ys) + ii, :] = apts[ii]
                proper_data['jetpt'][end_idx - len(ys) + ii, :] = jetpts[ii]

end_time = time.time()
print(f"==========complete in {(end_time-start_time)/60} mins===============")
