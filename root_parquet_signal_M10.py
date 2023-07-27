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


decay = "IMG_H_AATo4Tau_Hadronic_tauDR0p4_M10_signal"
local="/eos/uscms/store/group/lpcml/bbbam/Ntuples/gen_HToAATo4Tau_Hadronic_tauDR0p4_M10_ctau0To3_eta0To2p4_pythia8_2018UL/HtoAATo4Tau_Hadronic_tauDR0p4_M10_eta0To2p4_pythia8_signal/230725_041228/0000"
out_dir="/eos/uscms/store/user/bbbam/IMG/%s"%decay
out_dir_plots = "plot_signal_image_data_pkl"

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

mass_bins =np.arange(3.6, 14.4, .4)
pt_bins = np.arange(30,185,5)


rhFileList = '%s/output*.root'%(local)
rhFileList = glob.glob(rhFileList)
assert len(rhFileList) > 0
total_files = len(rhFileList)
print (" >> %d files found")%total_files


rhTree = ROOT.TChain("fevt/RHTree")
nEvts = 0
for filename in rhFileList:
  rhTree.Add(filename)

nEvts =  rhTree.GetEntries()
assert nEvts > 0
print(" >> nEvts:",nEvts)
if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
outStr ='%s/%s.parquet'%(out_dir, decay)
print(" >> Output file:",outStr)


iEvtStart = 0
# iEvtEnd   = 1001
iEvtEnd   = nEvts
assert iEvtEnd <= nEvts
print(" >> Processing entries: [",iEvtStart,"->",iEvtEnd,")")

nJets = 0
data = {} # Arrays to be written to parquet should be saved to data dict
a_mass_, a_pt_, jet_mass_, jet_pt_ = [], [], [], []


sw = ROOT.TStopwatch()
sw.Start()
for iEvt in range(iEvtStart,iEvtEnd):

    # Initialize event
    rhTree.GetEntry(iEvt)

    if iEvt % 1000 == 0:
        print(" .. Processing entry",iEvt)

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
    ams    = rhTree.a_m
    apts   = rhTree.a_pt
    # dRs    = rhTree.jetadR
    pts    = rhTree.jetPt
    m0s    = rhTree.jetM
    iphis  = rhTree.jetSeed_iphi
    ietas  = rhTree.jetSeed_ieta
    #pdgIds = rhTree.jetPdgIds
    njets  = len(ys)

    for i in range(njets):

        data['am']    = ams[i]
        data['apt']   = apts[i]
        data['y']     = ys[i]
        #data['dR']    = dRs[i]
        data['pt']    = pts[i]
        data['m0']    = m0s[i]
        data['iphi']  = iphis[i]
        data['ieta']  = ietas[i]
        #data['pdgId'] = pdgIds[i]
        a_mass_.append(ams[i])
        a_pt_.append(apts[i])
        jet_mass_.append(m0s[i])
        jet_pt_.append(pts[i])
        data['X_jet'] = crop_jet(X_CMSII, data['iphi'], data['ieta']) # (7, 125, 125)

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

output_dict = {}
output_dict["a_mass"] = a_mass_
output_dict["a_pt"] = a_pt_
output_dict["jet_mass"] = jet_mass_
output_dict["jet_pt"] = jet_pt_

with open("%s/data_for_validation_H_AATo4Tau_hadronic_dR0p4_M10_a_jet_mass_pt.pkl"%(out_dir_plots), "wb") as outfile:
    pickle.dump(output_dict, outfile, protocol=2) #protocol=2 for compatibility

print(" >> nJets:",nJets)
print(" >> Real time:",sw.RealTime()/60.,"minutes")
print(" >> CPU time: ",sw.CpuTime() /60.,"minutes")
print("========================================================")
