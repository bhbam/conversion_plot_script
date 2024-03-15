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

import argparse
parser = argparse.ArgumentParser(description='Process dataset 0-9')
parser.add_argument('-m', '--mass',     default='',    type=str, help='mass of signal -> 3p7,4,5,6,8,10,12,14')
parser.add_argument('-p', '--part',     default='0',     type=str, help='0 for single parquet, 1 and 2 to make two parquet')
args = parser.parse_args()
Mass = args.mass
part = args.part

decay ="GG_H_TauTau_samples_IMG"
# decay ="GG_H_TauTau_Hadronic_background_valid"


local="/eos/uscms/store/group/lpcml/bbbam/Ntuples_v2/GGHToTauTau_background/231106_174037/0001"


out_dir="/eos/uscms/store/user/bhbam/IMG_v2/background_for_actual_signals"


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

nEvts_ =  rhTree.GetEntries()
assert nEvts_ > 0
# nEvts = 425000
nEvts = 722400 # this not limit nEvts = nJets so brak statement after saving nJets break down

if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
outStr ='%s/%s.parquet'%(out_dir, decay)
print(" >> Output file:",outStr)

if part == '1':
    iEvtStart = 0
    iEvtEnd = nEvts//2
elif part == '2':
    iEvtStart   = nEvts//2
    iEvtEnd   = nEvts
else:
    iEvtStart = 0
    iEvtEnd = nEvts
print(" >> nEvts  all  :  ",nEvts_)
assert iEvtEnd <= nEvts
print(" >> Processing entries: [",iEvtStart,"->",iEvtEnd,")")

nJets = 0
data = {}
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
    ys     = rhTree.jet_IsTau
    # ams    = rhTree.jet_M
    # apts   = rhTree.a_pt
    # dRs    = rhTree.jetadR
    # pts    = rhTree.jet_Pt
    # m0s    = rhTree.jet_M
    iphis  = rhTree.jetSeed_iphi
    ietas  = rhTree.jetSeed_ieta
    #pdgIds = rhTree.jetPdgIds
    njets  = len(ys)

    for i in range(njets):

        # data['am']    = ams[i]
        # data['apt']   = apts[i]
        data['y']     = 0.
        #data['dR']    = dRs[i]
        # data['pt']    = pts[i]
        # data['m0']    = m0s[i]
        data['iphi']  = iphis[i]
        data['ieta']  = ietas[i]
        #data['pdgId'] = pdgIds[i]
        data['X_jet'] = crop_jet(X_CMSII, data['iphi'], data['ieta']) #

        # Create pyarrow.Table

        pqdata = [pa.array([d]) if (np.isscalar(d) or type(d) == list) else pa.array([d.tolist()]) for d in data.values()]

        table = pa.Table.from_arrays(pqdata, data.keys())

        if nJets == 0:
            writer = pq.ParquetWriter(outStr, table.schema, compression='snappy')

        writer.write_table(table)

        nJets += 1
        if nJets >= 722400: # added to break after nJets written to pq files

            writer.close()



            print(" >> nJets:",nJets)
            print(" >> Real time:",sw.RealTime()/60.,"minutes")
            print(" >> CPU time: ",sw.CpuTime() /60.,"minutes")
            print("========================================================")
            break
