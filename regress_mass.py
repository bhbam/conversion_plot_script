from __future__ import print_function
import numpy as np
np.random.seed(0)
import os, glob
import time
import argparse
import json
from collections import OrderedDict

#'''
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
import torch.nn as nn
#os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
#'''

import ROOT
from array import array

# Register command line options
parser = argparse.ArgumentParser(description='Run STEALTH selection.')
parser.add_argument('-s', '--sample', default='test', type=str, help='Sample name.')
parser.add_argument('-i', '--img_list', default='img_inputs.txt', type=str, help='List file of imgNtuple inputs.')
parser.add_argument('-g', '--gg_list', default='gg_inputs.txt', type=str, help='List file of ggNtuple inputs.')
parser.add_argument('-m', '--model', default='Models/model_epoch80_mae0.1906.pkl', type=str, help='Regressor model file.')
parser.add_argument('-o', '--outdir', default='MAntuples', type=str, help='Output directory.')
parser.add_argument('-l', '--log_file', default='log.txt', type=str, help='Log file.')
args = parser.parse_args()

sample = args.sample
img_list = args.img_list
gg_list = args.gg_list
model = args.model
outdir = args.outdir
log_file = args.log_file

flog = open(log_file, 'w+')
def logger(log_file, log):
    print(log)
    log_file.write('%s\n'%(log))

logger(flog, '>> Sample: %s'%sample)
logger(flog, '>> Model: %s'%model)
logger(flog, '>> Input imgNtuple list: %s'%img_list)
logger(flog, '>> Input ggSkim list: %s'%gg_list)
logger(flog, '>> Output maNtuple dir: %s'%outdir)
logger(flog, '>> Log file: %s'%log_file)

# Crop out EB shower from full EB image
def crop_EBshower(imgEB, ieta, iphi, window=32):

    # NOTE: image window here should correspond to the one used in RHAnalyzer
    off = window//2
    ieta = int(ieta[0])+1 # seed positioned at [15,15]
    iphi = int(iphi[0])+1 # seed positioned at [15,15]

    # Wrap-around on left side
    if iphi < off:
        diff = off-iphi
        img_crop = np.concatenate((imgEB[:,ieta-off:ieta+off,-diff:],
                                   imgEB[:,ieta-off:ieta+off,:iphi+off]), axis=-1)
    # Wrap-around on right side
    elif 360-iphi < off:
        diff = off - (360-iphi)
        img_crop = np.concatenate((imgEB[:,ieta-off:ieta+off,iphi-off:],
                                   imgEB[:,ieta-off:ieta+off,:diff]), axis=-1)
    # Nominal case
    else:
        img_crop = imgEB[:,ieta-off:ieta+off,iphi-off:iphi+off]

    return img_crop

# Crop out EB shower from full EB image with padding
def crop_EBshower_padded(imgEB, ieta, iphi, window=32):

    assert len(imgEB.shape) == 3, '!! len(imgEB.shape): %d != 3'%len(imgEB.shape)
    assert ieta[0] < imgEB.shape[1], '!! ieta:%d !< imgEB.shape[1]:%d'%(ieta[0], imgEB.shape[1])
    assert iphi[0] < imgEB.shape[2], '!! iphi:%d !< imgEB.shape[2]:%d'%(iphi[0], imgEB.shape[2])

    # NOTE: image window here should correspond to the one used in RHAnalyzer
    off = window//2
    ieta = int(ieta[0])+1 # seed positioned at [15,15]
    iphi = int(iphi[0])+1 # seed positioned at [15,15]

    # ------------------------------------------------
    # ieta (row) padding
    # ------------------------------------------------
    pad_lo, pad_hi = 0, 0
    # lower padding check
    if ieta >= off:
        ieta_lo = ieta-off
    else:
        pad_lo = abs(ieta-off)
        ieta_lo = 0
    # upper padding check
    if ieta+off <= imgEB.shape[1]:
        ieta_hi = ieta+off
    else:
        pad_hi = abs(ieta+off-imgEB.shape[1])
        ieta_hi = imgEB.shape[1]

    # ------------------------------------------------
    # iphi (col) wrap-around
    # ------------------------------------------------
    # Wrap-around on left side
    if iphi < off:
        diff = off-iphi
        img_crop_ = np.concatenate((imgEB[:, ieta_lo:ieta_hi, -diff:],
                                    imgEB[:, ieta_lo:ieta_hi, :iphi+off]), axis=-1)
    # Wrap-around on right side
    elif 360-iphi < off:
        diff = off - (360-iphi)
        img_crop_ = np.concatenate((imgEB[:, ieta_lo:ieta_hi, iphi-off:],
                                    imgEB[:, ieta_lo:ieta_hi, :diff]), axis=-1)
    # Nominal case
    else:
        img_crop_ = imgEB[:, ieta_lo:ieta_hi, iphi-off:iphi+off]

    # Add ieta padding if needed
    img_crop = np.pad(img_crop_, ((0,0), (pad_lo, pad_hi), (0,0)), 'constant') # pads with 0
    assert img_crop.shape[1] == window, '!! img_crop.shape[1]:%d != window:%d'%(img_crop.shape[1], window)
    assert img_crop.shape[2] == window, '!! img_crop.shape[2]:%d != window:%d'%(img_crop.shape[2], window)

    return img_crop

def rotate(imgEB, window=32):

    imgEB = np.float32(imgEB)
    assert len(imgEB.shape) == 3, '!! len(imgEB.shape): %d != 3'%len(imgEB.shape)
    assert imgEB.shape[1] == window
    assert imgEB.shape[2] == window

    #print('pre rotation:')
    #print(np.argwhere(imgEB == imgEB.max()))

    rot   = np.rot90(imgEB, axes=(1,2))
    shift = np.pad(rot, ((0,0),(0,1),(0,0)), 'constant')
    crop  = shift[:,1:,:]

    #print('post rotation:')
    #print(np.argwhere(crop == imgEB.max()))
    assert crop.shape[1] == window
    assert crop.shape[2] == window
    return crop

# Create an event list containing run:lumi:event:idx IDs for some input `tree`
# Defaults to ggNtuple so events can be skipped if not present here.
# NOTE: event id conventions for the ntuples are as follows:
# IMG -> runId:lumiId:eventId
# GG -> run:lumis:event
def get_evtlist(tree, is_ggntuple=True):
    nEvts = tree.GetEntries()
    idxs = []
    for iEvt in range(nEvts):
        # Initialize event
        tree.GetEntry(iEvt)
        if iEvt%100e3==0: logger(flog, '%d / %d'%(iEvt, nEvts))
        if is_ggntuple:
            eventId = [tree.run, tree.lumis, tree.event, iEvt]
        else:
            eventId = [tree.runId, tree.lumiId, tree.eventId, iEvt]
        idxs.append(eventId)
    # Array index on 1st element should be 0, and nEvts-1 on last element
    idxs = np.array(idxs)
    assert idxs[0,-1] == 0 and idxs[-1,-1] == nEvts-1
    return idxs

# Return the array index in `gg_evtlist` corresponding to event loaded in `img_tree`
# Each event uniquely identified by run, lumi, event no.
# Filter first by run then lumi before looking for event no.
# If evt not found, returns `-1` to trigger TTree event status == bad
def idx_where_run_lumi_evt(img_tree, gg_evtlist):

    # `eventlistf` must have shape (nevts,4) where
    # [:,0]: run
    # [:,1]: lumi
    # [:,2]: event
    # [:,3]: idx
    assert gg_evtlist.shape[-1] == 4

    # This is the target event ID from the main IMG TTree
    evtid = np.array([img_tree.runId, img_tree.lumiId, img_tree.eventId])

    # Find the index in gg_evtlistf corresponding to target img event
    # Filter by run
    iruns = np.argwhere(gg_evtlist[:,0] == evtid[0]).flatten()
    gg_evtlist = gg_evtlist[iruns]
    if len(gg_evtlist) == 0:
        return -1
    # Filter by lumi
    ilumis = np.argwhere(gg_evtlist[:,1] == evtid[1]).flatten()
    gg_evtlist = gg_evtlist[ilumis]
    if len(gg_evtlist) == 0:
        return -1
    # Filter by event
    ievts = np.argwhere(gg_evtlist[:,2] == evtid[2]).flatten()
    gg_evtlist = gg_evtlist[ievts]

    assert len(gg_evtlist) <= 1, '!! More than one evt match found!'

    if len(gg_evtlist) == 0:
        return -1
    else:
        return gg_evtlist.flatten()[-1]

# Load IMG ntuples as main TTree
logger(flog, '>> Setting img inputs as primary tree')
# Read in file list
logger(flog, '   .. opening img input list')
img_inputs = []
with open(img_list, 'r') as img_files:
    for f in img_files:
        img_inputs.append(f.strip('\n'))
logger(flog, '   .. found %d files'%len(img_inputs))
assert len(img_inputs) > 0
logger(flog, '   .. [ 0]: %s'%img_inputs[0])
logger(flog, '   .. [-1]: %s'%img_inputs[-1])
# Load files into tree
img_tree = ROOT.TChain("fevt/RHTree")
for f in img_inputs:
    img_tree.Add(f)
    #break
nEvts = img_tree.GetEntries()
logger(flog, '   .. Nevts: %d'%nEvts)

# Load ggNtuples as secondary tree
logger(flog, '>> Setting gg inputs as secondary tree')
# Read in file list
logger(flog, '   .. opening gg input list')
gg_inputs = []
with open(gg_list, 'r') as gg_files:
    for f in gg_files:
        gg_inputs.append(f.strip('\n'))
logger(flog, '   .. found %d files'%len(gg_inputs))
assert len(gg_inputs) > 0
logger(flog, '   .. [ 0]: %s'%gg_inputs[0])
logger(flog, '   .. [-1]: %s'%gg_inputs[-1])
# Load files into tree
gg_tree = ROOT.TChain('ggNtuplizer/EventTree')
for f in gg_inputs:
    gg_tree.Add(f)
nEvts_gg = gg_tree.GetEntries()
logger(flog, '   .. Nevts: %d'%nEvts_gg)
logger(flog, '   >> Collecting event indices from gg ntuples...')
gg_evtlist = get_evtlist(gg_tree)
logger(flog, '      .. done')

# Initialize output ntuple
# First, clone ggNtuple directory+variables
#if not os.path.isdir(outdir):
#    os.makedirs(outdir)
#file_out = ROOT.TFile("%s/%s_mantuple.root"%(outdir, sample), "RECREATE")
outpath = "%s/%s_mantuple.root"%(outdir, sample)
logger(flog, '>> Initializing output mantuple: %s'%outpath)
file_out = ROOT.TFile.Open(outpath, 'RECREATE')
file_out.mkdir("ggNtuplizer")
file_out.cd("ggNtuplizer")
tree_out = gg_tree.CloneTree(0)
#logger(flog, list(tree_out.GetListOfBranches()))

# Initialize add-on `ma` branch:
# In PyROOT, to declare a branch `ma` with variable size nPho(evt_i),
# need to declare a separate branch `nPho` whose value at fill time
# is the desired (variable) size of `ma`. The cloned ggNtuple already
# has such a branch `nPho` (otherwise would need to be created explicitly).
# !! NOTE: at fill time, need to check that ggtree.nPho corresponds to
# no. of presumed photon objects to be regressed, i.e.:
# ggtree.nPho == len(imgtree.SC_ieta) or len(imgtree.SC_iphi)
# The `ma` branch still needs to be initialized with an array of some
# arbitrarily large allocation `nPhoMax` which then gets reduced at fill time.
nPhoMax = 10
ma = np.zeros(nPhoMax, dtype='float32')
tree_out.Branch('ma', ma, 'ma[nPho]/F')
logger(flog, '   .. done.')

'''
# Load mass regressor model
logger(flog, '>> Loading model file...')
import torch_resnet_concat as networks
#epoch = 80
#neg_mass = 300
## AOD
#model_file = glob.glob('../EB_Pi0_massreg_Pt20To100_2017PU_ext3_tzfixed_wrapfix/MODELS/DoublePi0Pt20To100_m0To1600_pythia8_PU2017_genDR10_aodDR16_nPhoN_PhoNeg%dTo0_wgts_EBtzo25_AOD_m0o1.6_ResNet_blocks3_seedPos_FC128x0_MAEloss_lr0.0005_epochs80_n778k_run0/model_epoch%d_*.pkl'%(neg_mass, epoch))
model_file = model #'Models/model_epoch80_mae0.1906.pkl'
resnet = networks.ResNet(2, 3, [16, 32], 128, 0).cuda()
resnet.load_state_dict(torch.load(model_file)['model'])
'''
#############################
# Load mass regressor for inference
logger(flog,'>> Loading regressor model')
def load_model_json(model, path):
    # Newer pytorch version save models as zip archive
    # that can't be loaded by older pytorch versions
    # Save model as json in newer pytorch, load json in older
    # then copy contents to an empty model state_dict
    # ref: https://stackoverflow.com/questions/64141188/how-to-load-checkpoints-across-different-versions-of-pytorch-1-3-1-and-1-6-x-u
    data_dict = OrderedDict()
    with open(path, 'r') as f:
        data_dict = json.load(f)
    own_state = model.state_dict()
    for k, v in data_dict.items():
        #logger(flog,'Loading parameter:', k)
        if not k in own_state:
            logger(flog,'Parameter', k, 'not found in own_state!!!')
        if type(v) == list or type(v) == int:
            v = torch.tensor(v)
        own_state[k].copy_(v)
    model.load_state_dict(own_state)
    logger(flog,'.. model loaded from json.')
import torch_resnet_concat as networks
#epoch = 80
#neg_mass = 300
## AOD
#model_file = glob.glob('../EB_Pi0_massreg_Pt20To100_2017PU_ext3_tzfixed_wrapfix/MODELS/DoublePi0Pt20To100_m0To1600_pythia8_PU2017_genDR10_aodDR16_nPhoN_PhoNeg%dTo0_wgts_EBtzo25_AOD_m0o1.6_ResNet_blocks3_seedPos_FC128x0_MAEloss_lr0.0005_epochs80_n778k_run0/model_epoch%d_*.pkl'%(neg_mass, epoch))
#assert len(model_file) == 1
#model_file = model_file[0]
#'''
#model_file = '/uscms/home/mba2012/nobackup/h2aa/CMSSW_10_5_0/src/h2aa/ntupleAnalysis/Models/model_epoch80_mae0.1906.pkl'
#model_file = 'models/resnet/precise300/model_epoch90_val_mae0.1889.json'
model_file = model
resnet = networks.ResNet(2, 3, [16, 32], 128, 0).cuda()
if '.json' in model_file:
    load_model_json(resnet, model_file)
else:
    resnet.load_state_dict(torch.load(model_file)['model']) # doesnt work with models from newer pytorch
    #resnet.load_state_dict(torch.jit.load(model_file)['model']) #doesnt work with models from newer pytorch
resnet.eval()
#logger(flog, '   .. done.')

eb_scale = 25.
m0_scale = 1.6
def transform_y(y):
    return y/m0_scale
def inv_transform(y):
    return y*m0_scale

h = {}
#c, wd, ht = {}, int(440*1), int(400*1)

k = 'iphi'
h[k] = ROOT.TH1F(k, k, (360//5)+1, -5., 360.)
k = 'ieta'
h[k] = ROOT.TH1F(k, k, (170//5)+1, -5., 170.)

k = 'ma'
#c[k] = ROOT.TCanvas("c%s"%k,"c%s"%k,wd,ht)
h[k] = ROOT.TH1F(k, k, 56, -0.2, 1.2)
#h[k] = ROOT.TH1F(k, k, 48, 0., 1.2)

# Event range to process
iEvtStart = 0
iEvtEnd   = nEvts
#iEvtEnd   = 100#0#10
logger(flog, ">> Processing entries: [%d, %d)"%(iEvtStart, iEvtEnd))

def shapeEB(eb):
    return np.array(eb).reshape(1,170,360)

nWrite = 0
nEvts_gg_proc = 0
sw = ROOT.TStopwatch()
sw.Start()
for iEvt in range(iEvtStart,iEvtEnd):

    # Initialize event
    if iEvt%10e3==0: logger(flog, '%d / %d'%(iEvt, iEvtEnd-iEvtStart))
    # Main tree
    evt_status = img_tree.GetEntry(iEvt)
    if evt_status <= 0: continue
    #print('main img tree: good evt')
    # Secondary tree
    evt_status2 = gg_tree.GetEntry(idx_where_run_lumi_evt(img_tree, gg_evtlist))
    if evt_status2 <= 0: continue
    nEvts_gg_proc += 1
    #print('sec gg tree: good evt')

    # Check event ID
    assert((gg_tree.run == img_tree.runId) & (gg_tree.lumis == img_tree.lumiId) & (gg_tree.event == img_tree.eventId))

    # Check photon indices correspond to same objects
    #if list(gg_tree.phoEt) != list(img_tree.pho_pT):
    #    print('!!',iEvt)
    #    print('!! nPho: %d vs %d'%(gg_tree.nPho, len(img_tree.SC_ieta)))
    #    print('!! Et  : %s vs %s'%(','.join([str(e) for e in gg_tree.phoEt]),  ','.join([str(e) for e in img_tree.pho_pT])))
    #    print('!! Eta : %s vs %s'%(','.join([str(e) for e in gg_tree.phoEta]), ','.join([str(e) for e in img_tree.pho_eta])))
    #    print('!! Phi : %s vs %s'%(','.join([str(e) for e in gg_tree.phoPhi]), ','.join([str(e) for e in img_tree.pho_phi])))
    assert gg_tree.nPho == len(img_tree.SC_ieta), '%d vs %d'%(gg_tree.nPho, len(img_tree.SC_ieta))
    assert list(gg_tree.phoEta) == list(img_tree.pho_eta), '%s vs %s'%(','.join([str(e) for e in gg_tree.phoEta]), ','.join([str(e) for e in img_tree.pho_eta]))
    assert list(gg_tree.phoPhi) == list(img_tree.pho_phi), '%s vs %s'%(','.join([str(e) for e in gg_tree.phoPhi]), ','.join([str(e) for e in img_tree.pho_phi]))
    #assert list(gg_tree.phoEt) == list(img_tree.pho_pT),   '%s vs %s'%(','.join([str(e) for e in gg_tree.phoEt]),  ','.join([str(e) for e in img_tree.pho_pT])) # can differ slightly if MINIAOD-level corrections applied

    '''
    p4 = {}
    pho_idx = {}
    for i in range(len(gg_tree.phoPreselIdxs)):
        pho_idx[i] = gg_tree.phoPreselIdxs[i]
        # ggntuple photon vector
        p4['gg'] = ROOT.TVector3()
        p4['gg'].SetPtEtaPhi(gg_tree.phoEt[pho_idx[i]], gg_tree.phoEta[pho_idx[i]], gg_tree.phoPhi[pho_idx[i]])
        # img ntuple photon vector
        p4['img'] = ROOT.TVector3()
        p4['img'].SetPtEtaPhi(img_tree.pho_pT[pho_idx[i]], img_tree.pho_eta[pho_idx[i]], img_tree.pho_phi[pho_idx[i]])
        # Ensure photons are alike
        dR = ROOT.Math.VectorUtil.DeltaR(p4['gg'], p4['img'])
        assert dR == 0.
        h['ieta'].Fill(img_tree.SC_ieta[pho_idx[i]])
        h['iphi'].Fill(img_tree.SC_iphi[pho_idx[i]])
    '''

    # Initialize a separate array `ma_` that will be used to fill
    # tree branch `ma` (faster than initializing `ma` by element)
    if gg_tree.nPho > nPhoMax:
        logger(flog, ".. too many photons, skipping event...")
        continue
    ma_ = -1.*np.ones(gg_tree.nPho) # default value = -1.
    #print('nPhoMax pass')

    # Only regress barrel photons
    # Tag & probe viable for mass regressor as long as 1 EB photon
    idxs_roi = [i for i in range(gg_tree.nPho) if (img_tree.SC_ieta[i] >= 0) and (img_tree.SC_ieta[i] < 170)]
    if len(idxs_roi) < 1:
        continue
    #print('nROI pass')

    # Crop EB windows from full EB frames
    sc_cms = []
    ieta, iphi = [], []
    X_EBt = np.array(img_tree.EB_energyT).reshape(1,170,360) # Et
    X_EBz = np.array(img_tree.EB_energyZ).reshape(1,170,360) # Ez
    X_cms = np.concatenate([X_EBt, X_EBz], axis=0)
    for idx in idxs_roi:
        ieta.append([img_tree.SC_ieta[idx]])
        iphi.append([img_tree.SC_iphi[idx]])
        #sc_cms_ = crop_EBshower(X_cms, ieta[-1], iphi[-1])
        sc_cms_ = crop_EBshower_padded(X_cms, ieta[-1], iphi[-1])
        # rotate 90deg clockwise
        sc_cms_ = rotate(sc_cms_)
        sc_cms.append(sc_cms_)

    # Run inference
    ma_roi = inv_transform(resnet([torch.Tensor(sc_cms).cuda()/eb_scale,\
                               torch.Tensor(iphi).cuda()/360.,\
                               torch.Tensor(ieta).cuda()/170.
                              ])).tolist()

    # Overwrite default values at ROI indices
    for i,idx_roi in enumerate(idxs_roi):
        ma_[idx_roi] = ma_roi[i][0]
    #print(ma_)

    # Fill output branch and histograms
    for i in range(gg_tree.nPho):
        ma[i] = ma_[i]
        h['ma'].Fill(ma_[i])
        h['ieta'].Fill(img_tree.SC_ieta[i])
        h['iphi'].Fill(img_tree.SC_iphi[i])

    #break
    tree_out.Fill()
    nWrite += 1

sw.Stop()
logger(flog, ">> N events written / IMG processed: %d / %d"%(nWrite, iEvtEnd-iEvtStart))
logger(flog, ">> N events written / GG processed: %d / %d"%(nWrite, nEvts_gg_proc))
logger(flog, ">> N events written / GG total: %d / %d"%(nWrite, nEvts_gg))
logger(flog, ">> Real time: %f minutes"%(sw.RealTime()/60.))
logger(flog, ">> CPU time: %f minutes"%(sw.CpuTime() /60.))
flog.close()
