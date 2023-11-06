
# source = "source /cvmfs/sft.cern.ch/lcg/views/LCG_97/x86_64-centos7-gcc8-opt/setup.sh"
# print(source)
# os.system(source)
import numpy as np
np.random.seed(0)
import os, re, glob
import time
#import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import *

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)',s)]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

class ParquetDatasetTable(Dataset):
    def __init__(self, filename, cols=None):
        self.parquet = pq.ParquetFile(filename)
        self.cols = cols # read all columns
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols)
        return data
    def __len__(self):
        return self.parquet.num_row_groups

def merge_samples(dset, start, stop, idxs, decay):

    file_str = '%s.parquet'%(decay)
    print('>> Doing sample:',file_str)
    print('>> Output events: %d [ %d, %d )'%((stop-start), start, stop))

    # Write out the actual data
    print('>> Writing data...')
    now = time.time()
    for i, idx in enumerate(idxs[start:stop]):

        if i%1000 == 0:
            print(' >> Processed event:',i)

        t = dset.__getitem__(idx)
        data = {}

        data['X_jet'] = np.squeeze(t["X_jet"])
        data['ieta']  = float(np.squeeze(t["ieta"]))
        data['iphi']  = float(np.squeeze(t["iphi"]))

        if decay =='signals':
            data['am']    = float(np.squeeze(t["am"]))
            data['apt']   = float(np.squeeze(t["apt"]))
            data['y']     = 1.

        else:
            data['am']    = float(np.squeeze(t["jetM"]))
            data['apt']   = float(np.squeeze(t["jetPt"]))
            data['y']     = 0.

        pqdata = [pa.array([d]) if (np.isscalar(d) or type(d) == list) else pa.array([d.tolist()]) for d in data.values()]
        t = pa.Table.from_arrays(pqdata, list(data.keys()))
        if i == 0:
            writer = pq.ParquetWriter(file_str, t.schema, compression='snappy')

        writer.write_table(t)

    writer.close()
    print('>> E.T.: %f min'%((time.time()-now)/60.))

# subfile_name = 'DYToTauTau_M-50_13TeV'
# subfile_name = 'QCD_Pt-15to7000'
# subfile_name = 'TTToHadronic'
subfile_name = 'WJetsToLNu'
# subfile_name = 'A_TauTau_boosted'
decay='backgrounds' # name inse IMG directory
# decay='signals' # name inse IMG directory

# decay='Upsilon1s_ToTauTau_Hadronic_tauDR0p4_validation_no_pix_layers' # name inse IMG directory
cluster = 'FNAL'
local =''
outDir=''
if(cluster=='CERN'):
    outDir='/eos/user/b/bhbam/%s'%(decay)
if(cluster=='FNAL'):
    # local='/storage/local/data1/gpuscratch/bbbam/ParquetFiles_correctTrackerLayerHits_SecVtxInfoAdded/DYToTauTau_M-50_13TeV-powheg_pythia8/AODJets'
    # local='/storage/local/data1/gpuscratch/bbbam/ParquetFiles_correctTrackerLayerHits_SecVtxInfoAdded/QCD_Pt-15to7000_TuneCP5_Flat_13TeV_pythia8/AODJets'
    # local='/storage/local/data1/gpuscratch/bbbam/ParquetFiles_correctTrackerLayerHits_SecVtxInfoAdded/TTToHadronic_TuneCP5_13TeV_powheg-pythia8/AODJets'
    local='/storage/local/data1/gpuscratch/bbbam/ParquetFiles_correctTrackerLayerHits_SecVtxInfoAdded/WJetsToLNu_TuneCP5_13TeV_madgraphMLM-pythia8/AODJets'

    # local='/eos/uscms/store/user/bbbam/IMG_v2/IMG_aToTauTau_Hadronic_tauDR0p4_m3p6To14p8_dataset_2_unbaised_v2'

    # outDir='/eos/uscms/store/group/lpcml/bbbam/IMG/%s'%(decay)
    outDir='/eos/uscms/store/user/bhbam/IMG_classification/%s'%(decay)


files_ = '/%s/*.parquet*'%local
fs = glob.glob(files_)

assert len(fs) > 0
print(" >> %d files found"%len(fs))
sort_nicely(fs)

dset = ConcatDataset([ParquetDatasetTable(f) for f in fs])
nevts_in = len(dset)

# Shuffle
print('>> Input events:',nevts_in)
assert nevts_in > 0
idxs = np.random.permutation(nevts_in)

start, stop = 0, 425000
if not os.path.isdir(outDir):
        os.makedirs(outDir)
filename = '%s/%s'%(outDir,subfile_name)

merge_samples(dset, start, stop, idxs, filename)
print('_________________________________________________________\n')
