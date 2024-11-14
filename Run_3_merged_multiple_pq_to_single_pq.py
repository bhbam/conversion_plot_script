
# # Initilize the enviroment first: "source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh'"
import numpy as np
np.random.seed(0)
import os, re, glob
import time
#import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import *

import argparse
parser = argparse.ArgumentParser(description='Process dataset 0-9')
parser.add_argument('-m', '--mass_range',     default='m3p6To18',    type=str, help='select: m1p2To3p6 or m3p6To18')
parser.add_argument('-n', '--dataset',     default=0,    type=int, help='number of dataset used[0-9]')
args = parser.parse_args()
n = args.dataset
mass_range = args.mass_range


subsets= ['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009','0010']


subset = subsets[n]
print("processing dataset --->  ", subset)


local=f"/eos/uscms/store/user/bbbam/Run_3_IMG/IMG_aToTauTau_Hadronic_%s_pt30T0300/%s"%(mass_range, subset)
decay = f"IMG_aToTauTau_Hadronic_{mass_range}_pt30T0300_combined"
outDir="/eos/uscms/store/user/bbbam/Run_3_IMG/%s"%decay

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)',s)]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

files_ = '%s/*.parquet'%local
fs = glob.glob(files_)

assert len(fs) > 0
print(" >> %d files found"%len(fs))
sort_nicely(fs)



if not os.path.isdir(outDir):
    os.makedirs(outDir)

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

    file_str = '%s.parquet'%decay
    print('>> Doing sample:',file_str)
    print('>> Output events: %d [ %d, %d )'%((stop-start), start, stop))

    # Write out the actual data
    print('>> Writing data...')
    now = time.time()
    for i, idx in enumerate(idxs[start:stop]):

        if i%1000 == 0:
            print(' >> Processed event:',i)

        t = dset.__getitem__(idx)

        if i == 0:
            writer = pq.ParquetWriter(file_str, t.schema, compression='snappy')

        writer.write_table(t)

    writer.close()
    print('>> E.T.: %f min'%((time.time()-now)/60.))




dset = ConcatDataset([ParquetDatasetTable(f) for f in fs])
nevts_in = len(dset)
#for f in fs:
    #print(pq.ParquetFile(f).schema)
    # Schema MUST be consistent across input files!!

# Shuffle
print('>> Input events:',nevts_in)
idxs = np.random.permutation(nevts_in)

start, stop = 0, nevts_in

filename = '%s/%s_%s'%(outDir, decay, subset)

merge_samples(dset, start, stop, idxs, filename)
print('_____________________complete____________________________________\n')
