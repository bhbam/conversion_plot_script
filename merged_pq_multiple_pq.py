import os
# source = "source /cvmfs/sft.cern.ch/lcg/views/LCG_97/x86_64-centos7-gcc8-opt/setup.sh"
# print(source)
# os.system(source)

import numpy as np
np.random.seed(0)
import re, glob
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

def merge_samples(dset, start, stop, idxs,  decay):

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

        if i == 0:
            writer = pq.ParquetWriter(file_str, t.schema, compression='snappy')

        writer.write_table(t)

    writer.close()
    print('>> E.T.: %f min'%((time.time()-now)/60.))


decay='merged_IMG_aToTauTau_datase_1_reunbaied_mannul_v2_8_tracker_layer' # name inse IMG directory
# decay='Upsilon1s_ToTauTau_Hadronic_tauDR0p4_validation_no_pix_layers' # name inse IMG directory
cluster = 'FNAL'
local =''
outDir=''
if(cluster=='CERN'):
    outDir='/eos/user/b/bhbam/%s'%(decay)
if(cluster=='FNAL'):
    local='/eos/uscms/store/group/lpcml/bbbam/IMG/IMG_aToTauTau_datase_1_reunbaied_mannul_v2_8_tracker_layer'
    # local='/eos/uscms/store/user/bbbam/Upsilon1s_ToTauTau_Hadronic_tauDR0p4_eta0To2p4_pythia8_validationML/Upsilon1s_ToTauTau_Hadronic_tauDR0p4_validation/230314_172829/0000'
    outDir='/eos/uscms/store/group/lpcml/bbbam/IMG/%s'%(decay)

files_ = '/%s/*.parquet*'%local
fs = glob.glob(files_)
total_files = len(fs)
N=0
# total_files = 20
assert  total_files > 10
print(" >> %d files found"%len(fs))
sort_nicely(fs)
if not os.path.isdir(outDir):
    os.makedirs(outDir)
n0= N + (total_files//10)*0
n1= N + (total_files//10)*1
n2= N + (total_files//10)*2
n3= N + (total_files//10)*3
n4= N + (total_files//10)*4
n5= N + (total_files//10)*5
n6= N + (total_files//10)*6
n7= N + (total_files//10)*7
n8= N + (total_files//10)*8
n9= N + (total_files//10)*9
n10 = N +  total_files
part =[n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]
for n in range(len(part)-1):
    for f in range(part[n], part[n+1]):
        print("Input files :   ",fs[f])
    dset = ConcatDataset([ParquetDatasetTable(fs[f]) for f in range(part[n], part[n+1])])
    nevts_in = len(dset)
    filename = '%s/IMG_aToTauTau_datase_1_reunbaied_mannul_8_track_layers_train_merged_%d'%(outDir,N+n)
    print('>> Input events:',nevts_in)
    idxs = np.random.permutation(nevts_in)
    start, stop = 0, nevts_in
    merge_samples(dset, start, stop, idxs, filename)
    print('_________________________________________________________\n')
    del dset
