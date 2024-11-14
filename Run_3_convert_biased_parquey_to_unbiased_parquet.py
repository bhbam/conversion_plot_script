import os, glob, time, torch, pickle, random, h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset, ConcatDataset
import matplotlib.pyplot as plt
import pickle


import argparse
parser = argparse.ArgumentParser(description='Process dataset 0-9')
parser.add_argument('-m', '--mass_range',     default='m3p6To18',    type=str, help='select: m1p2To3p6 or m3p6To18')
parser.add_argument('-n', '--dataset',     default=0,    type=int, help='number of dataset used[0-9]')
args = parser.parse_args()
n = args.dataset
mass_range = args.mass_range


subsets= ['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009','0010']
event_per_bin = 200

subset = subsets[n]
print("processing dataset --->  ", subset)
print("Number of evebt expected in each mass and pT bins %d"%event_per_bin)

# local=f"/eos/uscms/store/user/bbbam/Run_3_IMG/IMG_aToTauTau_Hadronic_%s_pt30T0300/%s"%(mass_range, subset)
local=f"/storage/local/data1/gpuscratch/bbbam/IMG_aToTauTau_Hadronic_%s_pt30T0300_combined"%mass_range
decay = f"IMG_aToTauTau_Hadronic_{mass_range}_pt30T0300_unbiased"
# outDir="/eos/uscms/store/user/bbbam/Run_3_IMG_unbiased/%s"%decay
outDir="/storage/local/data1/gpuscratch/bbbam/Run_3_IMG_unbiased/%s"%decay

mass_bins = {"m1p2To3p6": np.arange(1.2, 3.7, .4), "m3p6To18": np.arange(3.6, 18.1, .4)}.get(mass_range, None)
pt_bins = np.arange(30,301,5)

hist_biased, xedges, yedges = np.histogram2d([], [], bins=[mass_bins, pt_bins])

class ParquetDatasetTable(Dataset):
    def __init__(self, filename, cols=None):
        self.parquet = pq.ParquetFile(filename)
        self.cols = cols # read all columns
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols)
        return data
    def __len__(self):
        return self.parquet.num_row_groups


fs = glob.glob(f"{local}/*{subset}*parquet")
assert len(fs) > 0
print(" >> %d files found"%len(fs))


dset = ConcatDataset([ParquetDatasetTable(f) for f in fs])
nevts_in = len(dset)
print('>> Input events:',nevts_in)
# nevts_in = 500
start, stop = 0, nevts_in
idxs = np.random.permutation(nevts_in)

if not os.path.isdir(outDir):
        os.makedirs(outDir)

print('>> Output events more than : %d [ %d, %d ]'%((stop-start), start, stop))
outStr = f"{outDir}/{decay}_unbiased_{subset}.parquet"
print(f">> Ouput file:  {outStr}")
index = 0
data = {}
new_table = True
for i, idx in enumerate(idxs[start:stop]):
    if i%10000 == 0:
        print(' >> Reading event:',i)
    t = dset.__getitem__(idx)
    ams = float(np.squeeze(t["am"]))
    apts= float(np.squeeze(t["apt"]))
    hist_index_mass = np.searchsorted(mass_bins, ams, side="right")-1
    hist_index_pt = np.searchsorted(pt_bins, apts, side="right")-1
    hist_biased[hist_index_mass, hist_index_pt] += 1
    events = hist_biased[hist_index_mass, hist_index_pt]
    if events > event_per_bin: continue


    data['am']    = ams
    data['X_jet'] = np.squeeze(t["X_jet"])
    data['ieta']  = float(np.squeeze(t["ieta"]))
    data['iphi']  = float(np.squeeze(t["iphi"]))
    data['m0']    = float(np.squeeze(t["m0"]))
    data['apt']   = apts
    data['jetpt'] = float(np.squeeze(t["jetpt"]))
    data['taudR'] = float(np.squeeze(t["taudR"]))

    pqdata = [pa.array([d]) if (np.isscalar(d) or type(d) == list) else pa.array([d.tolist()]) for d in data.values()]

    table = pa.Table.from_arrays(pqdata, list(data.keys()))

    if new_table:
        writer = pq.ParquetWriter(outStr, table.schema, compression='snappy')
        new_table = False

    writer.write_table(table)
    index += 1
writer.close()



print(f"Total jets written to parquet {index} from {nevts_in}")
print("---------------------Complete converted to unbiased parquet-----------------")
