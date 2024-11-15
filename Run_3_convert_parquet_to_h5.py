import os, glob, re
import shutil
import random
import json
import pyarrow.parquet as pq
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import time
from multiprocessing import Pool
import argparse
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process dataset 0-9')
parser.add_argument('-i', '--in_dir',     default='/storage/local/data1/gpuscratch/bbbam/Run_3_IMG_unbiased/IMG_aToTauTau_Hadronic_m3p6To18_pt30T0300_unbiased',   type=str, help='Input pqrquet directory')
parser.add_argument('-o', '--out_dir',    default='/storage/local/data1/gpuscratch/bbbam/Run_3_IMG_unbiased_h5/IMG_aToTauTau_Hadronic_m3p6To18_pt30T0300_unbiased_h5',    type=str, help='Output h5 directory')
args = parser.parse_args()
in_dir = args.in_dir
out_dir = args.out_dir

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)',s)]

def create_new_hdf5_file(filename, max_rows_per_file):
    hdf5_file = h5py.File(filename, 'w')
    dataset_names = ['all_jet', 'am', 'ieta', 'iphi', 'm0', 'apt', 'jetpt', 'taudR']
    total_samples = max_rows_per_file
    datasets = {
        name: hdf5_file.create_dataset(
        name,
        (total_samples, 13, 125, 125) if 'jet' in name else (total_samples, 1),
        dtype='float32',  # Specify an appropriate data type
        compression='lzf',  # Optional: add compression
        chunks = (32, 13, 125, 125) if 'jet' in name else (32, 1),
        ) for name in dataset_names
    }
    #hdf5_file.create_dataset('dataset', shape=(0, *data.shape[1:]), maxshape=(max_rows_per_file, *data.shape[1:]), dtype='float64')  # Adjust dtype as per your data type
    return hdf5_file

def append_data_to_hdf5(hdf5_file, start_index, end_index, df):
    #df = df[start_index:end_index]

    print("Writing to file", hdf5_file)
    xj = df.columns.get_loc('X_jet')
    am = df.columns.get_loc('am')
    ieta = df.columns.get_loc('ieta')
    iphi = df.columns.get_loc('iphi')
    m0 = df.columns.get_loc('m0')
    apt = df.columns.get_loc('apt')
    jetpt = df.columns.get_loc('jetpt')
    taudR = df.columns.get_loc('taudR')

    im = np.array(np.array(np.array(df.iloc[:, xj].tolist()).tolist()).tolist())
    am = np.array(df.iloc[:,am])
    ieta = np.array(df.iloc[:,ieta])
    iphi = np.array(df.iloc[:,iphi])
    m0 = np.array(df.iloc[:,m0])
    apt = np.array(df.iloc[:,apt])
    jetpt = np.array(df.iloc[:,jetpt])
    taudR = np.array(df.iloc[:,taudR])

    hdf5_file["all_jet"][start_index:end_index, :, :, :] = im
    hdf5_file["am"][start_index:end_index, :]   = am.reshape(df.shape[0],1).tolist()
    hdf5_file["ieta"][start_index:end_index, :] = ieta.reshape(df.shape[0],1).tolist()
    hdf5_file["iphi"][start_index:end_index, :] = iphi.reshape(df.shape[0],1).tolist()
    hdf5_file["m0"][start_index:end_index, :]   = m0.reshape(df.shape[0],1).tolist()
    hdf5_file["apt"][start_index:end_index, :]   = apt.reshape(df.shape[0],1).tolist()
    hdf5_file["jetpt"][start_index:end_index, :]   = jetpt.reshape(df.shape[0],1).tolist()
    hdf5_file["taudR"][start_index:end_index, :]   = taudR.reshape(df.shape[0],1).tolist()

    return hdf5_file


def process_files(args):
    batch_size = 4096
    file_path = args[0]
    h5py_file = args[1]
    parquet = pq.ParquetFile(file_path)
    total_samples = parquet.num_row_groups
    hdf5_file = create_new_hdf5_file(h5py_file,total_samples)
    batch_iter = parquet.iter_batches(batch_size,use_threads=True)

    start_index = 0
    bat = 0
    for batch in batch_iter:
        df = batch.to_pandas(use_threads=True)
        end_index = start_index + df.shape[0]
        print("total----->",total_samples , " Batch no.", bat, "Data frame shape", df.shape, " Start idx:", start_index, " end idx:", end_index)

        if end_index<=total_samples:

            append_data_to_hdf5(hdf5_file, start_index, end_index, df)
            start_index += df.shape[0]

        bat +=1

parquet_files = glob.glob(f"{in_dir}/*parquet")
h5_dir = out_dir
if not os.path.exists(h5_dir):
    os.makedirs(h5_dir)

batch_size = 4096
inputfile_list = []
outputfile_list = []

for f in parquet_files:
    opFile       = f.split("/")[-1].split(".")[0]
    h5_file = h5_dir+opFile+".h5"
    inputfile_list.append(f)
    outputfile_list.append(h5_file)
    tic = time.time()
args = list(zip(inputfile_list,outputfile_list))
print("----------------------------------------")
print("arg --------", args)

with Pool(len(parquet_files)) as p:
    p.map(process_files,args)
toc = time.time()
print("It took ", toc-tic, "Done----")
