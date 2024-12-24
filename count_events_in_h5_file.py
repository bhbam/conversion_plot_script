import h5py, random
import numpy as np
import os, glob, json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--Mass',     default='3p7',    type=str, help='select signal mass as str')
parser.add_argument('-b', '--batch_size', type=int, default=3200,
                    help='input batch size for training')
args = parser.parse_args()

# local = f"/eos/uscms/store/user/bbbam/Run_3_IMG_from_Ruchi/signals_normalized/IMG_HToAATo4Tau_Hadronic_signal_mass_{args.Mass}_GeV_normalized"
local = f"/eos/uscms/store/user/bbbam/Run_3_IMG_from_Ruchi/signals_normalized_combined"
total =0
file =glob.glob(f'{local}/*.h5')
batch_size =args.batch_size
for file_ in file:
    data = h5py.File(f'{file_}', 'r')
    num_images = len(data["am"])
    total = total + num_images
    print("in file:", file_ ,"--", "Total number of events->", num_images)
print("---Done---Total Events--->", total)
