import h5py
import os
import glob
import logging
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO)

def combine_h5_files(master_folder, out_dir, dest_file, batch_size):
    source_files = np.sort(glob.glob(f'{master_folder}/*.h5'))  # Ensure to match only .h5 files
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(f'{out_dir}/{dest_file}', 'w') as h5_dest:
        initialized_datasets = {}
        for file_name in source_files:
            try:
                with h5py.File(file_name, 'r') as h5_source:
                    copy_datasets(h5_source, h5_dest, initialized_datasets, batch_size)
                    logging.info(f"Copied data from {file_name}")
            except Exception as e:
                logging.error(f"Failed to process file {file_name}: {e}")

def copy_datasets(source, dest, initialized_datasets, batch_size):
    for name, item in source.items():
        if isinstance(item, h5py.Dataset):
            if name not in initialized_datasets:
                shape = (0,) + item.shape[1:]  # Initialize with zero rows
                maxshape = (None,) + item.shape[1:]

                dest.create_dataset(
                    name,
                    shape=shape,
                    dtype=item.dtype,
                    compression='lzf',
                    chunks=(batch_size,) + item.shape[1:],
                    maxshape=maxshape
                )
                initialized_datasets[name] = dest[name]

            dest_dataset = initialized_datasets[name]
            new_size = dest_dataset.shape[0] + item.shape[0]
            dest_dataset.resize(new_size, axis=0)
            dest_dataset[-item.shape[0]:] = item[:]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data_path', default='/eos/uscms/store/user/bbbam/Run_3_IMG_mass_reg_unphy_m0To3p6/IMG_Tau_hadronic_massregssion_samples_m1p8To3p6_pt30To300_v2_valid',
                        help='input data path')
    parser.add_argument('--output_data_path', default='/eos/uscms/store/user/bbbam/Run_3_IMG_mass_reg_unphy_m0To3p6/IMG_Tau_hadronic_massregssion_samples_m0To3p6_pt30To300_v2_original_combined_valid',
                        help='output data path')
    parser.add_argument('--output_data_file', default='IMG_Tau_hadronic_massregssion_samples_m0To3p6_pt30To300_v2_original_combined_valid.h5',
                        help='output data file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training')
    args = parser.parse_args()
    combine_h5_files(args.input_data_path, args.output_data_path, args.output_data_file, args.batch_size)
    logging.info("---Process is complete----")
