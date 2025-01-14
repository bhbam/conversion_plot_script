import os, time, random
import json
import numpy as np
import h5py
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--input_file', default='/eos/uscms/store/user/bbbam/Run_3_img/IMG_aToTauTau_m1p2T018_combined_normalized_h5/IMG_aToTauTau_Hadronic_m1p2To18_pt30T0300_unbiased_train_normalized.h5',
                    help='input data path')
parser.add_argument('--output_data_path', default='/eos/uscms/store/user/bbbam/Run_3_img_neagative_mass',
                    help='output data path')
parser.add_argument('--batch_size', type=int, default=3200,
                    help='input batch size for conversion')
parser.add_argument('--chunk_size', type=int, default=32,
                    help='chunk size')
parser.add_argument('--in_size', type=int, default=-1,
                    help='number of input to process')
args = parser.parse_args()

start_time=time.time()
chunk_size = args.chunk_size
infile = args.input_file
out_dir = args.output_data_path
in_size = args.in_size
batch_size = args.batch_size


unphy_bins = np.arange(-1.2,1.3,0.4)



data = h5py.File(f'{infile}', 'r')
num_images = data["all_jet"].shape[0] if in_size == -1 else in_size
outdir = out_dir

if not os.path.exists(outdir):
    os.makedirs(outdir)

prefix = infile.split('/')[-1].split('.')[0]
outfile = f'{prefix}_unphysica_negative_mass.h5'

with h5py.File(f'{outdir}/{outfile}', 'w') as proper_data:
    dataset_names = ['all_jet', 'am', 'ieta', 'iphi', 'apt']
    datasets = {
    name: proper_data.create_dataset(
        name,
        shape= (0,13, 125, 125) if 'all_jet' in name else (0,1),
        maxshape=(None, 13, 125, 125) if 'all_jet' in name else (None, 1),
        dtype='float32',  # Specify an appropriate data type
        compression='lzf',
        chunks=(chunk_size, 13, 125, 125) if 'all_jet' in name else (chunk_size, 1),
    ) for name in dataset_names
        }
    orig_num_am = 0
    start_idx_, end_idx_, start_idx, end_idx = 0, 0, 0, 0
    for start_idx_ in tqdm(range(0, num_images, batch_size)):
        end_idx_ = min(start_idx_ + batch_size, num_images)
        images_batch = data["all_jet"][start_idx_:end_idx_, :, :, :]
        am_batch = data["am"][start_idx_:end_idx_, :]
        ieta_batch = data["ieta"][start_idx_:end_idx_, :]
        iphi_batch = data["iphi"][start_idx_:end_idx_, :]
        apt_batch = data["apt"][start_idx_:end_idx_, :]



        lowest_mass_mask = am_batch < 1.6
        images_batch = images_batch[lowest_mass_mask.flatten()]
        am_batch = am_batch[lowest_mass_mask]
        ieta_batch = ieta_batch[lowest_mass_mask]
        iphi_batch = iphi_batch[lowest_mass_mask]
        apt_batch = apt_batch[lowest_mass_mask]
        orig_num_am = orig_num_am + len(am_batch)
        start_idx = max(start_idx, end_idx)


        if len(images_batch) <1 : continue

        new_am_batch = []
        new_images_batch =[]
        new_ieta_batch = []
        new_iphi_batch = []
        new_apt_batch = []

        for i in range(len(am_batch)):
            # Generate a random mass in each bin
            temp_am_batch = np.array([[np.random.uniform(low, high)] for low, high in zip(unphy_bins[:-1], unphy_bins[1:])])
            new_am_batch.append(temp_am_batch)

            # dublicate image for each unphysical mass bins
            temp_images_batch = np.repeat(images_batch[i][np.newaxis, ...], len(unphy_bins)-1, axis=0)
            new_images_batch.append(temp_images_batch)
            temp_ieta_batch = np.repeat(ieta_batch[i][np.newaxis, ...], len(unphy_bins)-1, axis=0)
            new_ieta_batch.append(temp_ieta_batch)
            temp_iphi_batch = np.repeat(iphi_batch[i][np.newaxis, ...], len(unphy_bins)-1, axis=0)
            new_iphi_batch.append(temp_iphi_batch)
            temp_apt_batch = np.repeat(apt_batch[i][np.newaxis, ...], len(unphy_bins)-1, axis=0)
            new_apt_batch.append(temp_apt_batch)


        new_am_batch = np.concatenate(new_am_batch, axis=0)
        new_images_batch = np.concatenate(new_images_batch, axis=0)
        new_ieta_batch = np.concatenate(new_ieta_batch, axis=0)
        new_iphi_batch = np.concatenate(new_iphi_batch, axis=0)
        new_apt_batch = np.concatenate(new_apt_batch, axis=0)

        np.random.shuffle(new_am_batch)
        np.random.shuffle(new_images_batch)
        np.random.shuffle(new_ieta_batch)
        np.random.shuffle(new_iphi_batch)
        np.random.shuffle(new_apt_batch)
        # plt.hist(np.concatenate(new_am_batch), bins=np.arange(-1.2,1.3,0.4))
        # plt.show()
        # print("start_idx. ", start_idx)
        end_idx   = min(start_idx + new_images_batch.shape[0], num_images)
        # print("end_idx. ", end_idx)
        # print("len(new_am_batch)", new_am_batch.shape)
        # print("len(new_images_batch)",new_images_batch.shape)
        # print("len(new_ieta_batch)", new_ieta_batch.shape)
        # print("len(new_iphi_batch)",new_iphi_batch.shape)
        # print("len(new_apt_batch)",new_apt_batch.shape)
        for name, dataset in datasets.items():
            dataset.resize((end_idx,13, 125, 125) if 'all_jet' in name else (end_idx,1))

        proper_data['all_jet'][start_idx:end_idx,:,:,:] = new_images_batch
        proper_data['am'][start_idx:end_idx] = new_am_batch
        proper_data['ieta'][start_idx:end_idx] = new_ieta_batch.reshape(-1, 1)
        proper_data['iphi'][start_idx:end_idx] = new_iphi_batch.reshape(-1, 1)
        proper_data['apt'][start_idx:end_idx] = new_apt_batch.reshape(-1, 1)

        # print("_____________________________________________________________")
data.close()
end_time=time.time()
print(f"-----All Processes Completed in {(end_time-start_time)/60} minutes-----------------")
