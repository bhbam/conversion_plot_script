import h5py
import numpy as np
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_bin(args):
    """Process a single bin and return selected indices."""
    i, j, mass_bins, pt_bins, mass, pt, bin_count = args
    bin_mask = (
        (mass >= mass_bins[i]) & (mass < mass_bins[i + 1]) &
        (pt >= pt_bins[j]) & (pt < pt_bins[j + 1])
    )
    bin_indices = np.where(bin_mask)[0]
    if len(bin_indices) >= bin_count:
        bin_indices = np.random.choice(bin_indices, bin_count, replace=False)
    return bin_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_file', default='/eos/uscms/store/group/lpcml/bbbam/Run_3_IMG_ATo2Tau_m0To18_from_miniAOD_combined_April_2026/Run_3_IMG_ATo2Tau_m0To18_from_miniAOD_combined_April_2026_valid.h5',
                        help='input data ')
    parser.add_argument('--output_data_path', default='/eos/uscms/store/group/lpcml/bbbam/Run_3_IMG_ATo2Tau_m0To18_from_miniAOD_combined_unbiased_April_2026',
                        help='output data path')
    parser.add_argument('--output_data_file', default='Run_3_IMG_ATo2Tau_m0To18_from_miniAOD_combined_unbiased_April_2026_valid.h5',
                        help='output file name')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for conversion')
    parser.add_argument('--chunk_size', type=int, default=32,
                        help='chunk size')
    parser.add_argument('--bin_count', type=int, default=200,
                        help='number entry in each bin needed')
    args = parser.parse_args()

    # Define mass and pt bin edges
    mass_bins = np.arange(0, 18.1, 0.4)
    pt_bins = np.arange(30, 301, 5)

    # Open the input file
    with h5py.File(args.input_data_file, "r") as data:
        # Load metadata
        mass = data["am"][:, 0]
        pt = data["apt"][:, 0]
        print("Original total events:", len(mass))

        # Use multiprocessing to process bins in parallel
        with Pool(cpu_count()) as pool:
            tasks = [(i, j, mass_bins, pt_bins, mass, pt, args.bin_count)
                     for i in range(len(mass_bins) - 1)
                     for j in range(len(pt_bins) - 1)]
            selected_indices = pool.map(process_bin, tasks)

        # Flatten the list of selected indices and sort them
        selected_indices = np.concatenate(selected_indices)
        selected_indices.sort()  # Ensure indices are in increasing order
        print("Final total events:", len(selected_indices))

        # Ensure output directory exists
        if not os.path.exists(args.output_data_path):
            os.makedirs(args.output_data_path)

        # Create output file and datasets
        with h5py.File(f'{args.output_data_path}/{args.output_data_file}', 'w') as output_data:
            dataset_names = ['all_jet', 'am', 'ieta', 'iphi', 'apt', 'jet_mass', 'jet_pt']
            datasets = {
                name: output_data.create_dataset(
                    name,
                    shape=(len(selected_indices), 5, 125, 125) if 'all_jet' in name else (len(selected_indices), 1),
                    dtype='float32',
                    compression='lzf',
                    chunks=(args.chunk_size, 5, 125, 125) if 'all_jet' in name else (args.chunk_size, 1),
                ) for name in dataset_names
            }

            # Process and save data in chunks
            batch_size = args.batch_size
            for start in tqdm(range(0, len(selected_indices), batch_size), desc="Processing image chunks"):
                end = min(start + batch_size, len(selected_indices))
                chunk_indices = selected_indices[start:end]
                datasets["am"][start:end] = data["am"][chunk_indices]
                datasets["apt"][start:end] = data["apt"][chunk_indices]
                datasets["ieta"][start:end] = data["ieta"][chunk_indices]
                datasets["iphi"][start:end] = data["iphi"][chunk_indices]
                datasets["all_jet"][start:end] = data["all_jet"][chunk_indices]
                datasets["jet_mass"][start:end] = data["jet_mass"][chunk_indices]
                datasets["jet_pt"][start:end] = data["jet_pt"][chunk_indices]

    print(f"Flat distribution created and saved to {args.output_data_path}/{args.output_data_file}")

if __name__ == "__main__":
    main()
