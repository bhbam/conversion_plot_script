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

    # Collect file lengths and initialize per-file data
    files_data = []

    dataset_names = ['all_jet', 'am', 'ieta', 'iphi', 'apt']

    total_length = 0

    # Determine total length and per-file lengths
    for file_name in source_files:
        try:
            h5_source = h5py.File(file_name, 'r')
            file_data = {}
            file_data['filename'] = file_name
            file_data['file'] = h5_source
            file_data['current_pos'] = 0
            file_data['buffer'] = None  # Will hold data buffer
            file_data['buffer_pos'] = 0  # Position in the buffer
            # Assume all datasets have the same length
            primary_dataset_name = dataset_names[0]
            primary_dataset = h5_source[primary_dataset_name]
            file_length = primary_dataset.shape[0]
            file_data['length'] = file_length
            total_length += file_length
            files_data.append(file_data)
        except Exception as e:
            logging.error(f"Failed to process file {file_name}: {e}")

    if total_length == 0:
        logging.error("No data to process.")
        return

    # Create the destination datasets with specified shapes and chunk sizes
    with h5py.File(f'{out_dir}/{dest_file}', 'w') as h5_dest:
        datasets = {}
        for name in dataset_names:
            # Determine shape and chunk size based on the dataset name
            if 'all_jet' in name:
                data_shape = (total_length, 13, 125, 125)
                chunk_shape = (32, 13, 125, 125)
            else:
                data_shape = (total_length, 1)
                chunk_shape = (32, 1)
            # Create the dataset
            datasets[name] = h5_dest.create_dataset(
                name,
                shape=data_shape,
                dtype='float32',
                compression='lzf',
                chunks=chunk_shape
            )

        # Initialize output buffers
        output_buffers = {name: [] for name in dataset_names}
        total_entries = 0
        write_position = 0  # Position in the destination datasets

        # Preload buffers for each file
        for file_data in files_data:
            preload_buffer(file_data, batch_size, dataset_names)

        # Main loop
        active_files = files_data.copy()

        while active_files:
            # Compute remaining lengths and probabilities
            remaining_lengths = np.array([f['length'] - f['current_pos'] for f in active_files])
            probabilities = remaining_lengths / np.sum(remaining_lengths)

            # Choose a file index based on probabilities
            file_indices = np.arange(len(active_files))
            selected_index = np.random.choice(file_indices, p=probabilities)
            selected_file_data = active_files[selected_index]

            # Fetch data from the selected file's buffer
            entries_needed = 1  # Number of entries to take in this iteration
            data_available = selected_file_data['buffer_length'] - selected_file_data['buffer_pos']

            if data_available >= entries_needed:
                # Take data from buffer
                buffer_start = selected_file_data['buffer_pos']
                buffer_end = buffer_start + entries_needed
                for name in dataset_names:
                    data_chunk = selected_file_data['buffer'][name][buffer_start:buffer_end]
                    output_buffers[name].append(data_chunk)
                selected_file_data['buffer_pos'] += entries_needed
                selected_file_data['current_pos'] += entries_needed
            else:
                # Not enough data in buffer, load new buffer
                preload_buffer(selected_file_data, batch_size, dataset_names)
                if selected_file_data['buffer_length'] == 0:
                    # No more data in file
                    selected_file_data['file'].close()
                    del active_files[selected_index]
                    logging.info(f"Finished processing file {selected_file_data['filename']}")
                    continue
                else:
                    continue  # Retry the iteration with new buffer

            total_entries += entries_needed

            # If total_entries >= chunk size (32), write data
            if total_entries >= 32:
                data_to_write = get_data_to_write(output_buffers, 32)
                # Write data to destination datasets
                for name in dataset_names:
                    data = data_to_write[name]
                    dest_dataset = datasets[name]
                    dest_dataset[write_position:write_position + 32] = data
                write_position += 32
                total_entries -= 32

        # After loop, write any remaining data in output buffers
        if total_entries > 0:
            data_to_write = get_data_to_write(output_buffers, total_entries)
            for name in dataset_names:
                data = data_to_write[name]
                dest_dataset = datasets[name]
                dest_dataset[write_position:write_position + total_entries] = data
            write_position += total_entries

        # Close all source files
        for file_data in files_data:
            if 'file' in file_data and file_data['file']:
                file_data['file'].close()

    logging.info("---Process is complete----")


def preload_buffer(file_data, batch_size, dataset_names):
    start = file_data['current_pos']
    end = min(start + batch_size, file_data['length'])
    entries_to_load = end - start
    if entries_to_load <= 0:
        # No more data to load
        file_data['buffer'] = None
        file_data['buffer_length'] = 0
        file_data['buffer_pos'] = 0
        return
    buffer_data = {}
    for name in dataset_names:
        data = file_data['file'][name][start:end]
        buffer_data[name] = data
    file_data['buffer'] = buffer_data
    file_data['buffer_length'] = entries_to_load
    file_data['buffer_pos'] = 0


def get_data_to_write(buffers, num_entries):
    data_to_write = {}
    for name in buffers.keys():
        data_list = buffers[name]
        entries_needed = num_entries
        data_arrays = []
        while entries_needed > 0 and data_list:
            data_array = data_list[0]
            data_len = data_array.shape[0]
            if data_len <= entries_needed:
                data_arrays.append(data_array)
                entries_needed -= data_len
                del data_list[0]
            else:
                data_arrays.append(data_array[:entries_needed])
                data_list[0] = data_array[entries_needed:]
                entries_needed = 0
        if data_arrays:
            data_to_write[name] = np.concatenate(data_arrays, axis=0)
        else:
            data_to_write[name] = np.array([], dtype='float32')
    return data_to_write


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data_path', default='/eos/uscms/store/user/bhbam/Run_3_IMG_mass_reg_unphy_m0To3p6/IMG_AToTau_Hadronic_mass_reg_m0To3p6_pt30To300_normalized',
                        help='input data path')
    parser.add_argument('--output_data_path', default='/eos/uscms/store/user/bhbam/Run_3_IMG_mass_reg_unphy_m0To3p6/IMG_AToTau_Hadronic_mass_reg_m0To3p6_pt30To300_normalized_combined',
                        help='output data path')
    parser.add_argument('--output_data_file', default='IMG_massregression_sample_m0To3p6_GeV_pt30To300_normalized_combined_train.h5',
                        help='output data file')
    parser.add_argument('--batch_size', type=int, default=320,
                        help='input batch size for training')
    args = parser.parse_args()
    combine_h5_files(args.input_data_path, args.output_data_path, args.output_data_file, args.batch_size)
    logging.info("---Process is complete----")
