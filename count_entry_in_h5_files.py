import h5py
import glob
import os

input_dir = "/eos/uscms/store/group/lpcml/bbbam/signals_background_h5_from_miniAOD_combined_seperately_valid_July_2026"

files = sorted(glob.glob(f"{input_dir}/*.h5"))

if not files:
    print(f"No .h5 files found in {input_dir}")
    exit()

total_events = 0

print(f"Found {len(files)} files in:\n{input_dir}\n")
print(f"{'File':<60} {'Events':>10}")
print("-" * 72)

for file_path in files:
    fname = os.path.basename(file_path)
    with h5py.File(file_path, "r") as f:
        n_events = f["all_jet"].shape[0]
    print(f"{fname:<60} {n_events:>10,}")
    total_events += n_events

print("-" * 72)
print(f"{'TOTAL':<60} {total_events:>10,}")
