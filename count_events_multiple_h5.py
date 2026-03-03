import h5py
import pickle
import glob

signal_masses = ['3p7','4','5','6','8']#,'10','12','14']
background_samples = ['QCD','HTo2Tau','TTbar','Wjets','DYto2L']

all_results = {}

# ---------------- SIGNAL ----------------
print("Processing ------ signal")
total_signal = 0

for mass in signal_masses:

    decay = f"IMG_signal_mass_{mass}_GeV"
    input_dir = f"/eos/uscms/store/group/lpcml/bbbam/signals_h5_Feb_2026/{decay}"
    input_files = glob.glob(f"{input_dir}/*.h5")

    total_events = 0
    file_count = 0

    for file_path in input_files:
        with h5py.File(file_path, "r") as f:
            n_events = f["all_jet"].shape[0]
            total_events += n_events
            file_count += 1

    print(f"\nSignal mass: {mass}")
    print(f"Total files: {file_count}")
    print(f"Total events: {total_events}")

    all_results[f"signal_{mass}"] = {
        "total_files": file_count,
        "total_events": total_events
    }

    total_signal += total_events

print(f"\nTotal signal events: {total_signal}")


# ---------------- BACKGROUND ----------------
print("\nProcessing ------ background")
total_background = 0

for bkg in background_samples:

    decay = f"IMG_background_{bkg}"
    input_dir = f"/eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_Feb_2026/{decay}"
    input_files = glob.glob(f"{input_dir}/*.h5")

    total_events = 0
    file_count = 0

    for file_path in input_files:
        with h5py.File(file_path, "r") as f:
            n_events = f["all_jet"].shape[0]
            total_events += n_events
            file_count += 1

    print(f"\nBackground sample: {bkg}")
    print(f"Total files: {file_count}")
    print(f"Total events: {total_events}")

    all_results[f"background_{bkg}"] = {
        "total_files": file_count,
        "total_events": total_events
    }

    total_background += total_events

print(f"\nTotal background events: {total_background}")


# ---------------- SAVE ----------------
with open("signal_background_event_counts_2.pkl", "wb") as f:
    pickle.dump(all_results, f)

print("\nSaved summary to signal_background_event_counts.pkl")
