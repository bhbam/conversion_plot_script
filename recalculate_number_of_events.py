import pickle
import numpy as np
# Load file
with open("signal_background_event_counts_2.pkl", "rb") as f:
    data = pickle.load(f)

signal_total = 0
background_total = 0

signal_files = 0
background_files = 0
samples = []
Total_events = []
Total_files  = []
for sample_name, info in data.items():
    samples.append(sample_name)
    print(f">>>>>>>>>>>>> {sample_name} >>>>>>>>>>>>>>>>>")

    print(f"Total files : {info['total_files']}")
    print(f"Total events: {info['total_events']}")
    Total_events.append(info['total_events'])
    Total_files.append(info['total_files'])
    event_per_file = info['total_events'] / info['total_files']
    print(f"Events per file : {event_per_file:.2f}\n")
    print(f"files to be processed : {315625/event_per_file}")

    # ---- Separate signal and background ----
    if sample_name.startswith("signal"):

        signal_total += info['total_events']
        signal_files += info['total_files']

    elif sample_name.startswith("background"):
        background_total += info['total_events']
        background_files += info['total_files']

# -------- Print Summary --------
print("============= SUMMARY =============")
print(f"Total SIGNAL events: {signal_total}")
print(f"Total SIGNAL files : {signal_files}")
print(f"Avg SIGNAL events/file: {signal_total/signal_files:.2f}\n")

print(f"Total BACKGROUND events: {background_total}")
print(f"Total BACKGROUND files : {background_files}")
print(f"Avg BACKGROUND events/file: {background_total/background_files:.2f}")

print("Samples : ", samples)
print("Total events : ", Total_events)
print("Total files : ", Total_files)

final_events = 130000
events_perfile = np.array(Total_events) / np.array(Total_files)
final_files = final_events / events_perfile
print("final number of files need to process ", final_files)
validation_files = np.array(Total_files) - final_files
print("Validation files :", validation_files)
