import h5py, random, os, glob, pickle
Total_events = 0
# test_dirs = glob.glob(f"/eos/uscms/store/group/lpcml/bbbam/signals_val_h5_Feb_2026/*.h5")
# test_dirs = glob.glob(f"/eos/uscms/store/group/lpcml/bbbam/signals_background_h5_Feb_2026_train/*.h5")
test_dirs = glob.glob(f"/storage/local/data1/gpuscratch/bbbam/signal_classifier_data_run3/train/*.h5")
print("Number of data sets ", len(test_dirs))
for file_ in test_dirs:
    with h5py.File(file_, "r") as data:
        # Load metadata
        y = data["y"][:, 0]
        Total_events = Total_events + len(y)
        print("In file_ : ", file_)
        print("Events:", len(y))
print("Total events:", Total_events)
# print(y[:1000])
# print("for pkl files")
# out_dir = glob.glob("/uscms/home/bbbam/nobackup/analysis_run3/Data_for_plots/ResNet_mapA_signal_backgrounds/*.pkl")
# for pkl_file in out_dir :
#     print("pkl  ", pkl_file)
#     infile = open(f"{pkl_file}", "rb")
#     data = pickle.load(infile)
#     # m_pred = data["m_pred"]
#     print("Length of m_pred:", len(data["m_pred"]))
