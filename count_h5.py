import h5py, random, os, glob, pickle
import numpy as np
Total_events = 0
Total_files = 0
am, apt = [], []
# test_dirs = glob.glob(f"/eos/uscms/store/group/lpcml/bbbam/signals_val_h5_Feb_2026/*.h5")
# test_dirs = glob.glob(f"/eos/uscms/store/group/lpcml/bbbam/Run_3_IMG_ATo2Tau_from_miniAOD/IMG_ATo2Tau_m0To3p6_pt30To300/*.h5")
test_dirs = glob.glob(f"/eos/uscms/store/group/lpcml/bbbam/Run_3_IMG_ATo2Tau_from_AOD_m0To18_combined_unbiased_April_2026/*train.h5")
# test_dirs = glob.glob(f"/storage/local/data1/gpuscratch/bbbam/signal_combined_backgriund_combined_test_Feb_2026/*.h5")
print("Number of data sets ", len(test_dirs))
for file_ in test_dirs:
    Total_files = Total_files +1
    with h5py.File(file_, "r") as data:
        # Load metadata
        am_ = data["am"][:, 0]
        apt_ = data["apt"][:, 0]
        am.append(am_)
        apt.append(apt_)
        Total_events = Total_events + len(am_)
        Total_events = Total_events + len(am_)
        print("In file_ : ", file_)
        print("Events:", len(am_))
        print("Events:", len(am_))

print("Total files :", Total_files)
print("Total events:", Total_events)
output_dict = {}
# output_dict["am"] = am
# output_dict["apt"] = apt
output_dict["am"] = np.concatenate(am)
output_dict["apt"] = np.concatenate(apt)
output_dict["Total_files"] = Total_files
output_dict["Total_events"] = Total_events
with open(f'IMG_ATo2Tau_m0To18_pt30To300_am_apt_from_AOD_unbiased_train.pkl', "wb") as outfile:
              pickle.dump(output_dict, outfile, protocol=2) #protocol=2 for compatibility
# print(y[:1000])
# print("for pkl files")
# out_dir = glob.glob("/uscms/home/bbbam/nobackup/analysis_run3/Data_for_plots/ResNet_mapA_signal_backgrounds/*.pkl")
# for pkl_file in out_dir :
#     print("pkl  ", pkl_file)
#     infile = open(f"{pkl_file}", "rb")
#     data = pickle.load(infile)
#     # m_pred = data["m_pred"]
#     print("Length of m_pred:", len(data["m_pred"]))
