import h5py
import pickle
import glob
import numpy as np

mass_regression_samples = ['m0To3p6', 'm3p6To18']
for mass in mass_regression_samples:
    print(f"Processing ------ sample ---> {mass}")
    decay = f"IMG_ATo2Tau_{mass}_pt30To300"
    input_dir = f"/eos/uscms/store/group/lpcml/bbbam/Run_3_H5_ATo2Tau_from_miniAOD_Jul_10_2026/{decay}" # train only
    input_files = glob.glob(f"{input_dir}/*.h5")
    file_count = 0
    am_list = []
    apt_list = []
    for file_path in input_files:
        with h5py.File(file_path, "r") as f:
            am = f["am"][:, 0]
            apt = f["apt"][:, 0]
            am_list.append(am)
            apt_list.append(apt)
            file_count += 1
    am_ = np.concatenate(am_list)
    apt_ = np.concatenate(apt_list)

    output_dict = {}
    output_dict["am"] = am_
    output_dict["apt"] = apt_
    out_file = f"am_apt_{mass}_from_{file_count}_h5_miniAOD_train_Jul_10_2026.pkl"
    with open(out_file, "wb") as outfile:
        pickle.dump(output_dict, outfile, protocol=2)
    print(f"\nSample done: {mass}")
    print(f"Saved: {out_file}")





# input_dir = f"/eos/uscms/store/group/lpcml/bbbam/Run_3_H5_ATo2Tau_from_miniAOD_test_m0To18_Jul_10_2026" # test only
# input_files = glob.glob(f"{input_dir}/*.h5")
# file_count = 0
# am_list = []
# apt_list = []
# for file_path in input_files:
#     with h5py.File(file_path, "r") as f:
#         am = f["am"][:, 0]
#         apt = f["apt"][:, 0]
#         am_list.append(am)
#         apt_list.append(apt)
#         file_count += 1
# am_ = np.concatenate(am_list)
# apt_ = np.concatenate(apt_list)
#
# output_dict = {}
# output_dict["am"] = am_
# output_dict["apt"] = apt_
# out_file = f"am_apt_m0To18_from_{file_count}_h5_miniAOD_test_Jul_10_2026.pkl"
# with open(out_file, "wb") as outfile:
#     pickle.dump(output_dict, outfile, protocol=2)
# print(f"Saved: {out_file}")
