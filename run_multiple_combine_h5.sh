#!/bin/bash

echo "Starting jobs at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_miniAOD_June_15_2026/IMG_signal_mass_3p7_GeV_miniAOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_backgrounds_h5_from_miniAOD_combined_seperately_June_15_2026 --output_data_file  IMG_signal_mass_3p7_GeV_miniAOD_combined_seperately_test.h5
echo "Finished 3p7 at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_miniAOD_June_15_2026/IMG_signal_mass_4_GeV_miniAOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_backgrounds_h5_from_miniAOD_combined_seperately_June_15_2026 --output_data_file  IMG_signal_mass_4_GeV_miniAOD_combined_seperately_test.h5
echo "Finished 4 at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_miniAOD_June_15_2026/IMG_signal_mass_5_GeV_miniAOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_backgrounds_h5_from_miniAOD_combined_seperately_June_15_2026 --output_data_file  IMG_signal_mass_5_GeV_miniAOD_combined_seperately_test.h5
echo "Finished 5 at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_miniAOD_June_15_2026/IMG_signal_mass_6_GeV_miniAOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_backgrounds_h5_from_miniAOD_combined_seperately_June_15_2026 --output_data_file  IMG_signal_mass_6_GeV_miniAOD_combined_seperately_test.h5
echo "Finished 6 at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_miniAOD_June_15_2026/IMG_signal_mass_8_GeV_miniAOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_backgrounds_h5_from_miniAOD_combined_seperately_June_15_2026 --output_data_file  IMG_signal_mass_8_GeV_miniAOD_combined_seperately_test.h5
echo "Finished 8 at $(date)"

# python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_miniAOD_June_15_2026/IMG_signal_mass_10_GeV_miniAOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_backgrounds_h5_from_miniAOD_combined_seperately_June_15_2026 --output_data_file  IMG_signal_mass_10_GeV_miniAOD_combined_seperately_test.h5
# echo "Finished 10 at $(date)"
#
# python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_miniAOD_June_15_2026/IMG_signal_mass_12_GeV_miniAOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_backgrounds_h5_from_miniAOD_combined_seperately_June_15_2026 --output_data_file  IMG_signal_mass_12_GeV_miniAOD_combined_seperately_test.h5
# echo "Finished 12 at $(date)"
#
# python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_miniAOD_June_15_2026/IMG_signal_mass_14_GeV_miniAOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_backgrounds_h5_from_miniAOD_combined_seperately_June_15_2026 --output_data_file  IMG_signal_mass_14_GeV_miniAOD_combined_seperately_test.h5
# echo "Finished 14 at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_from_miniAOD_June_15_2026/IMG_background_QCD_miniAOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_backgrounds_h5_from_miniAOD_combined_seperately_June_15_2026 --output_data_file  IMG_background_QCD_miniAOD_combined_seperately_test.h5
echo "Finished QCD at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_from_miniAOD_June_15_2026/IMG_background_HTo2Tau_miniAOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_backgrounds_h5_from_miniAOD_combined_seperately_June_15_2026 --output_data_file  IMG_background_HTo2Tau_miniAOD_combined_seperately_test.h5
echo "Finished HTo2Tau at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_from_miniAOD_June_15_2026/IMG_background_TTbar_miniAOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_backgrounds_h5_from_miniAOD_combined_seperately_June_15_2026 --output_data_file  IMG_background_TTbar_miniAOD_combined_seperately_test.h5
echo "Finished TTbar at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_from_miniAOD_June_15_2026/IMG_background_Wjets_miniAOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_backgrounds_h5_from_miniAOD_combined_seperately_June_15_2026 --output_data_file  IMG_background_Wjets_miniAOD_combined_seperately_test.h5
echo "Finished Wjets at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_from_miniAOD_June_15_2026/IMG_background_DYto2L_miniAOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_backgrounds_h5_from_miniAOD_combined_seperately_June_15_2026 --output_data_file  IMG_background_DYto2L_miniAOD_combined_seperately_test.h5
echo "Finished DYto2L at $(date)"

echo "All jobs completed at $(date)"
