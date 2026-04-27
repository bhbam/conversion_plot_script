#!/bin/bash

echo "Starting jobs at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_AOD_April_2026/IMG_signal_mass_3p7_GeV_AOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_background_h5_from_AOD_combined_April_2026 --output_data_file  IMG_signal_mass_3p7_GeV_AOD_combined.h5
echo "Finished 3p7 at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_AOD_April_2026/IMG_signal_mass_4_GeV_AOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_background_h5_from_AOD_combined_April_2026 --output_data_file  IMG_signal_mass_4_GeV_AOD_combined.h5
echo "Finished 4 at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_AOD_April_2026/IMG_signal_mass_5_GeV_AOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_background_h5_from_AOD_combined_April_2026 --output_data_file  IMG_signal_mass_5_GeV_AOD_combined.h5
echo "Finished 5 at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_AOD_April_2026/IMG_signal_mass_6_GeV_AOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_background_h5_from_AOD_combined_April_2026 --output_data_file  IMG_signal_mass_6_GeV_AOD_combined.h5
echo "Finished 6 at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_AOD_April_2026/IMG_signal_mass_8_GeV_AOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_background_h5_from_AOD_combined_April_2026 --output_data_file  IMG_signal_mass_8_GeV_AOD_combined.h5
echo "Finished 8 at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_AOD_April_2026/IMG_signal_mass_10_GeV_AOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_background_h5_from_AOD_combined_April_2026 --output_data_file  IMG_signal_mass_10_GeV_AOD_combined.h5
echo "Finished 10 at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_AOD_April_2026/IMG_signal_mass_12_GeV_AOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_background_h5_from_AOD_combined_April_2026 --output_data_file  IMG_signal_mass_12_GeV_AOD_combined.h5
echo "Finished 12 at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/signals_h5_from_AOD_April_2026/IMG_signal_mass_14_GeV_AOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_background_h5_from_AOD_combined_April_2026 --output_data_file  IMG_signal_mass_14_GeV_AOD_combined.h5
echo "Finished 14 at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_from_AOD_April_2026/IMG_background_QCD_AOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_background_h5_from_AOD_combined_April_2026 --output_data_file  IMG_background_QCD_AOD_combined.h5
echo "Finished QCD at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_from_AOD_April_2026/IMG_background_HTo2Tau_AOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_background_h5_from_AOD_combined_April_2026 --output_data_file  IMG_background_HTo2Tau_AOD_combined.h5
echo "Finished HTo2Tau at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_from_AOD_April_2026/IMG_background_TTbar_AOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_background_h5_from_AOD_combined_April_2026 --output_data_file  IMG_background_TTbar_AOD_combined.h5
echo "Finished TTbar at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_from_AOD_April_2026/IMG_background_Wjets_AOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_background_h5_from_AOD_combined_April_2026 --output_data_file  IMG_background_Wjets_AOD_combined.h5
echo "Finished Wjets at $(date)"

python combine_h5.py --input_data_path /eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_from_AOD_April_2026/IMG_background_DYto2L_AOD --output_data_path /eos/uscms/store/group/lpcml/bbbam/signals_background_h5_from_AOD_combined_April_2026 --output_data_file  IMG_background_DYto2L_AOD_combined.h5
echo "Finished DYto2L at $(date)"

echo "All jobs completed at $(date)"
