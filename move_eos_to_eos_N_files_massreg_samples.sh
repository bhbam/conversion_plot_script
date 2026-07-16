#!/bin/bash

eos_cmd="eos root://cmseos.fnal.gov"

# Define your samples
mass_regression_samples=(m0To3p6 m3p6To18)


# Max files per sample
max_files_to_move=(183 874)


##############################
# Move SIGNALS
##############################
for i in "${!mass_regression_samples[@]}"; do
    mass=${mass_regression_samples[$i]}
    max_files=${max_files_to_move[$i]}

    source_dir="/eos/uscms/store/group/lpcml/bbbam/Run_3_H5_ATo2Tau_from_miniAOD_Jul_10_2026/IMG_ATo2Tau_${mass}_pt30To300"
    target_dir="/eos/uscms/store/group/lpcml/bbbam/Run_3_H5_ATo2Tau_from_miniAOD_test_m0To18_Jul_10_2026"

    echo "======================================"
    echo "Processing SIGNAL mass ${mass} GeV (max $max_files files)"
    echo "======================================"

    $eos_cmd mkdir -p "$target_dir"
    count=0

    # Get file list first to avoid subshell issues
    files=$($eos_cmd ls "$source_dir" | grep '\.h5$')

    for file in $files; do
        if [ "$count" -ge "$max_files" ]; then
            break
        fi

        echo "Moving $file ..."
        if $eos_cmd mv "$source_dir/$file" "$target_dir/$file"; then
            ((count++))
            echo "Moved ($count/$max_files)"
        else
            echo "Failed: $file"
        fi
    done

    echo "Total moved for mass $mass GeV: $count"
    echo
done
