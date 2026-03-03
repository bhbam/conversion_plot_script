#!/bin/bash

eos_cmd="eos root://cmseos.fnal.gov"

# Define your samples
signal_masses=(3p7 4 5 6 8)
background_samples=(QCD HTo2Tau TTbar Wjets DYto2L)

# Max files per sample
max_files_signal=(112 112 113 112 113)
max_files_background=(681 941 510 121 92)

##############################
# Move SIGNALS
##############################
for i in "${!signal_masses[@]}"; do
    mass=${signal_masses[$i]}
    max_files=${max_files_signal[$i]}

    source_dir="/eos/uscms/store/group/lpcml/bbbam/signals_h5_Feb_2026/IMG_signal_mass_${mass}_GeV"
    target_dir="/eos/uscms/store/group/lpcml/bbbam/signals_val_h5_Feb_2026/IMG_signal_mass_${mass}_GeV"

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


##############################
# Move BACKGROUNDS
##############################
for i in "${!background_samples[@]}"; do
    sample=${background_samples[$i]}
    max_files=${max_files_background[$i]}

    source_dir="/eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_Feb_2026/IMG_background_${sample}"
    target_dir="/eos/uscms/store/group/lpcml/bbbam/backgrounds_val_h5_Feb_2026/IMG_background_${sample}"

    echo "======================================"
    echo "Processing BACKGROUND ${sample} (max $max_files files)"
    echo "======================================"

    $eos_cmd mkdir -p "$target_dir"
    count=0

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

    echo "Total moved for background $sample: $count"
    echo
done

echo "All done ✅"
