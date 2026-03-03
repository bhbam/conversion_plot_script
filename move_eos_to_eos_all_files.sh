#!/bin/bash

eos_cmd="eos root://cmseos.fnal.gov"

source_dir="/eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_Feb_2026"
base_target_dir="/eos/uscms/store/group/lpcml/bbbam/signals_background_h5_combined_Feb_2026"

# Create target directory if it doesn't exist
$eos_cmd mkdir -p "$base_target_dir"

echo "Listing .h5 files in $source_dir ..."
files=$($eos_cmd ls "$source_dir" | grep '\.h5$')

count=0

for file in $files; do
    echo "Moving $file ..."

    if $eos_cmd mv "$source_dir/$file" "$base_target_dir/$file"; then
        ((count++))
        echo "Moved ($count)"
    else
        echo "Failed: $file"
    fi
done

echo "Done. Total files moved: $count"
