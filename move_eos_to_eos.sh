#!/bin/bash

# Base directories
source_dir="/store/group/lpcml/bbbam/Run_3_IMG_mass_reg_unphy_m0To3p6/IMG_Tau_hadronic_massregssion_samples_m1p8To3p6_pt30To300_v2_valid"
base_target_dir="/store/user/bbbam/Run_3_IMG_mass_reg_unphy_m0To3p6/IMG_Tau_hadronic_massregssion_samples_m1p8To3p6_pt30To300_v2_valid"

# Loop through files
for i in {701..990}; do
   # Calculate target directory
   # target_dir=$((i / 70))
   # target_dir=$(printf "%04d" $target_dir)
   # target_dir_path="$base_target_dir/$target_dir"
   target_dir_path="$base_target_dir"

   # Ensure target directory exists
   eos root://cmseos.fnal.gov mkdir -p $target_dir_path

   # Move file with retry mechanism
   for attempt in {1..3}; do
       eos root://cmseos.fnal.gov mv "$source_dir/IMG_Tau_hadronic_massregssion_samples_m1p8To3p6_pt30To300_v2_$i.h5" "$target_dir_path/IMG_Tau_hadronic_massregssion_samples_m1p8To3p6_pt30To300_v2_$i.h5" \
           && { echo "Moved file IMG_Tau_hadronic_massregssion_samples_m1p8To3p6_pt30To300_v2_$i.h5 to $target_dir_path"; break; } \
           || { echo "Attempt $attempt failed for file IMG_Tau_hadronic_massregssion_samples_m1p8To3p6_pt30To300_v2_$i.h5"; sleep 2; }
   done
done
