#!/bin/bash

# Base directories
source_dir="/store/group/lpcml/bbbam/Ntuples_run3/GEN_SIM_AToTau_m1p8To3p6_pt30To300/RHAnalyzer_AToTau_Hadronic_m1p8To3p6_v2/250126_055706/0000"
base_target_dir="/store/group/lpcml/bbbam/Ntuples_run3/GEN_SIM_AToTau_m1p8To3p6_pt30To300/RHAnalyzer_AToTau_Hadronic_m1p8To3p6_v2"

# Loop through files
for i in {0..700}; do
   # Calculate target directory
   target_dir=$((i / 70))
   target_dir=$(printf "%04d" $target_dir)
   target_dir_path="$base_target_dir/$target_dir"

   # Ensure target directory exists
   eos root://cmseos.fnal.gov mkdir -p $target_dir_path

   # Move file with retry mechanism
   for attempt in {1..3}; do
       eos root://cmseos.fnal.gov mv "$source_dir/output_$i.root" "$target_dir_path/output_$i.root" \
           && { echo "Moved file output_$i.root to $target_dir_path"; break; } \
           || { echo "Attempt $attempt failed for file output_$i.root"; sleep 2; }
   done
done
