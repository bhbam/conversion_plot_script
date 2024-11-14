#!/bin/bash
for i in {0..1999}
do
   # Calculate the target directory based on the file index
   target_dir=$((i / 200))

   # Format the target directory as a 4-digit number with leading zeros
   target_dir=$(printf "%04d" $target_dir)

   # Move the file to the calculated target directory
   eos root://cmseos.fnal.gov mv /store/group/lpcml/bbbam/Ntuples_run3/GEN_SIM_ATo2Tau_m1p2To3p6_pt30To300_v4/RHAnalyzer_ATo4Tau_Hadronic_m1p2To3p6/241108_195542/00000/output_$i.root /store/group/lpcml/bbbam/Ntuples_run3/GEN_SIM_ATo2Tau_m1p2To3p6_pt30To300_v4/RHAnalyzer_ATo4Tau_Hadronic_m1p2To3p6/241108_195542/$target_dir/output_$i.root && echo "Moved file output_$i.root to directory $target_dir" || echo "Failed to move file output_$i.root"
done
