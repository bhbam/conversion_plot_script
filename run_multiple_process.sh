#!/bin/bash

echo "Starting jobs at $(date)"

python Run_3_root2h5_signal_jet_multiproc.py -m 3p7 -p signal
echo "Finished 3p7 at $(date)"

python Run_3_root2h5_signal_jet_multiproc.py -m 4 -p signal
echo "Finished 4 at $(date)"

python Run_3_root2h5_signal_jet_multiproc.py -m 5 -p signal
echo "Finished 5 at $(date)"

python Run_3_root2h5_signal_jet_multiproc.py -m 6 -p signal
echo "Finished 6 at $(date)"

python Run_3_root2h5_signal_jet_multiproc.py -m 8 -p signal
echo "Finished 8 at $(date)"

python Run_3_root2h5_signal_jet_multiproc.py -m 10 -p signal
echo "Finished 10 at $(date)"

python Run_3_root2h5_signal_jet_multiproc.py -m 12 -p signal
echo "Finished 12 at $(date)"

python Run_3_root2h5_signal_jet_multiproc.py -m 14 -p signal
echo "Finished 14 at $(date)"

python Run_3_root2h5_signal_jet_multiproc.py -m QCD -p background
echo "Finished QCD at $(date)"

python Run_3_root2h5_signal_jet_multiproc.py -m HTo2Tau -p background
echo "Finished HTo2Tau at $(date)"

python Run_3_root2h5_signal_jet_multiproc.py -m TTbar -p background
echo "Finished TTbar at $(date)"

python Run_3_root2h5_signal_jet_multiproc.py -m Wjets -p background
echo "Finished Wjets at $(date)"

python Run_3_root2h5_signal_jet_multiproc.py -m DYto2L -p background
echo "Finished DYto2L at $(date)"

echo "All jobs completed at $(date)"
