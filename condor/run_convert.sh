#!/bin/bash
set -e  # exit immediately if a command fails
echo " >> Starting job on $(date)"
echo " >> Host: $(hostname)"
echo " >> PWD: $(pwd)"

# Load CMS/LCG environment
# echo " >> Sourcing LCG environment..."
# source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh
# Condor sets X509_USER_PROXY automatically if use_x509userproxy = true
if [ -z "$X509_USER_PROXY" ]; then
    echo "ERROR: No proxy set in environment!"
    exit 1
fi

echo " >> Proxy being used: $X509_USER_PROXY"
if ! voms-proxy-info -all; then
    echo "ERROR: Proxy is invalid or expired."
    exit 1
fi

# Debug Python
echo " >> Python path: $(which python3)"
python3 --version

# Run python script (local copy after transfer)
echo " >> Running test.py"
python3 test.py

echo " >> Job finished at $(date)"
