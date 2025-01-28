
# Initilize the enviroment first: "source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh'"
import os, glob, re, ROOT
from multiprocessing import Pool

import argparse
parser = argparse.ArgumentParser(description='Process dataset 0-9')
parser.add_argument('-m', '--Mass',     default='m1p8To3p6',    type=str, help='select signal mass as str')


args = parser.parse_args()

Mass = args.Mass


local ={
'm1p8To3p6':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_run3/GEN_SIM_AToTau_m1p8To3p6_pt30To300/RHAnalyzer_AToTau_Hadronic_m1p8To3p6_v2/250126_055706/0000"

}.get(Mass, None)


decay = f"IMG_AToTau_Hadronic_massregssion_samples_{Mass}_pt30To300"
outDir=f"/eos/uscms/store/user/bbbam/Run_3_IMG_mass_reg_unphy_m0To3p6/{decay}"

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)',s)]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def run_process(process):
    os.system('python %s'%process)


rhFileList = '%s/output*.root'%(local)
rhFileList = glob.glob(rhFileList)
assert len(rhFileList) > 0
print (" >> %d files found"%len(rhFileList))
sort_nicely(rhFileList)


files_per_run = 16

file_idx_ = list(range( 0, len(rhFileList), files_per_run ))
n_iter_ = len( file_idx_ )
file_idx_.append( len(rhFileList) )
print ( file_idx_ )

for irun_ in range( n_iter_ ):
    files_ = rhFileList[ file_idx_[ irun_ ] : file_idx_[irun_+1] ]
    for idx_, file_ in enumerate(files_):
        print(' >> Input File[%d]: %s' % ( idx_, file_ ) )

    # Output path
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    print(' >> Output directory: %s'%outDir)

    proc_file = 'Run_3_convert_root2h5_mass_reg_unphysical_jet.py'
    processes = ['%s -i %s -o %s -d %s -n %d'%(proc_file, rhFile, outDir, decay, ( irun_*files_per_run + i + 1 )) for i,rhFile in enumerate( files_ )]
    print(' >> Process[0]: %s'%processes[0])

    pool = Pool(processes=len(processes))
    pool.map(run_process, processes)
print("--------------Process is complete----------------")
