
# Initilize the enviroment first: "source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh'"
import os, glob, re, ROOT
from multiprocessing import Pool

import argparse
parser = argparse.ArgumentParser(description='Process dataset 0-9')
parser.add_argument('-m', '--Mass',     default='3p7',    type=str, help='select signal mass as str')

args = parser.parse_args()

Mass = args.Mass


local ={
'3p7':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_run3/HToAATo4Tau_hadronic_tauDecay_M3p7_Run3_2023/RHAnalyzer_HToAATo4Tau_Hadronic_M3p7/241204_180445/0000"
,'4':None
,'5':None
,'6':None
,'8':None
,'10':None
,'12':None
,'14':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_run3/HToAATo4Tau_hadronic_tauDecay_M14_Run3_2023/RHAnalyzer_HToAATo4Tau_Hadronic_M14/241204_181504/0000"
}.get(Mass, None)

decay = f"IMG_HToAATo4Tau_Hadronic_signal_mass_{Mass}_GeV"
outDir=f"/eos/uscms/store/user/bbbam/Run_3_IMG/signals/{decay}"



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


files_per_run = 4

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

    proc_file = 'Run_3_convert_root2h5_signal_jet.py'
    processes = ['%s -i %s -o %s -d %s -n %d'%(proc_file, rhFile, outDir, decay, ( irun_*files_per_run + i + 1 )) for i,rhFile in enumerate( files_ )]
    print(' >> Process[0]: %s'%processes[0])

    pool = Pool(processes=len(processes))
    pool.map(run_process, processes)
print("--------------Process is complete----------------")
