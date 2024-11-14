
# Initilize the enviroment first: "source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh'"
import os, glob, re, ROOT
from multiprocessing import Pool

import argparse
parser = argparse.ArgumentParser(description='Process dataset 0-9')
parser.add_argument('-m', '--mass_range',     default='m3p6To18',    type=str, help='select: m1p2To3p6 or m3p6To18')
parser.add_argument('-n', '--dataset',     default=0,    type=int, help='number of dataset used[0-9]')
args = parser.parse_args()
n = args.dataset
mass_range = args.mass_range

subsets= ['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009','0010']


subset = subsets[n]
print("processing dataset --->  ", subset)

if mass_range =="m1p2To3p6":
    local='/eos/uscms/store/group/lpcml/bbbam/Ntuples_run3/GEN_SIM_ATo2Tau_m1p2To3p6_pt30To300_v4/RHAnalyzer_ATo4Tau_Hadronic_m1p2To3p6/241108_195542/%s'%subset
if mass_range =="m3p6To18":
    local='/eos/uscms/store/group/lpcml/bbbam/Ntuples_run3/GEN_SIM_ATo2Tau_m3p6To18_pt30To300_v2/RHAnalyzer_ATo4Tau_Hadronic_m3p6To18/241109_114917/%s'%subset
else:
    print("select correct mass range!!!!!!!!")
decay = f"IMG_aToTauTau_Hadronic_{mass_range}_pt30T0300"
outDir="/eos/uscms/store/user/bbbam/Run_3_IMG/%s/%s"%(decay,subset)



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

    proc_file = 'Run_3_convert_root2pq_jet.py'
    processes = ['%s -i %s -o %s -d %s -n %d'%(proc_file, rhFile, outDir, decay, ( irun_*files_per_run + i + 1 )) for i,rhFile in enumerate( files_ )]
    print(' >> Process[0]: %s'%processes[0])

    pool = Pool(processes=len(processes))
    pool.map(run_process, processes)
print("--------------Process is complete----------------")
