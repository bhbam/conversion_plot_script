
# Initilize the enviroment first: "source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh'"
import os, glob, re
from multiprocessing import Pool

import argparse
parser = argparse.ArgumentParser(description='Process dataset 0-9')
parser.add_argument('-m', '--Mass',     default='3p7',    type=str, help='select signal mass as str')
parser.add_argument('-p', '--process',     default='signal',    type=str, help='select signal or background')

args = parser.parse_args()

Mass = args.Mass




if args.process=='signal':
    local = f"/eos/uscms/store/user/bbbam/Run_3_IMG_from_Ruchi/signals/IMG_HToAATo4Tau_Hadronic_signal_mass_{Mass}_GeV"
    decay = f"IMG_HToAATo4Tau_Hadronic_signal_mass_{Mass}_GeV_normalized"
    outDir=f"/eos/uscms/store/user/bbbam/Run_3_IMG_from_Ruchi/signals_normalized/{decay}"

if args.process=='background':
    local = f"/eos/uscms/store/user/bbbam/Run_3_IMG_from_Ruchi/background/IMG_HToAATo4Tau_Hadronic_background_{Mass}_GeV"
    decay = f"IMG_HToAATo4Tau_Hadronic_background_{Mass}_normalized"
    outDir=f"/eos/uscms/store/user/bbbam/Run_3_IMG_from_Ruchi/background_normalized/{decay}"

if args.process=='massreg':
    local = f"/eos/uscms/store/user/bhbam/Run_3_IMG_mass_reg_unphy_m0To3p6/IMG_AToTau_Hadronic_massregssion_samples_m1p8To3p6_pt30To300"
    decay = f"IMG_AToTau_Hadronic_mass_reg_{Mass}_pt30To300_normalized"
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


rhFileList = '%s/*.h5'%(local)
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

    proc_file = 'h5_to_normalized_h5_conversion_dynamic_size.py'
    processes = ['%s --input_file %s --output_data_path %s'%(proc_file, rhFile, outDir) for rhFile in files_ ]
    print(' >> Process[0]: %s'%processes[0])

    pool = Pool(processes=len(processes))
    pool.map(run_process, processes)
print("--------------Process is complete----------------")
