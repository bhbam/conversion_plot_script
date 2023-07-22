
# cmd = "source /cvmfs/sft.cern.ch/lcg/views/LCG_97/x86_64-centos7-gcc8-opt/setup.sh"
import os, glob, re, ROOT
from multiprocessing import Pool

#print("ENVIRONMENT: ", os.environ)

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

decay='IMG_aToTauTau_datase_1_reunbaied_mannul_v2_8_tracker_layer' # name inse IMG directory
# decay='Upsilon1s_ToTauTau_Hadronic_tauDR0p4_validation_no_pix_layers' # name inse IMG directory
cluster = 'FNAL'
local =''
outDir=''
if(cluster=='CERN'):
    outDir='/eos/user/b/bhbam/%s'%(decay)
if(cluster=='FNAL'):
    local='/eos/uscms/store/user/bbbam/aToTauTau_Hadronic_tauDR0p4_m3p6To16_pT30To180_ctau0To3_eta0To1p4_pythia8_unbiased4ML_dataset_1/aToTauTau_Hadronic_tauDR0p4_m3p6To14_dataset_1_reunbaised_mannual_v2/230420_051900/0000/'
    # local='/eos/uscms/store/user/bbbam/Upsilon1s_ToTauTau_Hadronic_tauDR0p4_eta0To2p4_pythia8_validationML/Upsilon1s_ToTauTau_Hadronic_tauDR0p4_validation/230314_172829/0000'
    outDir='/eos/uscms/store/group/lpcml/bbbam/IMG/%s'%(decay)

# Paths to input files
# filelist = 'list_aToTauTau_Hadronic_tauDR0p4_m3p6To14_dataset_1_reunbaised_mannual_v2_ntuples.txt'
# with open(filelist) as list_:
#     content = list_.readlines()
# rhFileList = [x.strip() for x in content]
# print(" >> Input file list: %s"%rhFileList)
# assert len(rhFileList) > 0
# print(" >> %d files found"%len(rhFileList))
# sort_nicely(rhFileList)

rhFileList = '%s/output*.root'%(local)
print " >> Input file list: %s"%rhFileList
rhFileList = glob.glob(rhFileList)
assert len(rhFileList) > 0
print " >> %d files found"%len(rhFileList)
sort_nicely(rhFileList)


files_per_run = 10

file_idx_ = range( 0, len(rhFileList), files_per_run )
n_iter_ = len( file_idx_ )
file_idx_.append( len(rhFileList) )
print ( file_idx_ )

for irun_ in range( n_iter_ ):
    #to do run 10,42
    # if (irun_ < 43) : continue
    #if (irun_ > 2) : continue
    files_ = rhFileList[ file_idx_[ irun_ ] : file_idx_[irun_+1] ]
    for idx_, file_ in enumerate(files_):
        print(' >> Input File[%d]: %s' % ( idx_, file_ ) )

    # Output path
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    print(' >> Output directory: %s'%outDir)

    proc_file = 'convert_root2pq_jet.py'
    processes = ['%s -i %s -o %s -d %s -n %d'%(proc_file, rhFile, outDir, decay, ( irun_*files_per_run + i + 1 )) for i,rhFile in enumerate( files_ )]
    print(' >> Process[0]: %s'%processes[0])

    pool = Pool(processes=len(processes))
    pool.map(run_process, processes)
