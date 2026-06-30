
# Initilize the enviroment first: "source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh'"
import os, glob, re, ROOT
from multiprocessing import Pool

import argparse
parser = argparse.ArgumentParser(description='Process dataset 0-9')
parser.add_argument('-m', '--Mass',     default='3p7',    type=str, help='select signal mass as str')
parser.add_argument('-p', '--process',     default='signal',    type=str, help='select signal or background')

args = parser.parse_args()

Mass = args.Mass


# local ={
# '3p7':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M3p7_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_3p7_GeV/260225_052605/0000"
# ,'4':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M4_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_4_GeV/260225_055101/0000"
# ,'5':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M5_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_5_GeV/260225_131350/0000"
# ,'6':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M6_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_6_GeV/260225_131414/0000"
# ,'8':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M8_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_8_GeV/260225_131435/0000"
# ,'10':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M10_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_10_GeV/260225_131506/0000"
# ,'12':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M12_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_12_GeV/260225_131530/0000"
# ,'14':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M14_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_14_GeV/260225_131603/0000"
# ,'QCD':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/QCD_PT-15to7000_TuneCP5_13p6TeV_pythia8/MLAnalyzer_ntuples_using_miniAOD_QCD/260226_160954/0000"
# ,'HTo2Tau':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/GluGluHToTauTau_M-125_TuneCP5_13p6TeV_powheg-pythia8/MLAnalyzer_ntuples_using_miniAOD_HTo2Tau/260226_164125/0000"
# ,'TTbar':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/TT_TuneCP5_13p6TeV_powheg-pythia8/MLAnalyzer_ntuples_using_miniAOD_TTbar/260226_160423/0000"
# ,'Wjets':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/WtoLNu_2Jets_TuneCP5_13p6TeV_amcatnloFXFX_pythia8_v1/MLAnalyzer_ntuples_using_miniAOD_Wjets/260226_161832/0000"
# ,'DYto2L':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/DYto2L_M-50_TuneCP5_13p6TeV_pythia8/MLAnalyzer_ntuples_using_miniAOD_DTTo2L/260226_170136/0000"
# }.get(Mass, None)


local ={
'3p7':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_June15_2026/HToAATo4Tau_hadronic_tauDecay_M3p7_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_June153p7_GeV/*/0000"
,'4':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_June15_2026/HToAATo4Tau_hadronic_tauDecay_M4_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_June154_GeV/*/0000"
,'5':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_June15_2026/HToAATo4Tau_hadronic_tauDecay_M5_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_June155_GeV/*/0000"
,'6':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_June15_2026/HToAATo4Tau_hadronic_tauDecay_M6_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_June156_GeV/*/0000"
,'8':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_June15_2026/HToAATo4Tau_hadronic_tauDecay_M8_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_June158_GeV/*/0000"
# ,'10':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M10_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_10_GeV/260225_131506/0000"
# ,'12':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M12_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_12_GeV/260225_131530/0000"
# ,'14':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M14_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_14_GeV/260225_131603/0000"
,'QCD':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_June15_2026/QCD_PT-15to7000_TuneCP5_13p6TeV_pythia8/MLAnalyzer_ntuples_using_miniAOD_June15QCD/*/0000"
,'HTo2Tau':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_June15_2026/GluGluHToTauTau_M-125_TuneCP5_13p6TeV_powheg-pythia8/MLAnalyzer_ntuples_using_miniAOD_June15HTo2Tau/*/0000"
,'TTbar':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_June15_2026/TT_TuneCP5_13p6TeV_powheg-pythia8/MLAnalyzer_ntuples_using_miniAOD_June15TTbar/*/0000"
,'Wjets':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_June15_2026/WtoLNu_2Jets_TuneCP5_13p6TeV_amcatnloFXFX_pythia8_v1/MLAnalyzer_ntuples_using_miniAOD_June15Wjets/*/0000"
,'DYto2L':"/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_June15_2026/DYto2L_M-50_TuneCP5_13p6TeV_pythia8/MLAnalyzer_ntuples_using_miniAOD_June15DTTo2L/*/0000"
}.get(Mass, None)
max_number_files ={
'3p7':226
,'4':226
,'5':228
,'6':232
,'8':240
,'QCD':295
,'HTo2Tau':333
,'TTbar':236
,'Wjets':933
,'DYto2L':1448

}.get(Mass, None)
# local = "/eos/uscms/store/user/bbbam/signal_background_rootfile_9999_ntuples"

mass_to_write=0

if args.process == 'signal':
    label = 1
    decay = f"IMG_signal_mass_{Mass}_GeV_miniAOD"
    outDir=f"/eos/uscms/store/group/lpcml/bbbam/signals_h5_from_miniAOD_June_15_2026/{decay}"
    # outDir=f"/storage/local/data1/gpuscratch/bbbam/{decay}"
    mass_to_write = {'3p7':3.7, '4':4, '5':5, '6':6, '8':8, '10':10, '12':12, '14':14}.get(Mass, None)

if args.process == 'background':
    label = 0
    decay = f"IMG_background_{Mass}_miniAOD"
    outDir=f"/eos/uscms/store/group/lpcml/bbbam/backgrounds_h5_from_miniAOD_June_15_2026/{decay}"
    mass_to_write = 0

# outDir=f"/eos/uscms/store/user/bbbam/signal_background_rootfile_9999_to_h5"

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


rhFileList = '%s/*.root'%(local)
rhFileList = glob.glob(rhFileList)
assert len(rhFileList) > 0
print (" >> %d files found"%len(rhFileList))
sort_nicely(rhFileList)

files_per_run = 8

file_idx_ = list(range( 0, len(rhFileList), files_per_run ))
n_iter_ = len( file_idx_ )
file_idx_.append( len(rhFileList) )
print ( file_idx_ )

for irun_ in range( n_iter_ ):
    files_ = rhFileList[ file_idx_[ irun_ ] : file_idx_[irun_+1] ]
    for idx_, file_ in enumerate(files_):
        print(' >> Input File[%d]: %s' % ( idx_, file_ ) )
    # if file_idx_[irun_+1] > max_number_files: # commented this to process all files
    #     exit()
    # Output path
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    print(' >> Output directory: %s'%outDir)

    # proc_file = 'Run_3_convert_root2h5_signal_jet.py'
    # proc_file = 'Run_3_convert_root2h5_signal_background_miniAOD_AOD.py'
    proc_file = 'Run_3_convert_root2h5_signal_background_miniAOD_dijet_ditau.py'
    processes = ['%s -i %s -m %s -y %d -o %s -d %s -n %d'%(proc_file, rhFile, mass_to_write, label, outDir, decay, ( irun_*files_per_run + i + 1 )) for i,rhFile in enumerate( files_ )]
    print(' >> Process[0]: %s'%processes[0])

    pool = Pool(processes=len(processes))
    pool.map(run_process, processes)
print("--------------Process is complete----------------")
