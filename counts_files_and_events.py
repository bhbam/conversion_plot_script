# Initialize the environment first:
# source /cvmfs/sft.cern.ch/lcg/views/LCG_105_swan/x86_64-el9-gcc13-opt/setup.sh

import os, glob, re, ROOT
ROOT.gROOT.SetBatch(True)

# ── Dataset paths (same dict as your main script) ──────────────────────────
datasets = {
    '3p7'    : "/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M3p7_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_3p7_GeV/260225_052605/0000",
    '4'      : "/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M4_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_4_GeV/260225_055101/0000",
    '5'      : "/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M5_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_5_GeV/260225_131350/0000",
    '6'      : "/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M6_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_6_GeV/260225_131414/0000",
    '8'      : "/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M8_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_8_GeV/260225_131435/0000",
    '10'     : "/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M10_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_10_GeV/260225_131506/0000",
    '12'     : "/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M12_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_12_GeV/260225_131530/0000",
    '14'     : "/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/HToAATo4Tau_hadronic_tauDecay_M14_Run3_2023/MLAnalyzer_ntuples_using_miniAOD_14_GeV/260225_131603/0000",
    'QCD'    : "/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/QCD_PT-15to7000_TuneCP5_13p6TeV_pythia8/MLAnalyzer_ntuples_using_miniAOD_QCD/260226_160954/0000",
    'HTo2Tau': "/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/GluGluHToTauTau_M-125_TuneCP5_13p6TeV_powheg-pythia8/MLAnalyzer_ntuples_using_miniAOD_HTo2Tau/260226_164125/0000",
    'TTbar'  : "/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/TT_TuneCP5_13p6TeV_powheg-pythia8/MLAnalyzer_ntuples_using_miniAOD_TTbar/260226_160423/0000",
    'Wjets'  : "/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/WtoLNu_2Jets_TuneCP5_13p6TeV_amcatnloFXFX_pythia8_v1/MLAnalyzer_ntuples_using_miniAOD_Wjets/260226_161832/0000",
    'DYto2L' : "/eos/uscms/store/group/lpcml/bbbam/Ntuples_Signal_Background_Feb24_2026/DYto2L_M-50_TuneCP5_13p6TeV_pythia8/MLAnalyzer_ntuples_using_miniAOD_DTTo2L/260226_170136/0000",
}

# ── Name of the TTree inside the ROOT files ─────────────────────────────────
# Change this if your tree has a different name, e.g. "tree", "events", etc.
TREE_NAME = "fevt/RHTree"

# ── Helpers (same as your main script) ──────────────────────────────────────
def alphanum_key(s):
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]

def sort_nicely(lst):
    lst.sort(key=alphanum_key)

# ── Main loop ────────────────────────────────────────────────────────────────
col_w = (12, 10, 14)  # column widths
header = f"{'Dataset':<{col_w[0]}}  {'# Files':>{col_w[1]}}  {'# Events':>{col_w[2]}}"
divider = "-" * (sum(col_w) + 6)

print(divider)
print(header)
print(divider)

grand_total_files  = 0
grand_total_events = 0

for name, path in datasets.items():
    if not os.path.isdir(path):
        print(f"{'[MISSING]':<{col_w[0]}}  {name} → path not found")
        print(f"  {path}")
        continue

    root_files = glob.glob(os.path.join(path, "*.root"))
    sort_nicely(root_files)
    n_files = len(root_files)

    if n_files == 0:
        print(f"{name:<{col_w[0]}}  {n_files:>{col_w[1]}}  {'(no .root files)':>{col_w[2]}}")
        continue

    # Count events by chaining all files and reading the TTree entry count
    chain = ROOT.TChain(TREE_NAME)
    n_bad = 0
    for f in root_files:
        if chain.Add(f, -1) == 0:   # -1 = don't read entries yet, returns 0 on failure
            n_bad += 1

    n_events = chain.GetEntries()

    grand_total_files  += n_files
    grand_total_events += n_events

    note = f"  ({n_bad} file(s) unreadable)" if n_bad else ""
    n_events_str = f"{n_events:,}"
    print(f"{name:<{col_w[0]}}  {n_files:>{col_w[1]}}  {n_events_str:>{col_w[2]}}{note}")

print(divider)
total_events_str = f"{grand_total_events:,}"
print(f"{'TOTAL':<{col_w[0]}}  {grand_total_files:>{col_w[1]}}  {total_events_str:>{col_w[2]}}")
print(divider)
