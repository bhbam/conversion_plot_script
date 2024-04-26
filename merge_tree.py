import os
import sys
import numpy as np
import argparse
import ROOT

## MAIN ##
def main():

    # Keep time
    sw = ROOT.TStopwatch()
    sw.Start()

    # Load input TTrees into TChain
    ggIn = ROOT.TChain("fevt/RHTree")
    # root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_1.root
    # root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_10.root
    # root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_11.root
    # root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_12.root
    # root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_13.root
    # root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_14.root
    # ggIn.Add('root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_15.root')
    # ggIn.Add('root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_16.root')
    # root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_2.root
    # root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_3.root
    # root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_4.root
    # root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_5.root
    # root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_6.root
    # root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_7.root
    # root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_8.root
    # root://cmsxrootd.fnal.gov//store/group/lpcml/bbbam/Samples/HIG_RunIISummer20UL18_HtoAAto4tau/220922_202510/0000/output_9.root
    # ggIn.Add('/uscms/home/bbbam/nobackup/final_root_files_Eff_study/2018_ExoEff_Background_rootfile/ZToEE_NNPDF30_13TeV-powheg_M*.root')
    # ggIn.Add('HIG_RunIISummer20UL18_HtoAAto4tau/output_16.root')
    # ggIn.Add('HIG_RunIISummer20UL18_HtoAAto4tau/output_1.root')
    ggIn.Add('/uscms/home/bbbam/nobackup/analysis/gen_information_H_AA_4Tau/CMSSW_10_6_20/src/Gen/GenInfo_only_H_AA_4Tau_M14_*.root')
    # ggIn.Add('/uscms/home/bbbam/nobackup/analysis/gen_information_H_AA_4Tau/CMSSW_10_6_20/src/Gen/')
    # ggIn.Add('/eos/cms/store/user/mandrews/ML/IMGs/SingleElectronPt50_FEVTDEBUG_n125k_IMG.root')
    nEvts = ggIn.GetEntries()
    print " >> nEvts: ",nEvts

    # Initialize output file as empty clone
    outFileStr = "/uscms/home/bbbam/nobackup/analysis/gen_information_H_AA_4Tau/CMSSW_10_6_20/src/Gen/HtoAAto4tau_merged_output_M14.root"
    # outFileStr = "/uscms/home/bbbam/nobackup/final_root_files_Eff_study/Background.root"
    # outFileStr = "/uscms/home/bbbam/nobackup/final_root_files_Eff_study/Full_datasetcd .root"
    outFile = ROOT.TFile(outFileStr, "RECREATE")
    ggOut = ggIn.CloneTree(-1)
    outFile.Write()
    print " >> Output file:",outFileStr

    sw.Stop()
    print " >> Real time:",sw.RealTime()/60.,"minutes"
    print " >> CPU time: ",sw.CpuTime() /60.,"minutes"


#_____ Call main() ______#
if __name__ == '__main__':
    main()
