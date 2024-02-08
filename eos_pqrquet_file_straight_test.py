### cd nobackup/storage/local/data1/gpuscratch/bbbam/IMG/ then copy required files from eos to here
### e.g:   xrdcp -r root://cmseos.fnal.gov//store/group/lpcml/ddicroce/IMG/AToTauTau_Signal/Run2018B_Tau_AOD_12Nov2019_UL2018_part1_signal/Run2018A_Tau_AOD_12Nov2019_UL2018_signal.parquet.99 .
### then pyhton run this script to see frist 10 lines in pandas data frame


from pyarrow.parquet import ParquetFile
import pyarrow as pa
import glob
#pf = ParquetFile('/storage/local/data1/gpuscratch/bbbam/IMG/train/HToTauTau_Hadronic_RHAnalyzer_M10_part1.parquet')
# pf = ParquetFile('/storage/local/data1/gpuscratch/bbbam/IMG_aToTauTau_unbaised/train/IMG_aToTauTau_ML.parquet.2')
# pf = ParquetFile('/storage/local/data1/gpuscratch/bbbam/IMG/AToTauTau/HToTauTau_Hadronic_RHAnalyzer_M12_part1.parquet')
# pf = ParquetFile('/storage/local/data1/gpuscratch/bbbam/IMG/Run2018B_Tau_AOD_12Nov2019_UL2018_part1_signal/Run2018B_Tau_AOD_12Nov2019_UL2018_part1_signal.parquet.110')
# pf = ParquetFile('/storage/local/data1/gpuscratch/bbbam/ParquetFiles_correctTrackerLayerHits_SecVtxInfoAdded/DYToTauTau_M-50_13TeV-powheg_pythia8/AODJets/DYToTauTau_M-50_13TeV-powheg_pythia8_9.parquet')
# pf = ParquetFile('/eos/uscms/store/user/bbbam/IMG_classification/merged_background/DYY.parquet')
# pf = ParquetFile('/storage/local/data1/gpuscratch/bbbam/classification/new/GGH_TauTau_valid.parquet')
# pf = ParquetFile('../H_AA_GG/data/IMG_aToTauTau_ML.parquet.100')
# pf = ParquetFile('/storage/local/data1/gpuscratch/bbbam/background/DYToTauTau_M-50_13TeV_train.parquet')
files = glob.glob('/storage/local/data1/gpuscratch/bbbam/aToTauTau_unboosted/IMG_ATotauTau_Hadronic_M10_unboosted_v2_0.parquet')
# files = glob.glob("/eos/uscms/store/user/bhbam/IMG_v2/signal/IMG_H_AATo4Tau_Hadronic_tauDR0p4_M5_signal_v2_1.parquet")
print("Total files   :", len(files))
for i in range (len(files)):
    pf = ParquetFile(files[i])
    print(files[i], "----->total number of data", pf.num_row_groups)
    first_tenrow= next(pf.iter_batches(batch_size=10))
    df=pa.Table.from_batches([first_tenrow]).to_pandas()
    print(df)
    # print(df.info())
