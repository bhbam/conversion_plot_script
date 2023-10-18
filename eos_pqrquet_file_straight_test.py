### cd nobackup/storage/local/data1/gpuscratch/bbbam/IMG/ then copy required files from eos to here
### e.g:   xrdcp -r root://cmseos.fnal.gov//store/group/lpcml/ddicroce/IMG/AToTauTau_Signal/Run2018B_Tau_AOD_12Nov2019_UL2018_part1_signal/Run2018A_Tau_AOD_12Nov2019_UL2018_signal.parquet.99 .
### then pyhton run this script to see frist 10 lines in pandas data frame


from pyarrow.parquet import ParquetFile
import pyarrow as pa
#pf = ParquetFile('/storage/local/data1/gpuscratch/bbbam/IMG/train/HToTauTau_Hadronic_RHAnalyzer_M10_part1.parquet')
# pf = ParquetFile('/storage/local/data1/gpuscratch/bbbam/IMG_aToTauTau_unbaised/train/IMG_aToTauTau_ML.parquet.2')
# pf = ParquetFile('/storage/local/data1/gpuscratch/bbbam/IMG/AToTauTau/HToTauTau_Hadronic_RHAnalyzer_M12_part1.parquet')
# pf = ParquetFile('/storage/local/data1/gpuscratch/bbbam/IMG/Run2018B_Tau_AOD_12Nov2019_UL2018_part1_signal/Run2018B_Tau_AOD_12Nov2019_UL2018_part1_signal.parquet.110')
pf = ParquetFile('/storage/local/data1/gpuscratch/bbbam/signal/IMG_H_AATo4Tau_Hadronic_tauDR0p4_M3p7_signal_v2.parquet')
# pf = ParquetFile('../H_AA_GG/data/IMG_aToTauTau_ML.parquet.100')
first_tenrow= next(pf.iter_batches(batch_size=10))
df=pa.Table.from_batches([first_tenrow]).to_pandas()
print(df)
print(df.info())
