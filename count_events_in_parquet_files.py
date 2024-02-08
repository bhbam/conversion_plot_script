from pyarrow.parquet import ParquetFile
import pyarrow as pa
import glob
# local='/storage/local/data1/gpuscratch/bbbam/ParquetFiles_correctTrackerLayerHits_SecVtxInfoAdded/DYToTauTau_M-50_13TeV-powheg_pythia8/AODJets'
# local='/storage/local/data1/gpuscratch/bbbam/ParquetFiles_correctTrackerLayerHits_SecVtxInfoAdded/QCD_Pt-15to7000_TuneCP5_Flat_13TeV_pythia8/AODJets'
# local='/storage/local/data1/gpuscratch/bbbam/ParquetFiles_correctTrackerLayerHits_SecVtxInfoAdded/TTToHadronic_TuneCP5_13TeV_powheg-pythia8/AODJets'
local='/storage/local/data1/gpuscratch/bbbam/ParquetFiles_correctTrackerLayerHits_SecVtxInfoAdded/WJetsToLNu_TuneCP5_13TeV_madgraphMLM-pythia8/AODJets'
# files = glob.glob("root://cmseos.fnal.gov//store/user/bbbam/IMG_v2/signals_Tau_classifier/*valid*.parquet")
files = glob.glob(f"{local}/*.parquet")
total = 0
print("Total files   :", len(files))
for i in range (len(files)):
    pf = ParquetFile(files[i])
    A = pf.num_row_groups
    total = total + A
    if i==0:
        print(files[i], "----->total number of data", A)
        first_tenrow= next(pf.iter_batches(batch_size=5))
        df=pa.Table.from_batches([first_tenrow]).to_pandas()
        print(df)
print("Total :  ", total)
