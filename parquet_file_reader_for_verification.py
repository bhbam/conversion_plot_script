import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile

# outStr='/storage/local/data1/gpuscratch/bbbam/signal/IMG_H_AATo4Tau_Hadronic_tauDR0p4_M3p7_signal_v2_1.parquet'
# outStr='/storage/local/data1/gpuscratch/bbbam/ParquetFiles_correctTrackerLayerHits_SecVtxInfoAdded/DYToTauTau_M-50_13TeV-powheg_pythia8/AODJets/DYToTauTau_M-50_13TeV-powheg_pythia8_9.parquet'
outStr='/storage/local/data1/gpuscratch/bbbam/classification/train/GG_H_TauTau_Hadronic_background_train.parquet'
# outStr='IMG_H_AATo4Tau_Hadronic_tauDR0p4_M5_signal_v2.parquet'
pqIn = pq.ParquetFile(outStr)
print(pqIn.metadata)
print(pqIn.schema)
# X = pqIn.read_row_group(0, columns=['y','secVtx_jet_dR','secVtx_Pt','X_jet']).to_pydict()
# #X = pqIn.read_row_group(0, columns=['y','am','dR','pt','m0','iphi','ieta','pdgId']).to_pydict()
# print(X)
