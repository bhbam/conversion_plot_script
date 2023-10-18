import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile

# outStr='/storage/local/data1/gpuscratch/bbbam/signal/IMG_H_AATo4Tau_Hadronic_tauDR0p4_M3p7_signal_v2.parquet'
outStr='IMG_H_AATo4Tau_Hadronic_tauDR0p4_M5_signal_v2.parquet'
pqIn = pq.ParquetFile(outStr)
print(pqIn.metadata)
print(pqIn.schema)
X = pqIn.read_row_group(0, columns=['y','am','iphi','ieta']).to_pydict()
#X = pqIn.read_row_group(0, columns=['y','am','dR','pt','m0','iphi','ieta','pdgId']).to_pydict()
print(X)
