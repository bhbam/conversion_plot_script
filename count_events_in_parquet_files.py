from pyarrow.parquet import ParquetFile
import pyarrow as pa
import glob
local="/eos/uscms//store/user/bbbam/IMG_v2/signals_Tau_classifier"
# files = glob.glob("root://cmseos.fnal.gov//store/user/bbbam/IMG_v2/signals_Tau_classifier/*valid*.parquet")
files = glob.glob(f"{local}/*.parquet")
total = 0
print("Total files   :", len(files))
for i in range (len(files)):
    pf = ParquetFile(files[i])
    A = pf.num_row_groups
    print(files[i], "----->total number of data", A)
    total = total + A
    first_tenrow= next(pf.iter_batches(batch_size=10))
    df=pa.Table.from_batches([first_tenrow]).to_pandas()
    print(df)
print("Total :  ", total)
