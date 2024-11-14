import ROOT
import glob
local = "/uscms/home/bbbam/nobackup/analysis/gen_information_H_AA_4Tau/CMSSW_10_6_20/src/Gen"
# Open the existing ROOT file in update mode
rhFileList = '%s/Unboosted_GenInfo_only_H_AA_4Tau_M8.root'%(local)
rhFileList = glob.glob(rhFileList)
assert len(rhFileList) > 0
total_files = len(rhFileList)
# print (" >> %d files found")%total_files
print("Root file ", rhFileList)


rhTree = ROOT.TChain("fevt/RHTree")
nEvts_start = 0
for filename in rhFileList:
  rhTree.Add(filename)
nEvts_end =  rhTree.GetEntries()
print("Total events  ", nEvts_end)

# outpath = "%s/updated_mantuple.root"%local
# print('>> Initializing output mantuple: %s'%outpath)
# file_out = ROOT.TFile.Open(outpath, 'RECREATE')
# file_out.mkdir("fevt")
# file_out.cd("fevt")
# tree_out = rhTree.CloneTree(0)
#
# nPhoMax = -9999.99
#
# ma = np.zeros(nPhoMax, dtype='float32')
# tree_out.Branch('ma', ma, 'ma[nPho]/F')
