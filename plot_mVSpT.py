#!/usr/bin/env python
import os, glob

import numpy as np

import ROOT
from ROOT import TCanvas, TPad, TFile, TPaveText, TLegend
from ROOT import gBenchmark, gStyle, gROOT, TStyle
from ROOT import TH1D, TF1, TGraphErrors, TMultiGraph

from math import sqrt

from array import array

import tdrstyle
tdrstyle.setTDRStyle()

import CMS_lumi
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-n', type=int, default=0, help='dataset number[0-9]')
args = parser.parse_args()
dataset = args.dataset
subset = ['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009']
# out_dir = 'plots_A_2Tau_m3p6To14p8_v2_dataset_2_unbiasing_status/%s/'%subset[dataset]
out_dir = 'plots_A_2Tau_m14p8To17p2_v2_dataset_2_unbiasing_status/%s/'%subset[dataset]
if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

#change the CMS_lumi variables (see CMS_lumi.py)
CMS_lumi.lumi_13TeV = '13 TeV'
CMS_lumi.writeExtraText = 1
#CMS_lumi.extraText = 'Preliminary'
CMS_lumi.extraText = 'Simulation'

iPos    = 0
iPeriod = 0

gStyle.SetOptFit(0)

def loadcanvas(name):
  canvas = TCanvas(name,name,400,20,1400,1000)
  canvas.SetFillColor(0)
  canvas.SetBorderMode(0)
  canvas.SetFrameFillStyle(0)
  canvas.SetFrameBorderMode(0)
  canvas.SetTickx(0)
  canvas.SetTicky(0)
  return canvas

def loadlegend(top, bottom, left, right):
  relPosX    = 0.001
  relPosY    = 0.005
  posX = 1 - right - relPosX*(1-left-right)
  posY = 1 - top - relPosY*(1-top-bottom)
  legendOffsetX = 0.0
  legendOffsetY = - 0.05
  textSize   = 0.05
  textFont   = 60
  legendSizeX = 0.4
  legendSizeY = 0.2
  legend = TLegend(posX-legendSizeX+legendOffsetX,posY-legendSizeY+legendOffsetY,posX+legendOffsetX,posY+legendOffsetY)
  legend.SetTextSize(textSize)
  legend.SetLineStyle(0)
  legend.SetBorderSize(0)
  return legend

histos={}
# eosDir='root://cmsxrootd.fnal.gov//store/user/ddicroce/test'
#eosDir='/eos/uscms/store/user/ddicroce/test'


# filelist = 'list_aToTauTau_Hadronic_tauDR0p4_m3p6To14_dataset_1_reunbaised_mannual_v2_ntuples.txt'
# #filelist = '/uscms/home/bbbam/nobackup/list_aToTauTau_Hadronic_tauDR0p4_m3p6To16_unbaised_without_antiparticle_ntuples.txt'
# #filelist= 'testlist_sim_Jul14.txt'
# with open(filelist) as list_:
#     content = list_.readlines()
# paths = [x.strip() for x in content]
# print(paths)
#
# for path in paths:

# local='/eos/uscms/store/group/lpcml/bbbam/aToTauTau_Hadronic_tauDR0p4_m3p6To16_pT30To180_ctau0To3_eta0To1p4_pythia8_unbiased4ML_dataset_1/aToTauTau_Hadronic_tauDR0p4_m3p6To14p8_dataset_1_v2/230825_060154/'
# local='/eos/uscms/store/group/lpcml/bbbam/aToTauTau_Hadronic_tauDR0p4_m3p6To16_pT30To180_ctau0To3_eta0To1p4_pythia8_unbiased4ML_dataset_2/aToTauTau_Hadronic_tauDR0p4_m3p6To14p8_dataset_2_v2/230825_055652/'
local='/eos/uscms/store/group/lpcml/bbbam/aToTauTau_Hadronic_tauDR0p4_m14To17p2_pT30To180_ctau0To3_eta0To2p4_pythia8_dataset_2/aToTauTau_Hadronic_tauDR0p4_m14p8To17p2_eta0To2p4_pythia8_unbiased4ML_v2_dataset_2/230826_063114/'
rhFileList = '%s/%s/output*.root'%(local,subset[dataset])
# print " >> Input file list: %s"%rhFileList
rhFileList = glob.glob(rhFileList)
assert len(rhFileList) > 0
print " >> %d files found"%len(rhFileList)
# sort_nicely(rhFileList)
files_ = []
firstfile = True
for path in rhFileList:
  files_.append( TFile.Open(path))
  # files_ = TFile.Open(path)
  tmp_2d = files_[-1].Get('fevt/h_a_m_pT')
  tmp_m  = files_[-1].Get('fevt/h_jet_ma')
  tmp_pt = files_[-1].Get('fevt/h_jet_pta')
  if (firstfile):
    histos['mVSpT'] = tmp_2d.Clone('mVSpT')
    histos['mass']  = tmp_m.Clone('m')
    histos['pt']    = tmp_pt.Clone('pt')
    firstfile = False
  if not (firstfile):
    histos['mVSpT'].Add(tmp_2d)
    histos['mass'].Add(tmp_m)
    histos['pt'].Add(tmp_pt)

print (histos['mVSpT'].GetNbinsX())
print (histos['mVSpT'].GetNbinsY())
binx = []
biny = []
binz = []

histos['mVSpT_inverted'] = histos['mVSpT'].Clone('mVSpT_inverted')
binint = histos['mVSpT'].Integral()
binmax = 0
for iBinX in range(histos['mVSpT'].GetNbinsX()):
  #binx.append(histos['mVSpT'].GetXaxis().GetBinUpEdge(iBinX+1))
  for iBinY in range(histos['mVSpT'].GetNbinsY()):
    binz.append(histos['mVSpT'].GetBinContent(iBinX+1,iBinY+1))
    if (histos['mVSpT'].GetBinContent(iBinX+1,iBinY+1) > binmax):
      binmax = histos['mVSpT'].GetBinContent(iBinX+1,iBinY+1)
print(binmax)

histos['mVSpT_ratio'] = histos['mVSpT'].Clone('mVSpT_ratio')
for iBinX in range(histos['mVSpT_ratio'].GetNbinsX()):
  for iBinY in range(histos['mVSpT_ratio'].GetNbinsY()):
    if (histos['mVSpT'].GetBinContent(iBinX+1,iBinY+1) == 0): continue   #Avoids division by 0, in the case that the bin content was 0
    histos['mVSpT_ratio'].SetBinContent(iBinX+1, iBinY+1, ((1/binmax)*(histos['mVSpT'].GetBinContent(iBinX+1,iBinY+1))) )
    histos['mVSpT_inverted'].SetBinContent(iBinX+1, iBinY+1, (1/binmax)*(binint/histos['mVSpT'].GetBinContent(iBinX+1,iBinY+1)))

histos['mass_inverted'] = histos['mass'].Clone('mass_inverted')
massint = histos['mass'].Integral()
massmax = 0
for iBinX in range(histos['mass'].GetNbinsX()):
  binx.append(histos['mass'].GetXaxis().GetBinUpEdge(iBinX+1))
  if (massmax < massint/histos['mass'].GetBinContent(iBinX+1)):
    massmax = massint/histos['mass'].GetBinContent(iBinX+1)
for iBinX in range(histos['mass_inverted'].GetNbinsX()):
  if (histos['mass'].GetBinContent(iBinX+1) == 0): continue  #Avoids division by 0, in the case that the bin content was 0
  histos['mass_inverted'].SetBinContent(iBinX+1, (1/massmax)*(massint/histos['mass'].GetBinContent(iBinX+1)))

histos['pt_inverted'] = histos['pt'].Clone('pt_inverted')
ptint = histos['pt'].Integral()
ptmax = 0
for iBinX in range(histos['pt'].GetNbinsX()):
  biny.append(histos['pt'].GetXaxis().GetBinUpEdge(iBinX+1))
  if (ptmax < ptint/histos['pt'].GetBinContent(iBinX+1)):
    ptmax = ptint/histos['pt'].GetBinContent(iBinX+1)
for iBinX in range(histos['pt_inverted'].GetNbinsX()):
  if (histos['pt'].GetBinContent(iBinX+1) == 0): continue  #Avoids division by 0, in the case that the bin content was 0
  histos['pt_inverted'].SetBinContent(iBinX+1, (1/ptmax)*(ptint/histos['pt'].GetBinContent(iBinX+1)))



# print(binx)
# print(biny)
# print(binz)

canvas = loadcanvas("c1")
canvas.cd()
histos['mVSpT'].GetXaxis().SetTitle("m^{a} (GeV)")
histos['mVSpT'].GetYaxis().SetTitle("p_{T}^{a} (GeV)")
histos['mVSpT'].SetMinimum(0)
histos['mVSpT'].Draw('COLZ TEXT')
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
canvas.SaveAs('%s/a_massVspT_biased.root'%(out_dir))
canvas.SaveAs('%s/a_massVspT_biased.png'%(out_dir))

canvas = loadcanvas("c2")
canvas.cd()
legend = loadlegend(canvas.GetTopMargin(), canvas.GetBottomMargin(), canvas.GetLeftMargin(), canvas.GetRightMargin())
histos['mass'].SetLineColor(2)
histos['mass'].SetLineWidth(3)
histos['mass'].SetXTitle("m^{a} (GeV)")
histos['mass'].SetYTitle("Jets")
histos['mass'].SetMinimum(0)
histos['mass'].Draw('COLZ TEXT')
# legend.AddEntry(histos['mass'], 'Biased','lf')
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# legend.Draw()
canvas.SaveAs('%s/a_m_biased.root'%(out_dir))
canvas.SaveAs('%s/a_m_biased.png'%(out_dir))

canvas = loadcanvas("c3")
canvas.cd()
legend = loadlegend(canvas.GetTopMargin(), canvas.GetBottomMargin(), canvas.GetLeftMargin(), canvas.GetRightMargin())
histos['pt'].SetLineColor(2)
histos['pt'].SetLineWidth(3)
histos['pt'].SetXTitle("p_{T}^{a} (GeV)")
histos['pt'].SetYTitle("Jets")
histos['pt'].SetMinimum(0)
histos['pt'].Draw('COLZ TEXT')
# legend.AddEntry(histos['pt'], 'Biased','lf')
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# legend.Draw()
canvas.SaveAs('%s/a_pt_biased.root'%(out_dir))
canvas.SaveAs('%s/a_pt_biased.png'%(out_dir))

canvas = loadcanvas("c4")
canvas.cd()
histos['mVSpT_ratio'].GetXaxis().SetTitle("m^{a} (GeV)")
histos['mVSpT_ratio'].GetYaxis().SetTitle("p_{T}^{a} (GeV)")
histos['mVSpT_ratio'].SetMinimum(0)
histos['mVSpT_ratio'].Draw('COLZ TEXT')
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
canvas.SaveAs('%s/a_massVspT_ratio_biased.root'%(out_dir))
canvas.SaveAs('%s/a_massVspT_ratio_biased.png'%(out_dir))

canvas = loadcanvas("c5")
canvas.cd()
legend = loadlegend(canvas.GetTopMargin(), canvas.GetBottomMargin(), canvas.GetLeftMargin(), canvas.GetRightMargin())
histos['mass_inverted'].SetLineColor(2)
histos['mass_inverted'].SetLineWidth(3)
histos['mass_inverted'].SetXTitle("m^{a} (GeV)")
histos['mass_inverted'].SetYTitle("Jets")
histos['mass_inverted'].Draw('COLZ TEXT')
# legend.AddEntry(histos['mass_inverted'], 'Biased','lf')
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# legend.Draw()
canvas.SaveAs('%s/a_m_inverted_biased.root'%(out_dir))
canvas.SaveAs('%s/a_m_inverted_biased.png'%(out_dir))

canvas = loadcanvas("c6")
canvas.cd()
legend = loadlegend(canvas.GetTopMargin(), canvas.GetBottomMargin(), canvas.GetLeftMargin(), canvas.GetRightMargin())
histos['pt_inverted'].SetLineColor(2)
histos['pt_inverted'].SetLineWidth(3)
histos['pt_inverted'].SetXTitle("p_{T}^{a} (GeV)")
histos['pt_inverted'].SetYTitle("Jets")
histos['pt_inverted'].Draw('COLZ TEXT')
# legend.AddEntry(histos['pt_inverted'], 'Biased','lf')
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# legend.Draw()
canvas.SaveAs('%s/a_pt_inverted_biased.root'%(out_dir))
canvas.SaveAs('%s/a_pt_inverted_biased.png'%(out_dir))

canvas = loadcanvas("c7")
canvas.cd()
histos['mVSpT_inverted'].GetXaxis().SetTitle("m^{a} (GeV)")
histos['mVSpT_inverted'].GetYaxis().SetTitle("p_{T}^{a} (GeV)")
histos['mVSpT_inverted'].SetMinimum(0)
histos['mVSpT_inverted'].Draw('COLZ TEXT')
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
canvas.SaveAs('%s/a_massVspT_inverted_biased.root'%(out_dir))
canvas.SaveAs('%s/a_massVspT_inverted_biased.png'%(out_dir))
print(">>>>>>>>>>>>>>>>>>  All plots were saved in %s directory")%out_dir
