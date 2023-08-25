#!/usr/bin/env python
import os
from glob import glob

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

outdir = 'plots_A_2Tau_massreg_gen_information_m14T017p2_unbaising'
if not os.path.isdir(outdir):
            os.makedirs(outdir)

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

files_ = []
firstfile = True
# filelist = 'list_aToTauTau_Hadronic_tauDR0p4_m3p6To14_dataset_1_reunbaised_mannual_v2_ntuples.txt'
#
# with open(filelist) as list_:
#     content = list_.readlines()
# paths = [x.strip() for x in content]
# # print(paths)
#
# for path in paths:
# local='/eos/uscms/store/user/bbbam/aToTauTau_Hadronic_tauDR0p4_m3p6To16_pT30To180_ctau0To3_eta0To1p4_pythia8_unbiased4ML_dataset_1/aToTauTau_Hadronic_tauDR0p4_m3p6To14_dataset_1_reunbaised_mannual_v2/230420_051900/0000/'
local='/eos/uscms/store/group/lpcml/bbbam/Ntuples/aToTauTau_Hadronic_tauDR0p4_m14To17p2_pT30To180_ctau0To3_eta0To2p4_pythia8/aToTauTau_Hadronic_tauDR0p4_m14To17p2_eta0To1p2_pythia8_unbiased4ML/230610_112656/0000/'
rhFileList = '%s/output*.root'%(local)
print " >> Input file list: %s"%rhFileList
rhFileList = glob.glob(rhFileList)
assert len(rhFileList) > 0
print " >> %d files found"%len(rhFileList)
# sort_nicely(rhFileList)
for path in rhFileList:
  files_.append( TFile.Open(path) )
  #print(file)

  tmp_h_jet_ma = files_[-1].Get('fevt/h_jet_ma')

  tmp_h_jet_m0 = files_[-1].Get('fevt/h_jet_m0')

  tmp_h_jet_pta = files_[-1].Get('fevt/h_jet_pta')
  tmp_h_jet_Tau1pT = files_[-1].Get('fevt/h_jet_Tau1pT')
  tmp_h_jet_Tau2pT = files_[-1].Get('fevt/h_jet_Tau2pT')

  tmp_h_jet_pT = files_[-1].Get('fevt/h_jet_pT')
  tmp_h_jet_E = files_[-1].Get('fevt/h_jet_E')

  tmp_h_jet_Tau1dR = files_[-1].Get('fevt/h_jet_Tau1dR')
  tmp_h_jet_Tau2dR = files_[-1].Get('fevt/h_jet_Tau2dR')

  tmp_h_jet_dR = files_[-1].Get('fevt/h_jet_dR')
  tmp_h_jet_TaudR = files_[-1].Get('fevt/h_jet_TaudR')
  tmp_h_jet_recoTau1dR = files_[-1].Get('fevt/h_jet_recoTau1dR')
  tmp_h_jet_recoTau2dR = files_[-1].Get('fevt/h_jet_recoTau2dR')
  tmp_h_jet_n1dR = files_[-1].Get('fevt/h_jet_n1dR')
  tmp_h_jet_n2dR = files_[-1].Get('fevt/h_jet_n2dR')

  tmp_h_jet_nJet = files_[-1].Get('fevt/h_jet_nJet')
  tmp_h_jet_NGenTaus = files_[-1].Get('fevt/h_jet_NGenTaus')
  tmp_h_jet_NrecoTaus = files_[-1].Get('fevt/h_jet_NrecoTaus')

  tmp_h_jet_eta = files_[-1].Get('fevt/h_jet_eta')
  tmp_h_jet_etaa = files_[-1].Get('fevt/h_jet_etaa')

  tmp_h_jet_phia = files_[-1].Get('fevt/h_jet_phia')

  if (firstfile):

    histos['h_jet_ma'] = tmp_h_jet_ma.Clone('h_jet_ma')

    histos['h_jet_m0'] = tmp_h_jet_m0.Clone('h_jet_m0')

    histos['h_jet_pta'] = tmp_h_jet_pta.Clone('h_jet_pta')
    histos['h_jet_Tau1pT'] = tmp_h_jet_Tau1pT.Clone('h_jet_Tau1pT')
    histos['h_jet_Tau2pT'] = tmp_h_jet_Tau2pT.Clone('h_jet_Tau2pT')

    histos['h_jet_pT'] = tmp_h_jet_pT.Clone('h_jet_pT')
    histos['h_jet_E'] = tmp_h_jet_E.Clone('h_jet_E')

    histos['h_jet_Tau1dR'] = tmp_h_jet_Tau1dR.Clone('h_jet_Tau1dR')
    histos['h_jet_Tau2dR'] = tmp_h_jet_Tau2dR.Clone('h_jet_Tau2dR')

    histos['h_jet_dR'] = tmp_h_jet_dR.Clone('h_jet_dR')
    histos['h_jet_TaudR'] = tmp_h_jet_TaudR.Clone('h_jet_TaudR')
    histos['h_jet_recoTau1dR'] = tmp_h_jet_recoTau1dR.Clone('h_jet_recoTau1dR')
    histos['h_jet_recoTau2dR'] = tmp_h_jet_recoTau2dR.Clone('h_jet_recoTau2dR')
    histos['h_jet_n1dR'] = tmp_h_jet_n1dR.Clone('h_jet_n1dR')
    histos['h_jet_n2dR'] = tmp_h_jet_n2dR.Clone('h_jet_n2dR')

    histos['h_jet_nJet'] = tmp_h_jet_nJet.Clone('h_jet_nJet')
    histos['h_jet_NGenTaus'] = tmp_h_jet_NGenTaus.Clone('h_jet_NGenTaus')
    histos['h_jet_NrecoTaus'] = tmp_h_jet_NrecoTaus.Clone('h_jet_NrecoTaus')

    histos['h_jet_eta'] = tmp_h_jet_eta.Clone('h_jet_eta')
    histos['h_jet_etaa'] = tmp_h_jet_etaa.Clone('h_jet_etaa')

    histos['h_jet_phia'] = tmp_h_jet_phia.Clone('h_jet_phia')

    firstfile = False

  if not (firstfile):

    histos['h_jet_ma'].Add(tmp_h_jet_ma)

    histos['h_jet_m0'].Add(tmp_h_jet_m0)

    histos['h_jet_pta'].Add(tmp_h_jet_pta)
    histos['h_jet_Tau1pT'].Add(tmp_h_jet_Tau1pT)
    histos['h_jet_Tau2pT'].Add(tmp_h_jet_Tau2pT)

    histos['h_jet_pT'].Add(tmp_h_jet_pT)
    histos['h_jet_E'].Add(tmp_h_jet_E)

    histos['h_jet_Tau1dR'].Add(tmp_h_jet_Tau1dR)
    histos['h_jet_Tau2dR'].Add(tmp_h_jet_Tau2dR)

    histos['h_jet_dR'].Add(tmp_h_jet_dR)
    histos['h_jet_TaudR'].Add(tmp_h_jet_TaudR)
    histos['h_jet_recoTau1dR'].Add(tmp_h_jet_recoTau1dR)
    histos['h_jet_recoTau2dR'].Add(tmp_h_jet_recoTau2dR)
    histos['h_jet_n1dR'].Add(tmp_h_jet_n1dR)
    histos['h_jet_n2dR'].Add(tmp_h_jet_n2dR)

    histos['h_jet_nJet'].Add(tmp_h_jet_nJet)
    histos['h_jet_NGenTaus'].Add(tmp_h_jet_NGenTaus)
    histos['h_jet_NrecoTaus'].Add(tmp_h_jet_NrecoTaus)

    histos['h_jet_eta'].Add(tmp_h_jet_eta)
    histos['h_jet_etaa'].Add(tmp_h_jet_etaa)

    histos['h_jet_phia'].Add(tmp_h_jet_phia)




canvas = loadcanvas("c1")
canvas.cd()
histos['h_jet_ma'].GetXaxis().SetTitle("m^{A} (GeV)")
histos['h_jet_ma'].GetYaxis().SetTitle("Events")

histos['h_jet_ma'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()

canvas.SaveAs('%s/h_jet_ma.png'%(outdir))



canvas = loadcanvas("c2")
canvas.cd()
histos['h_jet_m0'].GetXaxis().SetTitle("m^{jet} (GeV)")
histos['h_jet_m0'].GetYaxis().SetTitle("Events")

histos['h_jet_m0'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()

canvas.SaveAs('%s/h_jet_m0.png'%(outdir))


canvas = loadcanvas("c3")
canvas.cd()
histos['h_jet_pta'].GetXaxis().SetTitle("A^{pT} (GeV)")
histos['h_jet_pta'].GetYaxis().SetTitle("Events")

histos['h_jet_pta'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()

canvas.SaveAs('%s/h_jet_pta.png'%(outdir))


canvas = loadcanvas("c4")
canvas.cd()
histos['h_jet_Tau1pT'].GetXaxis().SetTitle("Tau1^{pT} (GeV)")
histos['h_jet_Tau1pT'].GetYaxis().SetTitle("Events")

histos['h_jet_Tau1pT'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
canvas.SaveAs('%s/h_jet_Tau1pT.png'%(outdir))


canvas = loadcanvas("c5")
canvas.cd()
histos['h_jet_Tau2pT'].GetXaxis().SetTitle("Tau2^{pT} (GeV)")
histos['h_jet_Tau2pT'].GetYaxis().SetTitle("Events")

histos['h_jet_Tau2pT'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
canvas.SaveAs('%s/h_jet_Tau2pT.png'%(outdir))


canvas = loadcanvas("c6")
canvas.cd()
histos['h_jet_pT'].GetXaxis().SetTitle("jet^{pT} (GeV)")
histos['h_jet_pT'].GetYaxis().SetTitle("Events")
histos['h_jet_pT'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
canvas.SaveAs('%s/h_jet_pT.png'%(outdir))


canvas = loadcanvas("c7")
canvas.cd()
histos['h_jet_E'].GetXaxis().SetTitle("jet^{E} (GeV)")
histos['h_jet_E'].GetYaxis().SetTitle("Events")
histos['h_jet_E'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
canvas.SaveAs('%s/h_jet_E.png'%(outdir))



canvas = loadcanvas("c8")
canvas.cd()
histos['h_jet_Tau1dR'].GetXaxis().SetTitle("jet_Tau1^{dR}")
histos['h_jet_Tau1dR'].GetYaxis().SetTitle("Events")
# histos['h_jet_Tau1dR'].SetMinimum(0)
histos['h_jet_Tau1dR'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# canvas.SaveAs('%s/h_jet_Tau1dR.root'%(outdir))
canvas.SaveAs('%s/h_jet_Tau1dR.png'%(outdir))


canvas = loadcanvas("c9")
canvas.cd()
histos['h_jet_Tau2dR'].GetXaxis().SetTitle("jet_Tau2^{dR}")
histos['h_jet_Tau2dR'].GetYaxis().SetTitle("Events")
# histos['h_jet_Tau2dR'].SetMinimum(0)
histos['h_jet_Tau2dR'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# canvas.SaveAs('%s/h_jet_Tau2dR.root'%(outdir))
canvas.SaveAs('%s/h_jet_Tau2dR.png'%(outdir))




canvas = loadcanvas("c10")
canvas.cd()
histos['h_jet_dR'].GetXaxis().SetTitle("jet^{dR}")
histos['h_jet_dR'].GetYaxis().SetTitle("Events")
# histos['h_jet_dR'].SetMinimum(0)
histos['h_jet_dR'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# canvas.SaveAs('%s/h_jet_dR.root'%(outdir))
canvas.SaveAs('%s/h_jet_dR.png'%(outdir))

canvas = loadcanvas("c11")
canvas.cd()
histos['h_jet_TaudR'].GetXaxis().SetTitle("tautau^{dR}")
histos['h_jet_TaudR'].GetYaxis().SetTitle("Events")
# histos['h_jet_TaudR'].SetMinimum(0)
histos['h_jet_TaudR'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# canvas.SaveAs('%s/h_jet_TaudR.root'%(outdir))
canvas.SaveAs('%s/h_jet_TaudR.png'%(outdir))


canvas = loadcanvas("c12")
canvas.cd()
histos['h_jet_recoTau1dR'].GetXaxis().SetTitle("recoTau1^{dR}")
histos['h_jet_recoTau1dR'].GetYaxis().SetTitle("Events")
# histos['h_jet_recoTau1dR'].SetMinimum(0)
histos['h_jet_recoTau1dR'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# canvas.SaveAs('%s/h_jet_recoTau1dR.root'%(outdir))
canvas.SaveAs('%s/h_jet_recoTau1dR.png'%(outdir))


canvas = loadcanvas("c13")
canvas.cd()
histos['h_jet_recoTau2dR'].GetXaxis().SetTitle("recoTau2^{dR}")
histos['h_jet_recoTau2dR'].GetYaxis().SetTitle("Events")
# histos['h_jet_recoTau2dR'].SetMinimum(0)
histos['h_jet_recoTau2dR'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# canvas.SaveAs('%s/h_jet_recoTau2dR.root'%(outdir))
canvas.SaveAs('%s/h_jet_recoTau2dR.png'%(outdir))


canvas = loadcanvas("c14")
canvas.cd()
histos['h_jet_n1dR'].GetXaxis().SetTitle("jet_n1^{dR}")
histos['h_jet_n1dR'].GetYaxis().SetTitle("Events")
# histos['h_jet_n1dR'].SetMinimum(0)
histos['h_jet_n1dR'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# canvas.SaveAs('%s/h_jet_n1dR.root'%(outdir))
canvas.SaveAs('%s/h_jet_n1dR.png'%(outdir))

canvas = loadcanvas("c15")
canvas.cd()
histos['h_jet_n2dR'].GetXaxis().SetTitle("jet_n2^{dR}")
histos['h_jet_n2dR'].GetYaxis().SetTitle("Events")
# histos['h_jet_n2dR'].SetMinimum(0)
histos['h_jet_n2dR'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# canvas.SaveAs('%s/h_jet_n2dR.root'%(outdir))
canvas.SaveAs('%s/h_jet_n2dR.png'%(outdir))


canvas = loadcanvas("c16")
canvas.cd()
histos['h_jet_nJet'].GetXaxis().SetTitle("nJet")
histos['h_jet_nJet'].GetYaxis().SetTitle("Events")
# histos['h_jet_nJet'].SetMinimum(0)
histos['h_jet_nJet'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# canvas.SaveAs('%s/h_jet_nJet.root'%(outdir))
canvas.SaveAs('%s/h_jet_nJet.png'%(outdir))

canvas = loadcanvas("c17")
canvas.cd()
histos['h_jet_NGenTaus'].GetXaxis().SetTitle("NGenTaus")
histos['h_jet_NGenTaus'].GetYaxis().SetTitle("Events")
# histos['h_jet_NGenTaus'].SetMinimum(0)
histos['h_jet_NGenTaus'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# canvas.SaveAs('%s/h_jet_NGenTaus.root'%(outdir))
canvas.SaveAs('%s/h_jet_NGenTaus.png'%(outdir))


canvas = loadcanvas("c18")
canvas.cd()
histos['h_jet_NrecoTaus'].GetXaxis().SetTitle("NrecoTaus")
histos['h_jet_NrecoTaus'].GetYaxis().SetTitle("Events")
# histos['h_jet_NrecoTaus'].SetMinimum(0)
histos['h_jet_NrecoTaus'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# canvas.SaveAs('%s/h_jet_NrecoTaus.root'%(outdir))
canvas.SaveAs('%s/h_jet_NrecoTaus.png'%(outdir))

canvas = loadcanvas("c19")
canvas.cd()
histos['h_jet_eta'].GetXaxis().SetTitle("jet^{eta}")
histos['h_jet_eta'].GetYaxis().SetTitle("Events")
# histos['h_jet_eta'].SetMinimum(0)
histos['h_jet_eta'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# canvas.SaveAs('%s/h_jet_eta.root'%(outdir))
canvas.SaveAs('%s/h_jet_eta.png'%(outdir))


canvas = loadcanvas("c20")
canvas.cd()
histos['h_jet_etaa'].GetXaxis().SetTitle("A^{eta}")
histos['h_jet_etaa'].GetYaxis().SetTitle("Events")
# histos['h_jet_etaa'].SetMinimum(0)
histos['h_jet_etaa'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# canvas.SaveAs('%s/h_jet_etaa.root'%(outdir))
canvas.SaveAs('%s/h_jet_etaa.png'%(outdir))




canvas = loadcanvas("c21")
canvas.cd()
histos['h_jet_phia'].GetXaxis().SetTitle("A^{phi}")
histos['h_jet_phia'].GetYaxis().SetTitle("Events")
# histos['h_jet_phia'].SetMinimum(0)
histos['h_jet_phia'].Draw()
CMS_lumi.CMS_lumi(canvas, iPeriod, iPos)
canvas.Update()
# canvas.SaveAs('%s/h_jet_phia.root'%(outdir))
canvas.SaveAs('%s/h_jet_phia.png'%(outdir))
