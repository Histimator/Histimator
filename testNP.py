from histimator.Sample import HistiSample

import math
import numpy as np

print '############################'
print '######Testing Samples#######'
print '############################'
nominal = [10.,20.,30.,40.]
sig = HistiSample('signal')
sig.SetHisto(nominal)
print sig.nominal
print '======Testing Systematics==='
print 'scaling up and down by 50%'
sig.SetHistoSys('first',[i*1.5 for i in nominal],[i*.5 for i in nominal])
print sig.nps

print "up nominal down are "
print sig.nps['first']['up']
print sig.nominal
print sig.nps['first']['down']

print '======Testing Evaluation==='
print 'the variation at -0.5'
flux = sig.Evaluate("first",-.5)
print flux

print 'leading to an estimated pdf at this value of'
print sig.nominal + flux


print '======Tertiary Systematics==='
print 'scaling by an increasing/decreasing linear amount'
sig.SetHistoSys('second',[(i*.1+1)*a for i,a in enumerate(nominal)],[(i*.1+1)*a for i,a in enumerate(nominal)])
print 'up nominal down are'
print sig.nps['second']['up']
print sig.nominal
print sig.nps['second']['down']

print 'scaling by an increasing/decreasing linear amount with opposite slope'
sig.SetHistoSys('third',[a+math.pow(a,4-i)*.001 for i,a in enumerate(nominal)],[a-math.pow(a,4-i)*.001 for i,a in enumerate(nominal)])
print 'up nominal down are'
print sig.nps['third']['up']
print sig.nominal
print sig.nps['third']['down']

print '======Evaluate Multiple Systematics==='
params = [('first',-.5),('second',.3),('third',1.2)]
print 'evaluating all three at the following values', params
flux = np.zeros(len(sig.nominal))
for par in params:
    flux += sig.Evaluate(par[0],par[1])
print sig.nominal + flux

print '############################'
print '######Testing Channels######'
print '############################'
print '====Define Second Sample==='
bkgnominal = [40.,40.,40.,40.]
bkg = HistiSample('signal')
bkg.SetHisto(bkgnominal)
print bkg.nominal
print '====Set Histo Sys==='
bkg.SetHistoSys('first',[i*1.5 for i in nominal],[i*.5 for i in nominal])
print bkg.nps

print '====Import Channel and Add Samples==='
from histimator.Channel import HistiChannel
pdf = HistiChannel('BinnedTemplate')
pdf.AddSample(sig)
pdf.AddSample(bkg)

print '====Evaluate all samples==='
params = [('first',-.5),('second',-.3),('third',1.2)]
print 'evaluate sig+background for parameters',params
print pdf.Pdf(params)

print '====Add Data To Channel==='
pdf.SetData([45,62,71,78])

print '====Define Channel and add Samples==='
from histimator.Model import HistiModel
model = HistiModel()
model.AddChannel(pdf)

print '############################'
print '######   Minimize!    ######'
print '############################'
print '====Import Model==='
from histimator.Estimator import LikelihoodFit
opt = LikelihoodFit('test')

print '====Minimize==='

print 'result is',opt.newNLL(model)
