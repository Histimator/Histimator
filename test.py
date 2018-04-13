import scipy.stats as st
import numpy  as np

from histimator.models import HistiModel, HistiChannel, HistiSample
from histimator import models
from histimator.estimator import BinnedLH

from iminuit import Minuit, describe
from pprint import pprint

def theory_model(x, mu):
    """poisson pdf, parameter lamb is the fit parameter"""
    return mu*st.norm(4,scale=1).pdf(x) + (1-mu)*st.expon(scale=4).pdf(x)


# Generate the model and data
ff = 0.3
np.random.seed(42)

bounds  = (0, 10)
xbin    = np.linspace(0,10,21)
data    = np.random.poisson(1000*theory_model(xbin, ff))
hist_s  = np.random.poisson(1000*ff*st.norm(4,scale=1).pdf(xbin)).astype(np.float64)
hist_b1  = np.random.poisson(500*(1-ff)*st.expon(scale=4).pdf(xbin)).astype(np.float64)
hist_b2  = np.random.poisson(500*(1-ff)*st.expon(scale=2).pdf(xbin)).astype(np.float64)
binedge = xbin-np.diff(xbin)[0]/2.0
binedge = np.append(binedge, [xbin.max() + np.diff(xbin)[0]/2.0])

# Create a Histomator model
m = HistiModel('model')

signal = HistiSample("signal")
signal.SetHisto((hist_s, binedge))
signal.AddNorm("SigXSecOverSM", 1.0,0,3)

background1 = HistiSample("background1")
background1.SetHisto((hist_b1, binedge))
background1.AddNorm("Background1Norm", 1.0,0,100)

background2 = HistiSample("background2")
background2.SetHisto((hist_b2, binedge))
background2.AddNorm("Background2Norm", 1.0,0,100)

chan = HistiChannel("SR")
chan.AddSample(signal)
chan.AddSample(background1)
chan.AddSample(background2)

m.AddChannel(chan)
chan.SetData(data)
m.AddChannel(chan)


print "---- printing model --- "
pprint (chan.__dict__)
for n_, s_ in chan.__dict__['samples'].items():
    print 'sample : ', n_
    pprint (s_.__dict__)
print "---------------------------- "

# Call BinnedLH estimator
blh = BinnedLH(m, bound=bounds, extended=True)

# Minimise everything
params = m.Parameters()
print 'params are ', params

print 'describing blg', describe(blh)
minimiser = Minuit(blh, **params)

#print 'about to test SigXSecOverSM at value', minimiser.values['SigXSecOverSM']

minimiser.migrad()
minimiser.hesse()
minimiser.minos()
print 'migrad gives values', minimiser.values


print "#######################################"
print "####### Adding Systematics ############"
print "#######################################"

m = HistiModel('model')

signal = HistiSample("signal")
signal.SetHisto((hist_s, binedge))
signal.AddNorm("SigXSecOverSM", 1.0,0,3)

background1 = HistiSample("background1")
background1.SetHisto((hist_b1, binedge))
background1.AddNorm("Background1Norm", 1.0,0,100)
background1.AddOverallSys("Syst1", .5,1.5)

background2 = HistiSample("background2")
background2.SetHisto((hist_b2, binedge))
background2.AddNorm("Background2Norm", 1.0,0,100)
background2.AddOverallSys("Syst2", .75,1.25)

chan = HistiChannel("SR")
chan.AddSample(signal)
chan.AddSample(background1)
chan.AddSample(background2)

m.AddChannel(chan)
chan.SetData(data)
m.AddChannel(chan)

print "---- printing model --- "
pprint (chan.__dict__)
for n_, s_ in chan.__dict__['samples'].items():
    print 'sample : ', n_
    pprint (s_.__dict__)
print "---------------------------- "

# Call BinnedLH estimator
blh = BinnedLH(m, bound=bounds, extended=True)

# Minimise everything
params = m.Parameters()
print 'params are ', params

print 'describing blg', describe(blh)
minimiser = Minuit(blh, **params)

print 'about to test SigXSecOverSM at value', minimiser.values['SigXSecOverSM']

minimiser.migrad()
minimiser.hesse()
minimiser.minos()

print 'migrad gives values', minimiser.values
