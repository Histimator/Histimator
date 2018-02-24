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
hist_b  = np.random.poisson(1000*(1-ff)*st.expon(scale=4).pdf(xbin)).astype(np.float64)
binedge = xbin-np.diff(xbin)[0]/2.0
binedge = np.append(binedge, [xbin.max() + np.diff(xbin)[0]/2.0])

# Create a Histomator model
m = HistiModel('model')

signal = HistiSample("signal")
signal.SetHisto((hist_s, binedge))
signal.AddNorm("SigXSecOverSM", 1.0,0,3)

background = HistiSample("background1")
background.SetHisto((hist_b, binedge))

chan = HistiChannel("SR")
chan.AddSample(signal)
chan.AddSample(background)

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
minimiser = Minuit(blh, SigXSecOverSM=0.5, error_SigXSecOverSM=0.1)

print 'about to test SigXSecOverSM at value', minimiser.values['SigXSecOverSM']

minimiser.migrad()
minimiser.minos()
print 'migrad gives SigXSecOverSM as value', minimiser.values['SigXSecOverSM']
