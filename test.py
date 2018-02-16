from histimator.models import HistiModel, HistiChannel, HistiSample
import numpy as np

from probfit import gen_toy, BinnedLH
from iminuit import Minuit

m = HistiModel('first model')

bound = (0, 10)
np.random.seed(0)
bkg = gen_toy(lambda x : x**2, 100, bound=bound) # a parabola background
sig= np.random.randn(50)+5  # a Gaussian signal
hsig= np.histogram(sig, bins=10, range=bound)
hbkg= np.histogram(bkg, bins=10, range=bound)

signal = HistiSample("signal")
signal.SetHisto(hsig)
signal.AddNorm("SigXSecOverSM", 0.5,0,3)

background = HistiSample("background1")
background.SetHisto(hbkg)



chan = HistiChannel("SR")
chan.AddSample(signal)
chan.AddSample(background)

m.AddChannel(chan)

data = gen_toy(lambda x : m.pdf(x,1), 150,(0,10))
chan.SetData(data)


blh = BinnedLH(m.pdf, data, bins=10, bound=bound, extended=True)

minimiser = Minuit(blh, SigXSecOverSM=0.5, error_SigXSecOverSM=1.)

print 'about to test SigXSecOverSM at value', minimiser.values['SigXSecOverSM']

minimiser.migrad()
print 'migrad gives SigXSecOverSM as value', minimiser.values['SigXSecOverSM']
