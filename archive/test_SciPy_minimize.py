from matplotlib import pyplot as plt

from iminuit import Minuit
import math

from histimator.Model import HistiModel
from histimator.Channel import HistiChannel
from histimator.Sample import HistiSample


data = [11.,12.,14.,17.,14.,12.,11.]
flat_background = [10,10,10,10,10,10,10]
peak = [2.,4.,8.,14.,8.,4.,2.]

model = HistiModel()
SR = HistiChannel("SignalRegion")
SR.SetData(data,"obs")

signal = HistiSample("signal")
signal.SetHisto(peak)
signal.AddNorm("mu",1,0,3)
SR.AddSample(signal)

background = HistiSample("background")
background.SetHisto(flat_background)
SR.AddSample(background)



model.AddChannel(SR)

pars = [('mu',0.5)]

print "mu = 0.5", model.Channels["SignalRegion"].Pdf(pars)

pars = [('mu',2.)]

print "mu = 2.", model.Channels["SignalRegion"].Pdf(pars)

#nll = model.NLL(2.)

#fitter = Minuit(model.NLL)
#fitter.migrad()

#x,y = fitter.profile('mu',subtract_min=True)
#plt.plot(x,y,alpha=0.5,linestyle='--',label=r'$-\ln\mathcal{L}(\theta$)')
#plt.title('negative log likelihood distribution')
#plt.xlabel(r'possible values for theta $\theta$')
#plt.ylabel(r'FCN')
#plt.legend()
#plt.savefig('examples/fitnorm.png')

from scipy.optimize import minimize
initial_pars = [1.]
par_bounds = [[.1,10.]]
result = minimize(model.NLL, initial_pars, method='SLSQP', bounds=par_bounds)

#try:
#    assert result.success
#except AssertionError:
#    print result

#print result.x


