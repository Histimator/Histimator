import timeit 
from functools import partial

from scipy.optimize import minimize
from iminuit import Minuit

import numpy as np
from histimator.models import HistiModel, HistiChannel, HistiSample
from histimator.estimator import BinnedLH


class Benchmark:
    def __init__(self,
            name = 'name',
            n_channels = 1,
            n_bins = 2,
            n_samples = 2,                
            n_nps = 0,
            ):
        self.name = name
        self.n_channels = n_channels
        self.n_bins = n_bins
        self.n_samples = n_samples
        self.n_nps = n_nps
        self.channels = {}
        self.samples = {}
        self.nps = {}
        np.random.seed(42)
        
    def gen_nps(self, n_nps):
        return np.random.choice(np.linspace(-2,2,201),n_nps)
                
        
    def make_variations(self, hist, slope):
        if len(hist) == 1:
            hist[0] = 1
        thebins = np.arange(len(hist))
        if slope < 0:
            thebins = thebins[::-1]
        slope = np.fabs(slope)
        gradient = lambda b, s: b*s
        sloped = np.vectorize(gradient)
        slope_vec = 1.+sloped(thebins, slope)
        p_sloped = np.divide(slope_vec,np.sum(slope_vec))
        n = np.sum(hist)
        factor = 1./np.sqrt(n)
        ep_sloped = n*p_sloped*factor
        array_up = hist + ep_sloped
        array_down = hist - ep_sloped
        return array_up, array_down

    def build(self):
        meas = HistiModel(self.name)
        if self.n_samples < 2: self.n_samples = 2 
        if len(self.nps) is not self.n_nps and self.n_nps > 0 :
            print "nps not set creating"
            some_nps = self.gen_nps(self.n_nps)
            nuis_i = 0
            for nuis in some_nps:
                nuis_name = "nuis_{}".format(nuis_i)
                nuis_i += 1
                self.nps[nuis_name] = nuis
        for channel in range(self.n_channels):
            c = HistiChannel("chan_{}".format(channel))
            for sample in range(self.n_samples):
                s = HistiSample("sample_{}".format(sample))
                if sample <1:
                    hist = np.linspace(1,self.n_bins,self.n_bins)
                    data = hist
                    s.SetHisto((hist,np.linspace(0,self.n_bins,self.n_bins+1)))
                    s.AddNorm("mu",1.0,0,3)
                else:
                    hist = np.ones(self.n_bins)*(10./(self.n_samples-1))
                    hist = hist*75./(hist.sum()*(self.n_samples-1))
                    s.SetHisto((hist, np.linspace(0,self.n_bins,self.n_bins+1)))
                    variation = np.random.rand()*.1
                    s.AddOverallSys("BackgroundNormSyst{}".format(sample),1.-variation,1.+variation)
                    for nuis in self.nps.keys():
                        up, down = self.make_variations(hist, self.nps[nuis])
                        up[up<0.1] = 0.1 #ensure positive
                        down[down<0.1] = 0.1
                        up = up*78./up.sum()
                        down = down*73./down.sum()
                        s.AddHistoSys("BackgroundShapeSyst{}".format(nuis),down,up)
                    data += hist
                c.AddSample(s)
            c.SetData(data)
            meas.AddChannel(c)
        self.m = meas

    def TimeBenchmark(self, m, mode = 'scipy'):
        if mode is 'scipy':
            blh = BinnedLH(m, minimiser='scipy')
            params = m.Parameters('scipy')
            minimiser = minimize(blh, params, method='bfgs')
        else:
            blh = BinnedLH(m)
            params = m.Parameters()
            minimiser = Minuit(blh, **params)
            minimiser.migrad()

    def Time(self):
        x = []
        y1 = []
        y2 = []
        for i in range(2,11):
            self.n_bins = 100 
            self.n_samples = i
            self.build()
            timer = timeit.Timer(partial(self.TimeBenchmark, self.m))
            t = timer.repeat(10, 1)
            y = np.mean(t)
            e = np.std(t)
            x.append(i)
            y1.append((y,e))
            timer = timeit.Timer(partial(self.TimeBenchmark, self.m, "minuit"))
            t = timer.repeat(10, 1)
            y = np.mean(t)
            e = np.std(t)
            y2.append((y,e))
        y1 = np.asarray(y1)
        y2 = np.asarray(y2)
        return x, y1, y2
