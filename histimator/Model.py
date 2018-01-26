from iminuit import Minuit
import math

#histimator imports
from .Data import HistiData
from .Channel import HistiChannel

class HistiModel:
    def __init__(self, name="ANewHistiModel"):
        self.name = name
        self.Channels = {}
        self.Norms = {}
    def AddChannel(self, channel):
        if channel.data.IsSet:
            self.Channels[channel.name] = channel
        else:
            print "No Data Set - Using Asimov"
            #To Do impliment Asimov

        for norm in channel.Norms:
            self.Norms[norm] = channel.Norms[norm]
    
    def binnedloglikelihood(self, observed, expected):
        assert len(observed) == len(expected)
        return sum([observed[bin_i]*math.log(expected[bin_i]) for bin_i in range(len(observed))]) - sum(expected)

    def NLL(self, mu):
        pars = []
        for norm in self.Norms:
            pars.append((norm, mu))
        for chan in self.Channels:
            shape = [0 for i in self.Channels[chan].Pdf([("dummy",1.)])]
        nll = None
        for chan in self.Channels:
            expected = self.Channels[chan].Pdf(pars)
            if nll:
                nll += -self.binnedloglikelihood(self.Channels[chan].data.obs, expected)
            else:
                nll = -self.binnedloglikelihood(self.Channels[chan].data.obs, expected)

        return nll
