from iminuit import Minuit
import math

from .Data import HistiData
from .Sample import HistiSample

class HistiChannel:
    def __init__(self, name):
        self.name = name
        self.Samples = {}
        self.Norms = {}
        self.data = HistiData()
    def SetData(self, datahist, dataname):
        if dataname:
            self.data.name = dataname
        self.data = self.data.SetData(datahist)

    def AddSample(self, sample):
        if sample.Histo:
            self.Samples[sample.name] = sample
            for norm in sample.Norms:
                self.Norms[norm] = sample.Norms[norm]
        else:
            "print that sample has no template"
        return self
        
    def Pdf(self, *pars):
        for sample in self.Samples:
            assert self.Samples[sample].Histo
            hist = [0 for i in self.Samples[sample].Histo]
        for sample in self.Samples:
            thissample = self.Samples[sample].Evaluate(pars)
            assert len(thissample) == len(hist)
            hist = [hist[i]+thissample[i] for i in range(len(hist))]
        return hist
