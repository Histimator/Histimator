import numpy as np

class HistiChannel(object):
                
    def __init__(self, name):
        self.name = name
        self.nps = {}
        self.samples = {}
        self.nominal = None
        self.IsDataSet = False #Else set with asimov
    def AddSample(self, sample):
        self.samples[sample.name] = sample
        if self.nominal is None:
            self.nominal = sample.nominal
        else:
            self.nominal +=sample.nominal
        for nuis in sample.nps:
            self.nps[nuis] = sample.nps[nuis]

    def SetData(self, data):
        self.IsDataSet = True
        self.data = np.asarray(data)

    def Pdf(self, params):
        flux = np.zeros(len(self.nominal))
        for sample in self.samples:
            sample = self.samples[sample]
            for par in params:
                if par[0] in sample.nps:
                    flux += sample.Evaluate(par[0],par[1])
        return self.nominal + flux
