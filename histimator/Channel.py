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
        for nuis in sample.nps.dtype.names:
            if 'up' in nuis or 'down' in nuis:
                np_name = nuis.replace('up_','').replace('down_',"")
                self.nps[np_name] = np_name

    def SetData(self, data):
        self.IsDataSet = True
        self.data = np.asarray(data)

    def Pdf(self, params):
        flux = np.zeros(len(self.nominal))
        for sample in self.samples:
            sample = self.samples[sample]
            for par in params:
                if 'up_' + par[0] in sample.nps.dtype.names or 'down_' + par[0] in sample.nps.dtype.names:
                    flux += sample.Evaluate(par[0],par[1])
        return self.nominal + flux
