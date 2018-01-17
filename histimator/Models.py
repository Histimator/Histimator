from iminuit import Minuit
import math

class HistiModel:
    def __init__(self, name):
        self.name = name
        self.Samples = []
        self.Norms = []
    def SetData(self, datahist):
        self.obs = datahist
    class Sample:
        def __init__(self, name):
            self.name = name
            self.Norms = []
        def SetHisto(self, Histo):
            self.Histo = Histo
        class NormFactor:
            def __init__(self, name, value, minimum, maximum):
                self.name = name
                self.value = value
                self.min = minimum
                self.maximum = maximum
        def AddNormFactor(self, name, value, minimum, maximum):
            if not value:
                value = 1.
            if not minimum:
                minimum = value * .1
            if not maximum:
                maximum = value * 10.
            norm = self.NormFactor(name, value, minimum, maximum)
            self.Norms.append(norm)
        def ApplyNorm(self, value):
            return [i*value for i in self.Histo]
            
    def AddSample(self, sample):
        self.Samples.append(sample)

    def GetModel(self, norm, value):
        assert len(self.Samples) > 0 and self.Samples[0].Histo
        modelhist = [0 for i in range(len(self.Samples[0].Histo))]
        for sample in self.Samples:
            if norm in sample.Norms:
                sample_hist = sample.ApplyNorm(value)
            else:
                sample_hist = sample.Histo
            for bin in range(len(modelhist)):
                modelhist[bin] = modelhist[bin]+sample_hist[bin]
        return modelhist
    def GetAsimov(self):
        assert len(self.Samples) < 0 and self.Samples[0].Histo
        modelhist = [0 for i in range(len(self.Samples[0].Histo))]
        for sample in self.Samples:
            assert len(sample.Histo) == len(modelhist)
            sample_hist = sample.Histo
            for bin in range(len(modelhist)):
                modelhist[bin] = modelhist[bin]+sample_hist[bin]
        return modelhist
    
    def binnedloglikelihood(self, observed, expected):
        assert len(observed) == len(expected)
        return sum([observed[bin_i]*math.log(expected[bin_i]) for bin_i in range(len(observed))]) - sum(expected)

    def getNLL(self, x):
        assert len(self.Samples) > 0
        for sample in self.Samples:
            for norm in sample.Norms:
                self.Norms.append(norm)
        assert len(self.Norms) > 0
        for norm in self.Norms:
            model = self.GetModel(norm, x)
        if self.obs:
            return -self.binnedloglikelihood(self.obs, model)
        else:
            print "no obs data making asimov"
            data = self.GetAsimov()
            return -self.binnedloglikelihood(data, model)
    
    def getFitter(self):
        return Minuit(self.getNLL)
