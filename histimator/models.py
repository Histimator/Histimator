from pdfs import HistogramPdf, NormedHist, OverallSys, HistoSys, HistiAddPdf
#from probfit import AddPdf
import math

class HistiModel(object):
    def __init__(self, name='TemplateModel'):
        self.name = name
        self.n_channels = 0
        self.channels = []
        self.pdf = None
        self.data = None
        self.binedges = None
        self.pois = {}
        self.nps = {}

    def AddChannel(self, channel):
        name = channel.name
        if name is None:
            name = 'channel_'+self.n_channels
        self.n_channels += 1
        self.channels.append(name)
        for sample in channel.samples:
            s = channel.samples[sample]
            for poi in s.pois:
                self.pois[poi] = s.pois[poi]
            for np in s.nps:
                self.nps[np] = s.nps[np]
            if self.pdf is None:
                self.pdf = s.pdf
                self.binedges = s.binedges
            else:
                self.pdf = HistiAddPdf(self.pdf, s.pdf)
        self.data = channel.data

    def Parameters(self):
        parameters = {'errordef': 1}
        for param in self.pois:
            name = param
            param = self.pois[param]
            parameters[name] = param['nom']
            parameters['limit_{}'.format(name)] = param['range']
#            parameters['error_{}'.format(name)] = (
#                param['range'][1]-param['range'][0]
#            )/2.
        for np in self.nps:
            name = np
            np = self.nps[np]
            parameters[name] = np['nom']
            parameters['limit_{}'.format(name)] = np['range']
#            parameters['error_{}'.format(name)] = (
#                np['range'][1]-np['range'][0]
#            )/2.
        return parameters


class HistiChannel(object):
    def __init__(self, name=None):
        self.name = name
        self.n_samples = 0
        self.samples = {}
        self.data = None

    def AddSample(self, sample):
        name = sample.name
        if name is None:
            name = 'sample_'+self.n_samples
        self.n_samples += 1
        self.samples[sample.name] = sample

    def SetData(self, data):
        self.data = data


class HistiSample(object):
    def __init__(self, name=None):
        self.name = name
        self.pois = {}
        self.nps = {}

    def SetHisto(self, hist):
        self.hist = hist
        self.bincontent, self.binedges = self.hist
        self.pdf = HistogramPdf(self.bincontent, self.binedges)

    def AddNorm(self, name='mu', nom=1, min=0, max=3):
        self.pois[name] = {'nom': nom, 'range': (min, max)}
        self.pdf = NormedHist(self.pdf, norm=name)

    def AddOverallSys(self, name, uncertainty_down, uncertainty_up, scheme=1.):
        self.nps[name] = {'nom':math.fabs(uncertainty_up-uncertainty_down),'range':(-50,50)}
        self.pdf = OverallSys(
            self.pdf, name, uncertainty_up, uncertainty_down, scheme
        )

    def AddHistoSys(self, name, uncertainty_down, uncertainty_up, scheme=1.):
        self.nps[name] = {'nom':math.fabs(sum(uncertainty_down)-sum(uncertainty_up)),'range':(-50,50)}
        assert len(uncertainty_down) == len(uncertainty_up) == len(self.bincontent)
        self.pdf = HistoSys(
            self.pdf, name, self.bincontent, uncertainty_up, uncertainty_down, scheme
        )
