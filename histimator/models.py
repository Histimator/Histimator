from pdfs import HistogramPdf, NormedHist, OverallSys, HistoSys, HistiAddPdf, HistiCombPdf
import numpy as np
import math

class HistiModel(object):
    def __init__(self, name='TemplateModel'):
        self.name = name
        self.n_channels = 0
        self.channels = []
        self.pdf = None
        self.data = []
        self.binedges = None
        self.pois = {}
        self.nps = {}

    def AddChannel(self, channel):
        name = channel.name
        if name is None:
            name = 'channel_'+self.n_channels
        self.n_channels += 1
        self.channels.append(name)
        if self.pdf:
            self.pdf = HistiCombPdf(self.pdf, channel.pdf)
            bwidth = np.diff(channel.pdf.binedges)
            self.data = np.hstack([self.data, channel.data*bwidth])
        else:
            self.pdf = channel.pdf
            bwidth = np.diff(channel.pdf.binedges)
            self.data = np.asarray(channel.data)*bwidth
        for poi in channel.pois:
            self.pois[poi] = channel.pois[poi]
        for nuis in channel.nps:
            self.nps[nuis] = channel.nps[nuis]
        
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
        for nuis in self.nps:
            name = nuis
            nuis = self.nps[nuis]
            parameters[name] = nuis['nom']
            parameters['limit_{}'.format(name)] = nuis['range']
#            parameters['error_{}'.format(name)] = (
#                nuis['range'][1]-nuis['range'][0]
#            )/2.
        return parameters


class HistiChannel(object):
    def __init__(self, name=None):
        self.name = name
        self.n_samples = 0
        self.samples = {}
        self.pois = {}
        self.nps = {}
        self.data = None
        self.pdf = None
        self.binedges = None

    def AddSample(self, sample):
        name = sample.name
        if name is None:
            name = 'sample_'+self.n_samples
        self.n_samples += 1
        self.samples[sample.name] = sample
        for poi in sample.pois:
            self.pois[poi] = sample.pois[poi]
        for nuis in sample.nps:
            self.nps[nuis] = sample.nps[nuis]

        if self.pdf is None:
            self.pdf = sample.pdf
            self.binedges = sample.binedges
        else:
            self.pdf = HistiAddPdf(self.pdf, sample.pdf)
    def SetData(self, data):
        self.data = np.asarray(data)


class HistiSample(object):
    def __init__(self, name=None):
        self.name = name
        self.pois = {}
        self.nps = {}

    def SetHisto(self, hist):
        self.hist = hist
        self.bincontent, self.binedges = self.hist
        self.pdf = HistogramPdf(self.bincontent, np.asarray(self.binedges))

    def AddNorm(self, name='mu', nom=1, min=0, max=3):
        self.pois[name] = {'nom': nom, 'range': (min, max)}
        self.pdf = NormedHist(self.pdf, norm=name)

    def AddOverallSys(self, name, uncertainty_down, uncertainty_up, scheme=1.):
        self.nps[name] = {'nom':math.fabs(uncertainty_up-uncertainty_down),'range':(-3,3)}
        self.pdf = OverallSys(
            self.pdf, name, uncertainty_down, uncertainty_up, scheme
        )

    def AddHistoSys(self, name, uncertainty_down, uncertainty_up, scheme=1.):
        self.nps[name] = {'nom':math.fabs(sum(uncertainty_down)-sum(uncertainty_up)),'range':(-3,3)}
        assert len(uncertainty_down) == len(uncertainty_up) == len(self.bincontent)
        self.pdf = HistoSys(
            self.pdf, name, self.bincontent, uncertainty_down, uncertainty_up, scheme
        )
