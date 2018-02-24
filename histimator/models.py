from pdfs import HistogramPdf, NormedHist, OverallSys
from probfit import AddPdf


class HistiModel(object):
    def __init__(self, name='TemplateModel'):
        self.name = name
        self.n_channels = 0
        self.channels = []
        self.pdf = None
        self.data = None
        self.binedges = None

    def AddChannel(self, channel):
        name = channel.name
        if name is None:
            name = 'channel_'+self.n_channels
        self.n_channels += 1
        self.channels.append(name)
        for sample in channel.samples:
            s = channel.samples[sample]
            if self.pdf is None:
                self.pdf = s.pdf
                self.binedges = s.binedges
            else:
                self.pdf = AddPdf(self.pdf, s.pdf)
        self.data = channel.data


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

    def SetHisto(self, hist):
        self.hist = hist
        self.bincontent, self.binedges = self.hist
        self.pdf = HistogramPdf(self.bincontent, self.binedges)

    def AddNorm(self, name='mu', nom=1, min=0, max=3):
        self.pdf = NormedHist(self.pdf, norm=name)

    def AddOverallSys(self, name, uncertainty_down, uncertainty_up, scheme=1.):
        self.pdf = OverallSys(
            self.pdf, name, uncertainty_up, uncertainty_down, scheme
        )
