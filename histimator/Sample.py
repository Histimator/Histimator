import numpy as np
from numpy.lib.recfunctions import append_fields

class HistiSample(object):
    def __init__(self, name):
        self.name = name
        self.norms = {}
        self.nps = np.array([])
    def SetHisto(self,hist):
        assert isinstance(hist, list)
        self.nominal = np.asarray(hist)
        
    def SetHistoSys(self, name, uphist, downhist, rewrite=False):
        assert len(self.nominal) == len(uphist) == len(downhist)
        if self.nps.dtype.names is None or 'nominal' not in self.nps.dtype.names:        
            self.nps = np.zeros(len(self.nominal), dtype=[
                ('nominal', np.float)
            ])
            self.nps['nominal'] = self.nominal
        if '_'.join(['up'  ,name]) in self.nps.dtype.names or '_'.join(['down'  ,name]) in self.nps.dtype.names:
            print '[WARNING] systematic already booked, use rewrite=True if you want to repalace the values'
            if rewrite :
                self.nps['_'.join(['up'  ,name])] = np.asarray(uphist)
                self.nps['_'.join(['down',name])] = np.asarray(downhist)
        else:
            self.nps = append_fields(self.nps, '_'.join(['up'  ,name]), uphist  ).data
            self.nps = append_fields(self.nps, '_'.join(['down',name]), downhist).data
    
    def PiecewiseLinear(self, alpha, I0, Iup, Idown):
        if alpha < 0:
            return (alpha*(I0-Idown))
        else:
            return (alpha*(Iup - I0))
        
    def Evaluate(self, name, value):
        f = np.vectorize(self.PiecewiseLinear)
        if '_'.join(['up'  ,name]) in self.nps.dtype.names or '_'.join(['down'  ,name]) in self.nps.dtype.names:
            return f(value, self.nps['nominal'],self.nps['_'.join(['up'  ,name])],self.nps['_'.join(['down',name])])
        else:
            return np.zeros(self.nps['nominal'].shape)
