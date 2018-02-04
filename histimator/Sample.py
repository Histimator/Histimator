import numpy as np

class HistiSample(object):
    def __init__(self, name):
        self.name = name
        self.norms = {}
        self.nps = {}
    def SetHisto(self,hist):
        assert isinstance(hist, list)
        self.nominal = np.asarray(hist)
        
    def SetHistoSys(self, name, uphist, downhist):
        self.nps[name] = self.Parametrize(len(self.nominal), uphist, downhist)

    def Parametrize(self, length, up, down):
        func = np.zeros(length, dtype={'names':('nominal','up','down'),'formats':('f8','f8','f8')})
        func['up'] = up
        func['down'] = down
        func['nominal'] = self.nominal
        return func
    
    def PiecewiseLinear(self, alpha, I0, Iup, Idown):
        if alpha < 0:
            return (alpha*(I0-Idown))
        else:
            return (alpha*(Iup - I0))
        
    def Evaluate(self, name, value):
        f = np.vectorize(self.PiecewiseLinear)
        func = self.nps[name]
        return f(value, func['nominal'],func['up'],func['down'])
