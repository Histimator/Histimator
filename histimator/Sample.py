from iminuit import Minuit
import math

from .NormFactor import HistiNorm

class HistiSample:
    def __init__(self, name):
        self.name = name
        self.Norms = {}
    def SetHisto(self, Histo):
        self.Histo = Histo
        return self
    def AddNorm(self, name, value, minimum, maximum):
        if not value:
            value = 1.
        if not minimum:
            minimum = value * .1
        if not maximum:
            maximum = value * 10.
        norm = HistiNorm(name, value, minimum, maximum)
        self.Norms[norm.name] = norm
        return self

    def Evaluate(self, pars):
        hist = self.Histo
        for parameter in pars:
            for par, value in parameter:
                if par in self.Norms:
                    print par, "parameter in", self.name
                    hist = [i*value for i in hist]
        return hist
