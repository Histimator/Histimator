from iminuit import Minuit
import math

class HistiNorm:
    def __init__(self, name, value, minimum, maximum):
        self.name = name
        self.value = value
        self.min = minimum
        self.maximum = maximum
        def ApplyNorm(self, value):
            return [i*value for i in self.Histo]
            
