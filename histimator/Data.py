from iminuit import Minuit
import math

class HistiData:
    def __init__(self, name="Data"):
        self.name = name
        self.IsSet = False
    def SetData(self, datahist):
        self.obs = datahist
        self.IsSet = True
        return self
