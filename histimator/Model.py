from iminuit import Minuit
import math

#histimator imports
from .Channel import HistiChannel

class HistiModel:
    def __init__(self, name="ANewHistiModel"):
        self.name = name
        self.Channels = {}
        self.nps = {}
    def AddChannel(self, channel):
        if channel.IsDataSet:
            self.Channels[channel.name] = channel
        else:
            print "No Data Set - Using Asimov"
            #To Do impliment Asimov

        for param in channel.nps:
            self.nps[param] = channel.nps[param]
    
