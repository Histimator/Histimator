# -*- coding: utf-8 -*-

"""Top-level package for Histimator."""
from backend import *
from estimator import *
from iterpolation import *
from models import *
from optimiser import *
from pdfs import *
from util import *


__author__ = """Yacine Haddad"""
__email__ = 'yhaddad@cern.ch'
__version__ = '0.1.0'


__all__ = [
    "BinnedLH",
    "BinnedLHProfile",
    "Interpolate",
    "HistiModel",
    "HistiChannel",
    "HistiSample",
    "HistogramPdf"
    "NormedHist",
    "MinimalFuncCode",
    "FakeFuncCode"
]
