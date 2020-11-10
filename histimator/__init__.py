# -*- coding: utf-8 -*-

"""Top-level package for Histimator."""
from __future__ import division, print_function, absolute_import

from .backend import *
from .estimator import *
from .models import *
from .optimiser import *
from .pdfs import *
from .util import *
from .statistics import *

__author__ = """Vincent Croft and Yacine Haddad"""
__email__ = 'yhaddad@cern.ch'
__version__ = '0.1.0'

__all__ = [
    "ProfileLikelihoodCalculator",
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
