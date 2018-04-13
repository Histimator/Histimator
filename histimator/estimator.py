"""Built-in estimator classes.
"""

# from . import backend as K
# if K.backend() == 'tensorflow':
# import tensorflow as tf


import numpy as np
import scipy.stats as st
import models
from scipy.stats import poisson
from .util import FakeFuncCode
from iminuit import describe
from iminuit.util import make_func_code

class BinnedLH(object):
    def __init__(self, model, data=None, bins=40, weights=None,
                 weighterrors=None, bound=None,
                 badvalue=1000000, extended=False,
                 use_w2=False, nint_subdiv=1):
        if isinstance(model, models.HistiModel):
            self.pdf = model.pdf
            self.binedges = model.binedges
            self.func_code = FakeFuncCode(self.pdf, dock=True)
            self.parameters = model.Parameters()
        else:
            print "ERROR model should be an instance of HistiModels"

        if hasattr(model, "data"):
            self.data = model.data
            if self.data is None:
                print "error: data is None, please feed the model with data"
            else:
                self.h = np.asarray(self.data)
                self.binned_data = data
                self.N = self.h.sum()
        else:
            print "error: model has no attribute data"

        self.use_w2 = use_w2
        self.extended = extended
        if bound is None:
            self.bound = min(data), max(data)
        else:
            self.bound = bound
        self.mymin, self.mymax = bound
        pdf_sig = describe(self.pdf)
        self.func_code = make_func_code(pdf_sig[1:])
        self.func_defaults = None

    def evaluatePDF(self, *arg):
        bwidth = np.diff(self.binedges)
        centre = self.binedges[:-1] + bwidth/2.0
        h_pred = np.asarray([self.pdf(centre[i], *arg) for i in range(bwidth.shape[0])]) * bwidth
        return h_pred

    def __call__(self, *arg):
        constraint = 0.
        h_pred = self.evaluatePDF(*arg)
        parameters = dict(zip(describe(self.pdf)[1:],arg))
#        print "parameters are", parameters
        constraints = []
        for par in parameters.keys():
            if "syst" in par.lower():
                constraints.append(parameters[par])
        constraint = st.norm(0,1).pdf(np.asarray(constraints)).sum()
        if constraint <= 0. or isNaN(constraint) : 
            constraint = 0.
#        else:
#            print "constraint at {} is {}".format(arg,constraint)
        h_meas = self.h
        if self.extended:
            return -st.poisson.logpmf(self.N, h_pred.sum())-poisson.logpmf(h_meas, h_pred).sum() - constraint
        else:
            return -poisson.logpmf(h_meas, h_pred).sum() - constraint

def isNaN(num):
    return num != num
