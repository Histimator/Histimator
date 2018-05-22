"""Built-in estimator classes.
"""

# from . import backend as K
# if K.backend() == 'tensorflow':
# import tensorflow as tf


import numpy as np
import scipy.stats as st
from .models import HistiModel
from .pdfs import cpoisson
# from scipy.stats import poisson
from .util import FakeFuncCode
from iminuit import Minuit, describe
from iminuit.util import make_func_code


class BinnedLH(object):
    def __init__(self, model, data=None, bins=40, weights=None,
                 weighterrors=None, bound=None,
                 badvalue=1000000, extended=False,
                 use_w2=False, nint_subdiv=1, minimiser = 'minuit'):
        if isinstance(model, HistiModel):
            self.pdf = model.pdf
            self.binedges = model.binedges
            self.func_code = FakeFuncCode(self.pdf, dock=True)
            self.parameters = model.Parameters()
        else:
            print("ERROR model should be an instance of HistiModels")
        self.minimiser = minimiser
        if hasattr(model, "data"):
            self.data = model.data
            if self.data is None:
                print("error: data is None, please feed the model with data")
            else:
                self.h = np.asarray(self.data)
                self.binned_data = data
                self.N = self.h.sum()
        else:
            print("error: model has no attribute data")

        self.use_w2 = use_w2
        self.extended = extended
        if bound is None:
            self.bound = (self.data[0], self.data[-1])
        else:
            self.bound = bound
        self.mymin, self.mymax = self.bound
        pdf_sig = describe(self.pdf)
        self.func_code = make_func_code(pdf_sig[1:])
        self.func_defaults = None

    def __call__(self, *arg):
        if self.minimiser is 'scipy':
            arg = tuple(arg[0].tolist())
        constraint = 0.
        h_pred = self.pdf.evaluatePdf(*arg)
        parameters = dict(zip(describe(self.pdf.evaluatePdf)[1:], arg))
        constraints = []
        for par in parameters.keys():
            if "syst" in par.lower():
                constraints.append(parameters[par])
        constraint = np.asarray([st.norm(con, 1.).pdf(0) for con in constraints]).prod()
        if constraint <= 0. or isNaN(constraint):
            constraint = 1.
        h_meas = self.h
        if self.extended:
            return -(cpoisson.logpdf(self.N, h_pred.sum()) +
                     cpoisson.logpdf(h_meas, h_pred).sum() + np.log(constraint))
        else:
            return -cpoisson.logpdf(h_meas, h_pred).sum() - np.log(constraint)


class ProfileLH:
    def __init__(self, model, data=None, bins=40, weights=None,
                 weighterrors=None, bound=None,
                 badvalue=1000000, extended=False,
                 use_w2=False, nint_subdiv=1, minimiser = 'minuit'):
        if isinstance(model, HistiModel):
            self.pdf = model.pdf
            self.binedges = model.binedges
            self.func_code = FakeFuncCode(self.pdf, dock=True)
            self.parameters = model.Parameters()
        else:
            print("ERROR model should be an instance of HistiModels")
        self.model = model
        self.minimiser = minimiser
        if hasattr(model, "data"):
            self.data = model.data
            if self.data is None:
                print("error: data is None, please feed the model with data")
            else:
                self.h = np.asarray(self.data)
                self.binned_data = data
                self.N = self.h.sum()
        else:
            print("error: model has no attribute data")



        self.params = model.Parameters()
        self.observables = model.pois.keys()

        pdf_sig = [i for i in describe(self.pdf) if i in model.pois.keys()]
        self.func_code = make_func_code(pdf_sig)
        self.func_defaults = None


    def __call__(self, *arg):
        pois = {key:arg[i] for i,key in enumerate(self.observables)}
        self.constrained_params = self.model.FixedParameters(pois)

        unconstrainedLH = BinnedLH(self.model)
        minimiser = Minuit(unconstrainedLH, **self.params)
        minimiser.migrad()
        unconstrainedBestFit = minimiser.get_fmin().fval

        constrainedLH = BinnedLH(self.model)
        newminimiser = Minuit(constrainedLH, **self.constrained_params)
        newminimiser.migrad()
        constrainedBestFit = newminimiser.get_fmin().fval
        return constrainedBestFit - unconstrainedBestFit

def isNaN(num):
    return num != num
