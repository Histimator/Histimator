"""Built-in estimator classes.
"""

# from . import backend as K
# if K.backend() == 'tensorflow':
# import tensorflow as tf


import numpy as np
import scipy.stats as st
from .models import HistiModel
from .pdfs import cpoisson
from .util import FakeFuncCode

from iminuit import describe
from iminuit.util import make_func_code

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

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
        constraint = st.norm(1., 1.).logpdf(np.asarray(constraints)).prod()
        if constraint <= 0. or isNaN(constraint):
            constraint = 0.
        h_meas = self.h
        if self.extended:
            return -(cpoisson.logpdf(self.N, h_pred.sum()) +
                     cpoisson.logpdf(h_meas, h_pred).sum() + constraint)
        else:
            return -cpoisson.logpdf(h_meas, h_pred).sum() - constraint

class RidgeFit:
    def __init__(self, model, data=None, npoints=None, alphas = None):
        self.model = model.pdf
        if data is None:
            self.data = model.data
        else:
            self.data = data
            
        if npoints is None:
            self.npoints = 1000.*len(self.data)
        else:
            self.npoints = npoints
        
        if alphas is None:
            self.alphas = np.array([1e-4,1e-3,1e-2,0.1,.5,1.,5.,10])
        else:
            self.alphas = alphas

    def __call__(self, *arg):
        print 'args are',arg
        lows = []
        highs = []
        for param in arg:
            lows.append(param[0])
            highs.append(param[1])
        print 'highs and lows',highs,lows
        y = np.random.uniform(lows, highs,(int(self.npoints),len(arg)))
        X = [self.model.evaluatePdf(*p) for p in y]

        ridge = Ridge()
        grid = GridSearchCV(estimator=ridge, param_grid=dict(alpha=self.alphas))
        grid.fit(X, y)
        return grid.best_estimator_.predict(self.data.reshape(1,-1))[0]

def isNaN(num):
    return num != num
