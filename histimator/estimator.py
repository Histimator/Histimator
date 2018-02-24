import numpy as np
import scipy.stats as st
import models
from scipy.stats import poisson
from util import FakeFuncCode


class BinnedLH(object):
    def __init__(self, model, data=None, bins=40, weights=None,
                 weighterrors=None, bound=None,
                 badvalue=1000000, extended=False,
                 use_w2=False, nint_subdiv=1):
        if isinstance(model, models.HistiModel):
            self.pdf = model.pdf
            self.binedges = model.binedges
            self.func_code = FakeFuncCode(self.pdf, dock=True)
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

    def __call__(self, *arg):
        self.last_arg = arg[1:]
        bwidth = np.diff(self.binedges)
        centre = self.binedges[:-1] + bwidth/2.0
        h_meas = self.h
        h_pred = np.asarray(
            [self.pdf(centre[i], arg[-1]) for i in range(bwidth.shape[0])]
        )
        h_pred = h_pred*bwidth
        if self.extended:
            return -st.poisson.logpmf(self.N, h_pred.sum())-poisson.logpmf(h_meas, h_pred).sum()
        else:
            return -poisson.logpmf(h_meas, h_pred).sum()


class BinnedLHProfile(BinnedLH):
    def __init__(self, model, data=None, bins=40, weights=None,
                 weighterrors=None, bound=None,
                 badvalue=1000000, extended=False,
                 use_w2=False, nint_subdiv=1):
        if isinstance(model, models.HistiModel):
            self.pdf = model.pdf
            self.binedges = model.binedges
            self.func_code = FakeFuncCode(self.pdf, dock=True)
