from iminuit import Minuit, describe
from iminuit.util import make_func_code
from .estimator import ProfileLH

class ProfileLikelihoodCalculator:
    def __init__(self, model):
        self.model = model
        self.test_stat = None
        self.best_fit = []

    def test_statistic(self): 
        params = self.model.Observables()
        plh = ProfileLH(self.model)
        self.test_stat = plh
        min = Minuit(plh, **params)
        min.migrad()
        self.best_fit = min.values

    def p_value(self):
        if self.test_stat is None:
            self.test_statistic()
        values = self.best_fit.values()
        return 2*self.test_stat(*values)
       
