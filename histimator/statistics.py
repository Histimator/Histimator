from iminuit import Minuit, describe
from iminuit.util import make_func_code
from .estimator import BinnedLH

class ProfileLikelihoodCalculator:
    def __init__(self, model):
        self.model = model
        self.pdf = model.pdf
        self.data = model.data
        pdf_sig = [i for i in describe(self.pdf) if i in self.model.pois.keys()]
        self.func_code = make_func_code(pdf_sig)
        self.func_defaults = None


    def __call__(self, *arg):
        print 'args are',arg
        params = self.model.Parameters()
        unconstrainedLH = BinnedLH(self.model)
        minimiser = Minuit(unconstrainedLH, **params)
        minimiser.migrad()
        unconstrainedBestFit = minimiser.get_fmin().fval
        print "unconstrainted ", unconstrainedBestFit
        pois = {}
        for i,key in enumerate(self.model.pois.keys()):
            pois[key] = arg[i]
        print 'pois are', pois
        print 'the argument is', arg
        constrained_params = self.model.FixedParameters(pois)
        print 'with parameters', constrained_params
        constrainedLH = BinnedLH(self.model)
        newminimiser = Minuit(constrainedLH, **constrained_params)
        newminimiser.migrad()
        constrainedBestFit = newminimiser.get_fmin().fval
        print 'constrained',constrainedBestFit
        return constrainedBestFit - unconstrainedBestFit
