import numpy as np
from scipy.optimize import minimize

class LikelihoodFit(object):
    def __init__(self, name, opt = 'minuit'):
        self.name = name
        if opt.lower() in ['minuit','scipy']:
            self.type = opt.lower()
        else: self.type = 'minuit'
    
    def newNLL(self, model):
        self.model = model
        names = model.nps.keys()
        init = np.ones(len(names))
#        ranges = [[i*.1,i*10.] for i in init]
        result = minimize(self.f, init, args=names, method = 'BFGS')
        if result.success:
            return zip(names,result.x)
        else:
            raise ValueError(result.message)

    def f(self, values, names):
        model = self.model
        params = zip(names, values)
        nll = None
        for chan in model.Channels:
            thechannel = model.Channels[chan]
            expected = thechannel.Pdf(params)
            if nll:
                nll -= (np.sum(thechannel.data*np.log(expected)) - np.sum(expected))
            else:
                nll = -(np.sum(thechannel.data*np.log(expected)) - np.sum(expected))
        return nll
        
