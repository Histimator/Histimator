import numpy as np
from scipy.optimize import minimize

class LikelihoodFit(object):
    def __init__(self, name, opt = 'minuit'):
        self.name = name
        if opt.lower() in ['minuit','scipy']:
            self.type = opt.lower()
        else: self.type = 'minuit'
    
    def NLL(self, model):
        self.model = model
        names = model.nps.keys()
        init = np.ones(len(names))
#        ranges = [[-i*1.,i*1.] for i in init]
        result = minimize(self.f, init, args=names, method = 'Nelder-Mead')
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
            if not thechannel.IsDataSet:
                thechannel.data = thechannel.nominal
            if nll:
                nll -= (np.sum(thechannel.data*np.log(expected)) - np.sum(expected))
            else:
                nll = -(np.sum(thechannel.data*np.log(expected)) - np.sum(expected))
        return nll
        
