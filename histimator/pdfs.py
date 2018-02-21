import numpy as np
from .util import MinimalFuncCode, FakeFuncCode
from iminuit import describe
from interpolation import Interpolate


class HistogramPdf(object):
    def __init__(self, hy, binedges, xname='x'):
        """
        A histogram PDF. User supplies a template histogram with bin contents and bin
        edges. The resulting PDF is NOT normalized.
        """
        self.hy= hy.astype(float)
        self.binedges= binedges
        if len(binedges)!= len(hy)+1:
            raise ValueError('binedges must be exactly one entry more than hy')
        varnames= [xname]
        self.func_code = MinimalFuncCode(varnames)
        self.func_defaults = None

    def integrate(self, bound, nint_subdiv=0, arg=None):
        edges= np.copy(self.binedges)
        [ib0,ib1]= np.digitize([bound[0],bound[1]], edges)
        ib0= max(ib0,0)
        ib1= min(ib1, len(edges)-1)
        edges[ib0-1]= max(edges[ib0-1],bound[0])
        edges[ib1]= min(edges[ib1],bound[1])
        ilo= max(0,ib0-1)
        ihi= ib1+1 if edges[ib1-1]!=edges[ib1] else ib1
        return (self.hy[ilo:ihi-1]*np.diff(edges[ilo:ihi])).sum()

    def __call__(self, *arg):
        x = arg[0]
        [i]= np.digitize([x], self.binedges)
        if i >0 and i<=len(self.hy):
            return self.hy[i-1]
        else:
            return 0.0

class NormedHist:
    def __init__(self, f, norm='N'):
        self.f = f
        if norm in describe(f):
            raise ValueError('%s is already taken pick something else for extname'%norm)
        self.func_code = FakeFuncCode(f,append=norm)
        self.func_defaults=None

    def __call__(self, *arg):
        N = arg[-1]
        fval = self.f(arg[0])
        return fval*N

    def integrate(self, bound, nint, *arg):
        N = arg[-1]
        ana = self.f.integrate(bound, nint, arg[:-1])
        return N*ana

class OverallSys:
    def __init__(self, f, OverallSys='NormSys', up=1., down=1., scheme=1.):
        self.f = f
        self.up = up
        self.down = down
        self.scheme = scheme
        if OverallSys in describe(f):
            raise ValueError('%s is already taken pick something else for systematicname'%OverallSys)
        self.func_code = FakeFuncCode(f,append=OverallSys)
        #print self.func_code.__dict__
        self.func_defaults=None

    def __call__(self, *arg):
        fval = self.f(arg[0])
        alpha = arg[-1]
        inter = Interpolate(self.scheme)
        if fval>0:
            return inter(alpha, fval, self.up, self.down)
        else:
            return 0

    def integrate(self, bound, nint, *arg):
        alpha = arg[-1]
        inter = Interpolate(self.scheme)
        mod = inter(alpha, 1., self.up, self.down)
        ana = self.f.integrate(bound, nint, arg[:-1])
        return mod*ana
