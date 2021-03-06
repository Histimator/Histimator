from __future__ import division
import numpy as np
from .util import MinimalFuncCode, FakeFuncCode
from .util import merge_func_code, mask_component_args
from iminuit import describe
from .interpolation import Interpolate
import scipy.special as sp
from scipy.stats.distributions import rv_continuous


class HistogramPdf(object):
    def __init__(self, hy, binedges, xname='x'):
        """
        A histogram PDF. User supplies a template histogram
        with bin contents and bin
        edges. The resulting PDF is NOT normalized.
        """
        self.hy = hy.astype(float)
        self.binedges = binedges
        if len(binedges) != len(hy) + 1:
            raise ValueError(
                'bin edges must be exactly one entry more than hy'
            )
        varnames = [xname]
        self.func_code = MinimalFuncCode(varnames)
        self.func_defaults = None

    def integrate(self, bound, nint_subdiv=0, arg=None):
        edges = np.copy(self.binedges)
        [ib0, ib1] = np.digitize([bound[0], bound[1]], edges)
        ib0 = max(ib0, 0)
        ib1 = min(ib1, len(edges)-1)
        edges[ib0-1] = max(edges[ib0-1], bound[0])
        edges[ib1] = min(edges[ib1], bound[1])
        ilo = max(0, ib0-1)
        ihi = ib1+1 if edges[ib1-1] != edges[ib1] else ib1
        return (self.hy[ilo:ihi-1]*np.diff(edges[ilo:ihi])).sum()

    def __call__(self, *arg):
        x = arg[0]
        [i] = np.digitize([x], self.binedges)
        if i > 0 and i <= len(self.hy):
            return self.hy[i-1]
        else:
            return 0.0


class NormedHist:
    def __init__(self, f, norm='N'):
        self.f = f
        if norm in describe(f):
            raise ValueError(
                '%s is already taken, please choose another name' % norm
            )
        self.func_code = FakeFuncCode(f, append=norm)
        self.func_defaults = None

        try:
            self.binedges = self.f.binedges
        except AttributeError:
            self.binedges = None

    def __call__(self, *arg):
        N = arg[-1]
        fval = self.f(arg[0])
        return fval*N

    def integrate(self, bound, nint, *arg):
        N = arg[-1]
        ana = self.f.integrate(bound, nint, arg[:-1])
        return N*ana


class OverallSys:
    def __init__(self, f, OverallSys='NormSys', down=1., up=1., scheme=1.):
        self.f = f
        self.up = up
        self.down = down
        self.scheme = scheme
        if OverallSys in describe(f):
            raise ValueError(
                '%s is already taken, please choose another name' % OverallSys
            )
        self.func_code = FakeFuncCode(f, append=OverallSys)
        self.func_defaults = None

        try:
            self.binedges = self.f.binedges
        except AttributeError:
            self.binedges = None

    def __call__(self, *arg):
        fval = self.f(*arg[:-1])
        alpha = arg[-1]
        inter = Interpolate(self.scheme)
        scale = inter(alpha, 1., self.down, self.up)
        return fval * scale

    def integrate(self, bound, nint, *arg):
        alpha = arg[-1]
        inter = Interpolate(self.scheme)
        mod = inter(alpha, 1., self.down, self.up)
        ana = self.f.integrate(bound, nint, arg[:-1])
        return mod*ana


class HistoSys:
    def __init__(self, f, HistoSys='ShapeSys', nom=[1.], down=[1.], up=[1.], scheme=1.):
        self.f = f
        self.up = np.asarray(up)/nom
        self.down = np.asarray(down)/nom
        self.scheme = scheme
        if HistoSys in describe(f):
            raise ValueError(
                '%s is already taken, please choose another name' % HistoSys
            )
        try:
            self.binedges = self.f.binedges
        except AttributeError:
            self.binedges = None

        self.func_code = FakeFuncCode(f, append=HistoSys)
        self.func_defaults = None

    def __call__(self, *arg):
        fval = self.f(*arg[:-1])
        alpha = arg[-1]
        inter = Interpolate(self.scheme)
        scale = inter(alpha, 1., self.down[int(arg[0]-0.5)], self.up[int(arg[0]-0.5)])
        return fval * scale

    def integrate(self, bound, nint, *arg):
        alpha = arg[-1]
        inter = Interpolate(self.scheme)
        mod = inter(alpha, 1., self.down[arg[0]], self.up[arg[0]])
        ana = self.f.integrate(bound, nint, arg[:-1])
        return mod*ana


class HistiAddPdf:
    def __init__(self, *arg):
        allf = list(arg)
        self.func_code, allpos = merge_func_code(*arg)
        funcpos = allpos[:len(arg)]
        for func in arg:
            try:
                self.binedges = func.binedges
            except AttributeError:
                print("One of these functions ({}) isn't binned", func.name)
        self.func_defaults = None
        self.arglen = self.func_code.co_argcount
        self.allf = arg # f function
        self.allpos = allpos # position for f arg
        self.numf = len(self.allf)
        self.argcache = [None] * self.numf
        self.cache = np.zeros(self.numf)
        self.hit = 0

    def __call__(self, *arg):
        ret = 0.
        for i in range(self.numf):
            thispos = self.allpos[i]
            this_arg = mask_component_args(thispos, *arg)
            tmp = self.allf[i](*this_arg)
            self.argcache[i] = this_arg
            self.cache[i] = tmp
            ret += tmp
        return ret

    def evaluatePdf(self, *arg):
        bwidth = np.diff(self.binedges)
        centre = self.binedges[:-1] + bwidth/2.0
        h_pred = np.asarray([self.__call__(centre[i], *arg) for i in range(bwidth.shape[0])]) * bwidth
        return h_pred


class HistiCombPdf:
    def __init__(self, *arg):
        allf = []
        self.binedges = []

        ### Needs major cleanup using Numpy.reshape incase one of
        ### the functions is already a HistiCombPdf
        ### points:
        ###
        ### a) bin edges are already list of numpy arrays
        ### b) functions already have a list of component functions
        ###                 ( whereas in AddPdf they are a tuple )
        ###
        ### in point b) this manifests as having two returns when
        ## evaluating and therefore the wrong shape. (2,n) vs (n,)

        for func in arg:
            if hasattr(func, 'binedges'):
                if isinstance(func.binedges, list):
                    for sub in func.binedges:
                        self.binedges.append(sub)
                else:
                    self.binedges.append(func.binedges)

            if hasattr(func, 'allf'):
                if isinstance(func.allf, list):
                    for f in func.allf:
                        allf.append(f)
                else:
                    allf.append(func)
            else:
                allf.append(func)
        self.func_code, allpos = merge_func_code(*tuple(allf))
        funcpos = allpos[:len(arg)]
        self.func_defaults = None
        self.arglen = self.func_code.co_argcount
        self.allf = allf # f function
        self.allpos = allpos # position for f arg
        self.numf = len(self.allf)
        self.argcache = [None]*self.numf
        self.cache = np.zeros(self.numf)
        self.hit = 0

    def __call__(self, *arg):
        ret = []
        for i in range(self.numf):
            print('this func', i, 'with bins', self.binedges)
            thispos = self.allpos[i]
            this_arg = mask_component_args(thispos, *arg)
            tmp = self.allf[i](*this_arg)
            self.argcache[i] = this_arg
            self.cache[i] = tmp
            ret.append(tmp)
        return tuple(ret)

    def evaluatePdf(self, *arg):
        h_pred = np.asarray([])
        binwidths = [np.diff(edges) for edges in self.binedges]
        for region in range(self.numf):
            bwidth = np.diff(self.binedges[region])
            centre = self.binedges[region][:-1] + bwidth/2.0
            h_new = [self.allf[region](centre[i], *arg) for i in range(bwidth.shape[0])]
            h_pred = np.hstack([h_pred, h_new])
        return h_pred


class cpoisson_gen(rv_continuous):
    """ Continuous Poisson distribution
    """
    def _logpdf(self, k, mu):
        Pk = np.where(mu > 0, sp.xlogy(k, mu) - sp.gammaln(k+1) - mu, 0)
        return Pk

    def _pdf(self, k, mu):
        # np.power(mu, n)*np.exp(-mu)/sp.gamma(n+1)
        return np.exp(self._logpmf(k, mu))

    def _cdf(self, x, mu):
        k = np.floor(x)
        return sp.pdtr(k, mu)


cpoisson = cpoisson_gen(name="cpoisson", longname='A Poisson')
