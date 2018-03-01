import numpy as np
from .util import MinimalFuncCode, FakeFuncCode, merge_func_code
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

class AddPdf:
    def __init__(self, prefix=None, factors=None, skip_prefix=None, *arg):
        if factors is not None and len(factors)!=len(arg):
            raise ValueError('factor is specified but has different length'
                             ' from arg.')
        allf = list(*arg)
        if factors is not None:
            allf += factors

        self.func_code, allpos = merge_func_code(*arg, prefix=prefix,
                                                 skip_first=True,
                                                 factor_list=factors,
                                                 skip_prefix=skip_prefix)

        funcpos = allpos[:len(arg)]
        factpos = allpos[len(arg):]

        self.func_defaults=None
        self.arglen = self.func_code.co_argcount
        self.allf = arg # f function
        self.factors = tuple(factors) if factors is not None else None# factor function
        self.allpos = allpos # position for f arg
        self.factpos = factpos # position for factor arg
        self.numf = len(self.allf)
        self.argcache = [None]*self.numf
        self.factor_argcache = [None]*self.numf
        self.cache = np.zeros(self.numf)
        self.factor_cache = np.zeros(self.numf)
        self.hit = 0

    def __call__(self, *arg):
        for i in range(self.numf):
            thispos = self.allpos[i]
            this_arg = construct_arg(arg, thispos)

            if self.argcache[i] is not None and fast_tuple_equal(this_arg, self.argcache[i], 0):
                tmp = self.cache[i]
                self.hit+=1
            else:
                tmp = self.allf[i](*this_arg)
                self.argcache[i]=this_arg
                self.cache[i]=tmp

            if self.factors is not None: # calculate factor
                factor_arg = construct_arg(arg, self.factpos[i])
                if self.factor_argcache[i] is not None and fast_tuple_equal(factor_arg, self.factor_argcache[i], 0):
                    tmp_factor = self.factor_cache[i]
                    self.hit+=1
                else:
                    tmp_factor = self.factors[i](*factor_arg)
                    self.factor_argcache[i] = factor_arg
                    self.factor_cache[i] = tmp_factor

                ret += tmp_factor*tmp
            else:
                ret += tmp
        return ret

    def parts(self):
        return [self._part(i) for i in range(self.numf)]

    def _part(self, findex):
        def tmp(*arg):
            thispos = self.allpos[findex]
            this_arg = construct_arg(arg, thispos)
            ret = self.allf[findex](*this_arg)
            if self.factors is not None:
                facpos = self.factpos[findex]
                facarg = construct_arg(arg, facpos)
                fac = self.factors[findex](*facarg)
                ret *= fac
            return ret

        tmp.__name__ = getattr(self.allf[findex],'__name__','unnamedpart')
        ret = FakeFunc(tmp)
        ret.func_code = self.func_code
        return ret

    def eval_parts(self,*arg):
        ret = list()
        for i in range(self.numf):
            tmp = self._part(i)(*arg)
            ret.append(tmp)
        return tuple(ret)

    def integrate(self, bound, nint, *arg):
        for findex in range(self.numf):
            fpos = self.allpos[findex]

            #docking off x and shift due to no x in arg
            this_arg = construct_arg(arg, fpos[1:]-1)
            thisf = self.allf[findex]
            fac = 1.

            if self.factors is not None:
                facpos = self.factpos[findex]
                # -1 accounting for no dependent variable in this arg
                facarg = construct_arg(arg, facpos-1)
                fac = self.factors[findex](*facarg)

            thisint = integrate1d(thisf, bound, nint, this_arg)
            ret += fac*thisint

        return ret
