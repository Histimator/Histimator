import numpy as np
from iminuit import describe


def divide(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0
    return c


class MinimalFuncCode:
    def __init__(self, arg):
        self.co_varnames = tuple(arg)
        self.co_argcount = len(arg)

    def append(self, varname):
        tmp = list(self.co_varnames)
        tmp.append(varname)
        self.co_varnames = tuple(tmp)
        self.co_argcount = len(self.co_varnames)


class FakeFuncCode:
    def __init__(self, f, prmt=None, dock=0, append=None):
        # f can either be tuple or function object
        self.co_varnames = describe(f)
        self.co_argcount = len(self.co_varnames)
        self.co_argcount -= dock
        self.co_varnames = self.co_varnames[dock:]

        if prmt is not None:  # rename parameters from the front
            for i, p in enumerate(prmt):
                self.co_varnames[i] = p

        if isinstance(append, str):
            append = [append]

        if append is not None:
            old_count = self.co_argcount
            self.co_argcount += len(append)
            self.co_varnames = tuple(
                list(self.co_varnames[:old_count]) +
                append +
                list(self.co_varnames[old_count:]))
