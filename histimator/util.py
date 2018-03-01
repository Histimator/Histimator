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

def merge_func_code(prefix=None, skip_first=False, factor_list=None,
                    skip_prefix=None, *arg):  # this needs a seiours refactor
    if prefix is not None and len(prefix) != len(arg):
        raise ValueError('prefix should have the same length as number of ',
                         'functions. Expect %d(%r)' % (len(arg), arg))
    all_arg = []
    skip_prefix = set([]) if skip_prefix is None else set(skip_prefix)
    for i, f in enumerate(arg):
        tmp = []
        first = skip_first
        for vn in describe(f):
            newv = vn
            if not first and prefix is not None and newv not in skip_prefix:
                newv = prefix[i] + newv
            first = False
            tmp.append(newv)
        all_arg.append(tmp)

    if factor_list is not None:
        for i, f in enumerate(factor_list):
            tmp = []
            for vn in describe(f):
                newv = vn
                if prefix is not None and newv not in skip_prefix:
                    newv = prefix[i] + newv
                first = False
                tmp.append(newv)
            all_arg.append(tmp)

    #now merge it
    #FIXME: do something smarter
    merge_arg = []
    for a in all_arg:
        for v in a:
            if v not in merge_arg:
                merge_arg.append(v)

    #build the map list of numpy int array
    pos = []
    for a in all_arg:
        tmp = []
        for v in a:
            tmp.append(merge_arg.index(v))
        pos.append(np.array(tmp, dtype=np.int))
    return MinimalFuncCode(merge_arg), pos
