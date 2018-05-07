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

def merge_func_code(*arg):

    """
    merge function arguments.::
        def f(x,y,z): return do_something(x,y,z)
        def g(x,z,p): return do_something(x,y,z)
        fc, pos = merge_func_code(f,g)
        #fc is now ('x','y','z','p')
    """

    all_arg = []
    for i, f in enumerate(arg):
        tmp = []
        first = False
        for vn in describe(f):
            newv = vn
            first = False
            tmp.append(newv)
        all_arg.append(tmp)

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

def mask_component_args(fpos, *arg):
    tmparg = []
    for pos in fpos:
        tmparg.append(arg[pos])
    return tuple(tmparg)

def rename(f, newarg):
    return FakeFunc(f, newarg)


class FakeFunc:
    def __init__(self, f, prmt=None):
        self.f = f
        self.func_code = FakeFuncCode(f, prmt)
        self.func_defaults = getattr(f, 'func_defaults', None)
        self.__name__ = getattr(f, '__name__', 'unnamed')

    def __call__(self, *arg):
        return self.f(*arg)
