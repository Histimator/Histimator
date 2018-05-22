from __future__ import absolute_import
from __future__ import print_function

import importlib
import sys
import os
import pkgutil

_BACKEND_NAME = 'scipy'

T = None

try:
    from . import scipy as T
except ImportError:
    pass


def use(backend="tensorflow"):
    print("os.path.abspath('.') == ", os.path.abspath('.'))
    print("os.getcwd('.')       == ", os.getcwd())
    for importer, package, _ in pkgutil.iter_modules(os.path.dirname(__file__)):
        name = package.split('.py')[0]
        print("module name : ", name)
        if name != backend:
            del sys.modules[name]
    global T
    T = __import__(backend, globals(), locals(), [], 1)
    _BACKEND_NAME = backend
    print("Histimator will use *%10s* backend!" % _BACKEND_NAME)
    return T
