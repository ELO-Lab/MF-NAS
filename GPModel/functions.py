from gplearn.functions import _Function
import numpy as np

def _protected_division(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.divide(x1, x2)
        return np.where(np.isinf(res), -10000000, res)


def _protected_sqrt(x1):
    with np.errstate(divide='ignore', invalid='ignore'):
        """Closure of square root for negative arguments."""
        res = np.sqrt(x1)
        return np.where(np.isnan(res), -100000, res)


def _protected_log(x1):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.log(x1)
        """Closure of log for zero and negative arguments."""
        return np.where(np.isnan(res),  -100000, res)

our_div = _Function(function=_protected_division, name='div', arity=2)
our_sqrt = _Function(function=_protected_sqrt, name='sqrt', arity=1)
our_log = _Function(function=_protected_log, name='log', arity=1)