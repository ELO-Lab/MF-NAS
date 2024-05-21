import math
import numpy as np

def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def define_H(n_obj, n_sol):
    H = 0
    while True:
        H += 1
        if nCr(H + n_obj - 1, n_obj - 1) > n_sol:
            return H - 1

def get_ref_point(n_obj, pof):
    H = define_H(n_obj, len(pof))
    r = np.ones(n_obj)
    r = r + 1 / H
    return r
