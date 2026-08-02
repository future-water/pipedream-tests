import numpy as np
import scipy.optimize

def compute_normal_depth(h, f_A, f_Pe, Q, n, S_o, *args):
    A = f_A(h, *args)
    Pe = f_Pe(h, *args)
    result = Q - np.sqrt(S_o) * A**(5/3) / Pe**(2/3) / n
    return result

def compute_critical_depth(h, f_A, f_B, Q, *args):
    A = f_A(h, *args)
    B = f_B(h, *args)
    g = 9.81
    result = 1 - Q**2 * B / g / A**3
    return result

def normal_depth(f_A, f_Pe, Q, n, S_o, *args, min_depth=1e-3, max_depth=100.):
    try:
        result = scipy.optimize.root_scalar(compute_normal_depth, method='bisect',
                                            bracket=[min_depth, max_depth],
                                            args=(f_A, f_Pe, Q, n,
                                                S_o, *args))
        h = result.root
    except ValueError:
        h = min_depth
    return h
    
def critical_depth(f_A, f_B, Q, *args, min_depth=1e-3, max_depth=100.):
    try:
        result = scipy.optimize.root_scalar(compute_critical_depth, method='bisect',
                                            bracket=[min_depth, max_depth],
                                            args=(f_A, f_B, Q, *args))
        h = result.root
    except ValueError:
        h = min_depth
    return h
