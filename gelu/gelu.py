import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    cdf_fun = np.vectorize(math.erf)
    cdf = 0.5 * (1 + cdf_fun(x * 2**-0.5))
    return cdf * x
