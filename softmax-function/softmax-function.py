import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x)
    row_max = np.max(x, axis=-1, keepdims=True)
    norm_exp = np.exp(x - row_max)
    norm_sum = np.sum(norm_exp, axis=-1, keepdims=True)

    return norm_exp / norm_sum
    