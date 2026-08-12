import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asarray(x)
    exp = np.exp(x)
    return (exp - exp**-1)/(exp + exp**-1)