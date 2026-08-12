import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    g = np.asarray(g)
    s = np.asarray(s)
    s = s * beta + (1-beta) * g**2
    # s = np.multiply(s,beta) + np.multiply(g*g, (1-beta))
    w = w - (lr / np.sqrt(s + eps)) * g
    return (w, s)