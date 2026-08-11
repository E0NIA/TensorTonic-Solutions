import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if len(y_pred) != len(y_true):
        return None
    return np.square(y_pred - y_true).mean()
    
