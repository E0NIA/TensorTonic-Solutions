import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    with np.errstate(all='ignore'):
        if strategy == 'mean':
            col_strat = np.nanmean(X, axis=0)
        else:
            col_strat = np.nanmedian(X, axis=0)

    fallback_value = 0.0
    col_strat = np.where(np.isnan(col_strat), fallback_value, col_strat)
    
    arr_filled = np.where(np.isnan(X), col_strat, X)
    # if still nan remains for pure nan columns
    return arr_filled
