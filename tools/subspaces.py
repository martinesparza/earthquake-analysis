"""
Module about communication subspaces
"""

import numpy as np
from scipy.linalg import null_space, orth


def get_output_null_projector(regressor, var_X=None):
    """
    Matrix that projects input data into the regressor's output-null subspace

    Parameters
    ----------
    regressor : sklearn.base.BaseEstimator
        fitted (linear) regressor
    var_X : np.array, optional
        if supplied, rotate the subspace axes according to
        explained variance in this array

    Returns
    -------
    projection matrix : np.ndarray
    """
    coef = regressor.coef_.copy()
    if coef.ndim == 1:
        coef = np.expand_dims(coef, 0)

    W = null_space(coef)
    if var_X is None:
        return W
    else:
        return  # rotate_by_var(W, var_X)


def get_output_potent_projector(regressor, var_X=None):
    """
    Get matrix that projects input data into the regressor's potent subspace

    Parameters
    ----------
    regressor : sklearn.base.BaseEstimator
        fitted (linear) regressor
    var_X : np.array, optional
        if supplied, rotate the subspace axes according to
        explained variance in this array

    Returns
    -------
    projection matrix : np.ndarray
    """
    W = orth(regressor.coef_.T)
    if var_X is None:
        return W
    else:
        return  # rotate_by_var(W, var_X)


def variance_across_arrays_in_subspace(arrs, W):
    return np.sum(np.var(np.stack([arr @ W for arr in arrs], axis=-1), axis=-1), axis=1)


def variance_in_subspace_df(df, signal, W):
    return variance_across_arrays_in_subspace(df[signal].values, W)
