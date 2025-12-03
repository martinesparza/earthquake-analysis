"""
Module about communication subspaces
"""

import numpy as np
from scipy import sparse
from scipy.linalg import null_space, orth
from sklearn.base import BaseEstimator


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


def variance_in_subspace(X, W):
    """
    Variance in a given subspace

    Parameters
    ----------
    X : 2D np.ndarray
        n_samples x n_features data array
    W : 2D np.ndarray
        n_features x n_components projection matrix

    Returns
    -------
    variance in the subspace
    """
    # Assume W is orthogonal
    W_norm = W / np.linalg.norm(W, axis=0)
    return np.trace(W_norm.T @ np.cov(X.T) @ W_norm)


def variance_in_dims(X, W):
    """
    Variance in each dimension of a given subspace

    Parameters
    ----------
    X : 2D np.ndarray
        n_samples x n_features data array
    W : 2D np.ndarray
        n_features x n_components projection matrix

    Returns
    -------
    np.array with the variance in the each dimension
    """
    W_norm = W / np.linalg.norm(W, axis=0)
    return np.diag(W_norm.T @ np.cov(X.T) @ W_norm)


class ReducedRankCommSubspace:
    """
    Reduced Rank comm space (find the reduced comm subspace between areas)

    Constructor parameters
    ----------------------
    rank : int
        rank constraint.
    reg : float (optional)
        regularization parameter
        (alpha in sklearn.linear_model.Ridge)
    """

    def __init__(self, rank, reg=None):
        self.rank = rank
        self.reg = reg if reg is not None else 0

    def __str__(self):
        return "Reduced Rank Regressor (rank = {})".format(self.rank)

    def fit(self, _X, _Y):
        """
        Fit reduced rank regressor to data.

        Parameters
        ----------
        _X : ndarray
            matrix of features with shape (n_samples x n_features)
        _Y : ndarray
            matrix of targets with shape (n_samples x n_target_features)

        Returns
        -------
        Sets attributes needed for prediction and returns None
        """
        if np.ndim(_X) == 1:
            _X = np.reshape(_X, (-1, 1))
        if np.ndim(_Y) == 1:
            _Y = np.reshape(_Y, (-1, 1))

        self.mean_input = _X.mean(axis=0)
        self.mean_output = _Y.mean(axis=0)

        X = _X - self.mean_input
        Y = _Y - self.mean_output

        CXX_inv = np.linalg.pinv((X.T @ X) + self.reg * sparse.eye(X.shape[1]))
        CXY = X.T @ Y
        W = CXX_inv @ CXY
        U, _S, _V = np.linalg.svd(W)

        self.Uw = U[:, : self.rank]
        self.projector_mx = self.Uw @ self.Uw.T
        print(self.projector_mx.shape)

        return
