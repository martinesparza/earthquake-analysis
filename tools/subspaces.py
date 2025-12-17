"""
Module about communication subspaces
"""

import numpy as np
from scipy import sparse
import scipy
from scipy.linalg import null_space, orth
from sklearn.base import BaseEstimator

import tools.dimensionality as dim


def project_signal(trial_data_, W, signal, out_fieldname):
    """
    Project a signal using a weight matrix

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    W : np.array
        projection matrix
        shape: N x D
    signal : str
        signal to project
    out_fieldname : str
        name of the field in which to store the projections

    Returns
    -------
    trial_data with the projections added
    """
    trial_data = trial_data_.copy()
    trial_data[out_fieldname] = [s @ W for s in trial_data[signal].values]

    return trial_data


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

        # Here you are asking: Which dirctions of X have the largest structured changes on Y
        # The projector matrix is orthogonal.
        # print(self.projector_mx.shape)

        return

    @property
    def coef_(self):
        return self.projector_mx.T


def compute_embedding(td, signal_x, signal_y, time_bin_window, model, k=None):

    X = np.stack(td[signal_x].values)
    y = np.stack(td[signal_y].values)
    _, _, n_feat_in = X.shape
    _, _, n_feat_out = y.shape

    # Slice time window and reshape
    X, y = (
        X[:, time_bin_window[0] : time_bin_window[1], :],
        y[:, time_bin_window[0] : time_bin_window[1], :],
    )
    X, y = X.reshape(-1, n_feat_in), y.reshape(-1, n_feat_out)

    if model == "pca":
        pca_model = dim.compute_pca(X, n_components=k)
        W = pca_model.components_.T
    else:
        model.fit(X, y)
        W = scipy.linalg.orth(model.coef_.T)

    return W


def compute_overlap_between_subspaces(emb_a, emb_b):
    angles = scipy.linalg.subspace_angles(emb_a, emb_b)
    return np.mean(np.cos(angles) ** 2)
