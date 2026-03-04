"""
Module about communication subspaces
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy import sparse
import scipy
from scipy.linalg import null_space, orth
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import cross_val_score

import tools.subspaces as subspaces
import tools.dimensionality as dim
import tools.decoding as decode
import tools.dataTools as dt


default_scorer = make_scorer(r2_score, multioutput="variance_weighted")


def frac_in_subspace(X: np.ndarray, U: np.ndarray, eps: float = 1e-12):
    """
    X: (T, N) or (..., T, N)
    U: (N, r) orthonormal basis for target subspace
    returns f: (..., T)
    """
    # component in subspace
    X_in = (X @ U) @ U.T
    num = np.sum(X_in**2, axis=-1)
    den = np.sum(X**2, axis=-1) + eps
    return num / den


def project_and_backproject_orthonormal(X: np.ndarray, U: np.ndarray):
    """
    X: (..., T, N) or (T, N)
    U: (N, r) with orthonormal columns

    returns:
      Z: (..., T, r) motor coords
      X_motor: (..., T, N) motor component in full space
    """
    # forward: z = x U
    Z = X @ U  # (.., T, r)

    # back: x_motor = z U^T
    X_hat = Z @ U.T  # (.., T, N)

    return Z, X_hat


def compute_rrr_between_areas_td(td, origin_signal, target_signal, rank, scorer):
    # Concatenate trials
    X = np.concatenate(td[origin_signal].values)
    Y = np.concatenate(td[target_signal].values)[:, :rank]

    # Build mask: keep rows with no NaNs in either X or Y
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)

    # Apply mask
    X_clean = X[mask]
    Y_clean = Y[mask]

    r2 = cross_val_score(
        decode.ReducedRankRegressorBence(rank=rank, reg=100),
        X_clean,
        Y_clean,
        cv=5,
        scoring=scorer,
    )
    return r2


def publicise_signal_td(
    td_,
    origin_signal,
    target_signal,
    rank,
    window_emb=(0, 500),
    scorer=make_scorer(r2_score, multioutput="variance_weighted"),
    target_lag=None,
    return_emb=False,
):
    td = td_.copy()
    td[f"{origin_signal}_potent_{target_signal}"] = td[origin_signal]

    #### Add lag before embedding to avoid wraps
    if target_lag is not None:
        arr_original = td[target_signal]
        arrs = dt.shift_time_no_wrap(np.stack(td[target_signal].values), shift=target_lag)
        td[target_signal] = [arr for arr in arrs]

    ##### compute private subspace ######
    emb = subspaces.compute_embedding(
        td,
        f"{origin_signal}_potent_{target_signal}",
        target_signal,
        (window_emb[0], window_emb[-1]),
        subspaces.ReducedRankCommSubspace(rank=rank),
        null=False,
    )
    td = subspaces.project_signal(
        td,
        emb,
        f"{origin_signal}_potent_{target_signal}",
        f"{origin_signal}_potent_{target_signal}",
    )

    r2 = compute_rrr_between_areas_td(td, origin_signal, target_signal, rank, scorer)
    dim_potent = np.stack(td[f"{origin_signal}_potent_{target_signal}"].values).shape[-1]
    var = variance_in_subspace(
        np.concatenate(td[f"{origin_signal}_potent_{target_signal}"].values),
        np.eye(dim_potent),
    )
    if target_lag is not None:
        td[target_signal] = arr_original
    return (td, var) if not return_emb else (td, var, emb)


def privatise_signal_td(
    td_,
    n_iter,
    origin_signal,
    target_signal,
    rank,
    window_emb=(0, 500),
    scorer=make_scorer(r2_score, multioutput="variance_weighted"),
    diagnostics=False,
    target_lag=None,
):
    td = td_.copy()
    td[f"{origin_signal}_null_{target_signal}"] = td[origin_signal]

    if target_lag is not None:
        arr_original = td[target_signal]

        arrs = dt.shift_time_no_wrap(np.stack(td[target_signal].values), shift=target_lag)
        td[target_signal] = [arr for arr in arrs]

    if diagnostics:
        r2s, vars = [], []
        dim_origin_signal = np.stack(td[origin_signal].values).shape[-1]
        r2 = compute_rrr_between_areas_td(td, origin_signal, target_signal, rank, scorer)
        var = variance_in_subspace(
            np.concatenate(td[origin_signal].values), np.eye(dim_origin_signal)
        )
        r2s.append(r2)
        vars.append(var)

    ##### compute private subspace ######
    for i in range(n_iter):
        emb = subspaces.compute_embedding(
            td,
            f"{origin_signal}_null_{target_signal}",
            target_signal,
            (window_emb[0], window_emb[-1]),
            subspaces.ReducedRankCommSubspace(rank=rank),
            null=True,
        )
        td = subspaces.project_signal(
            td,
            emb,
            f"{origin_signal}_null_{target_signal}",
            f"{origin_signal}_null_{target_signal}",
        )

        if diagnostics:
            dim_null = np.stack(td[f"{origin_signal}_null_{target_signal}"].values).shape[-1]
            r2 = compute_rrr_between_areas_td(
                td, f"{origin_signal}_null_{target_signal}", target_signal, rank, scorer
            )
            var = variance_in_subspace(
                np.concatenate(td[f"{origin_signal}_null_{target_signal}"].values),
                np.eye(dim_null),
            )
            r2s.append(r2)
            vars.append(var)

    if diagnostics:
        r2s, vars = np.array(r2s), np.array(vars)
        fig, ax = plt.subplots(1, 2)
        plt.suptitle(f"Removing {origin_signal} activity predictive of {target_signal}")
        ax[0].errorbar(range(n_iter + 1), r2s.mean(-1), yerr=r2s.std(-1), fmt="o-")
        ax[0].set_ylim(-0.1, 0.4)
        ax[0].axhline(0, color="r", linestyle="dashed")
        ax[0].set_xlabel("Iterations")
        ax[0].set_ylabel("R2")
        ax[1].errorbar(range(n_iter + 1), np.array(vars), fmt="o-")
        ax[1].set_xlabel("Iterations")
        ax[1].set_ylabel("Variance")
        plt.tight_layout()

    if target_lag is not None:
        td[target_signal] = arr_original
    return td if not diagnostics else (td, vars[-1])


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
    """
    Returns variance across trials and summed across components
    """
    return np.sum(np.var(np.stack([arr @ W for arr in arrs], axis=-1), axis=-1), axis=1)


def variance_across_arrays(arrs):
    """
    Returns variance across trials and summed across components
    """
    return np.sum(np.var(arrs, axis=0), axis=1)


def mean_across_arrays(arrs):
    """
    Returns variance across trials and summed across components
    """
    return np.sum(np.mean(arrs, axis=0), axis=1)


def variance_in_subspace_df_arr(df, signal, W):
    return variance_across_arrays_in_subspace(df[signal].values, W)


def variance_in_subspace_df(df, signal, W):
    return variance_in_subspace(np.concatenate(df[signal].values), W)


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


def compute_embedding_on_arr(arrx, arry, model):
    model.fit(arrx, arry)
    W = scipy.linalg.orth(model.coef_.T)
    return W


def compute_embedding(td, signal_x, signal_y, time_bin_window, model, k=None, null=False):

    X = np.stack(td[signal_x].values)
    _, _, n_feat_in = X.shape

    # Slice time window and reshape
    X = X[:, time_bin_window[0] : time_bin_window[1], :]
    X = X.reshape(-1, n_feat_in)

    if model == "pca":
        pca_model = dim.compute_pca(X, n_components=k)
        W = pca_model.components_.T
    else:
        if isinstance(signal_y, list):
            ys = []
            for col in signal_y:
                a = np.stack(td[col].values)  # (trials, T) or (trials, T, F)
                if a.ndim == 2:
                    a = a[..., None]
                ys.append(a)
            y = np.concatenate(ys, axis=-1)
        else:
            y = np.stack(td[signal_y].values)
        _, _, n_feat_out = y.shape
        y = y[:, time_bin_window[0] : time_bin_window[1], :]
        y = y.reshape(-1, n_feat_out)

        model.fit(X, y)
        if not null:
            W = scipy.linalg.orth(model.coef_.T)
        else:
            W = scipy.linalg.null_space(model.coef_.T)

    return W


def compute_overlap_between_subspaces(emb_a, emb_b):

    angles = scipy.linalg.subspace_angles(emb_a, emb_b)

    # Same dimension
    if emb_a.shape[-1] == emb_b.shape[-1]:
        overlap = np.mean(np.cos(angles) ** 2)

    # Different dimension
    else:
        min_d = np.min([emb_a.shape[-1], emb_b.shape[-1]])
        overlap = (1 / min_d) * np.sum(np.cos(angles) ** 2)

    return overlap
