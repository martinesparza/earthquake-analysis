import numpy as np
import scipy

from .utils import variance_in_subspace_df


def compute_embedding_on_arrs(arr_a, arr_b, model, null=False):

    model.fit(arr_a, arr_b)
    if null:
        W = scipy.linalg.null_space(model.coef_.T)
    else:
        W = scipy.linalg.orth(model.coef_.T)

    return W


def compute_embedding_on_trials(td_arr_a, td_arr_b, model, null=False):
    """Here im assuming td_arr_a has shape n_trials, n_time, n_features"""
    arr_a = td_arr_a.reshape(-1, td_arr_a.shape[-1])
    arr_b = td_arr_b.reshape(-1, td_arr_b.shape[-1])
    return compute_embedding_on_arrs(arr_a, arr_b, model, null)


def compute_embedding_on_td(td, signal_x, signal_y, model, window=(200, 450), null=False):
    """Here td_arr_a has shape (n_trials, n_time, n_features"""
    td_arr_a = np.stack(td[signal_x].values)[:, window[0] : window[1], :]
    td_arr_b = np.stack(td[signal_y].values)[:, window[0] : window[1], :]
    emb = compute_embedding_on_trials(td_arr_a, td_arr_b, model, null)
    return emb, variance_in_subspace_df(td, signal_x, emb)


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

        CXX_inv = np.linalg.pinv((X.T @ X) + self.reg * scipy.sparse.eye(X.shape[1]))
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
