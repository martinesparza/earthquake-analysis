import numpy as np
import scipy

from .utils import variance_in_subspace_df, variance_in_subspace
from .regression import cross_val_semedo_rrr_td


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


def compute_embedding_on_td(
    td,
    signal_x,
    signal_y,
    model,
    origin_rank=None,
    target_rank=30,
    window=(200, 450),
    null=False,
):
    if origin_rank is None:
        origin_rank = np.stack(td[signal_x].values).shape[-1]
    """Here td_arr_a has shape (n_trials, n_time, n_features"""
    td_arr_a = np.stack(td[signal_x].values)[:, window[0] : window[1], :origin_rank]
    td_arr_b = np.stack(td[signal_y].values)[:, window[0] : window[1], :target_rank]
    emb = compute_embedding_on_trials(td_arr_a, td_arr_b, model, null)
    # return emb, variance_in_subspace_df(td, signal_x, emb)
    var = variance_in_subspace(np.concatenate(td[signal_x].values)[:, :origin_rank], emb)
    return emb, var


def project_signal(td, W, signal, out_fieldname):
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
    trial_data = td.copy()
    trial_data[out_fieldname] = [s @ W for s in trial_data[signal].values]
    return trial_data


def project_signal_specific_rank(td, W, signal, out_fieldname, rank=None):
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
    trial_data = td.copy()
    if rank is None:
        rank = trial_data[signal].values[0].shape[-1]

    # print(rank)
    # print(W.shape)

    trial_data[out_fieldname] = [s[:, :rank] @ W for s in trial_data[signal].values]
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


def compute_potent_null_td(
    td_,
    signal_x,
    signal_y,
    origin_rank,
    target_rank,
    n_iter=2,
    window=(200, 450),
    fit_rank=True,
    rank=10,
):
    td = td_.copy()
    del td_

    print(f"Computing {signal_x} -> {signal_y} potent and null subspaces")

    # Compute initial reduced rank regression to optimise rank
    r2, opt_rank = cross_val_semedo_rrr_td(
        td=td,
        signal_x=signal_x,
        signal_y=signal_y,
        target_rank=target_rank,
        rank=rank,
        fit_rank=fit_rank,
    )
    comm_model = ReducedRankCommSubspace(rank=opt_rank)
    print(f"Prediction R2 = {r2.mean():.4f}. Optimal rank: {opt_rank}")

    # Compute embeddings
    emb_null, var_null_init = compute_embedding_on_td(
        td=td,
        signal_x=signal_x,
        signal_y=signal_y,
        model=comm_model,
        null=True,
        origin_rank=origin_rank,
        target_rank=target_rank,
        window=window,
    )
    emb_potent, var_potent = compute_embedding_on_td(
        td=td,
        signal_x=signal_x,
        signal_y=signal_y,
        model=comm_model,
        null=False,
        origin_rank=origin_rank,
        target_rank=target_rank,
        window=window,
    )

    # Project potent and null signals
    td = project_signal_specific_rank(
        td=td,
        W=emb_null,
        signal=signal_x,
        out_fieldname=f"{signal_x}_null_{signal_y}",
        rank=origin_rank,
    )  # potent
    td = project_signal_specific_rank(
        td=td,
        W=emb_potent,
        signal=signal_x,
        out_fieldname=f"{signal_x}_potent_{signal_y}",
        rank=origin_rank,
    )  # null

    ############# Loop to remove activity ####################
    print(f"Using {n_iter} iterations...")
    signal_null = f"{signal_x}_null_{signal_y}"
    for i in range(n_iter):
        r2, opt_rank = cross_val_semedo_rrr_td(
            td=td,
            signal_x=signal_null,
            signal_y=signal_y,
            target_rank=target_rank,
            rank=opt_rank,
            fit_rank=False,
        )
        print(f"\tIter: {i+1}. Prediction R2 = {r2.mean():.4f}")

        if (r2.mean() - r2.std()) < 0:
            pass
        else:
            print(f"Iteration number: {i+1}. R2 below 0, breaking")
            break

        emb_null, var_null = compute_embedding_on_td(
            td=td,
            signal_x=signal_null,
            signal_y=signal_y,
            model=comm_model,
            null=True,
            target_rank=target_rank,
            window=window,
        )
        td = project_signal_specific_rank(
            td, emb_null, signal=signal_null, out_fieldname=signal_null
        )

    r2, opt_rank = cross_val_semedo_rrr_td(
        td=td,
        signal_x=signal_null,
        signal_y=signal_y,
        target_rank=target_rank,
        rank=opt_rank,
        fit_rank=False,
    )
    print(f"\tFinal Prediction R2 = {r2.mean():.4f}")

    ############ Diagnostics ###################
    total_var = variance_in_subspace(
        np.concatenate(td[signal_x].values), np.eye(np.stack(td[signal_x].values).shape[-1])
    )
    total_var_origin_rank = variance_in_subspace(
        np.concatenate(td[signal_x].values)[:, :origin_rank], np.eye(origin_rank)
    )
    print("Diagnostics:")
    print(f"\tTotal {signal_x} var:         {total_var:.2f}")
    print(f"\tFraction Potent (dim: {opt_rank}) var:    {var_potent/total_var:.3f}")
    try:
        print(
            f"\tFraction Null (dim: {np.stack(td[signal_null].values).shape[-1]}) var:      {var_null/total_var:.3f}"
        )
    except:
        pass
    print(f"\tFraction Null init var:          {var_null_init/total_var:.3f}")
    print(f"\tFraction {signal_x} var:      {total_var_origin_rank/total_var:.3f}")

    return td
