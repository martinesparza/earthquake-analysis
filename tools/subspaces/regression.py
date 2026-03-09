import numpy as np
import scipy
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from tqdm import tqdm

import tools.decoding as decode
from .utils import default_scorer


def get_opt_rank(rank_list, cv_scores):
    """
    Calculate the optimal rank from cross-validation scores
    using the method in Semedo 2019

    "To find the optimal dimensionality for the RRR model (the value of m),
    we used 10-fold cross-validation and found the smallest number of dimensions
    for which predictive performance was within one SEM of the peak performance."
        - Semedo 2019

    Parameters
    ----------
    rank_list : list or 1D array
        list of ranks that produced the cv_scores
    cv_scores : n_rank x n_folds matrix or list of 1D arrays of length n_folds
        cross-validation scores for the different ranks

    Returns
    -------
    optimal rank
    """
    if isinstance(cv_scores, list):
        cv_scores = np.stack(cv_scores)

    assert cv_scores.ndim == 2
    assert np.shape(cv_scores)[0] == len(rank_list)

    cv_means = cv_scores.mean(axis=1)
    max_ind = np.argmax(cv_means)
    max_val = np.max(cv_means)
    max_sem = scipy.stats.sem(cv_scores[max_ind, :])

    # see which elements are within 1 SEM of the max score
    mask = cv_means > (max_val - max_sem)

    # sort in ascending order and take the largest alpha ~ simplest model
    return np.min(rank_list[mask])


class SemedoRRR(BaseEstimator):
    def __init__(
        self,
        alpha=1,
        fit_rank=False,
        cv=KFold(5, shuffle=False),
        rank=10,
    ):
        self.alpha = alpha
        self.fit_rank = fit_rank
        self.cv = cv
        self.rank = rank
        self.opt_ranks = []

    def fit(self, X, y):
        self.rrr = decode.ReducedRankRegressorBenceGPU(self.rank, reg=self.alpha).fit(X, y)
        return self

    def predict(self, X):
        return self.rrr.predict(X)

    @property
    def coef_(self):
        return self.rrr.coef_


def cross_val_semedo_rrr(
    X,
    y,
    alpha=0,
    cv=KFold(5, shuffle=False),
    fit_rank=False,
    ranks=np.arange(2, 30, step=2),
    default_rank=10,
):
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)

    # Apply mask
    X = X[mask]
    y = y[mask]
    if fit_rank:
        cv_scores = [
            cross_val_score(
                decode.ReducedRankRegressorBenceGPU(rank, alpha),
                X,
                y,
                cv=cv,
                scoring=default_scorer,
            )
            for rank in tqdm(ranks)
        ]

        opt_rank = get_opt_rank(ranks, cv_scores)
    else:
        opt_rank = default_rank

    r2 = cross_val_score(
        SemedoRRR(alpha=alpha, rank=opt_rank),
        X,
        y,
        cv=cv,
        scoring=default_scorer,
    )

    return r2, opt_rank


def cross_val_semedo_rrr_td(
    td,
    signal_x,
    signal_y,
    target_rank,
    alpha=0,
    cv=KFold(5, shuffle=False),
    fit_rank=False,
    rank=10,
):
    X = np.concatenate(td[signal_x].values)[:, :]
    y = np.concatenate(td[signal_y].values)[:, :target_rank]
    r2, opt_rank = cross_val_semedo_rrr(
        X, y, alpha, fit_rank=fit_rank, default_rank=rank, cv=cv
    )
    return r2, opt_rank
    # print(r2.mean(), opt_rank)
