import warnings

import numpy as np
import pyaldata
import scipy.stats
# from beneuro.reduced_rank import ReducedRankRegressor
from scipy.linalg import null_space, orth
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from tqdm.auto import tqdm

from tools.decoding import utils


def ridge_alpha_range(X, dmax_scales, z_score=False):
    """
    Determine the appropriate range for the ridge regularization parameter alpha based on the source data X


    Parameters
    ----------
    X : xr.DataArray
        source activities in ridge regression
    dmax_scales : array of floats
        list of scaling factors for the largest eigenvalue
        e.g. np.linspace(0.5, 1, n_alphas)
    z_score : bool, default False
        divide by the standard deviation of the activities or not

    Returns
    -------
    alpha_range : np.ndarray
        list of alpha values to try when doing cross-validation
    """
    dmax_scales = np.array(dmax_scales)

    if z_score:
        Z = pyaldata.z_score(X)
    else:
        Z = pyaldata.center(X)

    dmax = np.max(np.linalg.eigvals(Z.T @ Z))
    alpha_range = dmax * (1 - dmax_scales) / dmax_scales

    return alpha_range


def semedo_opt_alpha(alpha_range, cv_scores):
    """
    Select the optimal alpha parameter for ridge regression based on
    cross-validated test scores.
    Select the simplest model with test performance within 1 SEM of the highest test performance.

    Parameters
    ----------
    alpha_range : array-like
        candidate alpha values
    cv_scores : array-like
        cross-validated test scores corresponding
        to the values in alpha_range

    Returns
    -------
    optimal alpha : float
    """
    means = [np.mean(sc) for sc in cv_scores]

    max_score = np.max(means)
    max_score_sem = scipy.stats.sem(cv_scores[np.argmax(means)])

    # see which elements are within 1 SEM of the max score
    mask = means > (max_score - max_score_sem)

    # sort in ascending order and take the largest alpha ~ simplest model
    return np.max(alpha_range[mask])


def naive_opt_alpha(alpha_range, cv_scores):
    """
    Select the alpha value with the highest cross-validated test score

    Parameters
    ----------
    alpha_range : array-like
        candidate alpha values
    cv_scores : array-like
        cross-validated test scores corresponding
        to the values in alpha_range

    Returns
    -------
    optimal alpha : float
    """
    means = [np.mean(sc) for sc in cv_scores]

    return alpha_range[np.argmax(means)]


def fit_semedo_ridge(X, y, n_alpha=100, selection="semedo", cv=KFold(5, shuffle=True)):
    """
    Fit a ridge regression model by first finding the range of alpha values based on the input data,
    then select the best regressor using the method outlined in Semedo et al 2019

    Parameters
    ----------
    X : 2D np.array
        input data
        shape: n_samples x n_features
    y : np.array
        target values
    n_alpha : int, optional, default 100
        number of alpha values to try
    selection : str
        method to select the final alpha
        'semedo' or 'naive'
    cv : int or KFold
        cross-validation option to pass to cross_val_score

    Returns
    -------
    ridge : sklearn.linear_model.Ridge
        regressor fitted to X and y using the optimal alpha value
    """
    candidate_alphas = ridge_alpha_range(X, np.linspace(0.5, 1.0, n_alpha))
    candidate_alphas = np.real(candidate_alphas)
    cv_scores = [
        cross_val_score(Ridge(alpha), X, y, cv=cv, scoring=utils.default_scorer)
        for alpha in tqdm(candidate_alphas)
    ]
    if selection == "semedo":
        opt_alpha = np.real(semedo_opt_alpha(candidate_alphas, cv_scores))
    else:
        opt_alpha = np.real(naive_opt_alpha(candidate_alphas, cv_scores))

    return Ridge(opt_alpha).fit(X, y)


class SemedoRidge:
    def __init__(self, *args, n_alpha=100, selection="semedo", **kwargs):
        self.n_alpha = n_alpha
        self.selection = selection

    def fit(self, X, y):
        self.ridge = fit_semedo_ridge(X, y, self.n_alpha, self.selection)
        return self

    def predict(self, X):
        return self.ridge.predict(X)

    @property
    def coef_(self):
        return self.ridge.coef_


# class SemedoRRR:
#     # TODO have fit_alpha and fit_reg options to choose which parameter to fit
#     def __init__(
#         self,
#         rank,
#         *args,
#         n_alpha=100,
#         selection="semedo",
#         cv=KFold(10, shuffle=True),
#         **kwargs,
#     ):
#         self.rank = rank
#         self.n_alpha = n_alpha
#         self.cv = cv

#         if selection == "semedo":
#             self.selection_fn = semedo_opt_alpha
#         elif selection == "naive":
#             self.selection_fn = naive_opt_alpha
#         else:
#             raise ValueError(
#                 f"selection has to be one of ['semedo', 'naive'] but got {selection}"
#             )
#         self.selection = selection

#     def fit(self, X, y):
#         # self.ridge = fit_semedo_ridge(X, y, self.n_alpha, self.selection)
#         # return self
#         self.candidate_alphas = np.real(
#             ridge_alpha_range(X, np.linspace(0.5, 1.0, self.n_alpha))
#         )
#         self.cv_scores = [
#             cross_val_score(
#                 ReducedRankRegressor(self.rank, alpha),
#                 X,
#                 y,
#                 cv=self.cv,
#                 scoring=pysubspaces.utils.default_scorer,
#             )
#             for alpha in tqdm(self.candidate_alphas)
#         ]

#         self.opt_alpha = np.real(self.selection_fn(self.candidate_alphas, self.cv_scores))

#         self.rrr = ReducedRankRegressor(self.rank, reg=self.opt_alpha).fit(X, y)

#         return self

#     def predict(self, X):
#         return self.rrr.predict(X)

#     @property
#     def coef_(self):
#         return self.rrr.coef_


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


def get_output_potent_projector(regressor, var_X=None):
    """
    Matrix that projects input data into the regressor's output-potent subspace

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
        return rotate_by_var(W, var_X)


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
        return rotate_by_var(W, var_X)


def rotate_by_var(W, X):
    """
    Rotate the subspace axes according to explained variance in a signal

    Parameters
    ----------
    W : np.array
        orthonormal projection matrix
        shape: N x D
    X : np.array
        signal on which to calculate the
        explained variance per dimension
        shape: T x N

    Returns
    -------
    W_rot : np.array
        principal components in the subspace
    """
    # raise NotImplementedError
    return PCA().fit(pyaldata.center(X @ W)).transform(W)


@pyaldata.copy_td
def project_signal(trial_data, W, signal, out_fieldname):
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
    X = pyaldata.concat_trials(trial_data, signal)
    if not np.all(np.isclose(X.mean(axis=0), np.zeros(X.shape[1]))):
        warnings.warn(
            "Signal does not appear to be centered. Centering signal for projection only."
        )

    trial_data[out_fieldname] = [
        s @ W for s in pyaldata.tools.center_signal(trial_data, signal)[signal]
    ]

    return trial_data


def output_potent_and_null_anal(
    trial_data, model, input_signal, output_signal, out_fieldnames=None
):
    # TODO come up with default field names
    # fit model to input_signal and output_signal
    # get its output null and potent dimensions
    # order by variance using the input signal
    # project to them and store in out_fieldnames
    raise NotImplementedError


# def get_opt_reduced_rank_regressor(
#     df, from_area, to_area, cv=KFold(5, shuffle=True), reg=None
# ):
#     X = pyaldata.concat_trials(df, from_area)
#     Y = pyaldata.concat_trials(df, to_area)

#     max_possible_rank = max(Y.shape[1], X.shape[1])
#     rank_list = np.arange(1, max_possible_rank)

#     cv_scores = []
#     for rank in tqdm(rank_list):
#         cv_scores.append(
#             cross_val_score(
#                 ReducedRankRegressor(rank, reg=reg),
#                 X,
#                 Y,
#                 cv=cv,
#                 scoring=pysubspaces.utils.default_scorer,
#             )
#         )
#     cv_scores = np.stack(cv_scores)

#     opt_rrr = ReducedRankRegressor(get_opt_rank(rank_list, cv_scores)).fit(X, Y)

#     return opt_rrr
