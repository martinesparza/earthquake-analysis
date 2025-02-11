import warnings

import numpy as np
import pandas as pd
import pyaldata as pyal
from scipy.linalg import inv, qr, svd

from tools.dataTools import get_data_array
from tools.params import Params


def canoncorr(X: np.array, Y: np.array, fullReturn: bool = False) -> np.array:
    """
    Canonical Correlation Analysis (CCA)
    line-by-line port from Matlab implementation of `canoncorr`
    X,Y: (samples/observations) x (features) matrix, for both: X.shape[0] >> X.shape[1]
    fullReturn: whether all outputs should be returned or just `r` be returned (not in Matlab)

    returns: A,B,r,U,V
    A,B: Canonical coefficients for X and Y
    U,V: Canonical scores for the variables X and Y
    r:   Canonical correlations

    Signature:
    A,B,r,U,V = canoncorr(X, Y)
    """
    n, p1 = X.shape
    p2 = Y.shape[1]
    if p1 >= n or p2 >= n:
        warnings.warn("Not enough samples, might cause problems")

    # Center the variables
    X = X - np.mean(X, 0)
    Y = Y - np.mean(Y, 0)

    # Factor the inputs, and find a full rank set of columns if necessary
    Q1, T11, perm1 = qr(X, mode="economic", pivoting=True, check_finite=True)

    rankX = sum(
        np.abs(np.diagonal(T11)) > np.finfo(type((np.abs(T11[0, 0])))).eps * max([n, p1])
    )

    if rankX == 0:
        warnings.warn(f"stats:canoncorr:BadData = X")
    elif rankX < p1:
        warnings.warn("stats:canoncorr:NotFullRank = X")
        Q1 = Q1[:, :rankX]
        T11 = T11[rankX, :rankX]

    Q2, T22, perm2 = qr(Y, mode="economic", pivoting=True, check_finite=True)
    rankY = sum(
        np.abs(np.diagonal(T22)) > np.finfo(type((np.abs(T22[0, 0])))).eps * max([n, p2])
    )

    if rankY == 0:
        warnings.warn(f"stats:canoncorr:BadData = Y")
    elif rankY < p2:
        warnings.warn("stats:canoncorr:NotFullRank = Y")
        Q2 = Q2[:, :rankY]
        T22 = T22[:rankY, :rankY]

    # Compute canonical coefficients and canonical correlations.  For rankX >
    # rankY, the economy-size version ignores the extra columns in L and rows
    # in D. For rankX < rankY, need to ignore extra columns in M and D
    # explicitly. Normalize A and B to give U and V unit variance.
    d = min(rankX, rankY)
    L, D, M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver="gesdd")
    M = M.T

    A = inv(T11) @ L[:, :d] * np.sqrt(n - 1)
    B = inv(T22) @ M[:, :d] * np.sqrt(n - 1)
    r = D[:d]
    # remove roundoff errs
    r[r >= 1] = 1
    r[r <= 0] = 0

    if not fullReturn:
        return r

    # Put coefficients back to their full size and their correct order
    A[perm1, :] = np.vstack((A, np.zeros((p1 - rankX, d))))
    B[perm2, :] = np.vstack((B, np.zeros((p2 - rankY, d))))

    # Compute the canonical variates
    U = X @ A
    V = Y @ B

    return A, B, r, U, V


def get_ccs_between_two_areas(df, area1, area2, n_components=10):

    AllData_area1 = get_data_array(df, area=area1, model="pca")
    AllData_area2 = get_data_array(df, area=area2, model="pca")

    _, _, min_trials, min_time, _ = np.min(
        (AllData_area1.shape, AllData_area2.shape), axis=0
    )
    data1 = np.reshape(AllData_area1[:, :min_trials, :min_time, :], (-1, n_components))
    data2 = np.reshape(AllData_area2[:, :min_trials, :min_time, :], (-1, n_components))

    ccs = canoncorr(data1, data2)

    return ccs


# Function to roll the arrays
def _roll_array(arr, shift):
    return np.roll(arr, shift=shift, axis=0)  # Rolling along the time axis


def compute_shifted_cca_between_areas(
    df: pd.DataFrame,
    area_to_shift: str,
    area_to_compare: str,
    shift_start=-0.5,
    shift_end=0.5,
    shift_step=0.05,
    n_components=10,
):
    """Computes CCAs between areas by applying time shifts.

    Assumes you are providing a dataframe with trials only

    Parameters
    ----------
    df : pd.DataFrame
        trialdata with trials only
    area_to_shift : str
        area to shift
    area_to_compare : _type_
        area to compare
    shift_start : float, optional
        values from which to start in seconds, by default 0.5
    shift_end : float, optional
        values from which to end in seconds, by default 0.5
    shift_step : float, optional
        values for stepping t in seconds, by default 0.1
    n_components : int, optional
        components used for CCA, by default 10
    """

    if shift_step / Params.BIN_SIZE < 1:
        raise ValueError(f"Stepping time is lower than bin size")

    shifts = np.arange(
        start=int(shift_start / Params.BIN_SIZE),
        stop=int(shift_end / Params.BIN_SIZE),
        step=int(shift_step / Params.BIN_SIZE),
    )  # Example shift value

    df_tmp = df.copy()
    ccs_time_shifted = []

    for shift in shifts:
        df_tmp = df.copy()
        df_tmp[f"{area_to_shift}_rates"] = df_tmp[f"{area_to_shift}_rates"].apply(
            lambda arr: _roll_array(arr, shift)
        )
        df_tmp = pyal.restrict_to_interval(df_tmp, epoch_fun=Params.perturb_epoch)
        ccs_time_shifted.append(
            np.mean(
                get_ccs_between_two_areas(
                    df_tmp,
                    area1=area_to_shift,
                    area2=area_to_compare,
                    n_components=n_components,
                )[1:4]
            )
        )
    return ccs_time_shifted, shifts * Params.BIN_SIZE
