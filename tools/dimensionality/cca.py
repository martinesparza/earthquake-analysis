import warnings

import numpy as np
import pandas as pd
import pyaldata as pyal
from scipy.linalg import inv, qr, svd

from tools.params import Params


def canoncorr(X: np.array, Y: np.array, fullReturn: bool = False) -> np.array:
    """Run canonical correlation analysis (line-by-line port of MATLAB
    ``canoncorr``).

    Parameters
    ----------
    X : np.ndarray
        Observations x features, with many more rows than columns.
    Y : np.ndarray
        Observations x features, with many more rows than columns.
    fullReturn : bool, optional
        Return all outputs instead of only ``r`` (default is False).

    Returns
    -------
    r : np.ndarray
        Canonical correlations; the only output unless ``fullReturn``.
    A : np.ndarray
        Canonical coefficients for X; only if ``fullReturn``.
    B : np.ndarray
        Canonical coefficients for Y; only if ``fullReturn``.
    U : np.ndarray
        Canonical scores for X; only if ``fullReturn``.
    V : np.ndarray
        Canonical scores for Y; only if ``fullReturn``.
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


def get_ccs_between_two_areas_from_df(df, area1, area2):
    """Compute canonical correlations between two areas' PCA rates concatenated
    over rows.

    Parameters
    ----------
    df : pd.DataFrame
        Data with ``<area>_rates_pca`` columns.
    area1 : str
        First area name.
    area2 : str
        Second area name.

    Returns
    -------
    np.ndarray
        Canonical correlations from ``canoncorr``.
    """
    data1, data2 = df[f"{area1}_rates_pca"].values, df[f"{area2}_rates_pca"].values
    data1, data2 = np.concatenate(data1), np.concatenate(data2)
    ccs = canoncorr(data1, data2)

    return ccs


# Function to roll the arrays
def _roll_array(arr, shift, axis=0):
    """Circularly roll an array along an axis.

    Parameters
    ----------
    arr : np.ndarray
        Array to roll.
    shift : int
        Number of samples to roll.
    axis : int, optional
        Axis to roll along (default is 0).

    Returns
    -------
    np.ndarray
        Rolled array.
    """
    return np.roll(arr, shift=shift, axis=axis)  # Rolling along the time axis


def compute_shifted_cca_between_areas(
    df: pd.DataFrame,
    area_to_shift: str,
    area_to_compare: str,
    shift_start=-0.5,
    shift_end=0.5,
    shift_step=0.05,
    n_components=10,
):
    """Compute mean canonical correlations 2-4 per time shift (no-op: it rolls
    ``_rates`` but CCA reads ``_rates_pca``).

    Parameters
    ----------
    df : pd.DataFrame
        Trial-only data with ``_rates`` and ``_rates_pca`` columns.
    area_to_shift : str
        Area whose rates are rolled.
    area_to_compare : str
        Area it is compared against.
    shift_start : float, optional
        First shift in seconds (default is -0.5).
    shift_end : float, optional
        Last shift, exclusive, in seconds (default is 0.5).
    shift_step : float, optional
        Shift step in seconds, at least one bin (default is 0.05).
    n_components : int, optional
        Unused (default is 10).

    Returns
    -------
    ccs_time_shifted : list of float
        Mean of canonical correlations 2-4, one per shift.
    shifts_s : np.ndarray
        Shifts in seconds.

    Raises
    ------
    ValueError
        If ``shift_step`` is smaller than ``Params.BIN_SIZE``.
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
                get_ccs_between_two_areas_from_df(
                    df_tmp,
                    area1=area_to_shift,
                    area2=area_to_compare,
                    # n_components=n_components,
                )[1:4]
            )
        )
    return ccs_time_shifted, shifts * Params.BIN_SIZE
