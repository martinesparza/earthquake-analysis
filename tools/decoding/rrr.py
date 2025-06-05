"""
Module for Reduced Rank Regression utilities
"""

import warnings
from typing import Tuple

import cupy as cp
import numpy as np
import pyaldata as pyal
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import tools.decoding.decodeTools as decutils
from tools.dimensionality.cca import _roll_array
from tools.params import Params


def _get_data_for_rrr(df, area, condition, n_components=20, epoch=None, free_period=0):
    """Returns

    Args:
        df (_type_): _description_
        area (_type_): _description_
        condition (_type_): _description_
        n_components (int, optional): _description_. Defaults to 20.
        epoch (_type_, optional): _description_. Defaults to None.
        free_period (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    df_rrr = pyal.select_trials(df, df.trial_name == condition)

    # Get data
    if condition == "free":
        rates = df_rrr[f"{area}_rates"].values[free_period]
    elif condition == "trial":
        df_rrr = pyal.restrict_to_interval(df_rrr, epoch_fun=epoch)
        rates = np.concatenate(df_rrr[f"{area}_rates"].values, axis=0)

    else:
        rates = np.concatenate(df_rrr[f"{area}_rates"].values, axis=0)

    model = PCA(n_components=n_components, svd_solver="full")
    model.fit(rates)
    X = model.fit_transform(rates)
    return X, model.explained_variance_ratio_


def compute_rrr_on_df(
    df,
    areas,
    condition,
    k_folds=5,
    n_components=20,
    timepoints=None,
    verbose=True,
    free_period=0,
):

    results_rrr_ = {}
    kf = KFold(n_splits=k_folds, shuffle=False)

    for area_x in areas:
        results_rrr_[area_x] = {}

        X = _get_data_for_rrr(
            df,
            area_x,
            condition=condition,
            n_components=n_components,
            free_period=free_period,
        )
        rnd_timepoints = np.random.choice(X.shape[0], size=timepoints, replace=False)

        for area_y in areas:
            Y = _get_data_for_rrr(
                df,
                area_y,
                condition=condition,
                n_components=n_components,
                free_period=free_period,
            )

            # Get data
            r2 = []

            if timepoints is not None:
                X_subsampled = X[rnd_timepoints, :]
                Y = Y[rnd_timepoints, :]
            else:
                X_subsampled = X

            for train_index, test_index in kf.split(X_subsampled):

                X_train, X_test = X_subsampled[train_index], X_subsampled[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

                # Fit model
                model = ReducedRankRegression(r=10, lam=0.05, use_sklearn=False)
                model.fit(X=X_train, Y=Y_train)

                # Predict
                Y_pred_test = model.predict(X_test)
                multi_r2, col_r2 = decutils.multivariate_r2(Y_test, Y_pred_test)
                # print(f"{area_x} to {area_y}: {multi_r2:.3f}")
                r2.append(multi_r2)
            if verbose:
                print(f"{area_x} to {area_y}: {np.array(r2).mean():.3f}")

            results_rrr_[area_x][area_y] = np.array(r2)

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    # You can clear the memory pool by calling `free_all_blocks`.
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    if verbose:
        print("Freed memory pool")
    return results_rrr_


def delayed_rrr_on_df(
    df,
    areas,
    condition,
    shifts,
    k_folds=5,
    n_components=20,
    free_period=0,
    rank=10,
    lambda_=0.05,
    verbose=False,
    bin_size=Params.BIN_SIZE,
    epoch=Params.perturb_epoch_long,
):
    results_rrr = {}

    kf = KFold(n_splits=k_folds, shuffle=False)

    for area_x in areas:

        results_rrr[area_x] = {}

        # Get data
        X, _ = _get_data_for_rrr(
            df,
            area_x,
            condition=condition,
            n_components=n_components,
            free_period=free_period,
            epoch=epoch,
        )

        for area_y in areas:

            results_rrr[area_x][area_y] = {}

            # Get data
            Y, explained_variance_ratios = _get_data_for_rrr(
                df,
                area_y,
                condition=condition,
                n_components=n_components,
                free_period=free_period,
                epoch=epoch,
            )

            results_rrr[area_x][area_y]["vae_r2"] = {}
            results_rrr[area_x][area_y]["pc1_r2"] = {}

            for shift in shifts:

                # Introduce delay
                Y_delayed = _roll_array(Y, shift=int(shift / (bin_size * 1000)))

                # Initialize R2
                weighted_r2s = []
                pc1_r2s = []

                for train_index, test_index in kf.split(X):

                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y_delayed[train_index], Y_delayed[test_index]

                    # Fit model
                    model = ReducedRankRegression(r=rank, lam=lambda_, use_sklearn=False)
                    model.fit(X=X_train, Y=Y_train)

                    # Predict
                    Y_pred_test = model.predict(X_test)
                    vae_weighted_r2 = decutils.variance_explained_weighted_r2(
                        Y_test, Y_pred_test, explained_variance_ratios
                    )
                    col_r2 = decutils.columnwise_r2(Y_test, Y_pred_test)

                    weighted_r2s.append(vae_weighted_r2)
                    pc1_r2s.append(col_r2[0])

                if verbose:
                    print(f"{area_x} to {area_y}: {np.array(r2).mean():.3f}")

                results_rrr[area_x][area_y]["pc1_r2"][shift] = pc1_r2s
                results_rrr[area_x][area_y]["vae_r2"][shift] = weighted_r2s
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    # You can clear the memory pool by calling `free_all_blocks`.
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    if verbose:
        print("Freed memory pool")

    return results_rrr


class ReducedRankRegression:

    def __init__(self, r: int, lam: float, use_sklearn=True, verbose=False):
        if lam <= 0:
            raise ValueError("Regularisation parameter must be positive.")
        self.lam = lam
        self.rank = r
        self.use_sklearn = use_sklearn
        self.coef_: np.ndarray | None = None
        self.verbose = verbose
        return

    def center_XY_train(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.X_train_mean = np.mean(X, axis=0, keepdims=True)
        self.Y_train_mean = np.mean(Y, axis=0, keepdims=True)

        X_centered = X - self.X_train_mean
        Y_centered = Y - self.Y_train_mean

        return X_centered, Y_centered

    def fit(self, X: np.ndarray, Y: np.ndarray, fit_intercept=True):
        """Fits Brrr to X and Y train. Assumes X and Y are centered. See Mukherjee and Zhu, 2011

        Args:
            X (np.ndarray): Predictors. N (samples) x P (features)
            Y (np.ndarray): Responses. N (samples) x Q (features)
        """

        # Enforce float dtype
        if not np.issubdtype(X.dtype, np.floating):
            warnings.warn("Matrix X is not of type float. Changing to float")
            X = X.astype(float)
        if not np.issubdtype(Y.dtype, np.floating):
            warnings.warn("Matrix Y is not of type float. Changing to float")
            Y = Y.astype(float)

        X = cp.asarray(X)
        Y = cp.asarray(Y)

        if self.verbose:
            print(f"Fitting Reduced Rank Regression to data X: {X.shape} and Y: {Y.shape}")

        if fit_intercept:
            # Assuming X and Y are not centered
            X, Y = self.center_XY_train(X, Y)

        lam_mat = self.lam * cp.eye(X.shape[1])
        lam_mat_sqrt = cp.sqrt(self.lam) * cp.eye(X.shape[1])

        if self.use_sklearn:
            # Use the sklearn solver instead of the closed-form solution
            ridge = sklearn.linear_model.Ridge(alpha=self.lam, fit_intercept=False)
            b_ridge = ridge.fit(X, Y).coef_.T

        else:
            b_ridge = cp.linalg.pinv(X.T @ X + lam_mat) @ X.T @ Y

        # X_star = cp.vstack((X, lam_mat_sqrt))
        Y_star = cp.vstack((X, lam_mat_sqrt)) @ b_ridge
        _, _, Vt = cp.linalg.svd(Y_star @ b_ridge, full_matrices=False)

        self.coef_ = b_ridge @ Vt.T[:, : self.rank] @ Vt[: self.rank, :]

        r2, _ = decutils.multivariate_r2(Y, X @ self.coef_)
        if self.verbose:
            print("Proportion of variance explained training:", r2)
        return

    def predict(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet. ")
        if not np.issubdtype(X.dtype, np.floating):
            warnings.warn("Matrix X is not of type float. Changing to float")

        X = cp.asarray(X)

        return ((X - self.X_train_mean) @ self.coef_ + self.Y_train_mean).get()

    def get_weights(self) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.coef_
