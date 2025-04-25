"""
Module for Reduced Rank Regression utilities
"""

import warnings
from typing import Tuple

import cupy as cp
import numpy as np
import sklearn

import tools.decoding.decodeTools as decutils


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
