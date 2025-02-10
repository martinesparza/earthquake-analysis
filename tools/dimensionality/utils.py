import numpy as np
from sklearn.decomposition import PCA


def participation_ratio(explained_variances):
    """
    Estimate the number of "important" components based on explained variances

    Parameters
    ----------
    explained_variances : 1D np.ndarray
        explained variance per dimension

    Returns
    -------
    dimensionality estimated using participation ratio formula
    """
    return np.sum(explained_variances) ** 2 / np.sum(explained_variances**2)


def pca_pr(arr):
    """
    Estimate the data's dimensionality using PCA and participation ratio

    Parameters
    ----------
    arr : 2D array
        n_samples x n_features data

    Returns
    -------
    estimated dimensionality
    """

    model = PCA(n_components=arr.shape[-1], svd_solver="full")
    pca = model.fit(arr)
    return participation_ratio(pca.explained_variance_)
