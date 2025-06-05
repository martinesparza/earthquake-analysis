"""
Module for PCA-related utils
"""

from sklearn.decomposition import PCA


def compute_pca(arr, n_components=None):
    """Computes pca on array

    Args:
        arr (np.ndarray): Observations x features array
        n_components (int, optional): Number of component to use. Defaults to None.

    Returns:
        sklearn.PCA: PCA model
    """
    if n_components is None:
        n_components = arr.shape[1]
    model = PCA(n_components=n_components, svd_solver="full")
    model.fit(arr)
    return model


def get_explained_variance_ratio(arr, n_components=None):
    """Returns explained variance ratios

    Args:
        arr (np.ndarra): Observations x features array
        n_components (int, optional): Number of components to use. Defaults to None.

    Returns:
        list: explained variance ratio of each component
    """

    model = compute_pca(arr, n_components=n_components)
    return model.explained_variance_ratio_
