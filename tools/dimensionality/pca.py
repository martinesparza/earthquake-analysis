"""
Module for PCA-related utils
"""

from sklearn.decomposition import PCA


def compute_pca(arr, n_components=None):
    """Fit a full-SVD sklearn PCA to an observations x features array.

    Parameters
    ----------
    arr : np.ndarray
        Data, shape (n_observations, n_features).
    n_components : int, optional
        Components to keep (default is all features).

    Returns
    -------
    sklearn.decomposition.PCA
        Fitted model.
    """
    if n_components is None:
        n_components = arr.shape[1]
    model = PCA(n_components=n_components, svd_solver="full")
    model.fit(arr)
    return model


def get_explained_variance_ratio(arr, n_components=None):
    """Return the explained variance ratio of each PCA component.

    Parameters
    ----------
    arr : np.ndarray
        Data, shape (n_observations, n_features).
    n_components : int, optional
        Components to keep (default is all features).

    Returns
    -------
    np.ndarray
        Explained variance ratio per component.
    """

    model = compute_pca(arr, n_components=n_components)
    return model.explained_variance_ratio_
