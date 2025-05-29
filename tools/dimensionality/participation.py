import random

import numpy as np
import pandas as pd
import pyaldata as pyal
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import KernelCenterer
from sklearn.utils.graph import _fix_connected_components
from tqdm import tqdm


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


def pca_pr(arr, n_components=None):
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
    if n_components is None:
        n_components = arr.shape[-1]
    model = PCA(n_components=n_components, svd_solver="full")
    pca = model.fit(arr)
    return participation_ratio(pca.explained_variance_)


def geodesic_dist_matrix(X, n_neighbors=15, n_jobs=-1):
    """
    Estimate a geodesic distance matrix using nearest neighbors

    Parameters
    ----------
    X : 2D array
        n_samples x n_features data
    n_neighbors : int, default 15
        number of nearest neighbors
    n_jobs : int, default -1
        number of cores to use for nearest neighbors calculation
        -1 is use all available

    Returns
    -------
    estimated dimensionality
    """
    # borrowed from sklearn.manifold.Isomap to skip the kernel PCA embedding part because we only need the distance matrix
    nbrs_ = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="auto",
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=n_jobs,
    )
    nbrs_.fit(X)

    nbg = kneighbors_graph(
        nbrs_,
        n_neighbors,
        metric="minkowski",
        p=2,
        metric_params=None,
        mode="distance",
        n_jobs=n_jobs,
    )
    n_connected_components, labels = connected_components(nbg)
    if n_connected_components > 1:
        nbg = _fix_connected_components(
            nbrs_._fit_X,
            graph=nbg,
            n_connected_components=n_connected_components,
            component_labels=labels,
            mode="distance",
            metric=nbrs_.effective_metric_,
            **nbrs_.effective_metric_params_,
        )

    dist_matrix_ = shortest_path(nbg, method="auto", directed=False)

    return dist_matrix_


def geodesic_to_gram_matrix(D):
    """
    Transform the geodesic distance matrix to a Gram matrix used in kernel PCA

    Parameters
    ----------
    D : 2D np.array
        geodesic distance matrix

    Returns
    -------
    K : 2D np.array
    """
    G = D**2
    G *= -0.5
    K = KernelCenterer().fit_transform(G)

    return K


def isomap_pr(X, n_neighbors=15, n_jobs=-1):
    """
    Estimate the data's dimensionality using participation ratio. We compute eigenvalues
    using kPCA on the geodesic distance matrix from the Isomap algo. The kernel used is
    meant to recover inner products from the distances matrix by double centering.

    Parameters
    ----------
    X : 2D array
        n_samples x n_features data
    n_neighbors : int, default 15
        number of nearest neighbors to estimate the geodesic distances
    n_jobs : int, default -1
        number of cores to use for nearest neighbors calculation
        -1 is use all available

    Returns
    -------
    estimated dimensionality
    """
    G = geodesic_dist_matrix(X, n_neighbors=n_neighbors, n_jobs=n_jobs)

    K = geodesic_to_gram_matrix(G)

    evals = np.real(np.linalg.eigvals(K))

    return (np.sum(evals) ** 2) / (np.sum(evals**2))


def pcs_percent_vaf(arr, n_components=None, percent_vaf=80):
    if n_components is None:
        n_components = arr.shape[-1]

    model = PCA(n_components=n_components, svd_solver="full")
    model = model.fit(arr)
    return norm_pc_count


def get_pr_for_subsets_of_neurons(arr, niter=5, linear=True, verbose=False):
    results = []
    for num_neurons in np.arange(5, arr.shape[1] + 1, 10):
        if verbose:
            print(f"Neurons: {num_neurons}")
        prs = []
        for _ in range(niter):
            random_neurons = np.random.randint(0, arr.shape[1], size=num_neurons)
            if linear:
                pr = pca_pr(arr[:, random_neurons])
            else:
                pr = isomap_pr(arr[:, random_neurons])
            prs.append(pr)

        results.append(prs)
    return np.vstack(results)


def get_pcs_percentVAF_subsets_neurons(arr, niter=5, linear=True, verbose=False, vaf=80):
    results = []
    for num_neurons in np.arange(5, arr.shape[1] + 1, 10):
        if verbose:
            print(f"Neurons: {num_neurons}")
        prs = []
        for _ in range(niter):
            random_neurons = np.random.randint(0, arr.shape[1], size=num_neurons)
            norm_pc_count = pcs_percent_vaf(arr[:, random_neurons], vaf=vaf)
            prs.append(pr)

        results.append(prs)
    return np.vstack(results)
