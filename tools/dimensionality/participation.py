import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import KernelCenterer
from sklearn.utils.graph import _fix_connected_components
from tqdm import tqdm


def participation_ratio(explained_variances):
    """Estimate dimensionality as (sum of variances)^2 / sum of squared
    variances.

    Parameters
    ----------
    explained_variances : np.ndarray
        1D explained variance per dimension.

    Returns
    -------
    float
        Participation-ratio dimensionality.
    """
    return np.sum(explained_variances) ** 2 / np.sum(explained_variances**2)


def pca_pr(arr, n_components=None):
    """Estimate dimensionality as the participation ratio of PCA explained
    variances.

    Parameters
    ----------
    arr : np.ndarray
        Data, shape (n_samples, n_features).
    n_components : int, optional
        Components fitted (default is all features).

    Returns
    -------
    float
        Estimated dimensionality.
    """
    if n_components is None:
        n_components = arr.shape[-1]
    model = PCA(n_components=n_components, svd_solver="full")
    pca = model.fit(arr)
    return participation_ratio(pca.explained_variance_)


def geodesic_dist_matrix(X, n_neighbors=15, n_jobs=-1):
    """Estimate geodesic distances over a nearest-neighbour graph, joining
    disconnected components.

    Parameters
    ----------
    X : np.ndarray
        Data, shape (n_samples, n_features).
    n_neighbors : int, optional
        Neighbours per point (default is 15).
    n_jobs : int, optional
        Cores for neighbour search; -1 uses all (default is -1).

    Returns
    -------
    np.ndarray
        Geodesic distances, shape (n_samples, n_samples).
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
        # Convert to LIL before modification for efficiency
        nbg_lil = nbg.tolil()

        nbg = _fix_connected_components(
            nbrs_._fit_X,
            graph=nbg_lil,
            n_connected_components=n_connected_components,
            component_labels=labels,
            mode="distance",
            metric=nbrs_.effective_metric_,
            **nbrs_.effective_metric_params_,
        )

        # Convert fixed graph back to CSR
        nbg = nbg.tocsr()

    dist_matrix_ = shortest_path(nbg, method="auto", directed=False)

    return dist_matrix_


def geodesic_to_gram_matrix(D):
    """Convert geodesic distances to a double-centred Gram matrix (-D^2 / 2)
    for kernel PCA.

    Parameters
    ----------
    D : np.ndarray
        Geodesic distance matrix, shape (n, n).

    Returns
    -------
    np.ndarray
        Centred Gram matrix, shape (n, n).
    """
    G = D**2
    G *= -0.5
    K = KernelCenterer().fit_transform(G)

    return K


def isomap_pr(X, n_neighbors=15, n_jobs=-1):
    """Estimate dimensionality as the participation ratio of Isomap kernel
    eigenvalues.

    Parameters
    ----------
    X : np.ndarray
        Data, shape (n_samples, n_features).
    n_neighbors : int, optional
        Neighbours for geodesic distances (default is 15).
    n_jobs : int, optional
        Cores for neighbour search; -1 uses all (default is -1).

    Returns
    -------
    float
        Estimated dimensionality.
    """
    G = geodesic_dist_matrix(X, n_neighbors=n_neighbors, n_jobs=n_jobs)

    K = geodesic_to_gram_matrix(G)

    evals = np.real(np.linalg.eigvals(K))

    return (np.sum(evals) ** 2) / (np.sum(evals**2))


def normalised_components_for_vaf(arr, vaf=0.80, n_components=None):
    """Return the fraction of PCA components needed to reach a cumulative
    variance threshold.

    Parameters
    ----------
    arr : np.ndarray
        Data, shape (n_samples, n_features).
    vaf : float, optional
        Target cumulative variance fraction (default is 0.80).
    n_components : int, optional
        Components fitted (default is all features).

    Returns
    -------
    float
        Components needed divided by ``n_components``.
    """
    if n_components is None:
        n_components = arr.shape[-1]
    model = PCA(n_components=n_components, svd_solver="full")
    model.fit(arr)
    cumulative_vaf = np.cumsum(model.explained_variance_ratio_)
    num_components = np.searchsorted(cumulative_vaf, vaf) + 1
    return num_components / n_components


def get_pr_for_subsets_of_neurons(
    arr, n_iter=5, step=10, linear=True, verbose=False, n_neighbors=15
):
    """Compute the participation ratio on random neuron subsets of growing size
    (drawn with replacement).

    Parameters
    ----------
    arr : np.ndarray
        Data, shape (n_samples, n_neurons).
    n_iter : int, optional
        Random draws per subset size (default is 5).
    step : int, optional
        Increase in subset size, starting at 5 (default is 10).
    linear : bool, optional
        PCA ratio if True, else Isomap (default is True).
    verbose : bool, optional
        Print each subset size (default is False).
    n_neighbors : int, optional
        Isomap neighbours (default is 15).

    Returns
    -------
    np.ndarray
        Participation ratios, shape (n_sizes, n_iter).
    """
    results = []
    for num_neurons in tqdm(np.arange(5, arr.shape[1] + 1, step)):
        if verbose:
            print(f"Neurons: {num_neurons}")
        prs = []
        for i, _ in enumerate(range(n_iter)):
            random_neurons = np.random.randint(0, arr.shape[1], size=num_neurons)
            if linear:
                pr = pca_pr(arr[:, random_neurons])
            else:
                pr = isomap_pr(arr[:, random_neurons], n_neighbors=n_neighbors)
            prs.append(pr)

        results.append(prs)
    return np.vstack(results)


def get_pcs_percentVAF_subsets_neurons(arr, niter=5, linear=True, verbose=False, vaf=0.8):
    """Compute normalised components-for-VAF on random neuron subsets of size
    5, 15, 25, ... (drawn with replacement).

    Parameters
    ----------
    arr : np.ndarray
        Data, shape (n_samples, n_neurons).
    niter : int, optional
        Random draws per subset size (default is 5).
    linear : bool, optional
        Unused (default is True).
    verbose : bool, optional
        Print each subset size (default is False).
    vaf : float, optional
        Target cumulative variance fraction (default is 0.8).

    Returns
    -------
    np.ndarray
        Normalised component counts, shape (n_sizes, niter).
    """
    results = []
    for num_neurons in np.arange(5, arr.shape[1] + 1, 10):
        if verbose:
            print(f"Neurons: {num_neurons}")
        pc_counts = []
        for _ in range(niter):
            random_neurons = np.random.randint(0, arr.shape[1], size=num_neurons)
            norm_pc_count = normalised_components_for_vaf(arr[:, random_neurons], vaf=vaf)
            pc_counts.append(norm_pc_count)

        results.append(pc_counts)
    return np.vstack(results)
