import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import KernelCenterer
from sklearn.utils.graph import _fix_connected_components
from tqdm import tqdm
import torch
import warnings
import pathlib


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


def normalised_components_for_vaf(arr, vaf=0.80, n_components=None):
    """
    Calculate the minimum number of components required to reach the specified VAF threshold.

    Parameters:
    -----------

    Returns:
    --------
    int
        Number of components needed to reach or exceed the threshold.
    """
    if n_components is None:
        n_components = arr.shape[-1]
    model = PCA(n_components=n_components, svd_solver="full")
    model.fit(arr)
    cumulative_vaf = np.cumsum(model.explained_variance_ratio_)
    num_components = np.searchsorted(cumulative_vaf, vaf) + 1
    return num_components / n_components


def get_pr_for_subsets_of_neurons(
    arr, n_iter=100, step=10, linear=True, verbose=False, n_neighbors=15
):
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


def compute_pr_svd(X):
    """
    Compute participation ratio from data using SVD. 
    X should be (n_samples, n_features) or (n_neurons, n_timepoints)
    
    Parameters
    ----------
    X : 2D array
        Data array
        
    Returns
    -------
    float
        Participation ratio
    """
    # Use SVD directly for efficiency
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    # Participation ratio from singular values
    pr = (S**2).sum()**2 / (S**4).sum()
    return pr


def compute_pr_batched_svd(X_batch, device='cpu'):
    """
    Compute participation ratios for a batch of conditions using torch batched SVD.
    
    Parameters
    ----------
    X_batch : torch.Tensor
        Batch of data arrays, shape (batch_size, n_samples, n_features)
    device : str, default 'cpu'
        Device to use for computation ('cpu' or 'cuda')
        
    Returns
    -------
    np.ndarray
        Array of participation ratios for each batch element (batch_size,)
    """
    X_batch = X_batch.to(device)
    _, S, _ = torch.linalg.svd(X_batch, full_matrices=False)
    # Participation ratio from singular values for each batch element
    pr_batch = (S**2).sum(dim=1)**2 / (S**4).sum(dim=1)
    return pr_batch.cpu().numpy()


def compute_pr_incremental_batched(
    data_dict,
    n_time_increment,
    n_neurons_increment,
    n_iterations=10,
    output_file=None,
    device='cpu',
    verbose=True
):
    """
    Compute participation ratio for increasing numbers of neurons and timepoints.
    Uses batched SVD when computing PR for multiple conditions with the same shape.
    Saves results incrementally to file.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary with keys (area, condition) and values as data arrays (n_timepoints, n_neurons)
    n_time_increment : int
        Increment for number of timepoints to sample
    n_neurons_increment : int
        Increment for number of neurons to sample
    n_iterations : int, default 10
        Number of random sampling iterations per config
    output_file : str or Path, optional
        Path to save results as npz file. If provided, results are saved incrementally.
    device : str, default 'cpu'
        Device for torch batched SVD ('cpu' or 'cuda')
    verbose : bool, default True
        Print progress information
        
    Returns
    -------
    dict
        Results dict with structure:
        results[(area, condition)][(n_neurons, n_timepoints)] = array of PR values (n_iterations,)
    """
    import pathlib
    if output_file is not None:
        output_file = pathlib.Path(output_file)
    
    # Determine max neurons and timepoints across all data
    max_neurons = max(data.shape[1] for data in data_dict.values())
    max_timepoints = max(data.shape[0] for data in data_dict.values())
    
    # Initialize results structure
    results = {key: {} for key in data_dict.keys()}
    
    # Generate neuron and timepoint counts
    neuron_counts = np.arange(n_neurons_increment, max_neurons + 1, n_neurons_increment)
    timepoint_counts = np.arange(n_time_increment, max_timepoints + 1, n_time_increment)
    
    if verbose:
        print(f"Computing PR for {len(neuron_counts)} neuron levels and {len(timepoint_counts)} timepoint levels")
        print(f"Neuron counts: {neuron_counts[:5]}...{neuron_counts[-3:]}")
        print(f"Timepoint counts: {timepoint_counts[:5]}...{timepoint_counts[-3:]}")
    
    # Iterate through neuron counts (outer loop for incremental saving)
    for n_neuron_idx, n_neurons in enumerate(neuron_counts):
        if verbose:
            print(f"\nProcessing n_neurons={n_neurons} ({n_neuron_idx+1}/{len(neuron_counts)})")
        
        # Iterate through timepoint counts
        for n_timepoints in tqdm(timepoint_counts, desc="Timepoints", disable=not verbose):
            
            # Group data by shape for batched SVD
            shape_groups = {}
            for (area, condition), data in data_dict.items():
                # Ensure we don't exceed available data
                n_t = min(n_timepoints, data.shape[0])
                n_n = min(n_neurons, data.shape[1])
                shape_key = (n_n, n_t)
                
                if shape_key not in shape_groups:
                    shape_groups[shape_key] = []
                shape_groups[shape_key].append((area, condition, data, n_n, n_t))
            
            # Process each shape group
            for shape_key, group in shape_groups.items():
                n_n, n_t = shape_key
                
                if len(group) > 1 and device == 'cuda':
                    # Use batched SVD for multiple conditions with same shape
                    batch_data = []
                    batch_info = []
                    
                    for area, condition, data, n_n_actual, n_t_actual in group:
                        batch_info.append((area, condition))
                        batch_data.append(data[:n_t_actual, :n_n_actual].T)  # Transpose to (n_neurons, n_timepoints)
                    
                    X_batch = torch.from_numpy(np.array(batch_data)).float()
                    pr_values = compute_pr_batched_svd(X_batch, device=device)
                    
                    # Average over iterations by resampling
                    all_prs = [pr_values]
                    for _ in range(n_iterations - 1):
                        # Resample neurons
                        batch_data_resample = []
                        for area, condition, data, n_n_actual, n_t_actual in group:
                            sample_neurons = np.random.choice(
                                data.shape[1], size=n_n_actual, replace=False
                            )
                            sample_timepoints = np.random.choice(
                                data.shape[0], size=n_t_actual, replace=False
                            )
                            X_subset = data[np.sort(sample_timepoints)][:, sample_neurons].T
                            batch_data_resample.append(X_subset)
                        
                        X_batch_resample = torch.from_numpy(np.array(batch_data_resample)).float()
                        pr_vals = compute_pr_batched_svd(X_batch_resample, device=device)
                        all_prs.append(pr_vals)
                    
                    all_prs = np.array(all_prs)  # (n_iterations, batch_size)
                    for batch_idx, (area, condition) in enumerate(batch_info):
                        results[(area, condition)][(n_neurons, n_timepoints)] = all_prs[:, batch_idx]
                
                else:
                    # Process individually (or batched on CPU)
                    for area, condition, data, n_n_actual, n_t_actual in group:
                        prs_iter = []
                        
                        for iteration in range(n_iterations):
                            # Random sampling of neurons and timepoints
                            sample_neurons = np.random.choice(
                                data.shape[1], size=n_n_actual, replace=False
                            )
                            sample_timepoints = np.random.choice(
                                data.shape[0], size=n_t_actual, replace=False
                            )
                            
                            # Extract subset and compute PR
                            X_subset = data[np.sort(sample_timepoints)][:, sample_neurons]
                            pr = compute_pr_svd(X_subset.T)  # Transpose to (neurons, timepoints)
                            prs_iter.append(pr)
                        
                        results[(area, condition)][(n_neurons, n_timepoints)] = np.array(prs_iter)
        
        # Save results incrementally after each neuron level
        if output_file is not None:
            # Convert to saveable format
            save_dict = {}
            for (area, condition), pr_dict in results.items():
                for (n_n, n_t), pr_values in pr_dict.items():
                    key = f"{area}_{condition}_n{n_n}_t{n_t}"
                    save_dict[key] = pr_values
            
            np.savez(output_file, **save_dict)
            if verbose:
                print(f"Saved results to {output_file}")
    
    return results


def load_pr_results(output_file):
    """
    Load participation ratio results from npz file.
    
    Parameters
    ----------
    output_file : str or Path
        Path to npz file
        
    Returns
    -------
    dict
        Nested dict with structure results[(area, condition)][(n_neurons, n_timepoints)] = pr_values
    """
    output_file = pathlib.Path(output_file)
    loaded = np.load(output_file, allow_pickle=True)
    
    results = {}
    for key, pr_values in loaded.items():
        # Parse key: "area_condition_n{n_neurons}_t{n_timepoints}"
        parts = key.split('_')
        area = parts[0]
        condition = parts[1]
        n_neurons = int(parts[2][1:])  # Remove 'n' prefix
        n_timepoints = int(parts[3][1:])  # Remove 't' prefix
        
        if (area, condition) not in results:
            results[(area, condition)] = {}
        
        results[(area, condition)][(n_neurons, n_timepoints)] = pr_values
    
    return results