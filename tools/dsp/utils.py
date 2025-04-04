import numpy as np


def moving_window_mean(data, window_size):
    """Compute rolling mean specifying window size and ata

    Args:
        data (np.ndarray): 1d or 2d array
        window_size (int): Second for window

    Returns:
        means : nd.array
        time_bins : np.ndarray
    """

    means = []
    time_bins = []
    for i in range(len(data) - window_size + 1):
        if data.ndim == 2:
            window = data[i : i + window_size, :]
        elif data.ndim == 1:
            window = data[i : i + window_size]

        window_mean = np.mean(window, axis=0)
        means.append(window_mean)
        time_bins.append(i + window_size)
    return np.array(means), np.array(time_bins)


def sort_neurons_in_time_array(arr):
    """Given an array of neurons x time, sorts and zscores them.

    Args:
        arr (np.array): naurons x time

    Returns:
        np.array: sorted and z-scored array
    """
    max_firing_times = np.argmax(
        arr, axis=1
    )  # Get the time index of max firing for each neuron
    sorted_indices = np.argsort(max_firing_times)  # Sort indices based on max firing times
    sorted_firing_rates = arr[sorted_indices, :]  # Sort the matrix

    # Step 2: Z-score normalization
    # Compute the mean and variance across all neurons
    global_mean = np.mean(sorted_firing_rates)
    global_std = np.std(sorted_firing_rates)

    # Z-score the matrix
    z_scored_firing_rates = (sorted_firing_rates - global_mean) / global_std

    return z_scored_firing_rates
