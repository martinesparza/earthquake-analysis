import numpy as np


def compute_morlet_power(
    x: np.ndarray,
    fs: float,
    freqs: np.ndarray,
    n_cycles: float = 3.0,
    wavelet_support: float = 4.0,
) -> np.ndarray:
    """Compute per-frequency power with unit-energy Morlet wavelets, after mean
    removal.

    Parameters
    ----------
    x : np.ndarray
        Signal, shape (T,) or (T, n_features).
    fs : float
        Sampling rate in Hz.
    freqs : np.ndarray
        Centre frequencies in Hz.
    n_cycles : float, optional
        Cycles in the Gaussian envelope (default is 3.0).
    wavelet_support : float, optional
        Wavelet half-length in Gaussian sigmas (default is 4.0).

    Returns
    -------
    np.ndarray
        Power, shape (n_freqs, T, n_features); 3D even for 1D input.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]  # (T, 1)

    T, feat = x.shape
    freqs = np.atleast_1d(freqs).astype(float)
    F = freqs.size

    # denean per feature
    x0 = x - x.mean(axis=0, keepdims=True)
    power = np.empty((F, T, feat), dtype=float)

    for i, f in enumerate(freqs):

        sigma_t = n_cycles / (2 * np.pi * f)  # seconds
        t_max = wavelet_support * sigma_t
        t = np.arange(-t_max, t_max, 1.0 / fs)

        wavelet = np.exp(2j * np.pi * f * t) * np.exp(-(t**2) / (2 * sigma_t**2))
        # Unit-energy normalisation => comparable power across frequencies
        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))

        # Compute it in freqeucny domain
        n_conv = T + wavelet.size - 1
        n_fft = 1 << int(np.ceil(np.log2(n_conv)))
        X = np.fft.fft(x0, n_fft, axis=0)  # (n_fft, feat)
        W = np.fft.fft(wavelet, n_fft)[:, None]  # (n_fft, 1)
        conv = np.fft.ifft(X * W, axis=0)[:n_conv]  # (n_conv, feat)

        # crop to original length
        start = (wavelet.size - 1) // 2
        conv = conv[start : start + T, :]  # (T, feat)

        power[i, :, :] = np.abs(conv) ** 2

    return power


def moving_window_mean(data, window_size):
    """Compute the mean over a sliding window, advancing one sample at a time.

    Parameters
    ----------
    data : np.ndarray
        Array, shape (T,) or (T, n_columns).
    window_size : int
        Window length in samples, not seconds.

    Returns
    -------
    means : np.ndarray
        Mean of each window, one per window position.
    time_bins : np.ndarray
        Index one past the last sample of each window.
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
    """Sort neurons by peak-activity time, then z-score with one global mean
    and std.

    Parameters
    ----------
    arr : np.ndarray
        Activity, shape (n_neurons, T).

    Returns
    -------
    np.ndarray
        Sorted, z-scored array, shape (n_neurons, T).
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


def compute_moving_window_mean_on_array(df, signal, window_s):
    """Take a moving-window mean over a column concatenated across rows, so
    windows can span rows.

    Parameters
    ----------
    df : pd.DataFrame
        Trial data; the first row's ``bin_size`` sets the window.
    signal : str
        Column of (time x channels) arrays.
    window_s : float
        Window length in seconds.

    Returns
    -------
    means : np.ndarray
        Moving-window means.
    time_bins : np.ndarray
        Window end indices into the concatenated array.
    """
    arr = np.concatenate(df[signal].values, axis=0)
    means, time_bins = moving_window_mean(arr, int(window_s / df.bin_size.values[0]))
    return means, time_bins
