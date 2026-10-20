"""
LFP utils module
"""

import numpy as np
from scipy.signal import (
    butter,
    sosfiltfilt,
    hilbert,
    lfilter,
    get_window,
    firwin,
)


def sos_bandpass_filter(
    data: np.array, fs: float, freqs: tuple, order: int = 4
):
    """Band-pass filter along axis 0 with a zero-phase (forward-backward)
    Butterworth filter.

    Parameters
    ----------
    data : np.ndarray
        Signal, shape (T, n_channels).
    fs : float
        Sampling rate in Hz.
    freqs : tuple of float
        Pass band (low, high) in Hz.
    order : int, optional
        Butterworth order per pass (default is 4).

    Returns
    -------
    np.ndarray
        Filtered signal, same shape as ``data``.
    """

    low, high = freqs
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    filtered_data = sosfiltfilt(sos, data, axis=0)
    return filtered_data


def compute_hilbert_power(x):
    """Compute instantaneous power and phase from the Hilbert analytic signal
    along axis 0.

    Parameters
    ----------
    x : np.ndarray
        Real signal, shape (T,) or (T, n_channels).

    Returns
    -------
    power : np.ndarray
        Squared analytic magnitude, same shape as ``x``.
    phase : np.ndarray
        Instantaneous phase in radians, same shape as ``x``.
    """
    analytic_signal = hilbert(x, axis=0)
    instantaneous_phase = np.angle(analytic_signal)
    return np.abs(analytic_signal) ** 2, instantaneous_phase


def get_power_phase_in_freq_range(
    data: np.array, fs: float, freqs: tuple, order: int = 4
):
    """Band-pass filter a signal, then return its Hilbert power and phase.

    Parameters
    ----------
    data : np.ndarray
        Signal, shape (T, n_channels).
    fs : float
        Sampling rate in Hz.
    freqs : tuple of float
        Pass band (low, high) in Hz.
    order : int, optional
        Butterworth order per pass (default is 4).

    Returns
    -------
    power : np.ndarray
        Instantaneous power, shape (T, n_channels).
    phase : np.ndarray
        Instantaneous phase in radians, shape (T, n_channels).
    """
    filtered_data = sos_bandpass_filter(data, fs, freqs, order)
    power, phase = compute_hilbert_power(filtered_data)
    return power, phase


def causal_hilbert_fir(numtaps, window="hamming"):
    """Design a windowed Type III FIR Hilbert filter with constant delay
    (numtaps - 1) // 2.

    Parameters
    ----------
    numtaps : int
        Filter length; an even value is bumped to the next odd one.
    window : str, optional
        Window name for ``scipy.signal.get_window`` (default is "hamming").

    Returns
    -------
    np.ndarray
        Filter taps, shape (numtaps,) after the odd adjustment.
    """
    if numtaps % 2 == 0:
        numtaps += 1
    M = (numtaps - 1) // 2
    n = np.arange(-M, M + 1)
    h = np.zeros_like(n, dtype=float)
    odd = n % 2 != 0
    h[odd] = 2.0 / (np.pi * n[odd])
    return h * get_window(window, numtaps)


def sinefit(y, f0, fs=100, f_search=1.0, f_step=0.1):
    """Grid-search a sinusoid-plus-trend fit, returning its phase at the
    window's last sample.

    Parameters
    ----------
    y : np.ndarray
        1D signal window, shape (T,).
    f0 : float
        Centre of the frequency search in Hz.
    fs : float, optional
        Sampling rate in Hz (default is 100).
    f_search : float, optional
        Search half-width in Hz (default is 1.0).
    f_step : float, optional
        Grid spacing in Hz (default is 0.1).

    Returns
    -------
    phase : float
        Phase in radians at the last sample.
    f : float
        Best-fit frequency in Hz.
    b : np.ndarray
        Coefficients [offset, slope, cos, sin] at the best frequency.
    mu : float
        Mean removed from ``y`` before fitting.
    all_f : np.ndarray
        Frequencies searched in Hz.
    sse_ : np.ndarray
        Squared error at each frequency in ``all_f``.
    """
    mu = y.mean()
    y = y - mu
    t = (np.arange(len(y)) - (len(y) - 1)) / fs
    best = dict(sse=np.inf)
    all_f = np.arange(f0 - f_search, f0 + f_search + 1e-9, f_step)
    sse_ = []
    for f in all_f:
        A = np.column_stack(
            [
                np.ones_like(t),
                t,
                np.cos(2 * np.pi * f * t),
                np.sin(2 * np.pi * f * t),
            ]
        )
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = np.sum((y - A @ b) ** 2)
        sse_.append(sse)
        if sse < best["sse"]:
            best = dict(sse=sse, f=f, b=b)
    sse_ = np.array(sse_)
    phase = -np.arctan2(best["b"][3], best["b"][2])
    return phase, best["f"], best["b"], mu, all_f, sse_


def get_phase_at_perturb(x, onset, peak_freq, window=30):
    """Estimate each keypoint's causal phase at onset with ``sinefit`` (fs
    fixed at 100 Hz).

    Parameters
    ----------
    x : np.ndarray
        Keypoint signals, shape (T, n_keypoints).
    onset : int
        Onset sample index; must be at least ``window - 1``.
    peak_freq : float
        Stride frequency in Hz.
    window : int, optional
        Samples ending at onset, inclusive (default is 30).

    Returns
    -------
    np.ndarray
        Phase in radians, shape (n_keypoints,).
    """
    phases = []
    for kp_arr in x.T:
        phase, *_ = sinefit(kp_arr[onset - window + 1 : onset + 1], peak_freq)
        phases.append(phase)
    return np.array(phases)
