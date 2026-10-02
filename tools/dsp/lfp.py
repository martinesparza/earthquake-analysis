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
    """A forward-backward digital filter using cascaded second-order sections.

    Args:
        data (np.array): Time x channels
        fs (float): _description_
        freqs (tuple): _description_
        order (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """

    low, high = freqs
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    filtered_data = sosfiltfilt(sos, data, axis=0)
    return filtered_data


def compute_hilbert_power(x):
    """Extracts power from array using hilbert transform

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    analytic_signal = hilbert(x, axis=0)
    instantaneous_phase = np.angle(analytic_signal)
    return np.abs(analytic_signal) ** 2, instantaneous_phase


def get_power_phase_in_freq_range(
    data: np.array, fs: float, freqs: tuple, order: int = 4
):
    filtered_data = sos_bandpass_filter(data, fs, freqs, order)
    power, phase = compute_hilbert_power(filtered_data)
    return power, phase


def causal_hilbert_fir(numtaps, window="hamming"):
    """Causal FIR approximation of the Hilbert transform (windowed ideal quadrature filter).

    scipy.signal.remez(..., type="hilbert") is numerically unstable for a band this narrow
    and far from DC/Nyquist (returns NaNs) -- the classic windowed-sinc design is stable and
    is what this uses instead. Only odd taps get a nonzero coefficient (2 / (pi * n)); this
    is a Type III linear-phase filter, so its group delay is a single constant,
    (numtaps - 1) // 2 samples, matching the band-pass filter's own delay below.
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
    """Compute causal phase of each keypoint at onset

    Parameters
    ----------
    x : _type_
        _description_
    onset : _type_
        _description_
    window : _type_
        _description_
    peak_freq : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    phases = []
    for kp_arr in x.T:
        phase, *_ = sinefit(kp_arr[onset - window + 1 : onset + 1], peak_freq)
        phases.append(phase)
    return np.array(phases)
