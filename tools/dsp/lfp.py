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


def causal_phase_estimator(x, fs, center_freq, bandwidth=2, numtaps=71):
    """Causal band-pass + causal Hilbert -> instantaneous phase and amplitude.

    Both filters are linear-phase FIR (constant group delay), so the combined delay is
    known exactly: 2 * (numtaps - 1) // 2 samples. Output before that many samples have
    elapsed is filter transient, not a valid estimate -- see the returned `warmup`.
    """
    lo, hi = center_freq - bandwidth / 2, center_freq + bandwidth / 2
    bp_taps = firwin(numtaps, [lo, hi], pass_zero=False, fs=fs)
    hilb_taps = causal_hilbert_fir(numtaps)

    xf = lfilter(bp_taps, 1.0, x)  # causal band-pass, delay d
    xq = lfilter(hilb_taps, 1.0, xf)  # causal quadrature branch, +d more delay

    d = (numtaps - 1) // 2
    xr = np.full_like(xf, np.nan)
    # delay the real branch by d to align it with xq's extra delay
    xr[d:] = xf[:-d]

    # phase will be nans up until 2*d
    phase = np.arctan2(xq, xr)
    amp = np.hypot(xq, xr)
    warmup = 2 * d
    return phase, warmup


def causal_phase_arr(arr, fs, center_freqs, bandwidth=2, numtaps=71):
    phases = []
    for arr_ in arr.T:
        phase, warmup = causal_phase_estimator(
            arr_, fs, center_freqs, bandwidth, numtaps
        )
        phases.append(phase)
    return np.array(phases).T, warmup
