"""
LFP utils module
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert


def sos_bandpass_filter(data: np.array, fs: float, freqs: tuple, order: int = 4):
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


def get_power_in_freq_range(data: np.array, fs: float, freqs: tuple, order: int = 4):
    filtered_data = sos_bandpass_filter(data, fs, freqs, order)
    power, phase = compute_hilbert_power(filtered_data)
    return power, phase
