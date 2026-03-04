"""
Docstring for kinematics.utils
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.signal import savgol_filter
from scipy.stats import skew
from sklearn.mixture import GaussianMixture

import tools.dsp as dsp


def compute_perturb_distrurb_score(
    power, perturb_idx, start_idx, stop_idx=-200, dt: float = 0.01
):
    # z_score
    # power = (power - power[start_idx:perturb_idx].mean(0)) / power[start_idx:perturb_idx].std(
    #     0
    # )

    power = power - power[start_idx:perturb_idx].mean(0)
    # power = power - power[(perturb_idx - 100) : perturb_idx].mean(0)
    post = power[perturb_idx:stop_idx, :]

    disturb_score = np.nansum(post, axis=0) * dt

    return disturb_score


def add_power_metric_to_td(td):
    td = td.copy()
    td["disturb_score"] = pd.Series(np.nan * len(td), dtype="object")

    for idx, row in td.iterrows():
        if row.power is None:
            continue

        td.at[idx, "disturb_score"] = compute_perturb_distrurb_score(
            row.power,
            row.concat_perturb_time,
            start_idx=100,
        )

    return td


def compute_peak_freq_pre_perturb(bhv_arr, perturb_idx: int, nperseg=None, noverlap=None):
    freqs, psd = scipy.signal.welch(
        bhv_arr[perturb_idx - 300 : perturb_idx],
        fs=100,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=0,
    )
    return freqs[np.argmax(psd, axis=0)], freqs, psd


def compute_power_in_bhv_concat_td(td, freq_tresh=0.5, method="morlet", phase=True):
    td = td.copy()
    td["power"] = pd.Series([None] * len(td), dtype="object")
    td["phases"] = pd.Series([None] * len(td), dtype="object")

    n = 0
    for idx, row in td.iterrows():
        if row.trial_name != "trial":
            continue
        peak_freqs, freqs, psd = compute_peak_freq_pre_perturb(
            row.bhv_concat, perturb_idx=row.concat_perturb_time
        )
        peak_mean_freq = freqs[np.argmax(psd.mean(-1), axis=0)]
        if peak_mean_freq < 2:
            n = n + 1
            continue
        # if any(x < freq_tresh for x in peak_freqs):
        #     # idx = np.argsort(psd[:, 0], axis=0)[-2:]   # indices of top 2 values per column
        #     # top2_freqs = freqs[sorted(idx)]
        #     # print(top2_freqs, idx)
        #     n = n + 1
        #     # low_peak_idx = np.where(peak_freqs < freq_tresh)[0]
        #     # print(peak_freqs)
        #     fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        #     ax[0].plot(freqs, psd, alpha=0.5, color='gray')
        #     ax[0].plot(freqs, psd.mean(-1), color='k')
        #     ax[1].plot(row.bhv_concat)
        #     ax[1].axvline(row.concat_perturb_time, color='k', linestyle='dashed')
        #     peak_mean_freq = freqs[np.argmax(psd.mean(-1), axis=0)]
        #     ax[0].set_title(peak_mean_freq < 2)
        #     plt.show()
        #     continue

        powers, phases = [], []
        for i, peak_freq in enumerate(peak_freqs):
            peak_freq = peak_mean_freq  # remove this to get the per_keypoint_freq
            if method == "morlet":
                power = dsp.compute_morlet_power(
                    row.bhv_concat[:, i], fs=100, freqs=peak_freq
                )
            # compute phases
            _, instantaneous_phase = dsp.get_power_in_freq_range(
                row.bhv_concat[:, i], 100, (peak_freq - 1, peak_freq + 1)
            )
            phases.append(instantaneous_phase)
            powers.append(np.log10(np.squeeze(power)))

        phases = np.array(phases)
        powers = np.array(powers)
        if phase:
            td.at[idx, "phases"] = phases.T
        td.at[idx, "power"] = powers.T

    print(f"Skipped {n} trials")
    return td


def assess_bimodality(x, n_init=5, random_state=0):
    g = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=n_init,
        random_state=random_state,
    )

    x = x.reshape(-1, 1)
    g.fit(x)

    w = g.weights_.ravel()
    mu = g.means_.ravel()
    sd = np.sqrt(g.covariances_.ravel())

    order = np.argsort(mu)
    w, mu, sd = w[order], mu[order], sd[order]
    # print("weights:", w)
    # print("means:", mu)
    # print("sds:", sd)
    d = abs(mu[1] - mu[0]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2)
    print(f"d: {d:.3f}")
    bimodal = (w.min() > 0.10) and (d > 1.6)
    print(f"Bimodal: {bimodal}")

    return bimodal, mu


def starts_of_below_thresh_windows(arr, thresh: float, win: int):
    """
    Return start indices i such that arr[i:i+win] are ALL < thresh
    (sliding window of step 1). Its basically a convolution and then find values that match
    the window length

    Parameters
    ----------
    arr :
    thresh : float
    win : int
        window length (in samples).

    Returns
    -------
    starts : np.ndarray
        start indices of all valid windows.
    """
    arr = np.asarray(arr)
    if win <= 0:
        raise ValueError("win must be >= 1")

    mask = arr < thresh
    if win == 1:
        return np.flatnonzero(mask)

    # counts[j] = number of True values in mask[j:j+win]
    counts = np.convolve(mask.astype(np.int32), np.ones(win, dtype=np.int32), mode="valid")
    return np.flatnonzero(counts == win)


def immobile_starts_before_event(bhv_arr, thresh, event_onset=(100, 200), win=50) -> bool:
    """Detects if the animal is immobile in a give index window (before the perturbation)

    Parameters
    ----------
    bhv_arr : _type_
        array of kinematics of the mouse
    thresh : float,
        threshold under which the animal is immobile
    event_onset : tuple, optional
        onset of perturbation, by default 200
    win : int, optional
        minimum immobile duration in time points, by default 50

    Returns
    -------
    bool
        True if animal was immobile before perturbation
    """
    vel = np.gradient(bhv_arr, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    immobile_starts = starts_of_below_thresh_windows(speed, thresh, win)  # indices
    return np.any((immobile_starts > event_onset[0]) & (immobile_starts < event_onset[1]))


def valley_between_means(x, mu1, mu2, bins=300, smooth_win=31, poly=3):
    """
    Find the minimum of the distribution between two x-values by first smoothing it
    """

    x = np.asarray(x)
    lo, hi = min(mu1, mu2), max(mu1, mu2)

    hist, edges = np.histogram(x, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Smooth histogram. Thanks for suggestion, works better
    if smooth_win >= bins:
        smooth_win = bins - 1
    if smooth_win % 2 == 0:
        smooth_win += 1
    hist_s = savgol_filter(hist, smooth_win, poly)

    # Restrict to interval between means
    mask = (centers > lo) & (centers < hi)
    if not np.any(mask):
        raise ValueError("No histogram bins between means.")

    idx = np.argmin(hist_s[mask])
    x_valley = centers[mask][idx]
    return x_valley


def percentil_based_thresholding(arr, p):
    return np.percentile(arr, p)


def compute_immobile_thresh(td, p=5, plot=True):
    # compute speed and velocity
    vel = np.gradient(np.concatenate(td.bhv.values), axis=0)
    speed = np.linalg.norm(vel, axis=1)

    # Check bimodality and compute thresholds
    bimodal, mus = assess_bimodality(np.log(speed))
    if bimodal:
        x_min = valley_between_means(np.log(speed), mus[0], mus[1])
        thresh = 10**x_min
    else:
        x_min = percentil_based_thresholding(np.log(speed), p=p)
        thresh = 10**x_min

    print(f"Thresh: {thresh:.2f}")
    if plot:
        fig, ax = plt.subplots()
        ax.hist(np.log(speed), bins=100)
        ax.axvline(x_min, color="red", linestyle="--", label="Immobile thresh")
        ax.set_xlabel("Log speed")
        ax.legend()
    return thresh


def drop_immobile_trials_from_td(td, event_onset=(100, 200), win=50, p=5, plot=False):
    initial_count = len(td)

    thresh = compute_immobile_thresh(td, p=p, plot=plot)

    filtered_df = td[
        td["bhv"].apply(
            lambda arr: not immobile_starts_before_event(
                arr, thresh=thresh, event_onset=event_onset, win=win
            )
        )
    ]

    dropped_count = initial_count - len(filtered_df)
    print(
        f"Dropped {dropped_count} of {initial_count} rows ({dropped_count/initial_count:.2%})."
    )
    return filtered_df
