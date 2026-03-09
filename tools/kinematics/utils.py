"""
Docstring for kinematics.utils
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import savgol_filter
from scipy.stats import skew
from sklearn.mixture import GaussianMixture


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


def otsu_threshold(x, bins=256):
    """
    Parameter-free threshold via Otsu's method: finds the value t that maximises
    between-class variance of the two populations below/above t.

    Parameters
    ----------
    x    : 1-D array (e.g. log-speed values)
    bins : histogram resolution

    Returns
    -------
    float  threshold in the same units as x
    """
    hist, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = hist.astype(float) / hist.sum()

    best_t, best_var = centers[0], -1.0
    cumsum = np.cumsum(hist)
    cummean = np.cumsum(hist * centers)

    for i in range(1, len(centers)):
        w0 = cumsum[i - 1]
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = cummean[i - 1] / w0
        mu1 = (cummean[-1] - cummean[i - 1]) / w1
        between_var = w0 * w1 * (mu0 - mu1) ** 2
        if between_var > best_var:
            best_var = between_var
            best_t = centers[i]
    return best_t


def drop_immobile_trials(td, pre_perturb_window=(100, 200), min_immobile_bins=2, plot=False):
    """
    Drop perturbation trials where the animal stopped running for at least
    `min_immobile_bins` consecutive samples anywhere in `pre_perturb_window`.

    The immobility threshold is computed once per call using Otsu's method on
    the log-speed pooled across all trials — no bimodality assumption, no
    tunable percentile.

    Parameters
    ----------
    td : pd.DataFrame
        Trial-only pyalData table (trial_name == 'trial'), already filtered
        before calling this function.
    pre_perturb_window : tuple (start_bin, end_bin)
        Index window (in samples, 10 ms/bin) within each trial's `bhv` array
        to search for immobility. E.g. (100, 500) = 1–5 s before perturbation.
    min_immobile_bins : int
        Minimum consecutive below-threshold bins to count as immobile.
        2 bins = 20 ms at 10 ms/bin.
    plot : bool
        If True, show log-speed histogram with threshold.

    Returns
    -------
    pd.DataFrame  filtered trial table (index preserved).
    """
    # Pool speed across all trials
    vel = np.gradient(np.concatenate(td.bhv.values), axis=0)
    speed = np.linalg.norm(vel, axis=1)
    log_speed = np.log(np.maximum(speed, 1e-10))

    # Otsu threshold in log-space → convert back
    thresh_log = otsu_threshold(log_speed)
    thresh = np.exp(thresh_log)
    print(f"Otsu immobility threshold: {thresh:.4f}")

    if plot:
        fig, ax = plt.subplots()
        ax.hist(log_speed, bins=100)
        ax.axvline(thresh_log, color="red", linestyle="--", label=f"Otsu thresh (log={thresh_log:.2f})")
        ax.set_xlabel("Log speed")
        ax.legend()

    initial_count = len(td)
    filtered_df = td[
        td["bhv"].apply(
            lambda arr: not immobile_starts_before_event(
                arr, thresh=thresh, event_onset=pre_perturb_window, win=min_immobile_bins
            )
        )
    ]
    dropped_count = initial_count - len(filtered_df)
    print(f"Dropped {dropped_count} of {initial_count} rows ({dropped_count/initial_count:.2%}).")
    return filtered_df


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
