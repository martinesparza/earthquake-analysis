"""
Functions for computing the perturbation disturbance score from kinematic data.

Pipeline (high-level entry point: compute_perturb_score):
  1. Stack keypoint xyz arrays → `bhv` (T × 36)  [dt.add_bhv]
  2. Prepend preceding intertrial segment → `bhv_concat`  [dt.concat_previous_intertrial_signal]
  3. Index perturbation onset and trial start inside bhv_concat  [dt.add_concat_perturb_time/trial_start]
  4. Estimate dominant gait frequency; compute Morlet log-power → `power` (T × n_keypoints)
  5. Baseline-subtract and integrate post-perturbation power → `disturb_score` (n_keypoints,)
"""

import numpy as np
import pandas as pd
import scipy

import tools.dsp as dsp
import tools.dataTools as dt

# ---------------------------------------------------------------------------
# Low-level signal processing
# ---------------------------------------------------------------------------


def compute_peak_freq_pre_perturb(bhv_arr, perturb_idx: int, nperseg=300, noverlap=None):
    """
    Estimate the dominant frequency of each keypoint in the 300-sample window
    before the perturbation using Welch's method.

    Returns
    -------
    peak_freqs : np.ndarray  (n_keypoints,)
    freqs      : np.ndarray  frequency axis
    psd        : np.ndarray  (n_freqs, n_keypoints)
    """
    freqs, psd = scipy.signal.welch(
        bhv_arr[perturb_idx - 300 : perturb_idx],
        fs=100,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=0,
    )
    return freqs[np.argmax(psd, axis=0)], freqs, psd


def compute_perturb_score_row(power, perturb_idx, start_idx, stop_idx=-300, dt: float = 0.01):
    """
    Compute the disturbance score as the baseline-subtracted integral of log-power
    in the post-perturbation window.

    Parameters
    ----------
    power       : np.ndarray  (T, n_keypoints)  log-power time series
    perturb_idx : int         index of perturbation onset in the concatenated signal
    start_idx   : int         start of the pre-perturbation baseline window
    stop_idx    : int         end of the post-perturbation window (negative = from end)
    dt          : float       time step in seconds (default 0.01 s = 10 ms)

    Returns
    -------
    disturb_score : np.ndarray  (n_keypoints,)
    """
    power = power - power[start_idx:perturb_idx].mean(0)
    post = power[perturb_idx:stop_idx, :]
    return np.nansum(post, axis=0) * dt


# ---------------------------------------------------------------------------
# Trial-table (pyalData) level functions
# ---------------------------------------------------------------------------


def compute_power_in_bhv_concat_td(td, freq_tresh=2, method="morlet", phase=True):
    """
    For each perturbation trial, estimate the dominant gait frequency pre-perturbation,
    then compute Morlet log-power (and instantaneous phase) at that frequency for every
    keypoint in `bhv_concat`.

    Trials where peak_mean_freq < 2 Hz (animal not locomoting) are skipped.

    Adds columns
    ------------
    power  : np.ndarray  (T, n_keypoints)  log10 Morlet power
    phases : np.ndarray  (T, n_keypoints)  instantaneous phase
    """
    td = td.copy()
    td["power"] = pd.Series([None] * len(td), dtype="object", index=td.index)
    td["phases"] = pd.Series([None] * len(td), dtype="object", index=td.index)

    n_skipped = 0
    for idx, row in td.iterrows():
        if row.trial_name != "trial":
            continue
        peak_freqs, freqs, psd = compute_peak_freq_pre_perturb(
            row.bhv_concat, perturb_idx=row.concat_perturb_time
        )
        peak_mean_freq = freqs[np.argmax(psd.mean(-1), axis=0)]
        if peak_mean_freq < freq_tresh:
            n_skipped += 1
            continue

        powers, phases = [], []
        for i, peak_freq in enumerate(peak_freqs):
            peak_freq = peak_mean_freq  # use shared freq across keypoints
            if method == "morlet":
                power = dsp.compute_morlet_power(
                    row.bhv_concat[:, i], fs=100, freqs=peak_freq
                )
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

    print(f"Skipped {n_skipped} trials")
    return td


def add_perturb_score_td(td):
    """
    Compute `disturb_score` per trial from the `power` column.

    Adds column
    -----------
    disturb_score : np.ndarray  (n_keypoints,)
        Baseline-subtracted integral of log-power post-perturbation.
    """
    td = td.copy()
    td["disturb_score"] = pd.Series([None] * len(td), dtype="object", index=td.index)

    for idx, row in td.iterrows():
        if not isinstance(row.power, np.ndarray):
            continue
        td.at[idx, "disturb_score"] = compute_perturb_score_row(
            row.power,
            row.concat_perturb_time,
            start_idx=100,
        )
    return td


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------


def compute_perturb_score(df, bhv_fields, feature_dims=None):
    """
    Full pipeline to build bhv_concat and compute the per-trial disturbance score.

    Steps
    -----
    1. Stack keypoint xyz arrays → `bhv` (T × 36)
    2. Prepend preceding intertrial segment → `bhv_concat` (T_prev + T × n_features),
       using only the z-dimension of each keypoint (feature_dims).
    3. Record perturbation onset (`concat_perturb_time`) and trial start
       (`concat_trial_start`) inside bhv_concat.
    4. Estimate dominant gait frequency; compute Morlet log-power per keypoint
       → `power` (T × n_keypoints). Trials with peak_mean_freq < 2 Hz are skipped.
    5. Integrate baseline-subtracted post-perturbation power → `disturb_score`
       (n_keypoints,).

    Parameters
    ----------
    df           : pd.DataFrame  pyalData trial table (after dsp.preprocess + dt.add_pca_df)
    bhv_fields   : list[str]     keypoint names to include (e.g. ALL_BHV_FIELDS, 12 keypoints)
    feature_dims : np.ndarray    column indices into bhv for bhv_concat; defaults to
                                 z-dimension (every 3rd col starting at 2)

    Returns
    -------
    pd.DataFrame  with added columns: bhv, bhv_concat, concat_perturb_time,
                  concat_trial_start, power, phases, disturb_score
    """
    if feature_dims is None:
        feature_dims = np.arange(start=2, stop=len(bhv_fields) * 3, step=3)

    df = dt.add_bhv(df, bhv_fields=bhv_fields)
    df = dt.concat_previous_intertrial_signal(df, "bhv", features=feature_dims)
    df = dt.add_concat_perturb_time(df)
    df = dt.add_concat_trial_start(df)
    df = compute_power_in_bhv_concat_td(df)
    df = add_perturb_score_td(df)
    return df
