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
# Shared timing constants (10 ms bins / 100 Hz throughout)
# ---------------------------------------------------------------------------

PRE_PERTURB_WINDOW = 300  # samples (3 s) used to estimate the pre-perturbation PSD
BASELINE_START_SAMPLES = (
    100  # samples (1 s) into bhv_concat where the baseline window starts
)
POST_PERTURB_END_OFFSET = -300
# samples (3 s) excluded from the end of the integration window

# ---------------------------------------------------------------------------
# Low-level signal processing
# ---------------------------------------------------------------------------


def compute_peak_freq_pre_perturb(
    bhv_arr, perturb_idx: int, nperseg=PRE_PERTURB_WINDOW, noverlap=None, nfft=1024
):
    """
    Estimate the dominant frequency of each keypoint in the PRE_PERTURB_WINDOW-sample
    window before the perturbation using Welch's method.

    Returns
    -------
    peak_freqs : np.ndarray  (n_keypoints,)
    freqs      : np.ndarray  frequency axis
    psd        : np.ndarray  (n_freqs, n_keypoints)
    """
    freqs, psd = scipy.signal.welch(
        bhv_arr[perturb_idx - PRE_PERTURB_WINDOW : perturb_idx],
        fs=100,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        axis=0,
    )
    return freqs[np.argmax(psd, axis=0)], freqs, psd


def compute_perturb_score_row(
    power,
    perturb_idx,
    start_idx,
    stop_idx=POST_PERTURB_END_OFFSET,
    bin_size: float = 0.01,
):
    """
    Compute the disturbance score as the baseline-subtracted integral of log-power
    in the post-perturbation window.

    Parameters
    ----------
    power       : np.ndarray  (T, n_keypoints)  log-power time series
    perturb_idx : int         index of perturbation onset in the concatenated signal
    start_idx   : int         start of the pre-perturbation baseline window
    stop_idx    : int         end of the post-perturbation window (negative = from end)
    bin_size    : float       time step in seconds (default 0.01 s = 10 ms)

    Returns
    -------
    disturb_score : np.ndarray  (n_keypoints,)
    """
    power = power - power[start_idx:perturb_idx].mean(0)
    post = power[perturb_idx:stop_idx, :]
    return np.nansum(post, axis=0) * bin_size


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

    # only the columns this loop needs, and only trial rows to avoid iterrows()
    trial_cols = td.loc[td.trial_name == "trial", ["bhv_concat", "concat_perturb_time"]]

    n_total = len(trial_cols)
    n_skipped = 0
    for idx, bhv_concat, perturb_idx in trial_cols.itertuples():
        _, freqs, psd = compute_peak_freq_pre_perturb(bhv_concat, perturb_idx=perturb_idx)
        peak_mean_freq = freqs[np.argmax(psd.mean(-1), axis=0)]
        if peak_mean_freq < freq_tresh:
            n_skipped += 1
            continue

        # every keypoint shares peak_mean_freq, so compute_morlet_power /
        # get_power_phase_in_freq_range can run on all keypoints in one vectorised
        # call (both already operate along axis=0 per column) instead of looping
        if method == "morlet":
            power = dsp.compute_morlet_power(bhv_concat, fs=100, freqs=peak_mean_freq)
            td.at[idx, "power"] = np.log10(np.squeeze(power, axis=0))

        if phase:
            _, instantaneous_phase = dsp.get_power_phase_in_freq_range(
                bhv_concat, 100, (peak_mean_freq - 1, peak_mean_freq + 1)
            )
            td.at[idx, "phases"] = instantaneous_phase

    n_passed = n_total - n_skipped
    print(
        f"{n_passed} / {n_total} trials pass the peak_mean_freq >= {freq_tresh} Hz gate "
        f"({n_passed / n_total:.1%})"
    )
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
            start_idx=BASELINE_START_SAMPLES,
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
