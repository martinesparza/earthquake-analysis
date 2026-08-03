"""
Functions for computing the perturbation disturbance score from kinematic data.

Pipeline (high-level entry point: compute_perturb_score):
  0. Add `bhv` from all known keypoint/angle columns if not already present  [dt.add_bhv]
  1. Select one dimension (x/y/z) of each `on_keypoints` field from `bhv`  [get_keypoint_dim_indices]
  2. Prepend preceding intertrial segment on those columns → `bhv_concat`  [dt.concat_previous_intertrial_signal]
  3. Index perturbation onset and trial start inside bhv_concat  [dt.add_concat_perturb_time/trial_start]
  4. Estimate dominant gait frequency; compute Morlet log-power → `power` (T × n_keypoints)
  5. Baseline-subtract and integrate post-perturbation power → `disturb_score` (n_keypoints,)
"""

import warnings

import numpy as np
import pandas as pd
import scipy

import tools.dsp as dsp
import tools.dataTools as dt
from tools.params import Params

# ---------------------------------------------------------------------------
# Shared timing constants (10 ms bins / 100 Hz throughout)
# ---------------------------------------------------------------------------

PRE_PERTURB_WINDOW = (
    300  # samples (3 s) used to estimate the pre-perturbation PSD
)
BASELINE_START_SAMPLES = (
    100  # samples (1 s) into bhv_concat where the baseline window starts
)
POST_PERTURB_END_OFFSET = -300
# samples (3 s) excluded from the end of the integration window

# ---------------------------------------------------------------------------
# Low-level signal processing
# ---------------------------------------------------------------------------


def compute_peak_freq_pre_perturb(
    bhv_arr,
    perturb_idx: int,
    nperseg=PRE_PERTURB_WINDOW,
    noverlap=None,
    nfft=1024,
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


def compute_power_in_bhv_concat_td(
    td, freq_tresh=2, method="morlet", phase=True
):
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
    trial_cols = td.loc[
        td.trial_name == "trial", ["bhv_concat", "concat_perturb_time"]
    ]

    n_total = len(trial_cols)
    n_skipped = 0
    for idx, bhv_concat, perturb_idx in trial_cols.itertuples():
        _, freqs, psd = compute_peak_freq_pre_perturb(
            bhv_concat, perturb_idx=perturb_idx
        )
        peak_mean_freq = freqs[np.argmax(psd.mean(-1), axis=0)]
        if peak_mean_freq < freq_tresh:
            n_skipped += 1
            continue

        # every keypoint shares peak_mean_freq, so compute_morlet_power /
        # get_power_phase_in_freq_range can run on all keypoints in one vectorised
        # call (both already operate along axis=0 per column) instead of looping
        if method == "morlet":
            power = dsp.compute_morlet_power(
                bhv_concat, fs=100, freqs=peak_mean_freq
            )
            td.at[idx, "power"] = np.log10(np.squeeze(power, axis=0))

        if phase:
            _, instantaneous_phase = dsp.get_power_phase_in_freq_range(
                bhv_concat, 100, (peak_mean_freq - 1, peak_mean_freq + 1)
            )
            td.at[idx, "phases"] = instantaneous_phase

    n_passed = n_total - n_skipped
    print(
        f"compute_power_in_bhv_concat_td: dropping {n_skipped} trial(s) with peak_mean_freq < 2 Hz"
    )
    # print(
    #     f"{n_passed} / {n_total} trials pass the peak_mean_freq >= {freq_tresh} Hz gate "
    #     f"({n_passed / n_total:.1%})"
    # )
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
    td["disturb_score"] = pd.Series(
        [None] * len(td), dtype="object", index=td.index
    )

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

_DIM_TO_COL = {"x": 0, "y": 1, "z": 2}


def _keypoint_width(df, field):
    """Number of columns `field` contributes to the `bhv` design matrix."""
    arr = np.asarray(df[field].values[0])
    return 1 if arr.ndim == 1 else arr.shape[1]


def get_keypoint_dim_indices(df, keypoints, dim="z"):
    """
    Column indices selecting a single dimension (x/y/z) of each field in `keypoints`.

    Each field's width (1 for scalar fields like `*_angle`, 3 for xyz keypoint
    fields) is read from `df` directly, so this is correct even when fields have
    mixed widths. Offsets are computed purely from `keypoints`' own widths, in the
    given order — i.e. `keypoints` is assumed to lay out contiguously in `bhv` in
    this order, not merely be some subset scattered through a wider field list.

    Fields not recognised in `Params.keypoints` trigger a warning and are skipped.

    Parameters
    ----------
    df        : pd.DataFrame  trial table with a `bhv` column already built.
    keypoints : list[str]     field names to select a dimension from, in order.
    dim       : "x" | "y" | "z"  dimension to select for every field.

    Returns
    -------
    np.ndarray  column indices into `bhv`, one per keypoint recognised in
                `Params.keypoints`.
    """
    if dim not in _DIM_TO_COL:
        raise ValueError(
            f"dim must be one of {list(_DIM_TO_COL)}, got {dim!r}"
        )
    dim_col = _DIM_TO_COL[dim]

    missing = [f for f in keypoints if f not in Params.keypoints]
    if missing:
        warnings.warn(
            f"oscillating field(s) not found in bhv_fields, skipping: {missing}"
        )

    widths = {f: _keypoint_width(df, f) for f in keypoints}
    offsets = dict(zip(keypoints, np.cumsum([0] + list(widths.values())[:-1])))

    present = [f for f in keypoints if f in offsets]
    too_narrow = [f for f in present if widths[f] <= dim_col]
    if too_narrow:
        raise ValueError(
            f"field(s) {too_narrow} have < {dim_col + 1} column(s), "
            f"cannot select dim={dim!r}"
        )

    return np.array([offsets[f] + dim_col for f in present])


def compute_perturb_score(
    df,
    on_keypoints=Params.oscillating_keypoints,
    feature_dims="z",
    drop_trials=True,
):
    """
    Full pipeline to build bhv_concat and compute the per-trial disturbance score.

    Steps
    -----
    0. Add `bhv` via `dt.add_bhv(df, bhv_fields=["all"])` if not already present.
    1. Select one dimension (`feature_dims`) of each `on_keypoints` field from `bhv`,
       then prepend the preceding intertrial segment on those columns → `bhv_concat`
       (T_prev + T × n_keypoints).
    2. Record perturbation onset (`concat_perturb_time`) and trial start
       (`concat_trial_start`) inside bhv_concat.
    3. Estimate dominant gait frequency; compute Morlet log-power per keypoint
       → `power` (T × n_keypoints). Trials with peak_mean_freq < 2 Hz are skipped.
    4. Integrate baseline-subtracted post-perturbation power → `disturb_score`
       (n_keypoints,).

    Parameters
    ----------
    df           : pd.DataFrame  pyalData trial table. If it has no `bhv` column
                   yet, one is added from all known keypoints/angles before
                   dimension selection.
    on_keypoints : list[str]  keypoint fields to compute gait frequency/power for;
                   defaults to `Params.oscillating_keypoints`. Passed to
                   `get_keypoint_dim_indices`, which assumes these fields lay out
                   contiguously in `bhv`, in this order.
    feature_dims : "x" | "y" | "z" | np.ndarray
                   dimension to select per keypoint in `on_keypoints`, or an
                   explicit array of column indices into `bhv` (bypasses
                   `on_keypoints`/`get_keypoint_dim_indices` entirely).

    Returns
    -------
    pd.DataFrame  with added columns (plus `bhv` if it wasn't already present):
                  bhv_concat, concat_perturb_time, concat_trial_start, power,
                  phases, disturb_score
    """
    if "bhv" not in df.columns:
        print("bhv column not found adding it")
        df = dt.add_bhv(df, bhv_fields=["all"])

    if isinstance(feature_dims, str):
        feature_dims = get_keypoint_dim_indices(
            df, keypoints=on_keypoints, dim=feature_dims
        )
    df = dt.concat_previous_intertrial_signal(df, "bhv", features=feature_dims)
    df = dt.add_concat_perturb_time(df)
    df = dt.add_concat_trial_start(df)
    df = compute_power_in_bhv_concat_td(df)
    df = add_perturb_score_td(df)
    if drop_trials:
        df = df[df["power"].apply(lambda x: isinstance(x, np.ndarray))]

    return df
