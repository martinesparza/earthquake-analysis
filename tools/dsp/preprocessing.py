import traceback

import numpy as np
import pandas as pd
import scipy.stats
import pyaldata as pyal
from tqdm import tqdm

from tools.params import Params
import tools.dataTools as dt
import tools.kinematics as kin


# Define the function
def _insert_nans_and_extend_to_spikes_shape_inplace(
    df, idx_col, value_col, ref_col
):
    """Fill dropped samples of a field with NaN and pad it to ``ref_col``
    length, in place.

    Parameters
    ----------
    df : pd.DataFrame
        Trial data; ``idx_col`` and ``value_col`` are overwritten.
    idx_col : str
        Column of sample-index vectors.
    value_col : str
        Column of values aligned to ``idx_col``.
    ref_col : str
        Column whose first dimension sets the padded length.

    Returns
    -------
    pd.DataFrame
        Always empty; the result is the in-place edit of ``df``.
    """
    new_rows = []

    for i, row in df.iterrows():
        idx_seq = list(row[idx_col])
        val_seq = list(row[value_col])
        spike_len = row[ref_col].shape[0]

        # Fill missing indices in the existing idx sequence
        full_idx = list(range(idx_seq[0], idx_seq[-1] + 1))
        full_vals = []

        idx_pointer = 0
        for j in full_idx:
            if idx_pointer < len(idx_seq) and idx_seq[idx_pointer] == j:
                full_vals.append(val_seq[idx_pointer])
                idx_pointer += 1
            else:
                full_vals.append(np.nan)
                print(
                    f"Missing index {j} in trial: {df.trial_name[i]} and id: {df.trial_id[i]}, inserting NaN."
                )

        # Extend idx to match spikes length
        if full_idx[-1] + 1 < spike_len:
            for j in range(full_idx[-1] + 1, spike_len):
                full_idx.append(j)
                full_vals.append(np.nan)
                print(
                    f"Extending index to {j} in trial: {df.trial_name[i]} and id: {df.trial_id[i]}, inserting NaN."
                )

        # Update the DataFrame in-place
        df.at[i, idx_col] = np.array(full_idx)
        df.at[i, value_col] = np.array(full_vals)

    return pd.DataFrame(new_rows)


def preprocess(
    df: pd.DataFrame,
    only_trials: bool = True,
    trial_selection_criteria: None | list = None,
    repair_time_varying_fields: None | list = None,
    combine_time_bins=True,
    std=0.05,
) -> pd.DataFrame:
    """Drop low-firing neurons, select and bin trials, and turn spikes into
    smoothed rates.

    Parameters
    ----------
    df : pd.DataFrame
        Session trial data; repaired fields are edited in place.
    only_trials : bool, optional
        Keep only 'trial' rows (default is True).
    trial_selection_criteria : list of str, optional
        Extra ``pyal.select_trials`` queries (default is None).
    repair_time_varying_fields : list of str, optional
        Field names without ``idx_``/``values_`` prefix (default is None).
    combine_time_bins : bool, optional
        Combine 10 ms bins to ``Params.BIN_SIZE`` (default is True).
    std : float, optional
        Smoothing kernel std in seconds (default is 0.05).

    Returns
    -------
    pd.DataFrame
        Table with rates, ``sol_level_id`` and ``sol_contra_ipsi``.

    Raises
    ------
    AssertionError
        If ``combine_time_bins`` and any ``bin_size`` is not 0.01.
    """
    spikes_columns = [col for col in df.columns if col.endswith("spikes")]
    if repair_time_varying_fields is not None:
        print(f"Repairing columns {repair_time_varying_fields}")
        for time_varying_field_to_repair in repair_time_varying_fields:
            _insert_nans_and_extend_to_spikes_shape_inplace(
                df,
                idx_col=f"idx_{time_varying_field_to_repair}",
                value_col=f"values_{time_varying_field_to_repair}",
                ref_col=f"{spikes_columns[0]}",
            )

    time_signals = [
        signal
        for signal in pyal.get_time_varying_fields(df)
        if "spikes" in signal
    ]
    print(time_signals)

    # Remove low firing neurons
    for signal in time_signals:
        df = pyal.remove_low_firing_neurons(df, signal, 1)

    # Select trials
    if only_trials:
        df = pyal.select_trials(df, "trial_name == 'trial'")

    if trial_selection_criteria is not None:
        for condition in trial_selection_criteria:
            df = pyal.select_trials(df, condition)

    # Combine time bins
    if combine_time_bins:
        assert np.all(df.bin_size == 0.01), "bin size is not consistent!"
        df = pyal.combine_time_bins(df, int(Params.BIN_SIZE / 0.01))
        # df = pyal.combine_time_bins(df, int(02.0 / 0.01))
        print(f"Combined every {int(Params.BIN_SIZE / 0.01)} bins")

    # Sqrt transformation for homoscedasticity
    for signal in time_signals:
        df = pyal.sqrt_transform_signal(df, signal)

    # Transformation into firing rates
    df = pyal.add_firing_rates(df, "smooth", std=std)  # 0.05
    for signal in time_signals:
        print(
            f"Resulting {signal} ephys data shape is (NxT): {df[signal][0].T.shape}"
        )

    df["sol_level_id"] = [
        Params.sol_dir_to_level[dir_] if trial_name == "trial" else None
        for dir_, trial_name in zip(
            df["values_Sol_direction"], df["trial_name"]
        )
    ]

    df["sol_contra_ipsi"] = [
        Params.sol_dir_to_contra_ipsi[dir_] if trial_name == "trial" else None
        for dir_, trial_name in zip(
            df["values_Sol_direction"], df["trial_name"]
        )
    ]

    return df


def load_and_preprocess_trials_from_sess(
    session,
    data_dir="/mnt/mouse-data-london/proc/",
    bhv_fields=Params.oscillating_keypoints,
    std=0.05,
    run_pca=True,
    rates=True,
    exclude_first_free_from_pca=True,
    cv_thresh=0.255,
):
    """Load a session, run PCA, drop unsteady trials, then add perturbation
    scores (skipped on failure).

    Parameters
    ----------
    session : str
        Session name, e.g. "M061_2025_03_04_10_00".
    data_dir : str, optional
        Data root (default is "/mnt/mouse-data-london/proc/").
    bhv_fields : list of str, optional
        Keypoints for ``bhv`` and the score (default is oscillating keypoints).
    std : float, optional
        Smoothing kernel std in seconds (default is 0.05).
    run_pca : bool, optional
        Add PCA columns (default is True).
    rates : bool, optional
        PCA on rates if True, else on spikes (default is True).
    exclude_first_free_from_pca : bool, optional
        Drop the first free-locomotion row before PCA (default is True).
    cv_thresh : float, optional
        Max running-speed CV for a kept trial (default is 0.255).

    Returns
    -------
    pd.DataFrame
        All row types, with PCA and ``perturb_score*`` columns when computed.
    """

    # 1 & 2 — load + preprocess
    print(f"\n##### Loading and preprocessing ######")
    df = pyal.load_pyaldata(data_dir + session[:4] + "/" + session)
    df = preprocess(df, only_trials=False, combine_time_bins=False, std=std)

    # 3 — pca excluding the first row (free locomotion period)
    if run_pca:
        print(f"\n##### Running PCA ######")
        if rates:
            if exclude_first_free_from_pca:
                df = dt.add_pca_df(df.iloc[1:])
            else:
                df = dt.add_pca_df(df)
        else:  # spikes
            spike_fields = [
                col for col in df.columns if col.endswith("_spikes")
            ]
            if exclude_first_free_from_pca:
                df = dt.add_pca_df(df.iloc[1:], pca_fields=spike_fields)
            else:
                df = dt.add_pca_df(df, pca_fields=spike_fields)

    # 3 - drop immobile trials
    print(f"\n##### Dropping unsteady running trials ######")
    has_bhv = False
    if "shoulder_center" in df.columns:
        has_bhv = True
        df = kin.drop_unsteady_running_trials(df, cv_thresh=cv_thresh)
    else:
        print(f"No behaviour found")

    # 4 — perturbation metric (requires behavioural data; skips if absent)
    if has_bhv:
        print(f"\n##### Calculating perturbation metric ######")
        # drop_unsteady_running_trials (step 3) always builds `bhv` from
        # Params.oscillating_keypoints if it wasn't already present -- that's
        # deliberate for its own immobility check, but it means `bhv` may not
        # match this function's own `bhv_fields`. Rebuild it here so
        # compute_perturb_score's `on_keypoints` offsets are computed against
        # a `bhv` that actually has that layout, regardless of what step 3 left.
        df = dt.add_bhv(df, bhv_fields=bhv_fields)
        try:
            df = kin.compute_perturb_score(
                df,
                on_keypoints=bhv_fields,
                feature_dims="z",
            )
        except Exception as e:
            print(
                f"  WARNING: compute_perturb_score failed ({e}). "
                f"Skipping disturbance metric and trial dropping."
            )

    if "perturb_score" in df.columns:
        df["perturb_score_mean"] = df["perturb_score"].apply(np.mean)
        df["perturb_score_log_sum"] = df["perturb_score"].apply(
            lambda a: np.log10(np.abs(np.sum(a)))
        )
    return df


def _drop_unperturbed_trials(
    df_tr: pd.DataFrame, dstrb_idx: np.ndarray, thresh_val: float = -2.0
) -> pd.DataFrame:
    """Keep trials with ``disturb_mean`` below minus its SEM and above
    ``thresh_val``.

    Parameters
    ----------
    df_tr : pd.DataFrame
        Trial table with a ``disturb_mean`` column.
    dstrb_idx : np.ndarray
        Ascending argsort of ``disturb_mean``.
    thresh_val : float, optional
        Hard lower bound on ``disturb_mean`` (default is -2.0).

    Returns
    -------
    pd.DataFrame
        Kept rows of ``df_tr``.
    """
    disturb_vals = df_tr["disturb_mean"].values
    threshold = -scipy.stats.sem(disturb_vals)

    sorted_vals = disturb_vals[dstrb_idx]
    n_below_threshold = int(np.searchsorted(sorted_vals, threshold))

    by_rank = np.zeros(len(df_tr), dtype=bool)
    by_rank[dstrb_idx[:n_below_threshold]] = True
    by_value = disturb_vals > thresh_val

    mask = by_rank & by_value
    print(
        f"  threshold = {threshold:.3f}  |  "
        f"trials by rank: {n_below_threshold}  |  "
        f"trials kept: {mask.sum()}"
    )
    return df_tr.iloc[np.where(mask)[0]]


def drop_unperturbed_or_stopped_trials(
    trial_td: pd.DataFrame,
    thresh_val: float = -2,
    field_sem="perturb_score",
    field_val="perturb_score_mean",
    stat="sem",
    return_count=False,
) -> pd.DataFrame:
    """Keep trials whose score mean plus SEM/STD is below zero and whose mean
    exceeds ``thresh_val``.

    Parameters
    ----------
    trial_td : pd.DataFrame
        Trial table with ``field_sem`` and ``field_val`` columns.
    thresh_val : float, optional
        Floor on ``field_val``, dropping stopped trials (default is -2).
    field_sem : str, optional
        Per-keypoint array column (default is "perturb_score").
    field_val : str, optional
        Scalar column for the floor (default is "perturb_score_mean").
    stat : {"sem", "std"}, optional
        Spread added to the mean (default is "sem").
    return_count : bool, optional
        Also return the two boolean masks (default is False).

    Returns
    -------
    pd.DataFrame
        Kept rows of ``trial_td``.
    by_sem : pd.Series of bool
        Per-trial SEM/STD check; only if ``return_count``.
    by_value : pd.Series of bool
        Per-trial floor check; only if ``return_count``.

    Raises
    ------
    ValueError
        If ``stat`` is not "sem" or "std".
    """

    def _sem_crosses(arr):
        """Return True if ``mean + SEM`` of a non-empty array is below zero."""
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return False
        mean = arr.mean()
        sem = arr.std() / np.sqrt(arr.size)
        return (mean + sem) < 0

    def _std_crosses(arr):
        """Return True if ``mean + STD`` of a non-empty array is below zero."""
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return False
        return (arr.mean() + arr.std()) < 0

    if stat == "sem":
        by_sem = trial_td[field_sem].apply(_sem_crosses)
    elif stat == "std":
        by_sem = trial_td[field_sem].apply(_std_crosses)
    else:
        raise ValueError(
            "Please use either 'sem' or 'std' as dropping statistics"
        )

    by_value = trial_td[field_val] > thresh_val
    mask = by_sem & by_value

    n_total = len(trial_td)
    print(
        f"  drop_unperturbed_trials: "
        f"dropped by stat={stat}: {(~by_sem).sum()}  |  dropped by value: {(~by_value).sum()}  |  "
        f"kept: {mask.sum()}/{n_total} ({100*mask.sum()/n_total:.1f}%)"
    )
    if return_count:
        return trial_td.loc[mask], by_sem, by_value
    return trial_td.loc[mask]


def load_sessions_for_trial_analyses(
    sessions: list[str],
    data_dir: str = "/data/bnd-data/raw/",
    rel_start: int = -200,
    rel_end: int = 300,
    thresh_val: float = -2,
    use_sem_dropping: bool = True,
    rates=True,
    std=0.05,
    std_dropping=True,
) -> dict[str, dict | None]:
    """Load and window sessions around ``idx_sol_on``; returns None for all, as
    it calls undefined functions.

    Parameters
    ----------
    sessions : list of str
        Session names.
    data_dir : str, optional
        Data root (default is "/data/bnd-data/raw/").
    rel_start : int, optional
        Window start in bins from ``idx_sol_on`` (default is -200).
    rel_end : int, optional
        Window end in bins from ``idx_sol_on`` (default is 300).
    thresh_val : float, optional
        Hard floor on ``disturb_mean`` (default is -2).
    use_sem_dropping : bool, optional
        SEM-based instead of rank-based dropping (default is True).
    rates : bool, optional
        Passed to the session loader (default is True).
    std : float, optional
        Smoothing kernel std in seconds (default is 0.05).
    std_dropping : bool, optional
        Passed as ``std=`` to the SEM dropper (default is True).

    Returns
    -------
    dict of {str : dict or None}
        Session to ``{"td": windowed table}``, or None on failure.
    """

    results = {}

    for sess in tqdm(sessions, desc="Loading sessions"):
        print(f"\n{'='*60}\n{sess}\n{'='*60}")
        try:
            df_tr, dstrb_idx = load_and_process_session(
                sess, data_dir=data_dir, rates=rates, std=std
            )
            if "disturb_mean" in df_tr.columns:
                print(f"Thresh val dropping {thresh_val}")
                if use_sem_dropping:
                    df_design = drop_trials_sem_crosses_zero(
                        df_tr, thresh_val=thresh_val, std=std_dropping
                    )
                else:
                    df_design = drop_unperturbed_trials(
                        df_tr, dstrb_idx, thresh_val=thresh_val
                    )
            else:
                print("  No disturb_mean — skipping trial dropping.")
                df_design = df_tr

            perturb_td = pyal.restrict_to_interval(
                df_design,
                start_point_name="idx_sol_on",
                rel_start=rel_start,
                rel_end=rel_end,
            )

            results[sess] = {"td": perturb_td}

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results[sess] = None

    n_ok = sum(v is not None for v in results.values())
    print(f"\nLoaded {n_ok}/{len(sessions)} sessions successfully.")
    return results
