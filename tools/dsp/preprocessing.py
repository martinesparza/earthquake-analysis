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
def _insert_nans_and_extend_to_spikes_shape_inplace(df, idx_col, value_col, ref_col):
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
    """
    Preprocessing steps to manipulate trial data structure

    Parameters
    ----------
    df : pd.DataFrame
        Trial data structure of a session

    Returns
    -------
    df : pd. DataFrame
        Trial data with operations performed

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
        signal for signal in pyal.get_time_varying_fields(df) if "spikes" in signal
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
        print(f"Resulting {signal} ephys data shape is (NxT): {df[signal][0].T.shape}")

    df["sol_level_id"] = [
        Params.sol_dir_to_level[dir_] if trial_name == "trial" else None
        for dir_, trial_name in zip(df["values_Sol_direction"], df["trial_name"])
    ]

    df["sol_contra_ipsi"] = [
        Params.sol_dir_to_contra_ipse[dir_] if trial_name == "trial" else None
        for dir_, trial_name in zip(df["values_Sol_direction"], df["trial_name"])
    ]

    return df


def load_and_process_session(
    session,
    data_dir="/data/bnd-data/raw/",
    bhv_fields=None,
    min_immobile_bins=5,
    rates=True,
    std=0.05,
):
    """
    Full pipeline for a session:
      1. Load raw pyalData
      2. Preprocess (remove low-firing neurons, keep 10 ms bins)
      3. Fit PCA on all rows except the first (free locomotion period)
      4. Compute perturbation metric (bhv_concat, Morlet power, disturb_score)
      5. Filter to perturbation trials, drop immobile, drop NaN scores
      6. Add disturb_mean / disturb_sum and sort by disturbance

    Returns
    -------
    df_tr      : pd.DataFrame  filtered, annotated trial table
    dstrb_idx  : np.ndarray    indices that sort df_tr ascending by disturbance
    """
    if bhv_fields is None:
        bhv_fields = Params.oscillating_key_points

    # 1 & 2 — load + preprocess
    df = pyal.load_pyaldata(data_dir + session[:4] + "/" + session)
    df = preprocess(df, only_trials=False, combine_time_bins=False, std=std)

    # 3 — PCA excluding the first row (free locomotion period)
    if rates:
        df = dt.add_pca_df(df.iloc[1:])
    else:  # spikes
        spike_fields = [col for col in df.columns if col.endswith("_spikes")]
        df = dt.add_pca_df(df.iloc[1:], pca_fields=spike_fields)
    # 4 — perturbation metric (requires behavioural data; skip gracefully if absent)
    has_perturbation_metric = True
    if has_perturbation_metric:
        try:
            df = kin.compute_perturbation_metric(df, bhv_fields=bhv_fields)
        except Exception as e:
            print(
                f"  WARNING: compute_perturbation_metric failed ({e}). "
                f"Skipping disturbance metric and trial dropping."
            )
            has_perturbation_metric = False

    # 5 — filter trials
    df_tr = pyal.select_trials(df, df.trial_name == "trial")

    if has_perturbation_metric:
        df_tr = kin.drop_immobile_trials(df_tr, min_immobile_bins=min_immobile_bins)
        mask = df_tr["disturb_score"].apply(
            lambda x: isinstance(x, np.ndarray) and not np.any(np.isnan(x))
        )
        df_tr = df_tr.loc[mask]

        # 6 — derived metrics + sort order
        df_tr["disturb_mean"] = df_tr["disturb_score"].apply(np.mean)
        df_tr["disturb_sum"] = df_tr["disturb_score"].apply(
            lambda a: np.log10(np.abs(np.sum(a)))
        )
        disturbances = np.sum(np.stack(df_tr.disturb_score.values), axis=1)
        dstrb_idx = np.argsort(disturbances)
    else:
        dstrb_idx = np.arange(len(df_tr))

    return df_tr, dstrb_idx


def drop_unperturbed_trials(
    df_tr: pd.DataFrame, dstrb_idx: np.ndarray, thresh_val: float = -2.0
) -> pd.DataFrame:
    """
    Keep only trials with a strong mechanical perturbation.

    A trial is kept if it satisfies BOTH:
      - its disturbance rank falls below the SEM-based threshold, AND
      - its disturb_mean exceeds thresh_val

    Parameters
    ----------
    df_tr      : trial DataFrame with a 'disturb_mean' column
    dstrb_idx  : argsort indices (ascending disturbance) from load_and_process_session
    thresh_val : hard lower bound on disturb_mean (default -2.0)

    Returns
    -------
    Filtered DataFrame.
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


def drop_trials_sem_crosses_zero(
    df_tr: pd.DataFrame,
    thresh_val: float = -2,
    field_sem="disturb_score",
    field_val="disturb_mean",
    std=False,
) -> pd.DataFrame:
    """
    Drop trials whose per-trial SEM (across keypoints) does not push the
    disturbance mean below zero, and optionally remove extreme outliers.

    For each trial, disturb_score is a (n_keypoints,) array.  We compute:
        mean = disturb_score.mean()
        sem  = disturb_score.std() / sqrt(n_keypoints)

    A trial is kept only when BOTH:
      - mean + sem < 0   (±1 SEM interval lies entirely below zero)
      - disturb_mean > thresh_val  (hard lower bound to remove outliers)

    Parameters
    ----------
    df_tr      : pd.DataFrame  trial table with a 'disturb_score' column.
    thresh_val : float         hard lower bound on disturb_mean (default -2.0).

    Returns
    -------
    Filtered DataFrame.
    """

    def _sem_crosses(arr):
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return False
        mean = arr.mean()
        sem = arr.std() / np.sqrt(arr.size)
        return (mean + sem) < 0

    def _std_crosses(arr):
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return False
        return (arr.mean() + arr.std()) < 0

    if not std:
        by_sem = df_tr[field_sem].apply(_sem_crosses)
    else:
        by_sem = df_tr[field_sem].apply(_std_crosses)
    by_value = df_tr[field_val] > thresh_val
    mask = by_sem & by_value

    n_total = len(df_tr)
    print(
        f"  drop_trials_sem_crosses_zero: "
        f"by SEM: {by_sem.sum()}  |  by value: {by_value.sum()}  |  "
        f"kept: {mask.sum()}/{n_total} ({100*mask.sum()/n_total:.1f}%)"
    )
    return df_tr.loc[mask]


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
    """
    Load, preprocess and slice a list of sessions into a ready-to-analyse dict.

    For each session:
      1. load_and_process_session  →  df_tr, dstrb_idx
      2. trial dropping            →  df_design
         - use_sem_dropping=False (default): drop_unperturbed_trials (rank + hard floor)
         - use_sem_dropping=True           : drop_trials_sem_crosses_zero (per-trial SEM + hard floor)
      3. pyal.restrict_to_interval →  perturb_td  (window: rel_start..rel_end bins)

    Parameters
    ----------
    sessions         : list of session strings, e.g. ['M061_2025_03_04_10_00', ...]
    data_dir         : path to raw data directory
    rel_start        : start bin relative to idx_sol_on (default -200 = -2 s at 10 ms)
    rel_end          : end bin relative to idx_sol_on   (default  300 = +3 s at 10 ms)
    thresh_val       : hard lower bound on disturb_mean passed to the dropping function
    use_sem_dropping : if True, use drop_trials_sem_crosses_zero instead of drop_unperturbed_trials

    Returns
    -------
    dict  {session: {'td': perturb_td} | None}
          None indicates the session failed to load.
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
