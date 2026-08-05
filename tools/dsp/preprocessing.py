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
        Params.sol_dir_to_contra_ipse[dir_] if trial_name == "trial" else None
        for dir_, trial_name in zip(
            df["values_Sol_direction"], df["trial_name"]
        )
    ]

    return df


def load_and_preprocess_trials_from_sess(
    session,
    data_dir="C:/data/raw/",
    bhv_fields=Params.oscillating_keypoints,
    std=0.05,
    run_pca=True,
    rates=True,
    exclude_first_free_from_pca=True,
    cv_thresh=0.225,
):
    """
    Full pipeline for a session (drop-before-compute ordering):
      1. Load raw pyalData
      2. Preprocess (remove low-firing neurons, keep 10 ms bins)
      3. Fit PCA on all rows except the first (free locomotion period); this
         also drops that first row from `df` for every step below
      4. Drop unsteady 'trial' rows (immobile or high-CV pre-perturbation
         running, via `kin.drop_unsteady_running_trials`) *before* computing
         the perturbation metric, so the Welch/Morlet power estimate isn't
         contaminated by unsteady running. Only runs if behavioural columns
         are present (checked via `"shoulder_center" in df.columns`);
         `intertrial`/`free` rows are left untouched.
      5. Compute the perturbation metric on the reduced table
         (`kin.compute_perturb_score` -> `power`, `phases`, `perturb_score`
         per trial); skipped, with a warning, if it raises or if no
         behavioural columns were found in step 4.
      6. Add `perturb_score_mean` / `perturb_score_log_sum` summary columns.

    Unlike `load_and_process_session`, this does not restrict the returned
    table to `trial_name == 'trial'` or sort by disturbance -- it returns the
    full table (all trial types, minus dropped unsteady trials) with the
    added perturbation-score columns.

    Returns
    -------
    df : pd.DataFrame  session table with PCA-reduced rates, unsteady trials
         dropped, and perturb_score / perturb_score_mean / perturb_score_log_sum
         columns added.
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
    has_bhv = False
    if "shoulder_center" in df.columns:
        has_bhv = True
        df = kin.drop_unsteady_running_trials(df, cv_thresh=cv_thresh)
    else:
        print(f"No behaviour found")

    # 4 — perturbation metric (requires behavioural data; skips if absent)
    if has_bhv:
        print(f"\n##### Calculating perturbation metric ######")
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

    df["perturb_score_mean"] = df["perturb_score"].apply(np.mean)
    df["perturb_score_log_sum"] = df["perturb_score"].apply(
        lambda a: np.log10(np.abs(np.sum(a)))
    )
    return df


def _drop_unperturbed_trials(
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


def drop_unperturbed_or_stopped_trials(
    trial_td: pd.DataFrame,
    thresh_val: float = -2,
    field_sem="perturb_score",
    field_val="perturb_score_mean",
    stat="sem",
) -> pd.DataFrame:
    """
    Drop trials that don't show a genuine perturbation response: either
    'unperturbed' (no confident post-perturbation power decrease) or
    'stopped' (an extreme power drop consistent with the animal halting
    outright, rather than a graded gait disruption).

    For each trial, `field_sem` (`perturb_score`) is a (n_keypoints,) array.
    We compute:
        mean = perturb_score.mean()
        sem  = perturb_score.std() / sqrt(n_keypoints)   (or std, via `stat`)

    A trial is kept only when BOTH:
      - mean + sem < 0                    ('unperturbed' check -- the
                                             ±1 SEM/STD interval lies entirely
                                             below zero, so the power drop
                                             isn't just noise)
      - perturb_score_mean > thresh_val   ('stopped' check -- a hard lower
                                             bound excluding extreme outliers
                                             more likely caused by the animal
                                             stopping outright than by a
                                             modulated gait response)

    Parameters
    ----------
    trial_td   : pd.DataFrame  trial table with `field_sem` / `field_val`
                 columns (defaults: 'perturb_score', 'perturb_score_mean').
    thresh_val : float          hard lower bound on `field_val` (default -2).
    field_sem  : str            per-keypoint array column used for the
                 SEM/STD check.
    field_val  : str            scalar column used for the hard-floor check.
    stat       : "sem" | "std"  which statistic pushes the mean below zero.

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
        f"dropped by {stat}: {(~by_sem).sum()}  |  dropped by value: {(~by_value).sum()}  |  "
        f"kept: {mask.sum()}/{n_total} ({100*mask.sum()/n_total:.1f}%)"
    )
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
