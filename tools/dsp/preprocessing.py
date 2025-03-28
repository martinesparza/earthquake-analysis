import numpy as np
import pandas as pd
import pyaldata as pyal

from tools.params import Params


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
    assert np.all(df.bin_size == 0.01), "bin size is not consistent!"
    df = pyal.combine_time_bins(df, int(Params.BIN_SIZE / 0.01))
    print(f"Combined every {int(Params.BIN_SIZE / 0.01)} bins")

    # Sqrt transformation for homoscedasticity
    for signal in time_signals:
        df = pyal.sqrt_transform_signal(df, signal)

    # Transformation into firing rates
    df = pyal.add_firing_rates(df, "smooth", std=0.05)
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
