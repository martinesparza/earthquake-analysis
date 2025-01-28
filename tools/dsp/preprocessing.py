import numpy as np
import pandas as pd
import pyaldata as pyal

from tools.params import Params


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
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
    time_signals = [
        signal for signal in pyal.get_time_varying_fields(df) if "spikes" in signal
    ]

    # Remove low firing neurons
    for signal in time_signals:
        df = pyal.remove_low_firing_neurons(df, signal, 1)

    # Select trials
    df = pyal.select_trials(df, "trial_name == 'trial'")  # Remove baseline

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

    return df
