import numpy as np
import pandas as pd
import pyaldata as pyal
from params import Params


def preprocess(df: pd.DataFrame):
    """
    Preprocessing steps to manipulate trial data structure

    Parameters
    ----------
    df : pd.DataFrame
        Trial data structure of a session

    Returns
    -------
    df_ : pd. DataFrame
        Trial data with operations performed

    """
    time_signals = [
        signal for signal in pyal.get_time_varying_fields(df) if "spikes" in signal
    ]

    # Remove low firing neurons
    for signal in time_signals:
        df_ = pyal.remove_low_firing_neurons(df, signal, 1)

    # Select trials
    df_ = pyal.select_trials(df_, "trial_name == 'trial'")  # Remove baseline

    # Combine time bins
    assert np.all(df_.bin_size == 0.01), "bin size is not consistent!"
    df_ = pyal.combine_time_bins(df_, int(Params.BIN_SIZE / 0.01))

    # Sqrt transformation for homoscedasticity
    for signal in time_signals:
        df_ = pyal.sqrt_transform_signal(df_, signal)

    # Transformation into firing rates
    df_ = pyal.add_firing_rates(df_, "smooth", std=0.05)

    return df_
