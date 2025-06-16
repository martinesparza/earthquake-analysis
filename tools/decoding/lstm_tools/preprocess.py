from typing import Callable

import numpy as np
import pandas as pd
import pyaldata as pyal

import tools.dataTools as dt


def unroll_data(data, trial_length):
    """
    Unrolls a 2D array of shape (time * trials, 3) into shape (trials, time, 3).

    Parameters:
    - data: 2D numpy array of shape (time * trials, 3)
    - trial_length: the number of time steps per trial (default: 129)

    Returns:
    - unrolled_data: 3D numpy array of shape (trials, time, 3)
    """
    # Get the number of trials and time steps
    n_trial = data.shape[0] // trial_length  # Time per trial
    n_features = data.shape[1]  # Should be 3

    # Reshape the data into (trials, time, features)
    unrolled_data = data.reshape(n_trial, trial_length, n_features)

    # Transpose to get (trials, time, 3)
    # unrolled_data = unrolled_data.transpose(1, 0, 2)

    return unrolled_data


def create_sliding_windows(data, labels, len_window=20):
    """
    Create sliding windows of data for training, with causal windows and respecting discontinuities across trials.

    Parameters:
    - data: numpy array of shape (n_trials, n_time_, n_dims)
    - labels: numpy array of shape (n_trials, n_time, n_dims)
    - len_window: the size of the sliding window

    Returns:
    - X: reshaped data with shape (n_windows, len_window, n_dims)
    - y: reshaped labels with shape (n_windows, n_dims) corresponding to the label after each window
    """
    n_trials, n_time_, n_dims = data.shape
    X = []
    y = []

    for trial in range(n_trials):
        # Extract the current trial data and labels
        trial_data = data[trial]
        trial_labels = labels[trial]

        # Slide the window across the trial's time axis
        for t in range(n_time_ - len_window):
            # Extract the window of data (causal, so we use [t:t+len_window])
            window_data = trial_data[t : t + len_window]

            # The label is the value immediately after the window
            window_label = trial_labels[
                t + len_window
            ]  # Predict the timepoint right after the window

            # Append to the lists
            X.append(window_data)
            y.append(window_label)

    # Convert the lists into numpy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y


def baseline_norm_labels(train_labels: np.ndarray, test_labels: np.ndarray) -> tuple:
    """Baseline normalizing of data

    Args:
        train_labels (np.ndarray): training labels (keypoints)
        test_labels (np.ndarray): testing labels

    Returns:
        tuple: train and test baseline normalised labels
    """

    means = np.mean(
        train_labels.reshape(-1, train_labels.shape[-1]),
        axis=0,
    )
    train_labels = train_labels - means

    means = np.mean(
        test_labels.reshape(-1, test_labels.shape[-1]),
        axis=0,
    )
    test_labels = test_labels - means
    return train_labels, test_labels


def _get_trialdata_and_labels_from_df(
    df: pd.DataFrame,
    bhv: list,
    area: list,
    n_components: int,
    epoch: Callable[[pd.Series], slice],
    sigma: float,
) -> tuple:
    """Transform trial data into pc space and get behaviour

    Args:
        df (pd.DataFrame): Session data
        bhv (list): Behavioural outputs e.g., ['right_knee']
        area (list): Brain areas e.g., ['MOp', 'CP']
        n_components (int): PCA components to use
        epoch (Callable[[pd.Series], slice]): Epoch to restrict data
        sigma (float): sigma for smoothing

    Returns:
        tuple: data, labels
    """

    arr_data, arr_bhv = dt.get_data_array(
        data_list=[df],
        trial_cat="values_Sol_direction",
        epoch=epoch,
        area=area,
        bhv=bhv,
        n_components=n_components,
        sigma=sigma,
    )
    _, n_targets, n_trials, n_time, n_comp = arr_data.shape
    _, n_targets, n_trials, n_time, n_keypoints = arr_bhv.shape

    data = arr_data.reshape((n_targets * n_trials, n_time, n_comp))
    labels = arr_bhv.reshape((n_targets * n_trials, n_time, n_keypoints))

    return data, labels


def preprocess(df: pd.DataFrame, cfg: dict) -> tuple:
    """Preprocess data and prepare it for lstm training

    Args:
        df (pd.DataFrame): Session trial data
        cfg (dict): preprocessing config

    Returns:
        tuple: data, labels
    """

    # Get epoch
    if cfg["epoch"] is not None:
        epoch = pyal.generate_epoch_fun(
            start_point_name="idx_sol_on",
            rel_start=int(cfg["epoch"][0] / df.bin_size.values[0]),
            rel_end=int(cfg["epoch"][1] / df.bin_size.values[0]),
        )
    else:
        epoch = None

    # Parse trial condition
    if cfg["condition"] == "trial":
        df_trials = pyal.select_trials(df, df.trial_name == "trial")
        data, labels = _get_trialdata_and_labels_from_df(
            df=df_trials,
            bhv=cfg["bhv"],
            area=cfg["area"],
            n_components=cfg["n_input_dims"],
            epoch=epoch,
            sigma=cfg["sigma"],
        )
    else:
        raise ValueError(f'Condition: {cfg["condition"]} not implemented yet')

    if cfg["window_data"]:
        data, labels = create_sliding_windows(data, labels, len_window=cfg["len_window"])

    return data, labels
