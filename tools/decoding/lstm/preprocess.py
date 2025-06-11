import numpy as np


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
