"""
General lstm utils
"""


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
