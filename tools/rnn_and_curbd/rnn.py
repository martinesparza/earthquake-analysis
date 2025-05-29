# imports
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/data/PyalData')
import pyaldata as pyal
import pylab

from tools.curbd import curbd

# import custom plotting functions
from tools.rnn_and_curbd import plotting as pltz

### Functions written for trial-avg rnn training ###

def average_by_trial(df, trial_categories):
    if not isinstance(df['all_rates'].iloc[0], np.ndarray):
        df['all_rates'] = df['all_rates'].apply(np.array)

    averaged_activity = []
    for cat in trial_categories:
        angle_group = df[df['values_Sol_direction'] == cat]['all_rates']

        angle_array = np.stack(angle_group.values)

        mean_activity = np.mean(angle_array, axis=0)

        averaged_activity.append(mean_activity)

    averaged_activity = np.array(averaged_activity)

    return averaged_activity

def get_reset_points(df, activity, areas, dtFactor):
    trial_len = df[areas[0]][0].shape[0]
    if all(df[col][0].shape[0] == trial_len for col in areas):
        print(f"Trial length: {trial_len}")
    else:
        print("Variable trial length!")

    reset_points = []
    for i in range(len(df)):
        reset_points.append(i * trial_len * dtFactor)  # alter for consideration of dtFactor

    return reset_points

def get_regions(df, brain_areas):
    num_neurons = [df[col][1].shape[1] for col in brain_areas]
    cumulative_sums = np.cumsum([0] + num_neurons[:-1])

    regions = [[col.split("_")[0], np.arange(start, start + num)] for col, start, num in
               zip(brain_areas, cumulative_sums, num_neurons)]
    regions = np.array(regions, dtype=object)

    return regions

def run_rnn(formated_rates, resets, regions_arr, data, mouse_num, **kwargs):
    rnn_model = train_rnn(formated_rates, resets, regions_arr, data.bin_size[0], **kwargs)

    figure = pltz.plot_model_accuracy(rnn_model, mouse_num)

    return rnn_model, figure

def train_rnn(activity, reset_points, regions, bin_size, **kwargs):
    params = {
        "dtFactor": 1,
        "tauRNN": 0.2,
        "ampInWN": 0.001,
        "g": 1.5,
        "nRunTrain": 200
    }
    params.update(kwargs)

    print(f"reset points length: {len(reset_points)}")
    print(f"last reset at: {max(reset_points)}")
    print(f"RNN input shape: {activity.shape}")

    model = curbd.trainMultiRegionRNN(
        activity,
        dtData=bin_size,
        regions=regions,
        resetPoints=reset_points,
        verbose=True,
        plotStatus=False,
        nRunFree=1,
        **params
    )
    return model

def combine_rnn_time_bins(rnn_model):

    rnn_output = rnn_model['RNN']
    dtData = rnn_model['dtData']
    tData = rnn_model['tData']
    tRNN = rnn_model['tRNN']

    # Set up array
    rnn_combine = np.zeros((rnn_output.shape[0], len(tData)))

    # For each time bin in tData, find corresponding RNN indices and average
    for i in range(len(tData)):
        t_start = tData[i]
        t_end = tData[i] + dtData
        idx = np.where((tRNN >= t_start) & (tRNN < t_end))[0]

        if len(idx) > 0:
            rnn_combine_bin = rnn_output[:, idx].mean(axis=1)
        else:
            rnn_combine_bin = np.zeros(rnn_output.shape[0])

        rnn_combine[:, i] = rnn_combine_bin
    print(
        f"RNN model output transformed from shape: {rnn_output.shape} to {rnn_combine.shape}. Matching the original data shape of {rnn_model['Adata'].shape}")
    return rnn_combine

def rescale_array(arr):
    """
    Rescales a NumPy array to the range [0, 1].

    Parameters:
        arr (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Rescaled array.
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    if arr_max == arr_min:
        return np.zeros_like(arr)  # Avoid division by zero if all values are the same

    return (arr - arr_min) / (arr_max - arr_min)

