# imports
import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from typing import Union, List
import sys
sys.path.append("/home/zms24/Desktop")
import PyalData.pyaldata as pyal
import pylab

sys.path.append("/home/zms24/Desktop/earthquake/earthquake-analysis/")
from tools.dsp.preprocessing import preprocess
from tools.curbd import curbd
from tools.rnn_and_curbd import plotting as pltz

### Functions written for trial-avg rnn training ###

def average_by_trial(df, trial_categories, trial_col_name='values_Sol_direction'):
    if not isinstance(df['all_rates'].iloc[0], np.ndarray):
        df['all_rates'] = df['all_rates'].apply(np.array)

    averaged_activity = []
    for cat in trial_categories:
        angle_group = df[df[trial_col_name] == cat]['all_rates']
        angle_array = np.stack(angle_group.values)
        mean_activity = np.mean(angle_array, axis=0)
        averaged_activity.append(mean_activity)

    try:
        averaged_activity_array = np.array(averaged_activity)
        return averaged_activity_array
    except ValueError:
        print("Warning: Averaged arrays have different shapes; returning a list instead.")
        return averaged_activity

    # averaged_activity = np.array(averaged_activity)

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

def get_intertrial_reset_points(df, activity, trial_types, areas, dtFactor):
    # check for length inconsistencies bewteen brain region within a row
    trial_len = df[areas[0]][0].shape[0]
    if all(df[col][0].shape[0] == trial_len for col in areas):
        print(f"Trial length: {trial_len}")
    else:
        print("Variable trial length!")

    # make reset points, but the trial length differ between trials

    reset_points = [0]
    for i in range(len(trial_types)):
        previous = reset_points[-1]
        next = previous + (trial_types[i] * dtFactor)
        reset_points.append(int(next)) 

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

def process_pyal_M044_files(pyal_files: Union[str, List[str]], rnn_model, root_dir: str = "/data/raw") -> dict:
    if isinstance(pyal_files, str):
        pyal_files = [pyal_files]

    dfs = []
    for pyal_file in pyal_files:
        # Extract subject ID and session ID
        subject_id = pyal_file.split("_")[0]
        session_id = "_".join(pyal_file.split("_")[1:-1])  # everything between subject and 'pyaldata.mat'
        
        # Construct full path: /data/raw/<subject>/<session>/<file>
        data_dir = os.path.join(root_dir, subject_id, f"{subject_id}_{session_id}")
        fname = os.path.join(data_dir, pyal_file)

        df = pyal.mat2dataframe(fname, shift_idx_fields=True)
        dfs.append(df)

    # Combine and preprocess - custom for M044 session
    df = pd.concat(dfs, ignore_index=True)
    df_ = preprocess(df, only_trials=True)
    areas = ["M1_rates", "Dls_rates"]
    df_["M1_rates"] = [df_["all_rates"][i][:,300:] for i in range(len(df_))]
    df_["Dls_rates"] = [df_["all_rates"][i][:,0:300] for i in range(len(df_))]

    # Metadata
    areas = [col for col in df_.columns if col.endswith("_rates") and col != "all_rates"]
    # perturbation time
    perturbation_time = df_.idx_sol_on[0]
    perturbation_time_sec = df_.idx_sol_on[0] * df_['bin_size'][0]
    # solenoids
    sol_angles = df_.values_Sol_direction.unique()
    sol_angles.sort()
    trial_labels = [f"solenoid {angle}" for angle in sol_angles]
    trial_avg_rates = average_by_trial(df_, sol_angles)
    shapes = [arr.shape[0] for arr in trial_avg_rates]

    # do trial avg
    trial_avg_rates = average_by_trial(df_, sol_angles)
    concat_rates = np.concatenate(trial_avg_rates, axis=0)
    trial_avg_activity = np.transpose(concat_rates)
    reset_points = get_reset_points(df_, trial_avg_activity, areas, rnn_model['params']['dtFactor'])
    regions_arr = get_regions(df_, areas)

    return {
        'df_': df_,
        'sol_angles': sol_angles,
        'trial_labels': trial_labels,
        'trial_avg_rates': trial_avg_rates,
        'shapes': shapes,
        'areas': areas,
        'concat_rates': concat_rates,
        'reset_points': reset_points,
        'regions_arr': regions_arr,
        'perturbation_time': perturbation_time,
        'perturbation_time_sec': perturbation_time_sec
    }

def process_pyal_M061_M062_files(pyal_files: Union[str, List[str]], rnn_model, root_dir: str = "/data/raw") -> dict:
    if isinstance(pyal_files, str):
        pyal_files = [pyal_files]

    dfs = []
    for pyal_file in pyal_files:
        # Extract subject ID and session ID (supporting session timestamps with 5 parts like "2025_03_04_10_00")
        parts = pyal_file.split("_")
        subject_id = parts[2]
        session_id = ("_".join(parts[3:])).split('.')[0]  # e.g., 2025_03_04_10_00

        # Construct the directory and use glob to find matching files
        data_dir = os.path.join(root_dir, subject_id, f"{subject_id}_{session_id}")
        file_pattern = os.path.join(data_dir, f"{subject_id}_{session_id}_pyaldata_*.mat")
        matched_files = glob.glob(file_pattern)

        if not matched_files:
            raise FileNotFoundError(f"No matching files found for pattern: {file_pattern}")

        for fname in matched_files:
            df = pyal.mat2dataframe(fname, shift_idx_fields=True)
            dfs.append(df)

    # Combine and preprocess - custom for M061 and M062
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop(columns="all_spikes") # the content is incorrect
    df_ = preprocess(df, only_trials=True)

    BIN_SIZE = df_['bin_size'][0]
    areas = [col for col in df_.columns if col.endswith("_rates") and col != "all_rates"]
    # areas = ["MOp_rates", "SSp_rates", "CP_rates", "VAL_rates"]
    df_ = pyal.merge_signals(df_, areas, "all_rates")

    # Correct trial length
    df_['trial_length'] = (df_['trial_length'] / (BIN_SIZE * 100)).astype(int)
    df_ = df_[df_['trial_length'] == 200]

    # Metadata
    areas = [col for col in df_.columns if col.endswith("_rates") and col != "all_rates"]
    perturbation_time = df_.idx_sol_on[0]
    perturbation_time_sec = perturbation_time * df_['bin_size'][0]
    sol_angles = df_.values_Sol_direction.unique()
    sol_angles.sort()
    trial_labels = [f"solenoid {int(angle)}" for angle in sol_angles]
    trial_avg_rates = average_by_trial(df_, sol_angles)
    shapes = [arr.shape[0] for arr in trial_avg_rates]

    concat_rates = np.concatenate(trial_avg_rates, axis=0)
    trial_avg_activity = np.transpose(concat_rates)
    reset_points = get_reset_points(df_, trial_avg_activity, areas, rnn_model['params']['dtFactor'])
    regions_arr = get_regions(df_, areas)

    return {
        'df_': df_,
        'sol_angles': sol_angles,
        'trial_labels': trial_labels,
        'trial_avg_rates': trial_avg_rates,
        'shapes': shapes,
        'areas': areas,
        'concat_rates': concat_rates,
        'reset_points': reset_points,
        'regions_arr': regions_arr,
        'perturbation_time': perturbation_time,
        'perturbation_time_sec': perturbation_time_sec,
        'bin_size': BIN_SIZE
    }

def restrict_time_interval(df, brain_areas, bin_start, length):
    interval_df = df.copy()
    interval_df[brain_areas] = df[brain_areas].apply(lambda row: row.apply(lambda arr: arr[bin_start:bin_start+length, :]))
    interval_df['trial_length'] = length
    return interval_df

