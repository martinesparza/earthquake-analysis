import math

import numpy as np
import matplotlib.pyplot as plt

# Add the project root to sys.path
import sys
sys.path.append('/data/PyalData')
import pyaldata

import pylab
import math
import os
import csv
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from IPython.core.pylabtools import figsize
from numpy.core.records import ndarray

from tools.curbd import curbd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

# core RNN and CURBD functions
def before_RNN(data, areas, categories, num_time_bins, dtFactor, printing=True):
    # preprocessing
    data_ = preprocessing_RNN(data, areas, num_time_bins)

    # getting trial avg data
    trial_avg_rates = average_by_trial(data_, categories)
    shapes = [arr.shape[0] for arr in trial_avg_rates]

    # concatenate data
    concat_rates = np.concatenate(trial_avg_rates, axis=0)

    if printing:
        print("Averaged activity shape:", trial_avg_rates.shape)
        print("Concatenated average activity shape:", concat_rates.shape)

    # set up for RNN
    formated_rates = np.transpose(concat_rates)
    resets = get_reset_points(data_, formated_rates, areas, dtFactor)
    regions_arr = get_regions(data_, areas)

    if print:
        print(f"Building {len(regions_arr)} region RNN network")
        print(f"Regions: {[region[0] for region in regions_arr]}\n")

    return data_, concat_rates, formated_rates, regions_arr, resets, shapes

def RNN(formated_rates, resets, regions_arr, data, mouse_num, **kwargs):
    rnn_model = train_RNN(formated_rates, resets, regions_arr, data.bin_size[0], **kwargs)

    figure = plot_model_accuracy(rnn_model, mouse_num)

    return rnn_model, figure

def PCA_and_CCA(concat_rates, rnn_model, num_components, arr_shapes, mouse_num, printing=True):
    data_rnn = rnn_model['RNN'].T
    data_real = rescale_array(concat_rates)

    # PCA
    pca_real, pca_data_real = PCA_fit_transform(data_real, num_components)
    pca_rnn, pca_data_rnn = PCA_fit_transform(data_rnn, num_components)

    variance_figure = plot_PCA_cum_var(pca_real, pca_rnn, mouse_num)
    PCA_figure = plot_PCA(pca_data_real, pca_data_rnn, arr_shapes, mouse_num)

    # CCA
    canonical_values, scores = CCA_compare(pca_data_real, pca_data_rnn, num_components)
    canonical_values = np.array(canonical_values)

    if printing:
        print(f"CCA score of real data and RNN data aligment: {scores[0]}")
        print(f"CCA score for control on real data: {scores[1]}")
        print(f"CCA score for control on rnn data: {scores[2]}")

    CCA_figure = plot_CCA(canonical_values[:, 0], canonical_values[:, 1],
                          ['Real & RNN data', 'Control - Real data', 'Control - RNN data'], num_components, mouse_num)

    return scores, variance_figure, PCA_figure, CCA_figure

def CURBD(rnn_model, categories, event_time, mouse_num, normalization_method):
    curbd_arr, curbd_labels = curbd.computeCURBD(rnn_model)
    n_regions = curbd_arr.shape[0]

    trial_figure = plot_trial_currents(rnn_model, curbd_arr, curbd_labels, n_regions, categories, event_time, mouse_num,
                                       normalize=normalization_method)

    solenoid_level_figure = plot_sol_levels(rnn_model, curbd_arr, curbd_labels, n_regions, categories, event_time,
                                            mouse_num, normalize=normalization_method)

    currents_figure = plot_all_currents(rnn_model, curbd_arr, curbd_labels, n_regions, categories, event_time,
                                        mouse_num, normalize=normalization_method)

    return trial_figure, solenoid_level_figure, currents_figure

# helper functions
def preprocessing_RNN(df, brain_areas, num_time_bins):
    df_ = pyaldata.combine_time_bins(df, num_time_bins)

    for col in brain_areas:
        df_ = pyaldata.remove_low_firing_neurons(df_, col, 1)
    for col in brain_areas:
        df_ = pyaldata.transform_signal(df_, col, 'sqrt')

    df_ = pyaldata.merge_signals(df_, brain_areas, "all_spikes")

    df_ = pyaldata.add_firing_rates(df_, 'smooth')

    return df_

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

def train_RNN(activity, reset_points, regions, bin_size, **kwargs):
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

def PCA_fit_transform(data, n_components):
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_

    return pca, pca_data

def CCA_compare(data_real, data_rnn, num_comp):

    # CCA between real activity and simulated
    cca = CCA(n_components=num_comp, tol=1e-4, max_iter=2000)

    cca.fit(data_real, data_rnn)
    data_c, rnn_c = cca.transform(data_real, data_rnn)

    score = cca.score(data_real, data_rnn)

    # control CCA
    cca_ctrl_real = CCA(n_components=num_comp)
    cca_ctrl_real.fit(data_real, data_real)
    data_c_ctrl_real, rnn_c_ctrl_real = cca_ctrl_real.transform(data_real, data_real)

    cca_ctrl_rnn = CCA(n_components=num_comp)
    cca_ctrl_rnn.fit(data_rnn, data_rnn)
    data_c_ctrl_rnn, rnn_c_ctrl_rnn = cca_ctrl_rnn.transform(data_rnn, data_rnn)

    score_ctrl_real = cca_ctrl_real.score(data_real, data_real)
    score_ctrl_rnn = cca_ctrl_rnn.score(data_rnn, data_rnn)

    canonical_values = [[data_c, rnn_c], [data_c_ctrl_real, rnn_c_ctrl_real], [data_c_ctrl_rnn, rnn_c_ctrl_rnn]]
    scores = [score, score_ctrl_real, score_ctrl_rnn]

    return canonical_values, scores

def PCA_by_region(data, regions):
    num_regions = len(regions)
    PCA_data = []
    pcas = []
    for r in range(num_regions):
        # select region data
        neurons = len(regions[r][1])
        first_idx = regions[r][1][0]
        last_idx = regions[r][1][-1]
        region_data = data[:, first_idx:last_idx, ]
        # PCA and save
        pca = PCA(n_components=neurons - 1)
        PCA_data.append(pca.fit_transform(region_data))
        pcas.append(pca)

    return PCA_data, pcas

def get_last_row_number(filename):

    if not os.path.exists(filename):
        print(f"File '{filename}' does not exist.")
        return 0

    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        last_row = None
        for last_row in reader:
            pass
        print(last_row)
        if last_row:
            return int(last_row['model_idx'])
        else:
            return 0

def write_results_csv(model, scores, filename):
    model_idx = get_last_row_number(filename) + 1

    model_results = {"model_idx": model_idx,
                     "dtFactor": model['params']['dtFactor'],
                     "binSize": model['dtData'],
                     "tauRNN": model['params']['tauRNN'],
                     "ampInWN": model['params']['ampInWN'],
                     "pVar": model['pVars'][-1],
                     "chi2": model['chi2s'][-1],
                     "CCA_score": scores[0]}

    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['model_idx', 'dtFactor', 'binSize', 'tauRNN', 'ampInWN', 'pVar', 'chi2', 'CCA_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if file does not exist (so it's added just once)
        if not file_exists:
            writer.writeheader()

        writer.writerow(model_results)

    return model_idx

def save_plot(figures, filenames, folder_name, base_dir):
    """
    Saves the given figure to a new folder with a unique name for each run.

    Parameters:
        figures list of (matplotlib.figure.Figure): The figure to save.
        folder_name (str): Name of the folder to be created
        base_dir (str): The base directory to store all plot folders.

    Returns:
        str: The path where the figure was saved.
    """
    run_folder = os.path.join(base_dir, folder_name)
    os.makedirs(run_folder, exist_ok=True)
    file_paths = []

    for i in range(len(figures)):
        fig = figures[i]
        file_path = os.path.join(run_folder, f"{filenames[i]}.png")
        file_paths.append(file_path)
        fig.savefig(file_path, dpi=300)
        plt.close(fig)

    return file_paths

def save_results(model, scores, csv_filename, data_dir, figures, plot_filenames, printing=True):
    model_idx = write_results_csv(model, scores, csv_filename)

    graph_dir = os.path.join(data_dir, "RNN_graphs/")
    folder_name = f'RNN_figures_model_idx_{model_idx}'

    file_path = save_plot(figures, plot_filenames, folder_name, base_dir=graph_dir)
    if printing:
        print(file_path)


# plotting functions
def plot_neuron_activity(data, title, mouse_num):
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(data.T, cmap='viridis', shading='auto')
    plt.colorbar(label='Activity Level')
    plt.title(title + " - mouse " + mouse_num)
    plt.xlabel('Time Points')
    plt.ylabel('Neurons')
    plt.show()

def plot_model_accuracy(model, mouse_num):
    fig = pylab.figure(figsize=[6, 3])

    axn = fig.add_subplot(1, 2, 1)
    axn.plot(model['pVars'])
    axn.set_ylabel("pVar", fontsize=16)
    axn.set_xlabel("Iterations", fontsize=16)
    axn.set_ylim(0, 1)

    axn = fig.add_subplot(1, 2, 2)
    axn.plot(model['chi2s'])
    axn.set_ylabel("chi^2", fontsize=16)
    axn.set_xlabel("Iterations", fontsize=16)
    axn.set_ylim(0, max(model['chi2s']))
    fig.tight_layout()
    fig.suptitle(f"RNN Model accurancy - mouse {mouse_num}")

    return fig

def plot_PCA_cum_var(pca_real, pca_rnn, mouse_num):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca_real.explained_variance_ratio_), label='real activity')
    plt.plot(np.cumsum(pca_rnn.explained_variance_ratio_), label='RNN activity')

    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Cumulative Variance Explained by PCA - mouse {mouse_num}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return fig

    def plot_3PCs(fig, data, subplot_num, title):
        ax1 = fig.add_subplot(subplot_num, projection='3d', aspect='equal')
        for trial in range(data.shape[0]):
            ax1.plot(data[trial, :, 0],
                     data[trial, :, 1],
                     data[trial, :, 2], label=f'Trial {trial + 1}')
        ax1.set_title(title)
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('PC3')
        ax1.legend(loc='upper left')

def plot_3PCs(fig, data, subplot_num, title):
    ax1 = fig.add_subplot(subplot_num, projection='3d', aspect='equal')
    for trial in range(data.shape[0]):
        ax1.plot(data[trial, :, 0],
                 data[trial, :, 1],
                 data[trial, :, 2], label=f'Trial {trial + 1}')
    ax1.set_title(title)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.legend(loc='upper left')

def plot_PCA(real_data, rnn_data, original_shapes, mouse_num):
    reconstructed_data_avg = np.split(real_data, np.cumsum(original_shapes)[:-1])
    reconstructed_rnn = np.split(rnn_data, np.cumsum(original_shapes)[:-1])
    reconstructed_data_avg = np.array(reconstructed_data_avg)
    reconstructed_rnn = np.array(reconstructed_rnn)

    fig = plt.figure(figsize=(20, 10))

    plot_3PCs(fig, reconstructed_data_avg, 121, 'Recorded data')
    plot_3PCs(fig, reconstructed_rnn, 122, 'RNN simulated data')
    plt.suptitle(f'First 3PCs plot comparison - mouse {mouse_num}', fontsize=16)
    plt.tight_layout()
    plt.show()
    return fig

def plot_CCA(data_cs, rnn_cs, labels, num_comp, mouse_num):
    fig = plt.figure(figsize=(6, 4))
    for data_c, rnn_c, label in zip(data_cs, rnn_cs, labels):
        corrs = [np.corrcoef(data_c[:, i], rnn_c[:, i])[0, 1] for i in range(num_comp)]
        plt.plot(corrs, label=label)
    plt.xlabel('canonical coefficient index')
    plt.ylabel('CCA correlation')
    plt.title(f"CCA analysis results - mouse {mouse_num}")
    plt.legend()
    plt.show()

    return fig

def plot_PCs_by_region(real_data, rnn_data, original_shapes, regions, mouse_num):
    num_plots = len(regions)
    labels = regions[:, 0]
    re_real_data = []
    re_rnn_data = []

    for r in range(len(regions)):
        re_real = np.split(real_data[r], np.cumsum(original_shapes)[:-1])
        re_rnn = np.split(rnn_data[r], np.cumsum(original_shapes)[:-1])
        re_real = np.array(re_real)
        re_rnn = np.array(re_rnn)
        mean_real = np.mean(re_real, axis=0)
        mean_rnn = np.mean(re_rnn, axis=0)
        re_real_data.append(mean_real)
        re_rnn_data.append(mean_rnn)

    fig = plt.figure(figsize=(16, 6))
    for r in range(num_plots):
        axn = fig.add_subplot(1, num_plots, r + 1)
        axn.plot(re_real_data[r][:, 0], re_real_data[r][:, 1], label='experimental data', linewidth=3)
        axn.plot(re_rnn_data[r][:, 0], re_rnn_data[r][:, 1], label='RNN model data', linestyle='--', linewidth=3)
        axn.set_xlabel('PC1')
        axn.set_ylabel('PC2')
        axn.legend(fontsize=10, loc='upper left')
        axn.set_title(f"{labels[r]} activity", fontsize='xx-large')
    fig.suptitle(f"PCA of Model vs Experimental Trial Averaged Data - mouse {mouse_num}", fontsize=16)
    fig.tight_layout()
    return fig

def extract_trials(data, reset_points, trial_len):
    max_len = data.shape[1]
    reset_points = [point for point in reset_points if point <= max_len]
    num_neurons = data.shape[0]
    all_data = [[] for _ in range(num_neurons)]

    # split for each segment
    for neuron in range(num_neurons):
        arr = data[neuron]
        for p in range(len(reset_points)):
            point = reset_points[p]
            if point != 0:
                all_data[neuron].append(arr[reset_points[p - 1]:point])

    # split for trials and inter-trials
    trial_arr = []
    inter_trial_arr = []
    for neuron in all_data:
        neuron_trial = []
        neuron_inter = []
        for trial in neuron:
            if len(trial) == trial_len:
                neuron_trial.append(trial)
            else:
                neuron_inter.append(trial)
        trial_arr.append(np.array(neuron_trial))
        inter_trial_arr.append(neuron_inter)
    trial_arr = np.array(trial_arr)
    return all_data, trial_arr, inter_trial_arr

def plot_PCs_by_region_single(real_data, rnn_data, reset_points, regions, trial_len, mouse_num):
    num_plots = len(regions)
    labels = regions[:, 0]
    re_real_data = []
    re_rnn_data = []
    for r in range(len(regions)):
        all_data, trial_data, _ = extract_trials(real_data[r].T, reset_points, trial_len)
        all_rnn, trial_rnn, _ = extract_trials(rnn_data[r].T, reset_points, trial_len)
        re_real = np.array(trial_data)
        re_rnn = np.array(trial_rnn)
        print(re_real.shape, re_rnn.shape)
        mean_real = np.mean(re_real, axis=1)
        mean_rnn = np.mean(re_rnn, axis=1)
        print(mean_real.shape, mean_rnn.shape)
        re_real_data.append(mean_real)
        re_rnn_data.append(mean_rnn)

    fig = plt.figure(figsize=(12, 6))
    for r in range(num_plots):
        axn = fig.add_subplot(1, num_plots, r + 1)
        axn.plot(re_real_data[r][0, :], re_real_data[r][1, :], label='experimental data', linewidth=3)
        axn.plot(re_rnn_data[r][0, :], re_rnn_data[r][1, :], label='RNN model data', linestyle='--', linewidth=5)
        axn.set_xlabel('PC1')
        axn.set_ylabel('PC2')
        axn.legend(fontsize=10)
        axn.set_title(f"{labels[r]} activity", fontsize=12)
    fig.suptitle(f"PCA of Model vs Experimental extracted trial data - mouse {mouse_num}", fontsize=16)
    fig.tight_layout()
    return fig


def plot_rnn_weight_matrix(rnn_model, regions):
    matrix = rnn_model['J']
    neuron_num = rnn_model['RNN'].shape[0]

    # Extract boundaries dynamically
    boundaries = [region[1][-1] for region in regions]

    fig, ax = plt.subplots(figsize=[10, 6])
    cax = ax.pcolormesh(range(neuron_num), range(neuron_num), matrix, cmap="viridis")
    fig.colorbar(cax, label="Weight Strength")

    # Plot region boundaries
    for boundary in boundaries:
        ax.axvline(x=boundary + 0.5, color='red', linestyle='--', linewidth=1)
        ax.axhline(y=boundary + 0.5, color='red', linestyle='--', linewidth=1)

    # Compute midpoints for labels
    midpoints = [boundaries[0] / 2] + [(boundaries[i - 1] + boundaries[i]) / 2 for i in range(1, len(boundaries))]
    region_labels = [region[0] for region in regions]

    # Set axis labels dynamically
    ax.set_xticks(midpoints)
    ax.set_xticklabels(region_labels, fontsize='xx-large')
    ax.set_yticks(midpoints)
    ax.set_yticklabels(region_labels, fontsize='xx-large')

    ax.set_title('RNN weight matrix', fontsize='xx-large')
    ax.set_xlabel('target neuron', fontsize='xx-large')
    ax.set_ylabel('source neuron', fontsize='xx-large')

    return fig

def plot_trial_currents(model, curbd_arr, curbd_labels, n_regions, sol_angles, perturbation_time, mouse_num, normalize):
    '''
    :param model: rnn model
    :param curbd_arr: output from curbd
    :param curbd_labels: output from curbd
    :param n_regions: number of regions in brain CURBD
    :param sol_angles: all trial categories
    :param perturbation_time: needs to be in seconds
    :param mouse_num: mouse number (M044)
    :param normalize: either None, or 'z-score' or 'min-max'
    :return: figure
    '''
    fig = pylab.figure(figsize=[12, 8])
    count = 1
    for iTarget in range(n_regions):
        for iSource in range(n_regions):
            axn = fig.add_subplot(n_regions, n_regions, count)
            count += 1

            split_size = len(sol_angles)
            trim_length = curbd_arr[iTarget, iSource].shape[1] - (curbd_arr[iTarget, iSource].shape[1] % split_size)
            curbd_data = curbd_arr[iTarget, iSource][:, :trim_length]
            arr = np.array(np.split(curbd_data, len(sol_angles), axis=1))

            if normalize == 'z-score':
                mean_val = np.mean(arr)
                std_val = np.std(arr)
                arr = (arr - mean_val) / std_val
            elif normalize == 'min-max':
                arr_min = np.min(arr)
                arr_max = np.max(arr)
                arr = (arr - arr_min) / (arr_max - arr_min)

            mean_arr = np.mean(arr, axis=(0, 1))
            trials_mean = np.mean(arr, axis=1)

            time = np.array(np.split(model['tRNN'][:trim_length], len(sol_angles)))

            # plot CURBD means
            for i in range(len(trials_mean)):
                trial = trials_mean[i]
                axn.plot(time[0], trial, color='lightblue', linewidth=0.5)
            axn.plot(time[0], mean_arr, color='steelblue', linewidth=2, label='Average across trials')

            # perturbation line
            axn.axvline(perturbation_time, color='red', linestyle='--', linewidth=1, label='Perturbation time')

            axn.set_title(f'{curbd_labels[iTarget, iSource]} currents')
            axn.set_xlabel('Time (s)')
            axn.set_ylabel('Average Activity')
            axn.title.set_fontsize(16)
            axn.xaxis.label.set_fontsize(16)
            axn.yaxis.label.set_fontsize(16)
            axn.legend(loc='upper left')

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.suptitle(f'Currents sorted by Trial Type - {mouse_num}', fontsize='xx-large')

    return fig

def plot_sol_levels(model, curbd_arr, curbd_labels, n_regions, sol_angles, perturbation_time, mouse_num, normalize):
    fig = pylab.figure(figsize=[12, 8])
    count = 1
    for iTarget in range(n_regions):
        for iSource in range(n_regions):
            axn = fig.add_subplot(n_regions, n_regions, count)
            count += 1

            split_size = len(sol_angles)
            trim_length = curbd_arr[iTarget, iSource].shape[1] - (curbd_arr[iTarget, iSource].shape[1] % split_size)
            curbd_data = curbd_arr[iTarget, iSource][:, :trim_length]

            arr = np.array(np.split(curbd_data, len(sol_angles), axis=1))
            if normalize == 'z-score':
                mean_val = np.mean(arr)
                std_val = np.std(arr)
                arr = (arr - mean_val) / std_val
            elif normalize == 'min-max':
                arr_min = np.min(arr)
                arr_max = np.max(arr)
                arr = (arr - arr_min) / (arr_max - arr_min)

            trials_mean = np.mean(arr, axis=1)

            top_mean = np.mean(trials_mean[:5], axis=0)
            bottom_mean = np.mean(trials_mean[5:], axis=0)

            time = np.array(np.split(model['tRNN'][:trim_length], len(sol_angles)))

            # plot CURBD means
            for i in range(len(trials_mean)):
                if i >= 4:  # top solenoids
                    color = 'lightblue'
                if i < 4:  # bottom
                    color = 'peachpuff'
                trial = trials_mean[i]
                axn.plot(time[0], trial, color=color, linewidth=0.5)

            axn.plot(time[0], top_mean, color='deepskyblue', linewidth=2, label='Average across top solenoid trials')
            axn.plot(time[0], bottom_mean, color='orange', linewidth=2, label='Average across bottom solenoid trials')
            # perturbation line
            axn.axvline(perturbation_time, color='red', linestyle='--', linewidth=1, label='Perturbation time')

            axn.set_title(f'{curbd_labels[iTarget, iSource]} currents')
            axn.set_xlabel('Time (s)')
            axn.set_ylabel('Average Activity')
            axn.title.set_fontsize(16)
            axn.xaxis.label.set_fontsize(16)
            axn.yaxis.label.set_fontsize(16)
            axn.legend(fontsize='small')

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.suptitle(f'Currents sorted by Solenoid level - {mouse_num}', fontsize='xx-large')

    return fig

def plot_all_currents(model, curbd_arr, curbd_labels, n_regions, sol_angles, perturbation_time, mouse_num, normalize):
    fig = pylab.figure(figsize=[12, 8])

    for iTarget in range(n_regions):
        for iSource in range(n_regions):
            split_size = len(sol_angles)
            trim_length = curbd_arr[iTarget, iSource].shape[1] - (curbd_arr[iTarget, iSource].shape[1] % split_size)
            curbd_data = curbd_arr[iTarget, iSource][:, :trim_length]

            arr = np.array(np.split(curbd_data, len(sol_angles), axis=1))

            if normalize == 'z-score':
                mean_val = np.mean(arr)
                std_val = np.std(arr)
                arr = (arr - mean_val) / std_val
            elif normalize == 'min-max':
                arr_min = np.min(arr)
                arr_max = np.max(arr)
                arr = (arr - arr_min) / (arr_max - arr_min)

            mean_arr = np.mean(arr, axis=(0, 1))
            sem_arr = np.std(arr, axis=(0, 1)) / np.sqrt(arr.shape[0] * arr.shape[1])
            time = np.array(np.split(model['tRNN'][:trim_length], len(sol_angles)))

            pylab.plot(time[0], mean_arr, linewidth=2, label=f'{curbd_labels[iTarget, iSource]} current')
            pylab.fill_between(time[0], mean_arr - sem_arr, mean_arr + sem_arr, alpha=0.3)

    pylab.axvline(perturbation_time, color='red', linestyle='--', linewidth=1, label='Perturbation time')
    pylab.title(f'All currents - mouse {mouse_num}')
    pylab.xlabel('Time (s)')
    pylab.ylabel('Activity')
    pylab.legend(loc='upper left')
    pylab.show()
    return fig

def format_for_plotting(curbd_arr, curbd_labels, n_regions, reset_points):
    all_currents = []
    max_len = curbd_arr[0, 0].shape[1]
    reset_points = [point for point in reset_points if point <= max_len]

    for iTarget in range(n_regions):
        for iSource in range(n_regions):
            num_neurons = curbd_arr[iTarget, iSource].shape[0]
            # set up space for data
            new_row = [[] for _ in range(num_neurons)]

            # iterate over neurons
            for neuron in range(num_neurons):
                curr = curbd_arr[iTarget, iSource][neuron]
                for p in range(len(reset_points)):
                    point = reset_points[p]
                    if point != 0:
                        new_row[neuron].append(curr[reset_points[p - 1]:point])

            # new_row = np.array(new_row)
            all_currents.append(new_row)

    all_currents_labels = curbd_labels.flatten()
    return all_currents, all_currents_labels

def plot_currents_by_region(model, curbd_arr, curbd_labels, n_regions, sol_angles, perturbation_time, mouse_num, normalize=None):
    '''
    :param model: rnn model
    :param curbd_arr: output from curbd
    :param curbd_labels: output from curbd
    :param n_regions: number of regions in brain CURBD
    :param sol_angles: all trial categories
    :param perturbation_time: needs to be in seconds
    :param mouse_num: mouse number (M044)
    :param normalize: either None, or 'z-score' or 'min-max'
    :return: figure
    '''
    dtFactor = model['params']['dtFactor']
    bin_size = model['dtData']
    fig = pylab.figure(figsize=[12, 8])
    count = 1
    colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for iTarget in range(n_regions):
        for iSource in range(n_regions):
            axn = fig.add_subplot(n_regions, n_regions, count)
            count += 1

            split_size = len(sol_angles)
            trim_length = curbd_arr[iTarget, iSource].shape[1] - (curbd_arr[iTarget, iSource].shape[1] % split_size)
            curbd_data = curbd_arr[iTarget, iSource][:, :trim_length]
            arr = np.array(np.split(curbd_data, len(sol_angles), axis=1))

            if normalize == 'z-score':
                mean_val = np.mean(arr)
                std_val = np.std(arr)
                arr = (arr - mean_val) / std_val
            elif normalize == 'min-max':
                arr_min = np.min(arr)
                arr_max = np.max(arr)
                arr = (arr - arr_min) / (arr_max - arr_min)

            mean_arr = np.mean(arr, axis=(0, 1))
            sem_arr = np.std(arr, axis=(0, 1)) / np.sqrt(arr.shape[0] * arr.shape[1])
            time = np.linspace(0, arr.shape[2] * bin_size / dtFactor, arr.shape[2])
            # time = np.array(np.split(model['tRNN'][:trim_length], len(sol_angles)))

            axn.plot(time[0], mean_arr, color=colours[count-2], linewidth=2, label='Average across trials')
            pylab.fill_between(time[0], mean_arr - sem_arr, mean_arr + sem_arr, color=colours[count-2], alpha=0.3)

            # perturbation line
            axn.axvline(perturbation_time, color='red', linestyle='--', linewidth=1, label='Perturbation time')

            axn.set_title(f'{curbd_labels[iTarget, iSource]} mean current')
            axn.set_xlabel('Time (s)')
            axn.set_ylabel('Current Strength')

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.suptitle(f'Average current across all trials- mouse {mouse_num}', fontsize='xx-large')

    return fig

def plot_all_currents_separate(all_currents, all_currents_labels, perturbation_time, bin_size, dtFactor, mouse_num):
    fig = pylab.figure(figsize=[20, 12])
    count = 1
    n_regions = int(math.sqrt(len(all_currents)))
    colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    global_min, global_max = float('inf'), float('-inf')

    # First pass: find global y-limits
    mean_sem_data = []
    for i in range(len(all_currents)):
        current_data = np.array(all_currents[i])
        mean_current = np.mean(current_data, axis=(0, 1))
        sem_current = np.std(current_data, axis=(0, 1)) / np.sqrt(current_data.shape[0] * current_data.shape[1])

        y_lower = np.min(mean_current - sem_current)
        y_upper = np.max(mean_current + sem_current)

        global_min = min(global_min, y_lower)
        global_max = max(global_max, y_upper)

        mean_sem_data.append((mean_current, sem_current))

    # Second pass: plot with shared y-axis
    for i in range(len(all_currents)):
        current_label = all_currents_labels[i]
        colour = colours[i % len(colours)]

        axn = fig.add_subplot(n_regions, n_regions, count)
        count += 1

        current_data = np.array(all_currents[i])
        time_axis = np.linspace(0, (current_data.shape[2] * bin_size)/dtFactor, current_data.shape[2])

        mean_current, sem_current = mean_sem_data[i]

        axn.plot(time_axis, mean_current, linewidth=2, color=colour)
        axn.fill_between(time_axis, mean_current - sem_current, mean_current + sem_current, alpha=0.3, color=colour)
        axn.axvline(perturbation_time, color='red', linestyle='--', linewidth=1, label='Perturbation time')

        axn.set_title(f'{current_label} mean current', fontsize='xx-large')
        axn.set_xlabel('Time (s)', fontsize='xx-large')
        axn.set_ylabel('Current Strength', fontsize='xx-large')

        axn.set_ylim(global_min, global_max)

    fig.suptitle(f'Average current across all trials - Mouse {mouse_num}', fontsize='xx-large')
    fig.tight_layout()
    fig.show()

    return fig

def plt_curr_old(all_currents, all_currents_labels, perturbation_time, bin_size, dtFactor, mouse_num):
    fig = pylab.figure(figsize=[20, 12])
    count = 1
    n_regions = int(math.sqrt(len(all_currents)))
    colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    for i in range(len(all_currents)):
        current_data = np.array(all_currents[i])
        current_label = all_currents_labels[i]
        colour = colours[i % len(colours)]

        axn = fig.add_subplot(n_regions, n_regions, count)
        count += 1
        time_axis = np.linspace(0, (current_data.shape[2] * bin_size), current_data.shape[2])

        # Plot mean and SEM (standard error of the mean)
        mean_current = np.mean(current_data, axis=(0, 1))
        sem_current = np.std(current_data, axis=(0, 1)) / np.sqrt(current_data.shape[0] * current_data.shape[1])

        axn.plot(time_axis, mean_current, linewidth=2, color=colour)
        axn.fill_between(time_axis, mean_current - sem_current, mean_current + sem_current, alpha=0.3, color=colour)

        axn.axvline(perturbation_time, color='red', linestyle='--', linewidth=1, label='Perturbation time')

        axn.set_title(f'{current_label} mean current', fontsize='xx-large')
        axn.set_xlabel('Time (s)', fontsize='xx-large')
        axn.set_ylabel('Current Strength', fontsize='xx-large')

    fig.suptitle(f'Average current across all trials- mouse {mouse_num}', fontsize='xx-large')
    fig.tight_layout()
    fig.show()
    return fig

def PCA_of_currents(all_currents, all_currents_labels, perturbation_time, mouse_num):
    fig = pylab.figure(figsize=(16, 16))
    count = 1
    n_regions = int(math.sqrt(len(all_currents)))
    colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    for i in range(len(all_currents)):
        current_data = np.array(all_currents[i])
        current_label = all_currents_labels[i]
        colour = colours[i % len(colours)]

        axn = fig.add_subplot(n_regions, n_regions, count, projection='3d')
        count += 1

        mean_current = np.mean(current_data, axis=1).T # keep neuron dimension, but mean across trials

        pca = PCA(n_components=min(mean_current.shape[0], mean_current.shape[1]))
        pca_current = pca.fit_transform(mean_current)

        axn.plot(pca_current[:, 0],
                 pca_current[:, 1],
                 pca_current[:, 2], color=colour)
        axn.scatter(
            pca_current[perturbation_time, 0],
            pca_current[perturbation_time, 1],
            pca_current[perturbation_time, 2],
            color='red',
            s=50,
            marker='o'
        )
        axn.scatter(
            pca_current[0, 0],
            pca_current[0, 1],
            pca_current[0, 2],
            color='black',
            s=60,
            marker='x'
        )

        axn.xaxis.pane.fill = False
        axn.yaxis.pane.fill = False
        axn.zaxis.pane.fill = False
        axn.set_title(f'{current_label} mean current', fontsize='x-large')
        axn.set_xlabel('PC1')
        axn.set_ylabel('PC2')
        axn.set_zlabel('PC3')
        axn.grid(False)
        # axn.legend(loc='upper left')

    fig.suptitle(f'Average current across all trials- mouse {mouse_num}', fontsize='xx-large')
    fig.tight_layout()
    fig.show()

    return fig

def PCA_of_currents_plotly(all_currents, all_currents_labels, perturbation_time, mouse_num):
    n_regions = len(all_currents)
    n_rows = int(math.ceil(math.sqrt(n_regions)))
    n_cols = int(math.ceil(n_regions / n_rows))

    colours = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    subplot_titles = [f"{label} mean current" for label in all_currents_labels]
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[[{'type': 'scene'} for _ in range(n_cols)] for _ in range(n_rows)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.1
    )

    count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if count >= n_regions:
                break

            current_data = np.array(all_currents[count])
            current_label = all_currents_labels[count]
            colour = colours[count % len(colours)]

            # mean across trials, keep neuron dim, transpose for PCA
            mean_current = np.mean(current_data, axis=1).T

            # PCA
            pca = PCA(n_components=min(mean_current.shape[0], mean_current.shape[1], 3))
            pca_current = pca.fit_transform(mean_current)

            # Trajectory line
            fig.add_trace(go.Scatter3d(
                x=pca_current[:, 0],
                y=pca_current[:, 1],
                z=pca_current[:, 2],
                mode='lines',
                line=dict(color=colour, width=4),
                name=current_label,
                showlegend=False
            ), row=i+1, col=j+1)

            # Red dot at perturbation time
            fig.add_trace(go.Scatter3d(
                x=[pca_current[perturbation_time, 0]],
                y=[pca_current[perturbation_time, 1]],
                z=[pca_current[perturbation_time, 2]],
                mode='markers',
                marker=dict(color='red', size=6, symbol='circle'),
                showlegend=False
            ), row=i+1, col=j+1)

            # Black X at start
            fig.add_trace(go.Scatter3d(
                x=[pca_current[0, 0]],
                y=[pca_current[0, 1]],
                z=[pca_current[0, 2]],
                mode='markers',
                marker=dict(color='black', size=2, symbol='x'),
                showlegend=False
            ), row=i+1, col=j+1)

            fig.update_scenes(
                xaxis=dict(title='PC1', showgrid=False, zeroline=True, titlefont=dict(size=10)),
                yaxis=dict(title='PC2', showgrid=False, zeroline=True, titlefont=dict(size=10)),
                zaxis=dict(title='PC3', showgrid=False, zeroline=True, titlefont=dict(size=10)),
                bgcolor='white',
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
                aspectmode='cube',
                row=i+1, col=j+1)

            count += 1

    fig.update_layout(
        title_text=f'Average current across all trials - Mouse {mouse_num}',
        title_font_size=20,
        height=1000,
        width=1200,
        showlegend=False
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=15))
    fig.update_annotations(font_size=10)

    fig.show()
    return fig


#### FROM THE BeNeuro tool functions ####
import logging
from scipy.linalg import qr, svd, inv

def canoncorr(X: np.array, Y: np.array, fullReturn: bool = False) -> np.array:
    """
    Canonical Correlation Analysis (CCA)
    line-by-line port from Matlab implementation of `canoncorr`
    X,Y: (samples/observations) x (features) matrix, for both: X.shape[0] >> X.shape[1]
    fullReturn: whether all outputs should be returned or just `r` be returned (not in Matlab)
    returns: A,B,r,U,V
    A,B: Canonical coefficients for X and Y
    U,V: Canonical scores for the variables X and Y
    r:   Canonical correlations
    Signature:
    A,B,r,U,V = canoncorr(X, Y)
    """
    n, p1 = X.shape
    p2 = Y.shape[1]
    if p1 >= n or p2 >= n:
        logging.warning('Not enough samples, might cause problems')

    # Center the variables
    X = X - np.mean(X, 0)
    Y = Y - np.mean(Y, 0)

    # Factor the inputs, and find a full rank set of columns if necessary
    Q1, T11, perm1 = qr(X, mode='economic', pivoting=True, check_finite=True)

    rankX = sum(np.abs(np.diagonal(T11)) > np.finfo(type((np.abs(T11[0, 0])))).eps * max([n, p1]))

    if rankX == 0:
        logging.error(f'stats:canoncorr:BadData = X')
    elif rankX < p1:
        logging.warning('stats:canoncorr:NotFullRank = X')
        Q1 = Q1[:, :rankX]
        T11 = T11[:rankX, :rankX]

    Q2, T22, perm2 = qr(Y, mode='economic', pivoting=True, check_finite=True)
    rankY = sum(np.abs(np.diagonal(T22)) > np.finfo(type((np.abs(T22[0, 0])))).eps * max([n, p2]))

    if rankY == 0:
        logging.error(f'stats:canoncorr:BadData = Y')
    elif rankY < p2:
        logging.warning('stats:canoncorr:NotFullRank = Y')
        Q2 = Q2[:, :rankY]
        T22 = T22[:rankY, :rankY]

    # Compute canonical coefficients and canonical correlations.  For rankX >
    # rankY, the economy-size version ignores the extra columns in L and rows
    # in D. For rankX < rankY, need to ignore extra columns in M and D
    # explicitly. Normalize A and B to give U and V unit variance.
    d = min(rankX, rankY)
    L, D, M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver='gesdd')
    M = M.T

    A = inv(T11) @ L[:, :d] * np.sqrt(n - 1)
    B = inv(T22) @ M[:, :d] * np.sqrt(n - 1)
    r = D[:d]
    # remove roundoff errs
    r[r >= 1] = 1
    r[r <= 0] = 0

    if not fullReturn:
        return r

    # Put coefficients back to their full size and their correct order
    A[perm1, :] = np.vstack((A, np.zeros((p1 - rankX, d))))
    B[perm2, :] = np.vstack((B, np.zeros((p2 - rankY, d))))
    # Compute the canonical variates
    U = X @ A
    V = Y @ B

    return A, B, r, U, V