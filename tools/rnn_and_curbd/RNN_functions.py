import numpy as np
import matplotlib.pyplot as plt
import pyaldata
import pylab
import math
import os
import csv
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from tools.curbd import curbd

### core RNN and CURBD functions ###
def RNN(formated_rates, resets, regions_arr, data, mouse_num, graph = False, **kwargs):
    rnn_model = train_RNN(formated_rates, resets, regions_arr, data.bin_size[0], **kwargs)

    if graph:
        figure = plot_model_accuracy(rnn_model, mouse_num)
        return rnn_model, figure
    else:
        return rnn_model

def PCA_and_CCA(rnn_model, trial_num, mouse_num, printing=True):
    data_rnn = rnn_model['RNN'].T
    data_real = rnn_model['Adata'].T
    num_components = data_rnn.shape[1] - 1

    # PCA
    pca_real, pca_data_real = PCA_fit_transform(data_real, num_components)
    pca_rnn, pca_data_rnn = PCA_fit_transform(data_rnn, num_components)

    variance_figure = plot_PCA_cum_var(pca_real, pca_rnn, mouse_num)
    PCA_figure = plot_PCA(pca_data_real, pca_data_rnn, trial_num, mouse_num)

    # CCA
    canonical_values, scores = CCA_compare(pca_data_real, pca_data_rnn, num_components)
    canonical_values = np.array(canonical_values)

    if printing:
        print(f"CCA score of real data and RNN data aligment: {scores[0]}")
        print(f"CCA score for control on real data: {scores[1]}")
        print(f"CCA score for control on rnn data: {scores[2]}")

    # CCA_figure = plot_CCA(canonical_values[:, 0], canonical_values[:, 1],
    #                       ['Real & RNN data', 'Control - Real data', 'Control - RNN data'], num_components, mouse_num)

    return scores, variance_figure, PCA_figure

def PCA_by_region_helper(data, regions):
    num_regions = len(regions)
    PCA_data = []
    pcas = []
    for r in range(num_regions):
        # select region data
        neurons = len(regions[r][1])
        first_idx = regions[r][1][0]
        last_idx = regions[r][1][-1]
        region_data = data[:, first_idx:last_idx,]
        # PCA and save
        pca = PCA(n_components = neurons-1)
        PCA_data.append(pca.fit_transform(region_data))
        pcas.append(pca)
    return PCA_data, pcas

def PCA_by_region(concat_rates, rnn_model, regions, trial_num, mouse_num):
    data_rnn = rnn_model['Adata'].T
    data_real = rescale_array(concat_rates)

    PCA_real_data, pcas_real = PCA_by_region_helper(data_real, regions)
    PCA_rnn_data, pcas_rnn = PCA_by_region_helper(data_rnn , regions)
    
    # PCs_by_region_figure = plot_PCs(PCA_real_data, PCA_rnn_data, trial_num, regions, mouse_num)
    num_plots = len(regions)
    labels = regions[:, 0]
    re_real_data = []
    re_rnn_data = []
    
    for r in range(len(regions)):
        re_real = np.split(PCA_real_data[r], trial_num) 
        re_rnn = np.split(PCA_rnn_data[r], trial_num)
        re_real = np.array(re_real)
        re_rnn = np.array(re_rnn)
        mean_real = np.mean(re_real, axis=0)
        mean_rnn = np.mean(re_rnn, axis=0)
        re_real_data.append(mean_real)
        re_rnn_data.append(mean_rnn)
    
    fig = plt.figure(figsize=(12,6))
    for r in range(num_plots):
        axn = fig.add_subplot(1, num_plots, r + 1)
        axn.plot(re_real_data[r][ :, 0],re_real_data[r][ :, 1], label = 'experimental data', linewidth=3)
        axn.plot(re_rnn_data[r][:, 0],re_rnn_data[r][ :, 1], label = 'RNN model data', linestyle='--', linewidth=3)
        axn.set_xlabel('PC1')
        axn.set_ylabel('PC2')
        axn.legend(fontsize=14, loc='upper left')
        axn.set_title(f"{labels[r]} activity", fontsize=16)
    fig.suptitle(f"PCA of Model vs Experimental Trial Averaged Data - mouse {mouse_num}", fontsize=20)
    fig.tight_layout()

    return fig
   

### helper functions ###

def get_reset_points(df, activity, areas, dtFactor, printing = False):
    trial_len = df[areas[0]][0].shape[0]
    if all(df[col][0].shape[0] == trial_len for col in areas):
        if printing:
            print(f"Trial length: {trial_len}")
    else:
        print("Variable trial length!")

    reset_points = []
    for i in range(len(df)):
        point = i * trial_len * dtFactor
        if point >= (activity.shape[0]*dtFactor): # to avoid wierd indexing errors
            point = (activity.shape[0]*dtFactor)- 1
        reset_points.append(point)

    return reset_points, trial_len

def get_reset_points_continuous(df, dtFactor, printing=False):
    reset_points = []
    cumulative_time = 0

    for i in range(len(df)):
        trial_len = df['trial_length'][i]
        point = cumulative_time * dtFactor
        reset_points.append(point)

        if printing:
            print(f"Trial {i}: length = {trial_len}, reset point = {point}")

        cumulative_time += trial_len

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

    model = curbd.trainMultiRegionRNN(
        activity,
        dtData=bin_size,
        regions=regions,
        resetPoints=reset_points,
        verbose=False,
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

# def get_last_row_number(filename):

#     if not os.path.exists(filename):
#         print(f"File '{filename}' does not exist.")
#         return 0

#     with open(filename, 'r') as csvfile:
#         reader = csv.DictReader(csvfile)
#         last_row = None
#         for last_row in reader:
#             pass
#         print(last_row)
#         if last_row:
#             return int(last_row['model_idx'])
#         else:
#             return 0

# def write_results_csv(model, scores, filename):
#     model_idx = get_last_row_number(filename) + 1

#     model_results = {"model_idx": model_idx,
#                      "dtFactor": model['params']['dtFactor'],
#                      "binSize": model['dtData'],
#                      "tauRNN": model['params']['tauRNN'],
#                      "ampInWN": model['params']['ampInWN'],
#                      "pVar": model['pVars'][-1],
#                      "chi2": model['chi2s'][-1],
#                      "CCA_score": scores[0]}

#     file_exists = os.path.isfile(filename)

#     with open(filename, 'a', newline='') as csvfile:
#         fieldnames = ['model_idx', 'dtFactor', 'binSize', 'tauRNN', 'ampInWN', 'pVar', 'chi2', 'CCA_score']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         # Write header only if file does not exist (so it's added just once)
#         if not file_exists:
#             writer.writeheader()

#         writer.writerow(model_results)

#     return model_idx

# def save_plot(figures, filenames, folder_name, base_dir):
#     """
#     Saves the given figure to a new folder with a unique name for each run.

#     Parameters:
#         figures list of (matplotlib.figure.Figure): The figure to save.
#         folder_name (str): Name of the folder to be created
#         base_dir (str): The base directory to store all plot folders.

#     Returns:
#         str: The path where the figure was saved.
#     """
#     run_folder = os.path.join(base_dir, folder_name)
#     os.makedirs(run_folder, exist_ok=True)
#     file_paths = []

#     for i in range(len(figures)):
#         fig = figures[i]
#         file_path = os.path.join(run_folder, f"{filenames[i]}.png")
#         file_paths.append(file_path)
#         fig.savefig(file_path, dpi=300)
#         plt.close(fig)

#     return file_paths

# def save_results(model, scores, csv_filename, data_dir, figures, plot_filenames, printing=True):
    model_idx = write_results_csv(model, scores, csv_filename)

    graph_dir = os.path.join(data_dir, "RNN_graphs/")
    folder_name = f'RNN_figures_model_idx_{model_idx}'

    file_path = save_plot(figures, plot_filenames, folder_name, base_dir=graph_dir)
    if printing:
        print(file_path)

### plotting functions ###

def plot_neuron_activity(data, title, mouse_num):
    plt.figure(figsize=(12, 4))
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

    fig.suptitle(f"RNN Model accurancy - mouse {mouse_num}")
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.show()

    return fig

def plot_rnn_weight_matrix(rnn_model, regions):
    matrix = rnn_model['J']
    neuron_num = rnn_model['Adata'].shape[0]

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
    plt.show()

    return fig

def plot_3PCs(fig, real_data, rnn_data, subplot_num):
    ax1 = fig.add_subplot(subplot_num, projection='3d')

    # Plot all single trial trajectories for real_data in light grey
    for trial in range(real_data.shape[0]):
        ax1.plot(real_data[trial, :, 0],
                 real_data[trial, :, 1],
                 real_data[trial, :, 2], color='lightgrey', alpha=0.7)

    # Plot all single trial trajectories for rnn_data in light blue
    for trial in range(rnn_data.shape[0]):
        ax1.plot(rnn_data[trial, :, 0],
                 rnn_data[trial, :, 1],
                 rnn_data[trial, :, 2], color='lightblue', alpha=0.7)

    # Compute and plot the averaged trajectory for real_data in solid red
    avg_real_trajectory = np.mean(real_data, axis=0)
    ax1.plot(avg_real_trajectory[:, 0], avg_real_trajectory[:, 1], avg_real_trajectory[:, 2], 
             color='red', linewidth=3, label='Average Trajectory (Real)')

    # Compute and plot the averaged trajectory for rnn_data in dashed blue
    avg_rnn_trajectory = np.mean(rnn_data, axis=0)
    ax1.plot(avg_rnn_trajectory[:, 0], avg_rnn_trajectory[:, 1], avg_rnn_trajectory[:, 2], 
             color='blue', linestyle='--', linewidth=3, label='Average Trajectory (RNN)')

    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.legend(loc='upper left')

def plot_PCA(real_data, rnn_data, trial_num, mouse_num):
    reconstructed_data_avg = np.split(real_data, trial_num)
    reconstructed_rnn = np.split(rnn_data, trial_num)
    reconstructed_data_avg = np.array(reconstructed_data_avg)
    reconstructed_rnn = np.array(reconstructed_rnn)

    fig = plt.figure(figsize=(6, 5))

    # Now plot both real_data and rnn_data on the same subplot
    plot_3PCs(fig, reconstructed_data_avg, reconstructed_rnn, 111)

    plt.suptitle(f'Recorded vs. RNN data for 3PCs  - Mouse {mouse_num}', fontsize=16)
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

def plot_PCA_cum_var(pca_real, pca_rnn, mouse_num):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca_real.explained_variance_ratio_), label='real activity')
    plt.plot(np.cumsum(pca_rnn.explained_variance_ratio_), label='RNN activity')

    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Cumulative Variance Explained by PCA - mouse {mouse_num}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return fig

def format_for_plotting(curbd_arr, curbd_labels, n_regions, reset_points):
    all_currents = []
    max_len = curbd_arr[0, 0].shape[1]
    reset_points = [point for point in reset_points if point <= max_len]

    for iTarget in range(n_regions):
            for iSource in range(n_regions):
                num_neurons = curbd_arr[iTarget, iSource].shape[0]
                new_row = [[] for _ in range(num_neurons)]

                for neuron in range(num_neurons):
                    curr = curbd_arr[iTarget, iSource][neuron]
                    for p in range(len(reset_points)):
                        point = reset_points[p]
                        if point != 0:
                            new_row[neuron].append(curr[reset_points[p-1]:point])

                all_currents.append(new_row)

    all_currents_labels = curbd_labels.flatten()
    return all_currents, all_currents_labels

def plot_region_currents(all_currents, all_currents_labels, perturbation_time, bin_size, num_trials, mouse_num):

    fig = pylab.figure(figsize=[12, 6])
    count = 1

    for i in range(len(all_currents)):

        current_data = all_currents[i]
        current_label = all_currents_labels[i]
        time_axis = np.linspace(0, current_data.shape[2] * bin_size, current_data.shape[2])

        axn = fig.add_subplot(2, 2, count)
        count += 1

        # plotting each trial current
        for trial_index in range(num_trials):
            arr = current_data[:, trial_index, :]
            trial_mean = np.mean(arr, axis = 0)
            axn.plot(time_axis, trial_mean, color='lightblue', linewidth=0.5)

        # plotting mean current for trial
        mean_current = np.mean(current_data, axis=(0, 1))
        
        axn.plot(time_axis, mean_current, color='steelblue', linewidth=2)

        axn.axvline(perturbation_time, color='red', linestyle='--', linewidth=1, label = 'Perturbation time')

        axn.set_title(f'{current_label} currents')
        axn.set_xlabel('Time (s)')
        axn.set_ylabel('Average Activity')
        axn.title.set_fontsize(16)
        axn.xaxis.label.set_fontsize(16)
        axn.yaxis.label.set_fontsize(16)
        
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.suptitle(f'Currents seperated by origin and target region - {mouse_num}', fontsize='xx-large')
    plt.show()

    return fig
         
def plot_all_currents(all_currents, all_currents_labels, perturbation_time, bin_size, mouse_num, normalize):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i in range(len(all_currents)):
        current_data = all_currents[i]
        current_label = all_currents_labels[i]

        if normalize == 'min-max':
            min_across_trials = np.min(current_data, axis=(1, 2), keepdims=True)
            max_across_trials = np.max(current_data, axis=(1, 2), keepdims=True)

            scaled_data = (current_data - min_across_trials) / (max_across_trials - min_across_trials + 1e-8)

        elif normalize == 'z-score':
            mean_across_trials = np.mean(current_data, axis=(1, 2), keepdims=True)
            std_across_trials = np.std(current_data, axis=(1, 2), keepdims=True)

            scaled_data = (current_data - mean_across_trials) / (std_across_trials + 1e-8)

        time_axis = np.linspace(0, current_data.shape[2] * bin_size, current_data.shape[2])

        # Plot mean and SEM (standard error of the mean)
        mean_current = np.mean(scaled_data, axis=(0, 1))
        sem_current = np.std(scaled_data, axis=(0, 1)) / np.sqrt(scaled_data.shape[0] * scaled_data.shape[1])

        ax.plot(time_axis, mean_current, linewidth=2, label=f'{current_label} current')
        ax.fill_between(time_axis, mean_current - sem_current, mean_current + sem_current, alpha=0.3)

    ax.axvline(perturbation_time, color='red', linestyle='--', linewidth=1, label='Perturbation time')

    ax.set_title(f'All currents ({normalize}) - mouse {mouse_num}', fontsize='xx-large')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized Current Strength')
    ax.legend(loc='upper left')
    plt.show()

    return fig

def plot_all_currents_separate(all_currents, all_currents_labels, perturbation_time, bin_size, dtFactor, mouse_num, plot_single=True):
    fig = pylab.figure(figsize=[12, 8])
    count = 1
    n_regions = len(all_currents)
    colours = ['C3', 'C2', 'C1', 'C0']
          
    for i in range(len(all_currents)):
        current_data = all_currents[i]

        current_label = all_currents_labels[i]
        axn = fig.add_subplot(int(n_regions/2), int(n_regions/2), count)
        count += 1
        time_axis = np.linspace(0, current_data.shape[2] * bin_size/dtFactor, current_data.shape[2])

        # Plot mean and SEM (standard error of the mean)
        mean_current = np.mean(current_data, axis=(0, 1))
        sem_current = np.std(current_data, axis=(0, 1)) / np.sqrt(current_data.shape[0] * current_data.shape[1])
        if plot_single:
            mean_neurons = np.mean(current_data, axis=0)
            for j in range(0, mean_neurons.shape[0], 1):
                axn.plot(time_axis, mean_neurons[j].T, linewidth=0.5, color='lightblue', alpha=0.5)

        axn.plot(time_axis, mean_current, linewidth=2, color=colours[i])
        axn.fill_between(time_axis, mean_current - sem_current, mean_current + sem_current, alpha=0.3, color=colours[i])

        axn.axvline(perturbation_time, color='red', linestyle='--', linewidth=1, label='Perturbation time')

        axn.set_title(f'{current_label} mean current')
        axn.set_xlabel('Time (s)')
        axn.set_ylabel('Current Strength')
        # axn.set_ylim(min(mean_current)-0.005, max(mean_current)+0.005)

    fig.suptitle(f'Average current across all trials- mouse {mouse_num}', fontsize='xx-large')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    return fig

def plot_inter_trial_currents(all_currents, all_currents_labels, bin_size, dtFactor, mouse_num,
                               plot_single=True):
    n_regions = len(all_currents)
    max_time_length = max(len(trial) for region in all_currents for neuron in region for trial in neuron)

    padded_all_currents = []
    for region in all_currents:
        padded_region = []
        for neuron in region:
            padded_neuron = []
            for trial in neuron:
                padded_trial = np.pad(trial, (0, max_time_length - len(trial)), constant_values=np.nan)
                padded_neuron.append(padded_trial)
            padded_region.append(padded_neuron)
        padded_all_currents.append(np.array(padded_region))

    fig = pylab.figure(figsize=[12, 8])
    count = 1
    colours = ['C3', 'C2', 'C1', 'C0']

    # Loop over each region in all_currents
    for i in range(n_regions):
        current_data = np.array(padded_all_currents[i])
        current_label = all_currents_labels[i]

        # Generate time_axis based on max_time_length
        time_axis = np.linspace(0, max_time_length * bin_size / dtFactor, max_time_length)

        axn = fig.add_subplot(int(n_regions / 2), int(n_regions / 2), count)
        count += 1

        # Calculate the overall mean (over neurons and trials) manually
        mean_current = np.nanmean(current_data, axis=(0, 1))  # Mean over neurons and trials
        sem_current = np.nanstd(current_data, axis=(0, 1)) / np.sqrt(
            np.sum(~np.isnan(current_data), axis=(0, 1)))  # SEM ignoring NaNs

        axn.plot(time_axis, mean_current, linewidth=2, color=colours[i])
        axn.fill_between(time_axis, mean_current - sem_current, mean_current + sem_current, alpha=0.3, color=colours[i])

        if plot_single:
            mean_neurons = np.mean(current_data, axis=0)
            for j in range(0, mean_neurons.shape[0], 1):
                axn.plot(time_axis, mean_neurons[j].T, linewidth=0.5, color='lightblue', alpha=0.5)

        axn.set_title(f'{current_label} mean current')
        axn.set_xlabel('Time (s)')
        axn.set_ylabel('Current Strength')

    fig.suptitle(f'Average current across all trials - mouse {mouse_num}', fontsize='xx-large')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

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

def plot_PCs_by_region(real_data, rnn_data, reset_points, regions, trial_len, mouse_num):
    num_plots = len(regions)
    labels = regions[:, 0]
    re_real_data = []
    re_rnn_data = []
    for r in range(len(regions)):
        all_data, trial_data, _ = extract_trials(real_data[r].T, reset_points, trial_len)
        all_rnn, trial_rnn, _ = extract_trials(rnn_data[r].T, reset_points, trial_len)
        re_real = np.array(trial_data)
        re_rnn = np.array(trial_rnn)
        mean_real = np.mean(re_real, axis=1)
        mean_rnn = np.mean(re_rnn, axis=1)
        re_real_data.append(mean_real)
        re_rnn_data.append(mean_rnn)

    fig = plt.figure(figsize=(16, 6))
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
                all_data[neuron].append(arr[reset_points[p-1]:point])

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