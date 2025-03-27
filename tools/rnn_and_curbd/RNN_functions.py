import numpy as np
import matplotlib.pyplot as plt
import pyaldata
import pylab
import os
import csv
from tools.curbd import curbd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

### core RNN and CURBD functions ###
def RNN(formated_rates, resets, regions_arr, data, mouse_num, **kwargs):
    rnn_model = train_RNN(formated_rates, resets, regions_arr, data.bin_size[0], **kwargs)

    figure = plot_model_accuracy(rnn_model, mouse_num)

    return rnn_model, figure

def PCA_and_CCA(concat_rates, rnn_model, num_components, arr_shapes, mouse_num, printing=True):
    data_rnn = rnn_model['Adata'].T
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

def get_reset_points(df, activity, areas):
    trial_len = df[areas[0]][0].shape[0]
    if all(df[col][0].shape[0] == trial_len for col in areas):
        print(f"Trial length: {trial_len}")
    else:
        print("Variable trial length!")

    reset_points = []
    for i in range(len(df)):
        point = i * trial_len
        if point >= activity.shape[1]: # to avoid wierd indexing errors
            point = activity.shape[1]- 1
        reset_points.append(point)

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

### plotting functions ###

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
    axn.set_ylim(0, 3)
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
