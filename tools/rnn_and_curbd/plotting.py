# imports
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from IPython.core.pylabtools import figsize
from numpy.core.records import ndarray
from sklearn.decomposition import PCA

### Plotting for RNN ###

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

### Plotting for RNN model analysis (PCA, CCA) ###

def plot_pca_cum_var(pca_real, pca_rnn, mouse_num):
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

def plot_3pcs(fig, data, subplot_num, trial_labels, title):
    ax1 = fig.add_subplot(subplot_num, projection='3d', aspect='equal')

    # Support both NumPy arrays and lists of arrays
    if isinstance(data, np.ndarray):
        for trial in range(data.shape[0]):
            ax1.plot(data[trial, :, 0],
                     data[trial, :, 1],
                     data[trial, :, 2], label=trial_labels[trial])
    elif isinstance(data, list):
        for trial_idx, trial_data in enumerate(data):
            ax1.plot(trial_data[:, 0],
                     trial_data[:, 1],
                     trial_data[:, 2], label=trial_labels[trial_idx])
    else:
        raise ValueError("Unsupported data format for plot_3pcs. Expected ndarray or list of arrays.")

    ax1.set_title(title)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.legend(loc='upper left')

def plot_pca(real_data, rnn_data, original_shapes, mouse_num, trial_labels):
    reconstructed_data_avg = np.split(real_data, np.cumsum(original_shapes)[:-1])
    reconstructed_rnn = np.split(rnn_data, np.cumsum(original_shapes)[:-1])

    # Try to convert to NumPy arrays only if all shapes match
    def try_convert_to_array(data_list, label="Data"):
        shapes = [d.shape for d in data_list]
        if len(set(shapes)) == 1:
            return np.array(data_list)
        else:
            print(f"Warning: {label} has inconsistent shapes, returning list instead.")
            return data_list

    reconstructed_data_avg = try_convert_to_array(reconstructed_data_avg, label="Recorded data")
    reconstructed_rnn = try_convert_to_array(reconstructed_rnn, label="RNN data")

    fig = plt.figure(figsize=(20, 10))

    plot_3pcs(fig, reconstructed_data_avg, 121, trial_labels, 'Recorded data')
    plot_3pcs(fig, reconstructed_rnn, 122,trial_labels,'RNN simulated data')

    plt.suptitle(f'First 3PCs plot comparison - mouse {mouse_num}', fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig

def plot_cca(model_cca, ctrl1_cca, ctrl2_cca, mouse_num):

    fig = plt.figure(figsize=(6, 4))

    plt.plot(model_cca, label = "model vs experimental")
    plt.plot(ctrl1_cca, label = "control - experimental")
    plt.plot(ctrl2_cca, label = "control - model")
    plt.xlabel("neural mode")
    plt.ylabel("Correlation")
    plt.title(f"CCA analysis - assessing rnn model fit for mouse {mouse_num}")
    plt.legend(loc="upper right")
    plt.show()

    return fig

def plot_pca_by_region(real_data, rnn_data, original_shapes, regions, mouse_num):
    num_plots = len(regions)
    labels = regions[:, 0]
    re_real_data = []
    re_rnn_data = []

    for r in range(len(regions)):
        re_real = np.split(real_data[r], np.cumsum(original_shapes)[:-1])
        re_rnn = np.split(rnn_data[r], np.cumsum(original_shapes)[:-1])

        # Try to convert to arrays if shapes match
        def safe_mean(trial_list):
            shapes = [arr.shape for arr in trial_list]
            if len(set(shapes)) == 1:
                return np.mean(np.array(trial_list), axis=0)
            else:
                print(f"Warning: Inconsistent trial lengths in region {labels[r]}; averaging manually.")
                # Truncate all to min length
                min_len = min(arr.shape[0] for arr in trial_list)
                truncated = np.array([arr[:min_len] for arr in trial_list])
                return np.mean(truncated, axis=0)

        mean_real = safe_mean(re_real)
        mean_rnn = safe_mean(re_rnn)

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

def plot_firing_rates(rnn_model, reset_points, mouse_num):
    firing_rate_model = rnn_model['RNN'].mean(axis=0)
    firing_rate_real = rnn_model['Adata'].mean(axis=0)

    fig = plt.figure(figsize=(15, 3))
    plt.plot(firing_rate_model, label = "model")
    plt.plot(firing_rate_real, label = "experimental data")

    for trial in reset_points:
        if trial/2 < firing_rate_model.shape[0]:
            plt.axvline(x= trial/2, color='red', linestyle='--', linewidth=1, label = 'trial start' if trial==0 else "")

    plt.xlabel("Time points")
    plt.ylabel("Firing rate")
    plt.title(f"Firing rate over concatenated trial types - Mouse {mouse_num}")
    plt.legend(loc="upper right")
    plt.show()
    return fig

def plot_avg_firing_rates(rnn_model, original_shapes, perturbation_time, bin_size, mouse_num):
    # Split and reshape data
    reconstructed_rnn = np.split(rnn_model['RNN'].T, np.cumsum(original_shapes)[:-1])
    reconstructed_data_avg = np.split(rnn_model['Adata'].T, np.cumsum(original_shapes)[:-1])
    reconstructed_rnn = np.array(reconstructed_rnn)  # shape: (trials, time, neurons)
    reconstructed_data_avg = np.array(reconstructed_data_avg)

    # Global firing rate
    firing_rate_model = reconstructed_rnn.mean(axis=(0, 2))
    firing_rate_real = reconstructed_data_avg.mean(axis=(0, 2))

    # Per-neuron average firing rate
    neuron_avg_model = reconstructed_rnn.mean(axis=0) 
    neuron_avg_real = reconstructed_data_avg.mean(axis=0) 

    # Time axis
    time = np.arange(0, neuron_avg_model.shape[0] * bin_size, bin_size)

    # Create subplots
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[:, 0])  # left
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)  # Bottom-right sharing x-axis with ax2

    # --- Plot 1: Avg firing rate over time
    ax1.plot(time, firing_rate_model, label="Model")
    ax1.plot(time, firing_rate_real, label="Experimental data")
    ax1.axvline(x=perturbation_time, color='red', linestyle='--', linewidth=1, label='Perturbation')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Firing rate")
    ax1.set_title(f"Avg. firing rate over time - Mouse {mouse_num}")
    ax1.legend()

    # --- Plot 2: Heatmap for experimental data
    im2 = ax2.imshow(neuron_avg_real.T, aspect='auto', cmap='viridis',
                     extent=[time[0], time[-1], 0, neuron_avg_real.shape[1]],
                     origin='lower', vmin=0, vmax=1)
    ax2.axvline(x=perturbation_time, color='red', linestyle='--', linewidth=1)
    ax2.set_ylabel("Neuron index")
    ax2.set_title("Experimental avg firing rate per neuron")

    # --- Plot 3: Heatmap for model data
    im3 = ax3.imshow(neuron_avg_model.T, aspect='auto', cmap='viridis',
                     extent=[time[0], time[-1], 0, neuron_avg_model.shape[1]],
                     origin='lower', vmin=0, vmax=1)
    ax3.axvline(x=perturbation_time, color='red', linestyle='--', linewidth=1)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Neuron index")
    ax3.set_title("Model avg firing rate per neuron")
    
    cbar_ax = fig.add_axes([1, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label("Firing rate")

    plt.tight_layout()
    plt.show()
    return fig

def plot_intertrial_avg_firing_rates(rnn_model, original_shapes, trial_types, bin_size, mouse_num):
    # Split and reshape data
    reconstructed_rnn = np.split(rnn_model['RNN'].T, np.cumsum(original_shapes)[:-1])
    reconstructed_data_avg = np.split(rnn_model['Adata'].T, np.cumsum(original_shapes)[:-1])
    if not (len(reconstructed_data_avg)== len(reconstructed_rnn) == len(trial_types)):
        print("Issue in reshaping arrays! - address issue")
        return
    # Create figure layout
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(3, 4, height_ratios=[2, 1, 1])

    # Create figure layout
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(3, 4, height_ratios=[2, 1, 1])

    # Firing rate plots (top row)
    fr_plots = [fig.add_subplot(gs[0, i]) for i in range(len(trial_types))]

    # Heatmap plots - experimental (middle row), sharing x-axis with top row
    he_plots = [fig.add_subplot(gs[1, i], sharex=fr_plots[i]) for i in range(len(trial_types))]

    # Heatmap plots - model (bottom row), sharing x-axis with top row
    hr_plots = [fig.add_subplot(gs[2, i], sharex=fr_plots[i]) for i in range(len(trial_types))]

    for trial_type in range(len(trial_types)):
        exp_data = np.array(reconstructed_data_avg[trial_type])
        model_data = np.array(reconstructed_rnn[trial_type])

        # --- Plot 1: Avg firing rate over time
        exp_firing_rate = exp_data.mean(axis=1)
        model_firing_rate = model_data.mean(axis=1)

        time = np.arange(0, exp_data.shape[0] * bin_size, bin_size)

        fr_plots[trial_type].plot(time, model_firing_rate, label="Model")
        fr_plots[trial_type].plot(time, exp_firing_rate, label="Experimental data")
        fr_plots[trial_type].set_title(f"Average firing rate for {trial_types[trial_type]}")
        fr_plots[trial_type].legend()
        fr_plots[trial_type].set_ylabel("Firing rate")

        # --- Plot 2: Heatmap for experimental data
        im = he_plots[trial_type].imshow(exp_data.T, aspect='auto', cmap='viridis',
                                         extent=[time[0], time[-1], 0, exp_data.shape[1]],
                                         origin='lower', vmin=0, vmax=1)
        he_plots[trial_type].set_ylabel("Neuron index")
        he_plots[trial_type].set_title("Experimental avg. firing rate per neuron")

        # --- Plot 3: Heatmap for model data
        im2 = hr_plots[trial_type].imshow(model_data.T, aspect='auto', cmap='viridis',
                                          extent=[time[0], time[-1], 0, model_data.shape[1]],
                     origin='lower', vmin=0, vmax=1)
        hr_plots[trial_type].set_ylabel("Neuron index")
        hr_plots[trial_type].set_xlabel("Time (s)")
        hr_plots[trial_type].set_title("Model avg. firing rate per neuron")

    cbar_ax = fig.add_axes([1, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Firing rate")

    plt.tight_layout()
    plt.show()
    return fig


### Plotting CURBD results ###

def plot_currents_by_region(all_currents, all_currents_labels, perturbation_time, bin_size, dtFactor, mouse_num):
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

def plot_intertrial_currents(all_currents, all_currents_labels, original_shapes, bin_size, dtFactor, mouse_num):
    fig = pylab.figure(figsize=[20, 12])
    count = 1
    n_regions = int(math.sqrt(len(all_currents)))
    colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    global_min, global_max = float('inf'), float('-inf')

    # First pass: find global y-limits
    for i in range(len(all_currents)):
        # for plotting
        current_label = all_currents_labels[i]
        colour = colours[i % len(colours)]
        axn = fig.add_subplot(n_regions, n_regions, count)
        count += 1

        # Split data by intertrial length
        split_by_trial = []
        longest_trial = float('-inf')
        for trial in all_currents[i]:
            for i in range(4):
                data = np.array(trial[i])
                if data.shape[0] > longest_trial:
                    longest_trial = data.shape[0]
                split_by_trial.append(data)

        time_axis = np.linspace(0, (longest_trial * bin_size)/dtFactor, longest_trial)
        for trial in  split_by_trial:
            t = time_axis[:len(trial)]  # Match time to trial length
            axn.plot(t, trial, linewidth=2, color=colour)

        axn.set_title(f'{current_label} mean current', fontsize='xx-large')
        axn.set_ylabel('Current Strength', fontsize='xx-large')
        
    fig.suptitle(f'Average current across all trials - Mouse {mouse_num}', fontsize='xx-large')
    fig.tight_layout()
    fig.show()


    return fig

def plot_pca_currents(all_currents, all_currents_labels, perturbation_time, mouse_num, fig_size = None):
    if fig_size != None:
        fig = pylab.figure(figsize=fig_size)
    else:
        fig = pylab.figure(figsize=(10, 8))
    
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
            marker='o',
            label = "perturbation"
        )
        axn.scatter(
            pca_current[0, 0],
            pca_current[0, 1],
            pca_current[0, 2],
            color='black',
            s=60,
            marker='x',
            label = "trial start"
        )

        axn.xaxis.pane.fill = False
        axn.yaxis.pane.fill = False
        axn.zaxis.pane.fill = False
        axn.set_title(f'{current_label} mean current', fontsize='x-large')
        axn.set_xlabel('PC1')
        axn.set_ylabel('PC2')
        axn.set_zlabel('PC3')
        axn.grid(False)
        axn.legend()

    fig.suptitle(f'PCA of average current across all trials - mouse {mouse_num}', fontsize='xx-large')
    fig.tight_layout()
    fig.show()

    return fig