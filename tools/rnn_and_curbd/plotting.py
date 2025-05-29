# imports
import numpy as np
import matplotlib.pyplot as plt
import pylab

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from IPython.core.pylabtools import figsize
from numpy.core.records import ndarray

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

def plot_3pcs(fig, data, subplot_num, title):
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

def plot_pca(real_data, rnn_data, original_shapes, mouse_num):
    reconstructed_data_avg = np.split(real_data, np.cumsum(original_shapes)[:-1])
    reconstructed_rnn = np.split(rnn_data, np.cumsum(original_shapes)[:-1])
    reconstructed_data_avg = np.array(reconstructed_data_avg)
    reconstructed_rnn = np.array(reconstructed_rnn)

    fig = plt.figure(figsize=(20, 10))

    plot_3pcs(fig, reconstructed_data_avg, 121, 'Recorded data')
    plot_3pcs(fig, reconstructed_rnn, 122, 'RNN simulated data')
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
