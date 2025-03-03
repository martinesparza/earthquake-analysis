import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyaldata as pyal
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tools.params import Params


def plot_fr_raster(df, axes: list, sol_directions: list, trial=15, area="Dls"):
    data = []

    # example trial data for each target
    for sol in sol_directions:
        df_ = pyal.select_trials(df, df.values_Sol_direction == sol)
        fr = df_[f"{area}_rates"][trial]
        data.append(fr)

    data = np.array(data)  # 3d array of sol directions, time, neurons
    vmin = np.amin(data, axis=(0, 1))  # Minimum for each neuron across sols and time
    vmax = np.amax(data, axis=(0, 1))  # Maximum for each neuron across sols and time

    for solData, ax, sol_direction in zip(data, axes, sol_directions):
        solData -= vmin
        solData /= vmax - vmin
        im = ax.imshow(solData.T, aspect="auto", cmap="viridis")
        ax.axvline(x=df_.idx_sol_on[0], color="red", linestyle="--", linewidth=2)
        ax.tick_params("both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"Sol: {sol_direction}")

    axes[0].set_ylabel(f"{area} units ($n={solData.shape[1]}$)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
    return axes


def plot_heatmap_raster(
    df: pd.DataFrame,
    area: str,
    ax=None,
    show=True,
    num_ticks=10,
    show_colorbar=False,
    add_sol_onset=False,
):
    """Generate heatmap raster with a given array

    Args:
        arr (np.array): matrix of neuronx x time
        ax (ax, optional): Axes object of matplotlib. Defaults to None.
        show (bool, optional): Show plot or not. Defaults to True.
        num_ticks (int, optional): Number of ticks on the x axis. Defaults to 10.
        show_colorbar (bool, optional): Show color bar or not. Defaults to False.

    Returns:
        ax: Axes object
    """
    rates = np.concatenate(df[f"{area}_rates"].values, axis=0).T
    trial_length = df[f"{area}_rates"].values[0].shape[0]
    print(trial_length)

    if ax is None:
        fig, ax = plt.subplots(sharex="all", figsize=(12, 10))
    im = ax.imshow(rates, cmap="viridis", origin="lower", aspect="auto")
    if add_sol_onset:
        for time_bin in range(len(df)):
            sol_on = time_bin * (trial_length) + df.idx_sol_on.values[0]
            ax.axvline(
                x=sol_on,
                color="red",
                linestyle="--",
                linewidth=1,
            )

    time_bins = np.linspace(0, rates.shape[1] * 0.03, rates.shape[1])

    tick_positions = np.linspace(0, rates.shape[1] - 1, num_ticks)
    tick_labels = np.round(np.linspace(time_bins[0], time_bins[-1], num_ticks), 2)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    if show:
        plt.show()

    return ax
