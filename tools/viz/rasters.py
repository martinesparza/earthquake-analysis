import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tools.viz.utilityTools as vizutils
from tools.dataTools import get_trial_x_time_per_neuron
from tools.params import Params, colors


def plot_single_neuron_raster_and_psth(
    df, area, neuron_id, vmin=None, vmax=None, vline=None, figsize=None, ax=None, show=True
):
    """Plots psth and raster of a single neuron for every trial

    Args:
        df (_type_): _description_
        area (_type_): _description_
        neuron_id (_type_): _description_
        cmap (str, optional): _description_. Defaults to "copper_r".
        vmin (_type_, optional): _description_. Defaults to None.
        vmax (_type_, optional): _description_. Defaults to None.
        vline (_type_, optional): _description_. Defaults to None.
    """

    trials_arr = get_trial_x_time_per_neuron(df, area=area, neuron_id=neuron_id)
    n_trials, n_time_bins = trials_arr.shape
    total_spikes = np.sum(trials_arr)
    trial_duration_sec = n_time_bins * Params.BIN_SIZE
    print(total_spikes)
    avg_firing_rate = total_spikes / (n_trials * trial_duration_sec)  # in Hz

    print(f"Average firing rate: {avg_firing_rate:.2f} Hz")
    if ax is None:
        fig, ax = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex="all",
            height_ratios=[1, 4],
            gridspec_kw={"hspace": 0.05},
        )
    im = ax[1].imshow(
        trials_arr,
        aspect="auto",
        interpolation="nearest",
        cmap=vizutils.create_cmap_from_area(area),
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    psth = np.sum(trials_arr, axis=0)
    time = np.arange(trials_arr.shape[1])
    ax[0].plot(time, psth, color="k")
    ax[0].fill_between(time, psth, color=getattr(colors, area), alpha=0.6)

    if vline is not None:
        ax[1].axvline(x=vline, linestyle="--", color="k")

    xticks = np.linspace(0, trials_arr.shape[1], 7)  # 6 ticks
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels([f"{x * Params.BIN_SIZE:.2f}" for x in xticks])
    ax[1].set_xlabel("Time (ms)")

    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Trial")
    # ax[0].set_title(f"Neuron id: {neuron_id}. KSLabel: {df[f"{area}_KSLabel"][0][neuron_id]}")
    if show:
        plt.show()
    return ax


def plot_single_neuron_raster_and_psth_grid(
    df, area, neuron_ids, cmap="copper_r", vmin=None, vmax=None, vline=None
):

    n = len(neuron_ids)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows * 2,
        ncols,
        figsize=(ncols * 5, nrows * 5),
        sharex="col",
        gridspec_kw={"hspace": 0.03},
    )

    if nrows == 1:
        axes = np.array([axes])  # Ensure 2D indexing

    for idx, neuron_id in enumerate(neuron_ids):
        row = (idx // ncols) * 2
        col = idx % ncols

        try:
            trials_arr = get_trial_x_time_per_neuron(df, area=area, neuron_id=neuron_id)

            # PSTH
            psth_ax = axes[row, col]
            psth = np.sum(trials_arr, axis=0)
            time = np.arange(trials_arr.shape[1])
            psth_ax.plot(time, psth, color=getattr(colors, area))
            psth_ax.set_title(
                f"Neuron id: {neuron_id}. KSLabel: {df[f"{area}_KSLabel"][0][neuron_id]}"
            )

            psth_ax.set_xticks([])

            # Heatmap
            heatmap_ax = axes[row + 1, col]
            im = heatmap_ax.imshow(
                trials_arr,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                origin="upper",
                vmin=vmin,
                vmax=vmax,
            )

            if vline is not None:
                heatmap_ax.axvline(x=vline, linestyle="--", color="k")

            heatmap_ax.set_xlabel("Time (bins)")
            heatmap_ax.set_ylabel("Trial")

        except Exception as e:
            print(f"Error plotting neuron {neuron_id}: {e}")

    # Colour bar
    cbar = fig.colorbar(im, ax=axes[:, -1], shrink=0.6)
    cbar.set_label("Spike count")

    plt.tight_layout()
    plt.show()
    return


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

    if ax is None:
        fig, ax = plt.subplots(sharex="all", figsize=(15, 5))
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
    ax.set_title(f"{area}")
    ax.set_ylabel("Neurons")
    ax.set_xlabel("Time (s)")

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    if show:
        plt.show()

    return ax
