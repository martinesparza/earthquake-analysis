import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

# trial = 3
# area = "all"


# def plot_spiking_and_rates(df, trial, area, chan_limit=300):

#     with plt.style.context("seaborn-v0_8-bright"):
#         sns.set_theme(context="talk", style="ticks", font="Arial")
#         fig, ax = plt.subplots(1, 2, sharex="all", figsize=(12, 10))
#         im1 = ax[0].imshow(
#             df[f"{area}_spikes"][trial][:, :chan_limit].T, cmap="viridis", origin="lower"
#         )
#         divider = make_axes_locatable(ax[0])
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         plt.colorbar(im1, cax=cax)

#         im2 = ax[1].imshow(
#             df[f"{area}_rates"][trial][:, :chan_limit].T, cmap="viridis", origin="lower"
#         )
#         divider = make_axes_locatable(ax[1])
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         plt.colorbar(im2, cax=cax)
#         for ax_ in ax:
#             plt.title(f"Sol Direction: {df.loc[trial].values_Sol_direction}")
#             ax_.axvline(x=df.loc[trial].idx_sol_on, color="red", linewidth="1")
#             ax_.set_ylabel("Striatal Neurons")
#             ax_.set_xlabel("Time in sec. (30 ms bins)")
#             ax_.set_xticks([0, 50, 100])
#             ax_.set_xticklabels(["0", "1.6", "3.3"])
#         plt.show()
#         return df[f"{area}_rates"][trial][:, :chan_limit].T


def plot_heatmap_raster(
    arr: np.array, ax=None, show=True, num_ticks=10, show_colorbar=False
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
    if ax is None:
        fig, ax = plt.subplots(sharex="all", figsize=(12, 10))

    im = ax.imshow(arr, cmap="viridis", origin="lower", aspect="auto")

    time_bins = np.linspace(0, arr.shape[1] * 0.03, arr.shape[1])

    tick_positions = np.linspace(0, arr.shape[1] - 1, num_ticks)
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
