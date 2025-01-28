import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable



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
