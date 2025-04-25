import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_rrr_heatmap_from_dict_per_area(
    dict_, areas, figsize=(8, 6), ax=None, title=None, vmax=1
):

    arr = []
    for area_x in areas:
        arr_ = []
        for area_y in areas:
            arr_.append(dict_[area_x][area_y])
        arr.append(arr_)
    arr = np.array(arr)

    # Plot the heatmap of means

    with plt.style.context("seaborn-v0_8-bright"):
        sns.set_theme(context="poster", style="ticks")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            arr,
            cmap="GnBu",
            annot=True,
            fmt=".2f",
            xticklabels=areas,
            yticklabels=areas,
            square=False,
            vmin=0,
            vmax=vmax,
            ax=ax,
        )
        ax.set_xlabel("Response areas")
        ax.set_ylabel("Predictor areas")
        if title is None:
            ax.set_title("Reduced Rank Regression R2")
        else:
            ax.set_title(title)
    plt.show()
    return fig
