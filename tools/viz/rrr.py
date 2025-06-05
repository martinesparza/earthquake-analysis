import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import tools.viz.utilityTools as vizutils


def plot_rrr_heatmap_from_dict_per_area(
    dict_, areas, figsize=(8, 6), ax=None, title=None, vmin=0, vmax=1, cmap="RdBu"
):

    arr = []
    for area_x in areas:
        arr_ = []
        for area_y in areas:
            if len(dict_[area_x][area_y]) > 1:
                arr_.append(np.mean(dict_[area_x][area_y]))
            else:
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
            cmap=cmap,
            annot=True,
            fmt=".2f",
            xticklabels=areas,
            yticklabels=areas,
            square=False,
            vmin=vmin,
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


def plot_r2_with_errorbars(dict_, areas, figsize=(5, 7)):
    with plt.style.context("seaborn-v0_8-bright"):
        sns.set_theme(context="notebook", style="ticks")
        fig, axes = plt.subplots(len(areas), 1, figsize=figsize, sharex="all", sharey="none")
        for condition in dict_.keys():
            for ax, area_x in zip(axes, areas):
                arr = []
                for area_y in areas:
                    arr.append(dict_[condition][area_x][area_y])
                ax.errorbar(
                    np.arange(len(areas)),
                    np.mean(arr, axis=1),
                    yerr=np.std(arr, axis=1),
                    fmt="o",
                    capsize=5,
                    label=condition,
                )
                ax.set_ylabel(f"Train: {area_x}")
        axes[-1].set_xticks(np.arange(len(areas)))
        axes[-1].set_xticklabels(areas)
        # axes[-1].legend()
    plt.show()
    return
