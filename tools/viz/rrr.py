import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import tools.viz.utilityTools as vizutils

color_map_ = {
    "free": "blue",
    "free1": "r",
    "intertrial": "darkorange",
    "trial_short": "green",
    "trial_long": "darkgreen",
}


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


def plot_pairs_rrr(
    rrr_dict,
    pairs,
    x=np.arange(-200, 210, 10),
    r2_type="vae_r2",
    areas=["MOp", "SSp", "CP", "VAL"],
):
    for session in rrr_dict.keys():
        for pair in pairs:
            fig, axes = plt.subplots(4, 4, figsize=(9, 9), sharex="all", sharey="all")
            for ax_x, area_x in zip(axes, areas):
                for ax_y, area_y in zip(ax_x, areas):
                    vizutils.shaded_errorbar(
                        ax_y,
                        x=x,
                        y=np.array(
                            list(
                                rrr_dict[session][f"{pair[0]}"][f"{area_x}"][f"{area_y}"][
                                    f"{r2_type}"
                                ].values()
                            )
                        ),
                        label=f"{pair[0]}",
                        color=f"{color_map_[pair[0]]}",
                    )
                    vizutils.shaded_errorbar(
                        ax_y,
                        x=x,
                        y=np.array(
                            list(
                                rrr_dict[session][f"{pair[1]}"][f"{area_x}"][f"{area_y}"][
                                    f"{r2_type}"
                                ].values()
                            )
                        ),
                        label=f"{pair[1]}",
                        color=f"{color_map_[pair[1]]}",
                    )
                    ax_y.set_title(f"{area_x} -> {area_y}")
                    ax_y.axhline(y=0, color="k", linestyle="dashed", linewidth=1.5)
            for ax in axes[-1, :]:
                ax.set_xlabel("Time lag (ms)")

            for ax in axes[:, 0]:
                ax.set_ylabel("R2")

            axes[-1, -1].legend()
            axes[-1, -1].set_ylim([-0.5, 1.1])

            plt.suptitle(f"{session}", y=0.95)
