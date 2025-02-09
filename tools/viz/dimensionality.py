import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyaldata as pyal
import seaborn as sns
from sklearn.decomposition import PCA

from tools import params
from tools.viz import utilityTools as utility


def plot_VAF(
    ax,
    data_list: list[pd.DataFrame],
    epoch=None,
    areas=["all"],
    model=None,
    n_components=None,
    n_neighbors=10,
    linestyle="-",
    show=True,
):
    """
    Plot VAF for each area in areas list, averaged across sessions in data_list, with shaded errorbars.
    """
    if isinstance(data_list, pd.DataFrame):
        data_list = [data_list]
    if isinstance(areas, str):
        areas = [areas]
    if isinstance(areas, dict):
        units_per_area = list(areas.values())
        areas = list(areas.keys())
    else:
        units_per_area = None

    for i, area in enumerate(areas):
        field = f"{area}_rates" if units_per_area is None else "all_rates"
        VAF_per_area = []
        for session, df in enumerate(data_list):
            df_ = (
                pyal.restrict_to_interval(df, epoch_fun=epoch) if epoch is not None else df
            )
            rates = np.concatenate(df_[field].values, axis=0)
            if units_per_area is not None:
                rates = rates[:, units_per_area[i][0] : units_per_area[i][1]]

            n_components = rates.shape[-1]

            model = PCA(n_components=n_components, svd_solver="full")
            rates_model = model.fit(rates)
            if isinstance(model, PCA):
                explained_variance_ratio = model.explained_variance_ratio_
            cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
            VAF_per_area.append(cumulative_explained_variance_ratio)
        x_ = np.arange(1, n_components + 1)
        utility.shaded_errorbar(
            ax,
            x_,
            np.array(VAF_per_area).T,
            label=area,
            color=getattr(params.colors, area, "k"),
            linestyle=linestyle,
        )

    ax.set_xlabel("Number of PCs ")
    ax.set_ylabel("VAF (%)")
    ax.set_title("Variance accounted for by PCs")
    ax.axhline(y=0.8, color="red", linestyle="--")
    ax.legend()
    if show:
        plt.show()
    else:
        return ax


def plot_pairwise_corr(ax, df, areas, epoch):
    """
    Plot pairwise correlation for one session for each area in areas list.
    """
    if isinstance(areas, str):
        areas = [areas]
    if len(areas) == 1:
        ax = [ax]
    for i, area in enumerate(areas):
        field = f"{area}_rates"

        df_ = pyal.restrict_to_interval(df, epoch_fun=epoch) if epoch is not None else df
        rates = np.concatenate(df_[field].values, axis=0)
        correlation_matrix = np.corrcoef(rates.T)
        sns.heatmap(
            correlation_matrix,
            annot=False,
            fmt=".2f",
            cmap=params.colors.corr_cmap,
            cbar=True,
            ax=ax[i],
        )

        # Set titles and labels
        ax[i].set_title(area)
        ax[i].set_xlabel("Neuron #")
        ax[i].set_ylabel("Neuron #")
        plt.show()
