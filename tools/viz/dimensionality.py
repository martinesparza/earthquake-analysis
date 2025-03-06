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


def plot_participation_ratio_per_session(
    df, areas, epoch=None, trial_query=None, title=None
):
    results = {area: {"free": [], "inter": [], "trial": []} for area in areas}

    df_trials = pyal.select_trials(df, df.trial_name == "trial")
    df_trials_motion = df_trials[
        df_trials["idx_motion"].apply(lambda x: np.any(x < df_trials.idx_sol_on[0]))
    ]

    df_intertrials = pyal.select_trials(df, df.trial_name == "intertrial")
    df_free = pyal.select_trials(df, df.trial_name == "free")

    if epoch is not None:
        df_trials = pyal.restrict_to_interval(df_trials, epoch_fun=epoch)

    if trial_query is not None:
        print("Applying query")
        print(len(df_trials))
        df_trials = pyal.select_trials(df_trials, trial_query)
        print(len(df_trials))

    for area in areas:
        free_data = pyal.concat_trials(df_free, f"{area}_rates")
        results[area]["free"].append(pca_pr(free_data))

        trial_data = pyal.concat_trials(df_trials, f"{area}_rates")
        results[area]["trial"].append(pca_pr(trial_data))

        intertrial_data = pyal.concat_trials(df_intertrials, f"{area}_rates")
        results[area]["inter"].append(pca_pr(intertrial_data))

    fig, axes = plt.subplots(1, len(areas), sharey=True)

    for i, area in enumerate(areas):
        data = pd.DataFrame(
            {
                "PR": results[area]["free"]
                + results[area]["inter"]
                + results[area]["trial"],
                "Condition": ["free"] * len(results[area]["free"])
                + ["inter"] * len(results[area]["inter"])
                + ["trial"] * len(results[area]["trial"]),
            }
        )
        sns.stripplot(data=data, x="Condition", y="PR", s=8, color="blue", ax=axes[i])
        axes[i].set_title(area)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_latents(df, areas, trial_type="trial", n_components=10, motion_only=True):

    axes = []
    category = "values_Sol_direction"

    df = pyal.select_trials(df, df.trial_name == trial_type)
    sol_on_idx = df["idx_sol_on"][0]

    # Always remove last trial
    df = df[:-1]

    if motion_only:
        df = df[df["idx_motion"].apply(lambda x: np.any(x < df.idx_sol_on[0]))]

    # Define subplot grid dimensions
    n_rows = n_components  # Rows: trial types
    n_cols = len(areas)  # Columns: areas
    targets = np.unique(df[category])

    fig, axes = plt.subplots(n_components, n_cols, figsize=(20, 9), sharex="all")
    axes = np.array(axes).reshape(n_rows, n_cols)

    # Loop through areas (columns) and trial types (rows)
    for col, area in enumerate(areas):
        for row in range(n_components):
            rates = np.concatenate(
                df[area + "_rates"].values, axis=0
            )  # Shape: (239 trials, 15 timepoints, 87 units)

            # Fit PCA model
            rates_model = PCA(n_components=n_components, svd_solver="full").fit(rates)

            # Apply PCA to the dataframe
            df_ = pyal.apply_dim_reduce_model(df, rates_model, area + "_rates", "_pca")

            # Select the correct subplot
            ax = axes[row, col]

            # Loop through targets and plot averaged trials
            for tar in targets:
                df__ = pyal.select_trials(df_, df_[category] == tar)
                ex = pyal.get_sig_by_trial(df__, "_pca")
                ex = np.mean(ex, axis=2)[
                    :, :n_components
                ]  # Reduce to first 3 PCA components
                ax.plot(ex[:, row])

            # Titles and labels
            ax.axvline(x=sol_on_idx, color="r", linestyle="--")
            if row == 0:
                ax.set_title(f"{area}")
            if col == 0:
                ax.set_ylabel(f"PC {row}")
        ax.set_xlabel("Timebins (30ms)")
    plt.show()
    return
