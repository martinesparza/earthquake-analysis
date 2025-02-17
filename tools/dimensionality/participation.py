import random

import numpy as np
import pandas as pd
import pyaldata as pyal
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm


def participation_ratio(explained_variances):
    """
    Estimate the number of "important" components based on explained variances

    Parameters
    ----------
    explained_variances : 1D np.ndarray
        explained variance per dimension

    Returns
    -------
    dimensionality estimated using participation ratio formula
    """
    return np.sum(explained_variances) ** 2 / np.sum(explained_variances**2)


def pca_pr(arr, n_components=None):
    """
    Estimate the data's dimensionality using PCA and participation ratio

    Parameters
    ----------
    arr : 2D array
        n_samples x n_features data

    Returns
    -------
    estimated dimensionality
    """
    if n_components is None:
        n_components = arr.shape[-1]
    model = PCA(n_components=n_components, svd_solver="full")
    pca = model.fit(arr)
    return participation_ratio(pca.explained_variance_)


def plot_participation_ratio_per_session(
    df, areas, ax, epoch=None, n_components=None, num_iterations=15
):
    results = {area: {"free": [], "inter": [], "trial": []} for area in areas}

    df_trials = pyal.select_trials(df, df.trial_name == "trial")
    # df_trials = pyal.select_trials(df_trials, "idx_trial_end > 30365")
    # df_trials = df_trials[
    #     df_trials["idx_motion"].apply(lambda x: np.any(x < df_trials.idx_sol_on[0]))
    # ]

    df_intertrials = pyal.select_trials(df, df.trial_name == "intertrial")
    df_free = pyal.select_trials(df, df.trial_name == "free")

    if epoch is not None:
        df_trials = pyal.restrict_to_interval(df_trials, epoch_fun=epoch)

    lower_d_area = areas[np.argmin([df[f"{area}_rates"][0].shape[-1] for area in areas])]
    lowest_d = df[f"{lower_d_area}_rates"][0].shape[-1]

    for area in areas:

        if area != lower_d_area:
            print(
                f"Bootstrapping {area} data with {lowest_d} neurons and {num_iterations} iterations"
            )
            for i in tqdm(range(num_iterations)):
                selected_indices = random.sample(
                    range(df[f"{area}_rates"][0].shape[-1]), lowest_d
                )

                free_data = pyal.concat_trials(df_free, f"{area}_rates")
                results[area]["free"].append(
                    pca_pr(free_data[:, selected_indices], n_components=n_components)
                )

                trial_data = pyal.concat_trials(df_trials, f"{area}_rates")
                results[area]["trial"].append(
                    pca_pr(trial_data[:, selected_indices], n_components=n_components)
                )

                intertrial_data = pyal.concat_trials(df_intertrials, f"{area}_rates")
                results[area]["inter"].append(
                    pca_pr(intertrial_data[:, selected_indices], n_components=n_components)
                )
        else:
            free_data = pyal.concat_trials(df_free, f"{area}_rates")
            results[area]["free"].append(pca_pr(free_data, n_components=n_components))

            trial_data = pyal.concat_trials(df_trials, f"{area}_rates")
            results[area]["trial"].append(pca_pr(trial_data, n_components=n_components))

            intertrial_data = pyal.concat_trials(df_intertrials, f"{area}_rates")
            results[area]["inter"].append(
                pca_pr(intertrial_data, n_components=n_components)
            )

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
        sns.pointplot(
            data=data,
            x="Condition",
            y="PR",
            capsize=0.2,
            ax=ax[i],
            color="blue",
        )
        ax[i].set_title(area)

    return ax
