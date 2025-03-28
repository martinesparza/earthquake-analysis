import matplotlib.pyplot as plt
import numpy as np
import pyaldata as pyal

from tools import params


def compare_mean_firing(ax, df, trial_types=["trial", "intertrial"], areas=["M1", "Dls"]):
    mean_values = []
    std_errors = []

    i = 0
    for area in areas:
        # Concatenate all timepoints to get (timepoints * trials, neurons)
        for trial_type in trial_types:
            df___ = pyal.select_trials(df, f"trial_name == '{trial_type}'")
            rates = np.concatenate(df___[f"{area}_rates"].values, axis=0)

            # Compute mean firing rate per neuron
            mean_firing_rates = rates.mean(axis=0)

            # Compute overall mean and standard error
            mean_area = mean_firing_rates.mean()
            std_error = mean_firing_rates.std() / np.sqrt(
                len(mean_firing_rates)
            )  # Standard Error of the Mean (SEM)

            mean_values.append(mean_area)
            std_errors.append(std_error)

            # Jitter x positions for scatter plot to avoid overlap
            jitter_x = np.random.normal(i, 0.05, size=len(mean_firing_rates))

            # Scatter plot of individual neuron firing rates
            ax.scatter(
                jitter_x,
                mean_firing_rates,
                alpha=0.6,
                label=f"{area}",
                color=getattr(params.colors, area, "k"),
            )
            i += 1
    # Overlay mean firing rate with error bars
    ax.errorbar(
        range(len(areas) * len(trial_types)),
        mean_values,
        yerr=std_errors,
        fmt="o",
        capsize=5,
        markersize=5,
        color="k",
        label="Mean ± SEM",
    )

    # Labels and formatting
    ax.set_xticks(range(len(areas) * len(trial_types)))
    ax.set_xticklabels(trial_types * len(areas))
    ax.set_ylabel("Mean Firing Rate")
    ax.set_title("Mean firing rates, free trials")
    # ax.set_ylim([0, 50])
    ax.legend()

    plt.show()
