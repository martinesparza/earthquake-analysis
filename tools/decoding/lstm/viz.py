import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score


def plot_trial_lstm_example(
    preds,
    labels,
    area,
    keypoints,
    results_dir,
    trial_range=np.arange(2),
):
    os.makedirs(results_dir + "plots", exist_ok=True)
    times = np.arange(preds.shape[1])
    for keypoint in keypoints:
        for trial in trial_range:
            fig, axes = plt.subplots(3, 1, sharex="all", figsize=(13, 6))
            axes[-1].set_xlabel("Time (s)")
            for i, (dim_, ax) in enumerate(zip(["x", "y", "z"], axes)):
                ax.plot(times, labels[trial, :, i], label="label")
                ax.plot(times, preds[trial, :, i], label="pred")
                r2 = r2_score(
                    labels[trial, :, i], preds[trial, :, i], multioutput="raw_values"
                )
                ax.set_title(f"Area: {area} Keypoint: {keypoint}. R2: {r2}")
                # ax.axvline(x=0, color="r", linestyle="dashed", label="Perturb. onset")
                ax.set_ylabel(f"{dim_}")
            axes[-1].legend()
            fig.savefig(f"{results_dir}plots/{area}_{keypoint}_{trial}", bbox_inches="tight")


def plot_top_trials_lstm_example(
    preds, labels, area, keypoints, results_dir, perturb_onset, n_trials_to_plot: int = 2
):
    os.makedirs(results_dir + "plots", exist_ok=True)
    times = np.arange(preds.shape[1])

    # Loop over each trial in trial_range to compute R² for each trial
    r2_scores = []
    for trial in range(labels.shape[0]):
        # Calculate R² for each dimension (x, y, z)
        r2 = r2_score(
            labels[trial, :, :], preds[trial, :, :], multioutput="variance_weighted"
        )
        # Calculate average R² for the trial
        r2_scores.append(r2)

    # Sort trials by R² in descending order
    sorted_trials = np.argsort(r2_scores)[::-1]  # Sort in descending order

    # Select top 5 trials with highest R²
    top_trials = sorted_trials[:n_trials_to_plot]

    # Plot for top 5 trials
    for trial in top_trials:
        col_counter = 0
        for keypoint in keypoints:
            fig, axes = plt.subplots(3, 1, sharex="all", figsize=(13, 6))
            axes[-1].set_xlabel("Time (s)")

            if "angle" not in keypoint:
                for i, (dim_, ax) in enumerate(zip(["x", "y", "z"], axes)):
                    ax.plot(times, labels[trial, :, col_counter + i], label="label")
                    ax.plot(times, preds[trial, :, col_counter + i], label="pred")
                    ax.axvline(
                        x=perturb_onset, color="r", linestyle="dashed", label="Perturb. onset"
                    )
                    r2 = r2_score(
                        labels[trial, :, col_counter + i], preds[trial, :, col_counter + i]
                    )
                    ax.set_title(f"Area: {area} Keypoint: {keypoint}. R2: {r2:.2f}")
                    ax.set_ylabel(f"{dim_}")

                col_counter = col_counter + 3
            else:
                for i, (dim_, ax) in enumerate(zip(["x", "y", "z"], axes)):
                    ax.plot(
                        times, labels[trial, :, col_counter : col_counter + i], label="label"
                    )
                    ax.plot(
                        times, preds[trial, :, col_counter : col_counter + i], label="pred"
                    )
                    r2 = r2_scores[trial]
                    ax.axvline(
                        x=perturb_onset, color="r", linestyle="dashed", label="Perturb. onset"
                    )
                    ax.set_title(f"Area: {area} Keypoint: {keypoint}. R2: {r2:.2f}")
                    ax.set_ylabel(f"{dim_}")

                col_counter = col_counter + 1

            axes[-1].legend()
            fig.savefig(
                f"{results_dir}plots/{area}_{keypoint}_trial_{trial}", bbox_inches="tight"
            )
            plt.close()
