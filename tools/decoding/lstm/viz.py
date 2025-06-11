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
    bin_size=0.03,
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
