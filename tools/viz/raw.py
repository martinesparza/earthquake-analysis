import matplotlib.pyplot as plt
import numpy as np


# Plotting function
def plot_npx_traces(
    time, lf_data, ap_data, lf_label="LF Trace", ap_label="AP Trace", ax=None, show=True
):
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot LF trace
    ax[0].plot(time, lf_data, color="blue", lw=0.8)
    ax[0].set_ylabel("Amplitude (LF)", fontsize=12)
    ax[0].set_title(lf_label, fontsize=14)
    ax[0].grid(True, linestyle="--", alpha=0.5)

    # Plot AP trace
    ax[1].plot(time, ap_data, color="red", lw=0.8)
    ax[1].set_ylabel("Amplitude (AP)", fontsize=12)
    ax[1].set_xlabel("Time (s)", fontsize=12)
    ax[1].set_title(ap_label, fontsize=14)
    ax[1].grid(True, linestyle="--", alpha=0.5)

    # Adjust layout
    plt.tight_layout()
    if show:
        plt.show()

    return ax
