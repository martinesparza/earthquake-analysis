import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.extractors as se
import numpy as np

def plot_ap_and_lf_traces(spikeglx_recording, start_time, duration, channels=10, ap_label="AP Trace"):
    """
    Plots AP traces from a SpikeInterface SpikeGLX recording object.
    Each channel is plotted in a separate subplot without axis box edges.

    Parameters:
    - spikeglx_recording: SpikeInterface recording object (e.g., SpikeGLXRecordingExtractor).
    - start_time: Start time (in seconds) for the segment to plot.
    - duration: Duration (in seconds) of the trace to plot.
    - channels: Number of channels to plot.
    - ap_label: Label for the plot (used as the figure title).
    """
    # Extract sampling rates
    ap_sampling_rate = spikeglx_recording.get_sampling_frequency()
    
    # Convert time to sample indices
    start_sample_ap = int(start_time * ap_sampling_rate)
    end_sample_ap = int((start_time + duration) * ap_sampling_rate)

    # Get AP traces
    ap_trace = spikeglx_recording.get_traces(segment_index=0, start_frame=start_sample_ap, end_frame=end_sample_ap)
    
    # Time vector for plotting
    ap_time = start_time + (1 / ap_sampling_rate) * np.arange(ap_trace.shape[0])
    
    # Plot each channel in a separate subplot
    fig, axes = plt.subplots(channels, 1, figsize=(12, channels * 2), sharex=True)
    
    for i, ax in enumerate(axes):
        ax.plot(ap_time, ap_trace[:, 250 + i], lw=0.8, color="black")  # Plot channel i
        ax.axis("off")  # Remove axis edges and ticks
    
    # Add a title to the figure
    fig.suptitle(ap_label, fontsize=16, y=0.92)
    
    plt.tight_layout()
    plt.show()

# Usage example
# spikeglx_recording = se.read_spikeglx("path_to_spikeglx_folder")
# plot_ap_and_lf_traces(spikeglx_recording, start_time=0, duration=1, channels=10, ap_label="AP Trace")
