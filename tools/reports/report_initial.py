import pandas as pd
import pyaldata as pyal
from matplotlib import pyplot as plt

from tools.dsp.preprocessing import preprocess
from tools.viz.dimensionality import plot_latents, plot_VAF
from tools.viz.rasters import plot_heatmap_raster


def run_initial_report(df: pd.DataFrame, areas: list, trial_selection_criteria=None):

    # Preprocess
    df = preprocess(df, only_trials=False, trial_selection_criteria=trial_selection_criteria)

    # Take the three trial types
    df_trials = pyal.select_trials(df, df.trial_name == "trial")
    df_intertrials = pyal.select_trials(df, df.trial_name == "intertrial")
    df_free = pyal.select_trials(df, df.trial_name == "free")

    # Initial raster per region
    for area in areas:
        plot_heatmap_raster(
            df_trials[50:60],
            area=area,
            add_sol_onset=True,
        )

    # Variance account for
    fig, ax = plt.subplots()
    plot_VAF(
        ax=ax,
        data_list=df_trials,
        areas=areas,
    )

    # Latents
    plot_latents(df, areas=areas)
