import sys

sys.path.append("../../")

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyaldata as pyal
import seaborn as sns

import tools.dataTools as dt
import tools.decoding.decodeTools as decutils
import tools.decoding.rrr as rrr
import tools.viz.dimensionality as dim
import tools.viz.mean_firing as firing
import tools.viz.utilityTools as vizutils
from tools.curbd import curbd_gpu_v2
from tools.dsp.preprocessing import preprocess
from tools.params import Params, colors
from tools.reports.report_initial import run_initial_report

# Files
session = "M062_2025_03_20_14_00"
data_dir = f"/data/bnd-data/raw/M062/{session}"

areas = ["MOp", "SSp", "CP", "VAL"]
df = pyal.load_pyaldata(data_dir)

df_ = preprocess(df, only_trials=False, repair_time_varying_fields=["MotSen1_X", "MotSen1_Y"])

activity_ = []
for sol_dir in range(12):
    df_trials = pyal.select_trials(df_, df_.values_Sol_direction == 1)
    df_trials = df_trials.iloc[:-1]
    trial_length = df_trials.MOp_rates.values[0].shape[0]

    activity = []
    neurons = []
    for area in areas:
        neurons.append(df_trials[f"{area}_rates"][0].shape[-1])
        activity.append(pyal.concat_trials(df_trials, f"{area}_rates").T)

    # Activity
    activity = np.concatenate(activity)
    activity_.append(activity[np.newaxis, :, :])

activity_ = np.concatenate(activity_)

# Regions
regions = []

start_idx = 0
for area, size in zip(areas, neurons):
    regions.append([area, np.arange(start_idx, start_idx + size)])
    start_idx += size  # Update the starting index for the next region

gcurbd = curbd_gpu_v2.gCURBD(
    dt_data=df_.bin_size[0], dt_factor=3, regions=regions, train_epochs=150
)

print("Start fitting")
gcurbd.fit(activity_)

del gcurbd

# You can clear the memory pool by calling `free_all_blocks`.

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
mempool.free_all_blocks()
pinned_mempool.free_all_blocks()
print(mempool.used_bytes())  # 0
print(mempool.total_bytes())  # 0
print(pinned_mempool.n_free_blocks())  # 0
