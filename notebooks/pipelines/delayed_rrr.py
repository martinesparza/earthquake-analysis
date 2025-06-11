import pickle
import sys

sys.path.append("../../")
import numpy as np
import pyaldata as pyal

import tools.decoding.rrr as rrr
from tools.dsp.preprocessing import preprocess

# epochs
BIN_SIZE = 0.01
WINDOW_perturb = (0, 1.5)
perturb_epoch = pyal.generate_epoch_fun(
    start_point_name="idx_sol_on",
    rel_start=int(WINDOW_perturb[0] / BIN_SIZE),
    rel_end=int(WINDOW_perturb[1] / BIN_SIZE),
)

WINDOW_before_perturb = (-0.6, -0.1)
before_perturb_epoch = pyal.generate_epoch_fun(
    start_point_name="idx_sol_on",
    rel_start=int(WINDOW_perturb[0] / BIN_SIZE),
    rel_end=int(WINDOW_perturb[1] / BIN_SIZE),
)

WINDOW_perturb_long = (-1, 3)
perturb_epoch_long = pyal.generate_epoch_fun(
    start_point_name="idx_sol_on",
    rel_start=int(WINDOW_perturb_long[0] / BIN_SIZE),
    rel_end=int(WINDOW_perturb_long[1] / BIN_SIZE),
)

# Concatenate DF

areas = ["MOp", "SSp", "CP", "VAL"]

sessions = [
    # 'M046_2024_12_18_16_00'
    # 'M046_2024_12_19_13_30',
    "M061_2025_03_04_10_00",
    "M061_2025_03_05_14_00",
    "M061_2025_03_06_14_00",
    "M062_2025_03_19_14_00",
    "M062_2025_03_20_14_00",
    "M062_2025_03_21_14_00",
    "M063_2025_03_12_14_00",
    "M063_2025_03_13_14_00",
    "M063_2025_03_14_15_30",
]

data_dir = "/data/bnd-data/raw/"

df_ = []
for session in sessions:
    df = pyal.load_pyaldata(data_dir + session[:4] + "/" + session)
    df = preprocess(df, only_trials=False, combine_time_bins=False)

    df_.append(df)

bin_size_ms = 10
shifts = np.arange(-200, 210, bin_size_ms)  # Provide it in BINs
rrr_dict = {}
for session, tmp_df in zip(sessions, df_):
    rrr_dict[session] = {}
    print(f"Session: {session}")

    print("\tCondition free0")
    rrr_dict[session]["free"] = rrr.delayed_rrr_on_df(
        tmp_df,
        areas,
        condition="free",
        verbose=False,
        k_folds=5,
        free_period=0,
        shifts=shifts,
        bin_size=bin_size_ms / 1000,
    )

    print("\tCondition free1")
    rrr_dict[session]["free1"] = rrr.delayed_rrr_on_df(
        tmp_df,
        areas,
        condition="free",
        verbose=False,
        k_folds=5,
        free_period=1,
        shifts=shifts,
        bin_size=bin_size_ms / 1000,
    )

    print("\tCondition trial_short")
    rrr_dict[session]["trial_short"] = rrr.delayed_rrr_on_df(
        tmp_df,
        areas,
        condition="trial",
        verbose=False,
        k_folds=5,
        shifts=shifts,
        bin_size=bin_size_ms / 1000,
        epoch=perturb_epoch,
    )

    print("\tCondition trial_long")
    rrr_dict[session]["trial_long"] = rrr.delayed_rrr_on_df(
        tmp_df,
        areas,
        condition="trial",
        verbose=False,
        k_folds=5,
        shifts=shifts,
        bin_size=bin_size_ms / 1000,
        epoch=perturb_epoch_long,
    )

    print("\tCondition intertrial")
    rrr_dict[session]["intertrial"] = rrr.delayed_rrr_on_df(
        tmp_df,
        areas,
        condition="intertrial",
        verbose=False,
        k_folds=5,
        shifts=shifts,
        bin_size=bin_size_ms / 1000,
    )

results_path = "/home/me24/repos/earthquake-analysis/results/rrr/"
with open(results_path + "delayed_rrr_all_v2.pkl", "wb") as f:
    pickle.dump(rrr_dict, f)
