#!/usr/bin/env python3
"""
Bootstrap variance-in-disturbance-subspace analysis with efficient parallelisation.

Key fixes vs your original script:
1) Set BLAS/OMP thread env vars BEFORE importing numpy/scipy (prevents oversubscription).
2) Coarsen parallel tasks: parallelise over (n_neurons, area) and run n_iter serially inside each worker
   (reduces job count from 2560 -> 128 and avoids massive pickling/copy overhead).
3) Optionally pass only required columns to workers (df_small) to reduce serialization overhead.

Run (recommended):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python var_in_disturb_space_parallel.py
"""

# -------------------------
# MUST be first: env vars
# -------------------------
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# -------------------------
# Imports
# -------------------------
import pickle
import sys

import numpy as np
import pandas as pd
import scipy  # noqa: F401  (imported for side effects / compatibility)
import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns  # noqa: F401

from sklearn.linear_model import LinearRegression, Ridge  # noqa: F401
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    make_scorer,
)  # noqa: F401
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict  # noqa: F401

from dataclasses import dataclass  # noqa: F401
from tqdm import tqdm  # noqa: F401
from joblib import Parallel, delayed

import pyaldata as pyal

sys.path.append("../../")

import tools.dsp as dsp
from tools.params import Params  # noqa: F401
import tools.dimensionality as dim  # noqa: F401
import tools.decoding as decode  # noqa: F401
import tools.subspaces as subspaces
import tools.viz.utilityTools as vizutils  # noqa: F401
import tools.dataTools as dt
from tools.params import colors  # noqa: F401
import tools.reports as reports  # noqa: F401
import tools.kinematics as kin


# -------------------------
# Worker functions
# -------------------------
def one_bootstrap_job(
    df_base,
    area,
    n,
    i,
    time_window=(300, 500),
    n_rrr=12,
    n_keep=150,
    seed0=0,
):
    """
    One bootstrap iteration for a given (area, n_neurons).
    Returns (n, area, i, val) where val is disturb_var_ratio or None.
    """
    # Deterministic per (n, i, area)
    rng = np.random.default_rng(seed0 + 100000 * int(n) + 1000 * int(i) + (hash(area) % 1000))

    # Copy inside worker to avoid cross-talk between iterations
    df = df_base.copy()

    total_neurons = df[f"{area}_rates"].values[0].shape[1]
    if n > total_neurons:
        return (int(n), area, int(i), None)

    # NOTE: this samples WITH replacement (same as your original).
    # If you want without replacement, use rng.choice(total_neurons, size=n, replace=False)
    rand_neurons = rng.integers(0, total_neurons, size=int(n))
    df[f"{area}_rates"] = df[f"{area}_rates"].apply(lambda r: r[:, rand_neurons])

    # PCA + concat
    df = dt.add_pca_df(df, pca_fields=f"{area}_rates")
    df = dt.concat_previous_intertrial_signal(df, f"{area}_rates_pca")

    # Select trial rows and drop NaN disturb_score trials
    df_tr = pyal.select_trials(df, df.trial_name == "trial")
    mask = df_tr["disturb_score"].apply(lambda x: not np.any(np.isnan(x)))
    df_tr = df_tr.loc[mask]

    # Slice fields around perturb time
    df_tr_sliced = df_tr.copy()
    fields = ["power", f"{area}_rates_pca_concat"]
    for field in fields:
        df_tr_sliced[field] = df_tr.apply(
            lambda row: row[field][
                row["concat_perturb_time"] - 300 : row["concat_perturb_time"] + 400, :
            ],
            axis=1,
        )

    # Select n_keep lowest-disturbance trials (your logic)
    disturbances = np.sum(np.stack(df_tr_sliced.disturb_score.values), axis=1)
    dstrb_idx = np.argsort(disturbances)
    df_tr_sliced = df_tr_sliced.iloc[sorted(dstrb_idx[:n_keep])]

    # Fit subspace on the chosen time window
    arrx = np.stack(df_tr_sliced[f"{area}_rates_pca_concat"].values)[
        :, time_window[0] : time_window[1], :
    ]
    arrx = arrx.reshape(-1, arrx.shape[-1])

    arry = np.stack(df_tr_sliced.power.values)[:, time_window[0] : time_window[1], :]
    arry = arry.reshape(-1, arry.shape[-1])

    axis_disturbance = subspaces.compute_embedding_on_arr(
        arrx, arry, subspaces.ReducedRankCommSubspace(n_rrr)
    )

    # Compute variance ratio on full concat (all time)
    arrx_full = np.stack(df_tr_sliced[f"{area}_rates_pca_concat"].values)
    arrx_full = arrx_full.reshape(-1, arrx_full.shape[-1])

    total_var = np.sum(np.var(arrx_full, axis=0))
    disturb_var = subspaces.variance_in_subspace(arrx_full, axis_disturbance)
    return (int(n), area, int(i), float(disturb_var / total_var))


def one_n_area(
    df_base,
    area,
    n,
    n_iter,
    time_window=(300, 500),
    n_rrr=12,
    n_keep=150,
    seed0=0,
):
    """
    Coarse-grained worker: runs all n_iter bootstraps for a single (area, n).
    Returns (n, area, [vals...]) with vals excluding None.
    """
    vals = []
    for i in range(int(n_iter)):
        _, _, _, val = one_bootstrap_job(
            df_base=df_base,
            area=area,
            n=int(n),
            i=i,
            time_window=time_window,
            n_rrr=n_rrr,
            n_keep=n_keep,
            seed0=seed0,
        )
        if val is not None:
            vals.append(val)
    return (int(n), area, vals)


# -------------------------
# Main
# -------------------------
def main():
    data_dir = "/data/bnd-data/raw/"
    session = "M061_2025_03_06_14_00"

    df_ = pyal.load_pyaldata(data_dir + session[:4] + "/" + session)
    df_ = dsp.preprocess(
        df_,
        only_trials=False,
        combine_time_bins=False,
        # repair_time_varying_fields=['MotSen1_X', 'MotSen1_Y']
    )

    ALL_BHV_FIELDS = [
        "left_ankle",
        "left_elbow",
        "left_foot",
        "left_knee",
        "left_paw",
        "left_wrist",
        "right_ankle",
        "right_elbow",
        "right_foot",
        "right_knee",
        "right_paw",
        "right_wrist",
    ]

    feature_dims = np.arange(start=2, stop=len(ALL_BHV_FIELDS) * 3, step=3)

    n_iter = 20
    n_neurons = np.arange(20, 180, step=5)
    areas = ["MOp", "SSp", "CP", "VAL"]

    # Add behaviour and compute metrics (as in your original)
    df_ = dt.add_bhv(df_, bhv_fields=ALL_BHV_FIELDS)
    df_ = dt.concat_previous_intertrial_signal(df_, "bhv", features=feature_dims)
    df_ = dt.add_concat_perturb_time(df_)
    df_ = dt.add_concat_trial_start(df_)
    df_ = kin.compute_power_in_bhv_concat_td(df_)
    df_ = kin.add_power_metric_to_td(df_)

    # Reduce pickling overhead: keep only required columns for workers
    needed = ["trial_name", "disturb_score", "concat_perturb_time", "power"] + [
        f"{a}_rates" for a in areas
    ]
    df_small = df_[needed].copy()

    # Coarse jobs: (area, n) => 4 * 32 = 128 tasks
    jobs = [(area, int(n)) for n in n_neurons for area in areas]
    print(f"Total coarse tasks: {len(jobs)} (each runs n_iter={n_iter} bootstraps)")

    # Run parallel
    results = Parallel(n_jobs=-1, backend="loky", verbose=50)(
        delayed(one_n_area)(
            df_small,
            area,
            n,
            n_iter=n_iter,
            time_window=(300, 500),
            n_rrr=12,
            n_keep=150,
            seed0=0,
        )
        for (area, n) in jobs
    )

    # Aggregate
    bootstrapped_fracs = {int(n): {a: [] for a in areas} for n in n_neurons}
    for n, area, vals in results:
        bootstrapped_fracs[int(n)][area].extend(vals)

    # Save
    out_path = "bootstrapped_fracs.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bootstrapped_fracs, f)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
