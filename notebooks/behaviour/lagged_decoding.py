from __future__ import annotations

import sys

sys.path.append("../../")


import pickle
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyaldata as pyal
import scipy
import seaborn as sns
from joblib import Parallel, delayed

# from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)
from tqdm import tqdm
from tqdm.auto import tqdm

import tools.dataTools as dt
import tools.decoding as decode
import tools.dimensionality as dim
import tools.dsp as dsp
import tools.kinematics as kin
import tools.reports as reports
import tools.subspaces as subspaces
import tools.viz.utilityTools as vizutils
from tools.params import Params, colors


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

FORE_KEYWORDS = [
    "wrist",
    "paw",
    "elbow",
]

HIND_KEYWORDS = [
    "foot",
    "ankle",
    "knee",
]


def opt_ridge_alpha(x_cv, y_cv):
    alphas = np.logspace(-4, 6, 50)
    ridge_cv = RidgeCV(alphas=alphas)
    ridge_cv.fit(x_cv, y_cv)
    return ridge_cv.alpha_


def shift_time_no_wrap(arr, shift, fill_value=np.nan):
    """
    Shift along time axis=1 without wrap-around.
    Positive shift moves data to later times (right).
    Negative shift moves data to earlier times (left).
    """
    arr = np.asarray(arr)
    out = np.full_like(
        arr,
        fill_value=fill_value,
        dtype=float if np.isnan(fill_value) else arr.dtype,
    )

    if shift == 0:
        return arr.copy()

    T = arr.shape[1]

    if shift > 0:
        # t -> t+shift
        out[:, shift:T, :] = arr[:, : T - shift, :]
    else:
        s = -shift
        # t -> t-s
        out[:, : T - s, :] = arr[:, s:T, :]

    return out


def score_one_lag(val_, t, lag, y):
    # Build X for this lag/timebin
    val = shift_time_no_wrap(val_, lag)
    # val = np.array([dt.add_history(arr, n_hist=3) for arr in val])
    val = val[:, t : t + window, :]
    X = val.reshape(-1, val.shape[-1])

    # Fit/evaluate, and keep estimators to compute fold test predictions
    cv_out = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scorers,
        n_jobs=1,
        return_train_score=False,
        return_estimator=True,
    )

    # Build out-of-fold predictions + truths (concatenate folds)
    y_true_oof = []
    y_pred_oof = []
    test_indices = []  # optional: to reconstruct ordering

    # Recreate the exact same splits used by cross_validate
    # (safe because we passed a CV object, not an int)
    splits = list(cv.split(X, y))

    for (train_idx, test_idx), est in zip(splits, cv_out["estimator"]):
        y_true_oof.append(y[test_idx])
        y_pred_oof.append(est.predict(X[test_idx]))
        test_indices.append(test_idx)

    y_true_oof = np.concatenate(y_true_oof, axis=0)  # (N, y_dim)
    y_pred_oof = np.concatenate(y_pred_oof, axis=0)  # (N, y_dim)
    test_indices = np.concatenate(test_indices, axis=0)  # (N,)

    # OPTIONAL sanity check: compute OOF MSE directly (positive)
    mse_oof = mean_squared_error(y_true_oof, y_pred_oof)

    return {
        "r2_vw": cv_out["test_r2_vw"],  # (cv,)
        "r2_custom": cv_out["test_r2_custom"],  # (cv,)
        "mse": cv_out["test_mse"],  # flip to positive MSE per fold
        "y_true_oof": y_true_oof,  # (N, y_dim)
        "y_pred_oof": y_pred_oof,  # (N, y_dim)
        "test_idx_oof": test_indices,  # (N,)
        "mse_oof": mse_oof,  # scalar (OOF)
    }


########################### Begin pipeline ###############################

data_dir = "/data/bnd-data/raw/"
session = "M061_2025_03_06_14_00"
feature_dims = np.arange(start=2, stop=len(ALL_BHV_FIELDS) * 3, step=3)

df_ = pyal.load_pyaldata(data_dir + session[:4] + "/" + session)
df_ = dsp.preprocess(
    df_,
    only_trials=False,
    combine_time_bins=False,
    # repair_time_varying_fields=['MotSen1_X', 'MotSen1_Y']
)

df__ = dt.add_pca_df(df_)

df = dt.add_bhv(df__, bhv_fields=ALL_BHV_FIELDS)
df = dt.concat_previous_intertrial_signal(df, "MOp_rates_pca")
df = dt.concat_previous_intertrial_signal(df, "SSp_rates_pca")
df = dt.concat_previous_intertrial_signal(df, "CP_rates_pca")
df = dt.concat_previous_intertrial_signal(df, "VAL_rates_pca")
df = dt.concat_previous_intertrial_signal(df, "bhv", features=feature_dims)

# Compute metric
df = dt.add_concat_perturb_time(df)
df = dt.add_concat_trial_start(df)
df = kin.compute_power_in_bhv_concat_td(df)
df = kin.add_power_metric_to_td(df)

# Select trials and drop nans
# df = dt.add_bhv(df, bhv_fields=['hip_center'])
df = dt.concat_previous_intertrial_signal(df, "bhv")
df_tr = pyal.select_trials(df, df.trial_name == "trial")
df_tr = kin.drop_immobile_trials_from_td(df_tr, p=2, win=10)
mask = df_tr["disturb_score"].apply(lambda x: not np.any(np.isnan(x)))
df_tr = df_tr.loc[mask]

# Slice fields
df_tr_sliced = df_tr.copy()
fields = [
    "power",
    "phases",
    "MOp_rates_pca_concat",
    "SSp_rates_pca_concat",
    "VAL_rates_pca_concat",
    "CP_rates_pca_concat",
    "bhv_concat",
]
for field in fields:
    df_tr_sliced[field] = df_tr.apply(
        lambda row: row[field][
            row["concat_perturb_time"] - 300 : row["concat_perturb_time"] + 400, :
        ],
        axis=1,
    )


disturbances = np.sum(np.stack(df_tr_sliced.disturb_score.values), axis=1)
dstrb_idx = np.argsort(disturbances)

df_design = df_tr_sliced.copy()
df_design = df_design.iloc[sorted(dstrb_idx[:175])]


df_design["disturb_score"] = df_design["disturb_score"].apply(
    lambda a: np.log10(np.abs(np.sum(a)))
)
df_design_ = df_design.sample(frac=1, random_state=42).reset_index(drop=True)

# import numpy as np
# from sklearn.metrics import make_scorer, r2_score, mean_squared_error
# from tqdm.auto import tqdm

time_bins = np.arange(100, 450, step=1)
lags = np.arange(-25, 25, step=1)
window = 20


vw_r2_scorer = make_scorer(r2_score, multioutput="variance_weighted")
custom_r2_scorer = decode.get_custom_scorer()

scorers = {
    "r2_vw": vw_r2_scorer,
    "r2_custom": custom_r2_scorer,
    "mse": "neg_mean_squared_error",
}
perturb_td = pyal.restrict_to_interval(df_design_, "idx_sol_on", rel_start=-200, rel_end=300)
areas = ["MOp", "SSp", "CP", "VAL"]

results_global = {}
left = [
    "left_ankle",
    "left_elbow",
    "left_foot",
    "left_knee",
    "left_paw",
    "left_wrist",
]
right = ["right_ankle", "right_elbow", "right_foot", "right_knee", "right_paw", "right_wrist"]


other = [
    "hip_center",
    "shoulder_center",
    "tail_base",
    "tail_middle",
    "tail_tip",
    "left_shoulder",
    "right_shoulder",
]
for keypoint in left:
    print(keypoint)
    results_global[keypoint] = {}

    power_ = np.stack(perturb_td[f"{keypoint}"].values)
    for area in areas:
        print(area)
        val_ = np.stack(perturb_td[f"{area}_rates_pca"].values)

        alpha = opt_ridge_alpha(
            val_.reshape(-1, val_.shape[-1]), power_.reshape(-1, power_.shape[-1])
        )
        print(alpha)

        model = Ridge(alpha=alpha)

        # IMPORTANT: make CV object once so the split is identical across lags (recommended)
        cv = KFold(n_splits=5, shuffle=False)

        results_global[keypoint][area] = {}  # results[t][metric] -> arrays, plus preds

        for t in tqdm(time_bins, desc="time bins"):
            power = power_[:, t : t + window, :]
            y = power.reshape(-1, power.shape[-1])

            per_lag = []
            for lag in lags:
                per_lag.append(score_one_lag(val_, t, lag, y))

            results_global[keypoint][area][t] = {
                "r2_vw": np.stack([d["r2_vw"] for d in per_lag], axis=0),  # (n_lags, cv)
                # "r2_custom": np.stack(
                #     [d["r2_custom"] for d in per_lag], axis=0
                # ),  # (n_lags, cv)
                # "mse": np.stack(
                #     [d["mse"] for d in per_lag], axis=0
                # ),  # (n_lags, cv) (note: still neg if you keep 'neg_mean_squared_error')
                # "mse_oof": np.array([d["mse_oof"] for d in per_lag]),  # (n_lags,)
                # "y_true_oof": [d["y_true_oof"] for d in per_lag],
                # "y_pred_oof": [d["y_pred_oof"] for d in per_lag],
                # "test_idx_oof": [d["test_idx_oof"] for d in per_lag],
            }


with open("/data/equake_results/lagged_results_all_areas_left_keypoints.pkl", "wb") as f:
    pickle.dump(results_global, f)
