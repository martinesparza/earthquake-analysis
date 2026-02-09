import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import seaborn as sns

# from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

from dataclasses import dataclass

import pyaldata as pyal
import sys

sys.path.append("../../")

import tools.dsp as dsp
from tools.params import Params
import tools.dimensionality as dim
import tools.decoding as decode
import tools.subspaces as subspaces
import tools.viz.utilityTools as vizutils
import tools.dataTools as dt
from tools.params import colors
import tools.reports as reports
import tools.kinematics as kin

from tqdm import tqdm


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


bootstrapped_fracs = {}
n_iter = 20
n_neurons = np.arange(20, 180, step=5)
df_ = dt.add_bhv(df_, bhv_fields=ALL_BHV_FIELDS)

# Compute metric
df_ = dt.concat_previous_intertrial_signal(df_, "bhv", features=feature_dims)
df_ = dt.add_concat_perturb_time(df_)
df_ = dt.add_concat_trial_start(df_)
df_ = kin.compute_power_in_bhv_concat_td(df_)
df_ = kin.add_power_metric_to_td(df_)

for n in n_neurons:
    print(f"Number of neurons : {n}")
    frac_expl_var = {}
    for area in ["MOp", "SSp", "CP", "VAL"]:
        frac_expl_var[area] = []

        for i in range(n_iter):

            df = df_.copy()

            total_neurons = df[f"{area}_rates"].values[0].shape[1]
            if n > total_neurons:
                continue
            rand_neurons = np.random.randint(0, total_neurons, n)
            df[f"{area}_rates"] = df[f"{area}_rates"].apply(lambda r: r[:, rand_neurons])

            df = dt.add_pca_df(df, pca_fields=f"{area}_rates")
            df = dt.concat_previous_intertrial_signal(df, f"{area}_rates_pca")

            # Select trials and drop nans
            df_tr = pyal.select_trials(df, df.trial_name == "trial")
            # df_tr = kin.drop_immobile_trials_from_td(df_tr, p=2, win=10)
            mask = df_tr["disturb_score"].apply(lambda x: not np.any(np.isnan(x)))
            df_tr = df_tr.loc[mask]

            # Slice fields
            df_tr_sliced = df_tr.copy()
            fields = [
                f"power",
                f"{area}_rates_pca_concat",
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

            time_window = (300, 500)

            df_tr_sliced = df_tr_sliced.iloc[sorted(dstrb_idx[:150])]

            arrx = np.stack(df_tr_sliced[f"{area}_rates_pca_concat"].values)[
                :, time_window[0] : time_window[1], :
            ]
            _, _, feats = arrx.shape
            arrx = arrx.reshape(-1, feats)
            arry = np.stack(df_tr_sliced.power.values)[:, time_window[0] : time_window[1], :]
            _, _, feats = arry.shape
            arry = arry.reshape(-1, feats)

            axis_disturbance = subspaces.compute_embedding_on_arr(
                arrx, arry, subspaces.ReducedRankCommSubspace(12)
            )

            arrx = np.stack(df_tr_sliced[f"{area}_rates_pca_concat"].values)
            _, _, feats = arrx.shape
            arrx = arrx.reshape(-1, feats)

            total_var = np.sum(np.var(arrx, axis=0))
            disturb_var = subspaces.variance_in_subspace(arrx, axis_disturbance)
            disturb_var_ratio = disturb_var / total_var

            frac_expl_var[area].append(disturb_var_ratio)
    bootstrapped_fracs[n] = frac_expl_var

with open("bootstrapped_fracs.pkl", "wb") as f:
    pickle.dump(bootstrapped_fracs, f)
