

import os
import sys
sys.path.append("../")

import pyaldata as pyal
import pandas as pd
import numpy as np


# from tools.reports.report_initial import run_initial_report
from tools import dataTools as dt
import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import pyaldata as pyal
from tools.viz import mean_firing as firing
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from tools.params import Params
from tools.decoding import decodeTools as decode
from tools.dsp.preprocessing import preprocess
from tools.params import colors
from tools import dataTools as dt
import torch


# %% [markdown]
# start here

# %%
sessions = [
    # 'M062_2025_03_19_14_00',
    # 'M062_2025_03_20_14_00',
    # 'M062_2025_03_21_14_00',
    # 'M061_2025_03_04_10_00',
    # 'M061_2025_03_05_14_00',
    "M061_2025_03_06_14_00",
    # "M063_2025_03_12_14_00",  
    # "M063_2025_03_13_14_00",  
    # "M063_2025_03_14_15_30",
    # "M078_2025_08_05_15_30",
    # "M078_2025_08_06_15_00",
    # "M078_2025_08_07_13_30",
    # "M078_2025_08_08_10_30",
    # "M086_2025_12_09_16_00",
    # "M086_2025_12_10_15_00",
    # "M086_2025_12_11_15_00"

]
prep_dfs = []
only_trials = False
for session in sessions:
    print(session)
    animal = session.split('_')[0]
    data_dir = f"/data/bnd-data/raw/{animal}/{session}"
    for file in range(4):
        fname = os.path.join(data_dir, f"{session}_pyaldata_{file}.mat")
        if os.path.exists(fname):
            df = pyal.mat2dataframe(fname, shift_idx_fields=False)
            # concatenate the different parts of the session into one dataframe
            if file == 0:
                full_df = df
            else:
                full_df = pd.concat([full_df, df], ignore_index=True)
    prep_df = preprocess(full_df,only_trials=only_trials, repair_time_varying_fields=['MotSen1_X', 'MotSen1_Y'])
    prep_dfs.append(prep_df)

# %%
session_idx = 0
n_components = 10
prep_df = prep_dfs[session_idx]
area = "MOp"
field = f"{area}_rates"
prep_df =dt.add_velocity_fields(prep_df)
prep_df,bhv_names = dt.add_bhv(prep_df, bhv_fields=["all_keypoints_vel"])

# %%
# random windows for free
import numpy as np
from typing import List, Tuple, Optional
def split_free(
    free_duration_s: float = 7 * 60,      
    window_lengths_s: Tuple[float, ...] = (1.0, 3.0, 5.0),
    gap_s: float = 6.0,
    seed: Optional[int] = None,
    start_offset_s: float = 0.0,          # optional random offset before first window
) -> List[Tuple[float, float]]:
    """
    Create a sequential schedule of windows inside a continuous free period:
      [window] then [gap] then [window] then [gap] ...
    Window lengths are randomly drawn from window_lengths_s.

    Returns:
      List of (start_time_s, end_time_s) for each window (in seconds, relative to free start).
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    if start_offset_s > 0:
        max_offset = min(start_offset_s, max(0.0, free_duration_s - min(window_lengths_s)))
        t = float(rng.uniform(0.0, max_offset))
    windows: List[Tuple[float, float]] = []

    while True:
        L = float(rng.choice(window_lengths_s))
        if t + L > free_duration_s:
            break
        windows.append((t, t + L))
        t = t + L + gap_s  

        if t >= free_duration_s:
            break

    return windows
import numpy as np
import matplotlib.pyplot as plt

def window_metric(prep_df, field, windows, trial_name = "free0",metric = np.nanmean):
    """
    Assumes prep_df contains *binned* samples for free0 in time order.
    windows: list of (start_s, end_s) in seconds relative to free0 start.
    Returns: means (np.ndarray), plus (start_bin, end_bin) indices used.
    """
    # Extract the free0 time series for the chosen field
    free0 = prep_df[prep_df.trial_name == trial_name]
    x = np.concatenate(free0[field].values, axis = 0)

    means = []
    used_bins = []
    n = len(x)
    for (t0, t1) in windows:
        print
        i0 = int(np.floor(t0 / Params.BIN_SIZE))  # start-inclusive 
        i1 = int(np.ceil(t1 / Params.BIN_SIZE))   # end-exclusive-ish

        # Clamp to valid range
        i0 = max(0, min(i0, n))
        i1 = max(0, min(i1, n))

        if i1 <= i0:
            continue
        m = metric(x[i0:i1], axis=0)
        means.append(m)
        used_bins.append((i0, i1))

    return np.asarray(means), used_bins

def get_n_time(df, trial_name, field = "MOp_rates"):
    return np.concatenate(df[df['trial_name'] == trial_name][field].values, axis=0).shape[0]

# %%
free0_len = get_n_time(prep_df, "free0", field=field)
windows = split_free(
        free_duration_s=free0_len*Params.BIN_SIZE,
        window_lengths_s=[1,3,5],
        gap_s=1,
        seed=None,
        start_offset_s=0
    )
print(f"n_windows = {len(windows)}")
covered = sum(e - s for s, e in windows)
print(f"total window time = {covered:.1f}s ({covered/60:.2f} min)")

it = pyal.select_trials(prep_df,"trial_name=='intertrial'")

# %% [markdown]
# Cross-validated cross-condition decoding with:
# - K-fold splits for free windows + intertrial trials
# - optional matching (train/test separately) using matching behavioural features (can differ from decoded target)
# - optional global PCA vs condition-specific PCA
# - optional remove_means (demean Y within each segment)
# - incremental saving after EVERY result row (safe checkpointing)
# - final summary CSV

import os
import numpy as np
import pandas as pd
import scipy.linalg

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_squared_error

import pyaldata as pyal

from tools.params import Params
from tools import dataTools as dt
from tools.decoding import regression as reg


# -------------------------
# Utilities: safely add multiple bhv fields
# -------------------------

def add_bhv_as(df, bhv_fields, out_col):
    """
    dt.add_bhv writes to column 'bhv'. This helper calls it and renames 'bhv' -> out_col.
    """
    df2, names = dt.add_bhv(df, bhv_fields=bhv_fields)
    if "bhv" not in df2.columns:
        raise ValueError("dt.add_bhv did not create a 'bhv' column as expected.")
    df2 = df2.rename(columns={"bhv": out_col})
    return df2, names


# -------------------------
# Matching + feature building helpers
# -------------------------

def bhv_window_means(prep_df, windows_list, field, trial_name="free0", metric=np.nanmean, bin_size=Params.BIN_SIZE):
    free0 = prep_df[prep_df.trial_name == trial_name]
    x = np.concatenate(free0[field].values, axis=0)

    means = []
    used_bins = []
    n = len(x)
    for (t0, t1) in windows_list:
        i0 = int(np.floor(t0 / bin_size))
        i1 = int(np.ceil(t1 / bin_size))
        i0 = max(0, min(i0, n))
        i1 = max(0, min(i1, n))
        if i1 <= i0:
            continue
        means.append(metric(x[i0:i1], axis=0))
        used_bins.append((i0, i1))
    return np.asarray(means), used_bins


def bhv_intertrial_means(it_df, idx_list, field):
    return np.vstack(it_df[field].apply(lambda a: np.nanmean(a, axis=0)))[idx_list, :]


def match_by_bhv_pc1(free_means, it_means, free_ids, it_ids, n_comp=10, caliper_q=1.0):
    """
    One-to-one greedy nearest-neighbour matching (no replacement) from free windows to intertrial trials,
    using PC1 of behavioural means.

    caliper_q:
      1.0 keeps all; smaller (e.g. 0.8) keeps only better matches.
    """
    if len(free_means) == 0 or len(it_means) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    all_means = np.vstack([free_means, it_means])
    pca = PCA(n_components=min(n_comp, all_means.shape[1]), svd_solver="full").fit(all_means)

    free_pc1 = pca.transform(free_means)[:, 0:1]
    it_pc1   = pca.transform(it_means)[:, 0:1]

    nn = NearestNeighbors(n_neighbors=1).fit(it_pc1)
    dists, nbrs = nn.kneighbors(free_pc1, return_distance=True)
    dists = dists[:, 0]
    nbrs  = nbrs[:, 0]

    caliper = np.quantile(dists, caliper_q)
    ok = dists <= caliper

    free_ok = np.asarray(free_ids)[ok]
    it_ok   = np.asarray(it_ids)[nbrs[ok]]
    dist_ok = dists[ok]

    used = set()
    pairs = []
    for f_i, it_i, d in sorted(zip(free_ok, it_ok, dist_ok), key=lambda t: t[2]):
        if it_i in used:
            continue
        used.add(it_i)
        pairs.append((f_i, it_i))

    kept_free = np.array([p[0] for p in pairs], dtype=int)
    kept_it   = np.array([p[1] for p in pairs], dtype=int)
    return kept_free, kept_it


def build_lagged_xy(Xw, Yw, lags, remove_means=False):
    max_lag = int(np.max(lags))
    if Yw.ndim == 1:
        Yw = Yw[:, None]

    if remove_means:
        Yw = Yw - np.mean(Yw, axis=0, keepdims=True)

    T = min(len(Xw), len(Yw))
    Xw, Yw = Xw[:T], Yw[:T]

    if T <= max_lag:
        return None, None

    t = np.arange(max_lag, T)
    Xdelays = np.concatenate([Xw[t - lag, :] for lag in lags], axis=1)
    Yv = Yw[t, :]
    return Xdelays, Yv


def collect_intertrial_xy(it_df, idx_list, neural_field, kin_col, pca_model, lags, remove_means):
    Xs, Ys = [], []
    for idx in idx_list:
        Xw = it_df.iloc[idx][neural_field]
        Xw = pca_model.transform(Xw)

        Yw = it_df.iloc[idx][kin_col]
        Xd, Yv = build_lagged_xy(Xw, Yw, lags, remove_means=remove_means)
        if Xd is None:
            continue
        Xs.append(Xd)
        Ys.append(Yv)

    if len(Xs) == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))
    return np.vstack(Xs), np.vstack(Ys)


def collect_free0_xy(prep_df, windows_list, neural_field, kin_col, pca_model, lags, remove_means, free0_len, bin_size=Params.BIN_SIZE):
    Xs, Ys = [], []
    for (t0, t1) in windows_list:
        i0 = int(np.floor(t0 / bin_size))
        i1 = int(np.ceil(t1 / bin_size))
        i0 = max(0, min(i0, free0_len))
        i1 = max(0, min(i1, free0_len))
        if i1 <= i0:
            continue

        Xw = prep_df[neural_field][0][i0:i1, :]
        Xw = pca_model.transform(Xw)

        Yw = prep_df.iloc[0][kin_col][i0:i1, :]

        Xd, Yv = build_lagged_xy(Xw, Yw, lags, remove_means=remove_means)
        if Xd is None:
            continue
        Xs.append(Xd)
        Ys.append(Yv)

    if len(Xs) == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))
    return np.vstack(Xs), np.vstack(Ys)


def eval_decoder(ridge_model, X_test, Y_test):
    pred = ridge_model.predict(X_test)
    r2 = r2_score(Y_test, pred, multioutput="variance_weighted")
    rmse = np.sqrt(mean_squared_error(Y_test, pred))
    return float(r2), float(rmse)


def decoder_subspace_var_fraction(X, ridge_model):
    """
    Fraction of variance in X that lies in the feature-space subspace spanned by ridge weights.
    X: [n_samples, n_features] in the SAME feature space ridge sees (lagged PCA features).
    """
    W = ridge_model.coef_  # [n_targets, n_features]
    if W.ndim != 2 or W.shape[1] != X.shape[1]:
        raise ValueError(f"coef_/X feature mismatch: W {W.shape}, X {X.shape}")

    Q = scipy.linalg.orth(W.T)  # [n_features, rank]
    if Q.size == 0:
        return np.nan

    Xp = X @ Q
    var_pot = np.sum(np.var(Xp, axis=0))
    var_tot = np.sum(np.var(X, axis=0))
    if var_tot <= 0:
        return np.nan
    return float(var_pot / var_tot)


# -------------------------
# Incremental CSV writer
# -------------------------

def append_row_csv(row_dict, csv_path, header_written_flag):
    """
    Append a single row to csv_path. Writes header if header_written_flag is False and file doesn't exist.
    Returns updated header_written_flag.
    """
    row_df = pd.DataFrame([row_dict])
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    if (not file_exists) and (not header_written_flag):
        row_df.to_csv(csv_path, index=False, mode="w")
        return True
    else:
        row_df.to_csv(csv_path, index=False, mode="a", header=False)
        return True


# -------------------------
# Main CV runner with incremental saving
# -------------------------

def run_cv_decode_incremental(
    prep_df,
    windows,
    free0_len,
    neural_field="MOp_rates",
    kin_col_free0="bhv_decode",
    kin_col_intertrial="bhv_decode",
    match_bhv_col_free0="bhv_match",
    match_bhv_col_intertrial="bhv_match",
    n_splits=10,
    seed=0,
    n_components=30,
    lags=(0,1,2,3),
    match_opts=(False, True),
    remove_means_opts=(False, True),
    global_pca_opts=(False, True),
    caliper_q=1.0,
    save_path="decode_cv_results.csv",
    save_every_n_rows_summary=200,   # set None to disable periodic summary
):
    it = pyal.select_trials(prep_df, "trial_name=='intertrial'")
    all_neural = np.concatenate(prep_df[neural_field].values, axis=0)

    kf_win = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    kf_it  = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    win_indices = np.arange(len(windows))
    it_indices  = np.arange(len(it))

    win_splits = list(kf_win.split(win_indices))
    it_splits  = list(kf_it.split(it_indices))

    # If you want to resume by appending, leave existing file.
    # If you want to overwrite, delete it before running:
    # if os.path.exists(save_path): os.remove(save_path)
    header_written = False
    n_rows_written = 0

    # Keep minimal in-memory list for periodic summary (optional)
    rows_for_summary = []

    for fold in range(n_splits):
        win_tr_idx, win_te_idx = win_splits[fold]
        it_tr_idx,  it_te_idx  = it_splits[fold]

        windows_train = [windows[i] for i in win_tr_idx]
        windows_test  = [windows[i] for i in win_te_idx]
        it_train_ids  = it_tr_idx
        it_test_ids   = it_te_idx

        # Matching stats (train/test separately)
        free_means_tr, _ = bhv_window_means(prep_df, windows_train, field=match_bhv_col_free0, trial_name="free0")
        it_means_tr      = bhv_intertrial_means(it, it_train_ids, field=match_bhv_col_intertrial)

        free_means_te, _ = bhv_window_means(prep_df, windows_test, field=match_bhv_col_free0, trial_name="free0")
        it_means_te      = bhv_intertrial_means(it, it_test_ids, field=match_bhv_col_intertrial)

        for match in match_opts:
            if match:
                kept_free_tr, kept_it_tr = match_by_bhv_pc1(
                    free_means_tr, it_means_tr,
                    free_ids=np.arange(len(windows_train)),
                    it_ids=it_train_ids,
                    n_comp=10,
                    caliper_q=caliper_q
                )
                kept_free_te, kept_it_te = match_by_bhv_pc1(
                    free_means_te, it_means_te,
                    free_ids=np.arange(len(windows_test)),
                    it_ids=it_test_ids,
                    n_comp=10,
                    caliper_q=caliper_q
                )
                windows_train_kept = [windows_train[i] for i in kept_free_tr]
                windows_test_kept  = [windows_test[i]  for i in kept_free_te]
                it_train_kept = kept_it_tr
                it_test_kept  = kept_it_te
            else:
                windows_train_kept = windows_train
                windows_test_kept  = windows_test
                it_train_kept = it_train_ids
                it_test_kept  = it_test_ids

            for remove_means in remove_means_opts:
                for global_pca in global_pca_opts:

                    # Fit PCA(s)
                    if global_pca:
                        pca_global = PCA(n_components=n_components, svd_solver="full").fit(all_neural)
                        pca_free = pca_global
                        pca_it_  = pca_global
                    else:
                        # condition-specific PCA on TRAIN samples only
                        X_free_list = []
                        for (t0, t1) in windows_train_kept:
                            i0 = int(np.floor(t0 / Params.BIN_SIZE))
                            i1 = int(np.ceil(t1 / Params.BIN_SIZE))
                            i0 = max(0, min(i0, free0_len))
                            i1 = max(0, min(i1, free0_len))
                            if i1 > i0:
                                X_free_list.append(prep_df[neural_field][0][i0:i1, :])
                        X_free_train = np.concatenate(X_free_list, axis=0) if len(X_free_list) else np.zeros((0, all_neural.shape[1]))

                        X_it_list = [it.iloc[idx][neural_field] for idx in it_train_kept]
                        X_it_train = np.concatenate(X_it_list, axis=0) if len(X_it_list) else np.zeros((0, all_neural.shape[1]))

                        if X_free_train.shape[0] == 0 or X_it_train.shape[0] == 0:
                            continue

                        pca_free = PCA(n_components=n_components, svd_solver="full").fit(X_free_train)
                        pca_it_  = PCA(n_components=n_components, svd_solver="full").fit(X_it_train)

                    # TRAIN matrices
                    Xit_tr, Yit_tr = collect_intertrial_xy(
                        it, it_train_kept, neural_field, kin_col_intertrial, pca_it_, lags, remove_means
                    )
                    Xfr_tr, Yfr_tr = collect_free0_xy(
                        prep_df, windows_train_kept, neural_field, kin_col_free0, pca_free, lags, remove_means,
                        free0_len=free0_len
                    )

                    if Xit_tr.size == 0 or Xfr_tr.size == 0:
                        continue

                    ridge_it = reg.fit_semedo_ridge(Xit_tr, Yit_tr)
                    ridge_fr = reg.fit_semedo_ridge(Xfr_tr, Yfr_tr)

                    # TEST matrices in TRAIN PCA space of each decoder
                    Xit_te_for_it, Yit_te = collect_intertrial_xy(
                        it, it_test_kept, neural_field, kin_col_intertrial, pca_it_, lags, remove_means
                    )
                    Xfr_te_for_it, Yfr_te = collect_free0_xy(
                        prep_df, windows_test_kept, neural_field, kin_col_free0, pca_it_, lags, remove_means,
                        free0_len=free0_len
                    )

                    Xit_te_for_fr, _ = collect_intertrial_xy(
                        it, it_test_kept, neural_field, kin_col_intertrial, pca_free, lags, remove_means
                    )
                    Xfr_te_for_fr, _ = collect_free0_xy(
                        prep_df, windows_test_kept, neural_field, kin_col_free0, pca_free, lags, remove_means,
                        free0_len=free0_len
                    )

                    if (Xit_te_for_it.size == 0 or Xfr_te_for_it.size == 0 or
                        Xit_te_for_fr.size == 0 or Xfr_te_for_fr.size == 0):
                        continue

                    # Scores
                    r2_ii, rmse_ii = eval_decoder(ridge_it, Xit_te_for_it, Yit_te)
                    r2_if, rmse_if = eval_decoder(ridge_it, Xfr_te_for_it, Yfr_te)
                    r2_fi, rmse_fi = eval_decoder(ridge_fr, Xit_te_for_fr, Yit_te)
                    r2_ff, rmse_ff = eval_decoder(ridge_fr, Xfr_te_for_fr, Yfr_te)

                    # Variance fractions in decoder subspace (feature-space)
                    varfrac_ii = decoder_subspace_var_fraction(Xit_te_for_it, ridge_it)
                    varfrac_if = decoder_subspace_var_fraction(Xfr_te_for_it, ridge_it)
                    varfrac_fi = decoder_subspace_var_fraction(Xit_te_for_fr, ridge_fr)
                    varfrac_ff = decoder_subspace_var_fraction(Xfr_te_for_fr, ridge_fr)

                    row = {
                        "fold": fold,
                        "match": bool(match),
                        "remove_means": bool(remove_means),
                        "global_pca": bool(global_pca),
                        "caliper_q": float(caliper_q),
                        "n_components": int(n_components),
                        "lags": str(tuple(lags)),
                        "n_win_train": int(len(windows_train_kept)),
                        "n_win_test": int(len(windows_test_kept)),
                        "n_it_train": int(len(it_train_kept)),
                        "n_it_test": int(len(it_test_kept)),
                        "r2_ii": r2_ii, "rmse_ii": rmse_ii,
                        "r2_if": r2_if, "rmse_if": rmse_if,
                        "r2_fi": r2_fi, "rmse_fi": rmse_fi,
                        "r2_ff": r2_ff, "rmse_ff": rmse_ff,
                        "varfrac_ii": varfrac_ii,
                        "varfrac_if": varfrac_if,
                        "varfrac_fi": varfrac_fi,
                        "varfrac_ff": varfrac_ff,
                    }

                    # ---- Incremental save (every row) ----
                    header_written = append_row_csv(row, save_path, header_written)
                    n_rows_written += 1
                    rows_for_summary.append(row)

                    # Periodic summary checkpoint (optional)
                    if save_every_n_rows_summary is not None and (n_rows_written % save_every_n_rows_summary == 0):
                        df_tmp = pd.DataFrame(rows_for_summary)
                        metrics = ["r2_ii","r2_if","r2_fi","r2_ff","rmse_ii","rmse_if","rmse_fi","rmse_ff",
                                   "varfrac_ii","varfrac_if","varfrac_fi","varfrac_ff"]
                        df_summary_tmp = (
                            df_tmp
                            .groupby(["match","remove_means","global_pca","n_components","lags","caliper_q"], as_index=False)
                            .agg({m: ["mean","std"] for m in metrics})
                        )
                        summary_path = os.path.splitext(save_path)[0] + "_summary_partial.csv"
                        df_summary_tmp.to_csv(summary_path, index=False)

    # Final load & final summary from the saved CSV (most robust)
    df_res = pd.read_csv(save_path)

    metrics = ["r2_ii","r2_if","r2_fi","r2_ff","rmse_ii","rmse_if","rmse_fi","rmse_ff",
               "varfrac_ii","varfrac_if","varfrac_fi","varfrac_ff"]

    df_summary = (
        df_res
        .groupby(["match","remove_means","global_pca","n_components","lags","caliper_q"], as_index=False)
        .agg({m: ["mean","std"] for m in metrics})
    )

    summary_path = os.path.splitext(save_path)[0] + "_summary.csv"
    df_summary.to_csv(summary_path, index=False)

    return df_res, df_summary


# -------------------------
# Example usage
# -------------------------

# 1) Create matching features (e.g. all_keypoints_vel) in separate columns
prep_df, bhv_names_match = add_bhv_as(prep_df, bhv_fields=["all_keypoints_vel"], out_col="bhv_match")

# 2) Create decoded target features (e.g. all_keypoints) in separate columns
prep_df, bhv_names_decode = add_bhv_as(prep_df, bhv_fields=["all_keypoints"], out_col="bhv_decode")

# 3) Run CV with incremental saves
save_path = f"/home/il620/earthquake-analysis/notebooks/{sessions[session_idx]}_decode_cv.csv"

# If you want to overwrite rather than append:
# if os.path.exists(save_path): os.remove(save_path)

df_res, df_summary = run_cv_decode_incremental(
    prep_df=prep_df,
    windows=windows,
    free0_len=free0_len,
    neural_field="MOp_rates",
    kin_col_free0="bhv_decode",
    kin_col_intertrial="bhv_decode",
    match_bhv_col_free0="bhv_match",
    match_bhv_col_intertrial="bhv_match",
    n_splits=5,
    seed=0,
    n_components=30,
    lags=(0,1,2,3),
    match_opts=(False, True),
    remove_means_opts=(False, True),
    global_pca_opts=(False, True),
    caliper_q=1.0,
    save_path=save_path,
    save_every_n_rows_summary=200,  # set None to disable partial summaries
)

df_summary


