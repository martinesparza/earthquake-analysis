

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

from tools.decoding import regression as reg
# %% [markdown]
# start here

# %%
sessions = [
    # 'M062_2025_03_19_14_00',
    # 'M062_2025_03_20_14_00',
    # 'M062_2025_03_21_14_00',
    # 'M061_2025_03_04_10_00',
    # 'M061_2025_03_05_14_00',
    # "M061_2025_03_06_14_00",
    # "M063_2025_03_12_14_00",  
    # "M063_2025_03_13_14_00",  
    # "M063_2025_03_14_15_30",
    # "M078_2025_08_05_15_30",
    # "M078_2025_08_06_15_00",
    # "M078_2025_08_07_13_30",
    # "M078_2025_08_08_10_30",
    # "M086_2025_12_09_16_00",
    # "M086_2025_12_10_15_00",
    # "M086_2025_12_11_15_00",
    "M103_2026_02_17_14_00",
    # "M106_2026_02_24_15_00",


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


# for area in ["MOp", "SSp","CP","VAL"]:
#     rates_field = f"{area}_rates"
#     label_field = f"{area}_KSLabel"
#     out_field = f"{area}_rates_good"

#     labels0 = np.asarray(prep_df.iloc[0][label_field])
#     keep_good = (labels0 == "good")
#     print(area, np.sum(keep_good))
#     prep_df[out_field] = prep_df[rates_field].apply(lambda r: r[:, keep_good])
    

# %%
# random windows for free
import numpy as np
from typing import List, Tuple, Optional

def make_intertrial_sliding_blocks(it_df, block_size, step, prefix="inter", drop_incomplete=True):
    """
    Create overlapping sliding blocks from intertrial trials.

    Parameters
    ----------
    it_df : pd.DataFrame
        intertrial-only dataframe, reset_index(drop=True) recommended
    block_size : int
        number of trials per block (must be >= nW used in decoding)
    step : int
        stride between consecutive blocks (smaller => more overlap)
    prefix : str
        naming prefix for blocks
    drop_incomplete : bool
        if True, drop last block if it has < block_size trials

    Returns
    -------
    blocks : dict
        {f"{prefix}{b}": it_df.iloc[start:start+block_size].reset_index(drop=True)}
    """
    n = len(it_df)
    blocks = {}
    b = 0
    for start in range(0, n, step):
        end = start + block_size
        if end > n:
            if drop_incomplete:
                break
            end = n
        blk = it_df.iloc[start:end].reset_index(drop=True)
        if len(blk) < block_size and drop_incomplete:
            break
        blocks[f"{prefix}{b}"] = blk
        b += 1

    if len(blocks) == 0:
        raise ValueError("No sliding blocks created. Check block_size/step vs number of trials.")
    return blocks

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
import pandas as pd
import pyaldata as pyal

def make_intertrial_blocks(it_df, block_size, mode="early_late", drop_incomplete=True):
    """
    mode:
      - "early_late": returns inter0=head(block_size), inter1=tail(block_size)
      - "chunk": returns inter0, inter1, inter2, ... sequential chunks of length block_size
    """
    n = len(it_df)
    blocks = {}

    if mode == "early_late":
        if n < block_size:
            raise ValueError(f"Not enough intertrial trials ({n}) to take block_size={block_size}.")
        blocks["inter0"] = it_df.head(block_size).reset_index(drop=True)
        blocks["inter1"] = it_df.tail(block_size).reset_index(drop=True)
        return blocks

    if mode == "chunk":
        n_blocks = n // block_size if drop_incomplete else int(np.ceil(n / block_size))
        for b in range(n_blocks):
            i0 = b * block_size
            i1 = min((b + 1) * block_size, n)
            if (i1 - i0) < block_size and drop_incomplete:
                continue
            blocks[f"inter{b}"] = it_df.iloc[i0:i1].reset_index(drop=True)
        if len(blocks) == 0:
            raise ValueError("No blocks created; check block_size and drop_incomplete.")
        return blocks

    raise ValueError("mode must be 'early_late' or 'chunk'")


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
import scipy.linalg
from sklearn.metrics import r2_score, mean_squared_error

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


def collect_intertrial_xy_block(it_block_df, idx_list, neural_field, kin_col, pca_model, lags, remove_means):
    Xs, Ys = [], []
    for idx in idx_list:
        Xw = it_block_df.iloc[idx][neural_field]
        Xw = pca_model.transform(Xw)
        Yw = it_block_df.iloc[idx][kin_col]
        Xd, Yv = build_lagged_xy(Xw, Yw, lags, remove_means=remove_means)
        if Xd is None:
            continue
        Xs.append(Xd); Ys.append(Yv)
    if len(Xs) == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))
    return np.vstack(Xs), np.vstack(Ys)

def collect_free_xy(prep_df, windows_list, trial_name, neural_field, kin_col,
                    pca_model, lags, remove_means, bin_size=Params.BIN_SIZE):
    Xs, Ys = [], []
    free_df = prep_df[prep_df.trial_name == trial_name]
    if len(free_df) == 0:
        return np.zeros((0,0)), np.zeros((0,0))

    X_free = np.concatenate(free_df[neural_field].values, axis=0)
    Y_free = np.concatenate(free_df[kin_col].values, axis=0)
    T = min(X_free.shape[0], Y_free.shape[0])

    for (t0, t1) in windows_list:
        i0 = int(np.floor(t0 / bin_size))
        i1 = int(np.ceil(t1 / bin_size))
        i0 = max(0, min(i0, T))
        i1 = max(0, min(i1, T))
        if i1 <= i0:
            continue

        Xw = pca_model.transform(X_free[i0:i1])
        Yw = Y_free[i0:i1]
        Xd, Yv = build_lagged_xy(Xw, Yw, lags, remove_means=remove_means)
        if Xd is None:
            continue
        Xs.append(Xd); Ys.append(Yv)

    if len(Xs) == 0:
        return np.zeros((0,0)), np.zeros((0,0))
    return np.vstack(Xs), np.vstack(Ys)

    return np.vstack(Xs), np.vstack(Ys)


def eval_decoder(model, X_test, Y_test):
    pred = model.predict(X_test)
    r2 = r2_score(Y_test, pred, multioutput="variance_weighted")
    rmse = np.sqrt(mean_squared_error(Y_test, pred))
    return float(r2), float(rmse)


def decoder_subspace_var_fraction(X, ridge_model):
    W = ridge_model.coef_
    Q = scipy.linalg.orth(W.T)
    if Q.size == 0:
        return np.nan
    Xp = X @ Q
    var_pot = np.sum(np.var(Xp, axis=0))
    var_tot = np.sum(np.var(X, axis=0))
    return float(var_pot / var_tot) if var_tot > 0 else np.nan
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

def append_row_csv(row_dict, csv_path, header_written_flag):
    row_df = pd.DataFrame([row_dict])
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    if (not file_exists) and (not header_written_flag):
        row_df.to_csv(csv_path, index=False, mode="w")
        return True
    row_df.to_csv(csv_path, index=False, mode="a", header=False)
    return True
def run_cv_decode_pairs_incremental_free2(
    prep_df,
    free_blocks,          # dict: name -> {"trial_name": str, "windows": list}
    inter_blocks,         # dict: name -> it_block_df (len >= nW)
    neural_field="MOp_rates_good",
    kin_col="bhv",
    n_splits=5,
    seed=0,
    n_components=30,
    lags=(0,1,2,3),
    remove_means_opts=(False, True),
    global_pca_opts=(False, True),
    save_path="decode_pairs_cv.csv",
):
    # shared nW
    nW = min(len(v["windows"]) for v in free_blocks.values())
    # enforce same length
    for k in free_blocks:
        free_blocks[k]["windows"] = free_blocks[k]["windows"][:nW]
    for k, itb in inter_blocks.items():
        if len(itb) < nW:
            raise ValueError(f"{k} has {len(itb)} trials but needs >= nW={nW}")

    cond_names = list(free_blocks.keys()) + list(inter_blocks.keys())

    all_neural = np.concatenate(prep_df[neural_field].values, axis=0)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(kf.split(np.arange(nW)))

    header_written = False

    for fold, (tr_idx, te_idx) in enumerate(splits):
        for remove_means in remove_means_opts:
            for global_pca in global_pca_opts:

                pca_global = None
                if global_pca:
                    pca_global = PCA(n_components=n_components, svd_solver="full").fit(all_neural)

                trained = {}

                for train_cond in cond_names:
                    # ----- fit PCA -----
                    if global_pca:
                        pca_train = pca_global
                    else:
                        if train_cond in free_blocks:
                            trial_name = free_blocks[train_cond]["trial_name"]
                            windows_tr = [free_blocks[train_cond]["windows"][i] for i in tr_idx]

                            free_df = prep_df[prep_df.trial_name == trial_name]
                            X_free = np.concatenate(free_df[neural_field].values, axis=0)
                            T = X_free.shape[0]

                            X_list = []
                            for (t0, t1) in windows_tr:
                                i0 = int(np.floor(t0 / Params.BIN_SIZE))
                                i1 = int(np.ceil(t1 / Params.BIN_SIZE))
                                i0 = max(0, min(i0, T))
                                i1 = max(0, min(i1, T))
                                if i1 > i0:
                                    X_list.append(X_free[i0:i1])

                            if len(X_list) == 0:
                                continue
                            X_train_raw = np.concatenate(X_list, axis=0)

                        else:
                            it_block = inter_blocks[train_cond]
                            X_list = [it_block.iloc[i][neural_field] for i in tr_idx]
                            if len(X_list) == 0:
                                continue
                            X_train_raw = np.concatenate(X_list, axis=0)
                            if X_train_raw.shape[0] == 0:
                                continue

                        pca_train = PCA(n_components=n_components, svd_solver="full").fit(X_train_raw)

                    # ----- build TRAIN matrices -----
                    if train_cond in free_blocks:
                        trial_name = free_blocks[train_cond]["trial_name"]
                        windows_tr = [free_blocks[train_cond]["windows"][i] for i in tr_idx]
                        Xtr, Ytr = collect_free_xy(
                            prep_df, windows_tr, trial_name, neural_field, kin_col,
                            pca_train, lags, remove_means
                        )
                    else:
                        it_block = inter_blocks[train_cond]
                        Xtr, Ytr = collect_intertrial_xy_block(
                            it_block, tr_idx, neural_field, kin_col, pca_train, lags, remove_means
                        )

                    if Xtr.size == 0:
                        continue

                    ridge = reg.fit_semedo_ridge(Xtr, Ytr)
                    trained[train_cond] = (pca_train, ridge)

                # ----- evaluate all ordered pairs -----
                for train_cond, (pca_train, ridge) in trained.items():
                    for test_cond in cond_names:
                        if test_cond in free_blocks:
                            trial_name = free_blocks[test_cond]["trial_name"]
                            windows_te = [free_blocks[test_cond]["windows"][i] for i in te_idx]
                            Xte, Yte = collect_free_xy(
                                prep_df, windows_te, trial_name, neural_field, kin_col,
                                pca_train, lags, remove_means
                            )
                        else:
                            it_block = inter_blocks[test_cond]
                            Xte, Yte = collect_intertrial_xy_block(
                                it_block, te_idx, neural_field, kin_col, pca_train, lags, remove_means
                            )

                        if Xte.size == 0:
                            continue

                        r2, rmse = eval_decoder(ridge, Xte, Yte)
                        varfrac = decoder_subspace_var_fraction(Xte, ridge)

                        row = {
                            "fold": fold,
                            "train_cond": train_cond,
                            "test_cond": test_cond,
                            "remove_means": bool(remove_means),
                            "global_pca": bool(global_pca),
                            "n_components": int(n_components),
                            "lags": str(tuple(lags)),
                            "n_train_idx": int(len(tr_idx)),
                            "n_test_idx": int(len(te_idx)),
                            "r2": r2,
                            "rmse": rmse,
                            "varfrac": varfrac,
                        }
                        header_written = append_row_csv(row, save_path, header_written)

    df_res = pd.read_csv(save_path)
    df_summary = (
        df_res
        .groupby(["train_cond","test_cond","remove_means","global_pca","n_components","lags"], as_index=False)
        .agg(r2_mean=("r2","mean"), r2_std=("r2","std"),
             rmse_mean=("rmse","mean"), rmse_std=("rmse","std"),
             varfrac_mean=("varfrac","mean"), varfrac_std=("varfrac","std"))
    )
    df_summary.to_csv(os.path.splitext(save_path)[0] + "_summary.csv", index=False)
    return df_res, df_summary


def run_cv_decode_pairs_incremental(
    prep_df,
    windows,
    inter_blocks,                 # dict: name -> it_block_df (length == len(windows))
    free0_len,
    neural_field="SSp_rates",
    kin_col="bhv_decode",
    n_splits=5,
    seed=0,
    n_components=30,
    lags=(0,1,2,3),
    remove_means_opts=(False, True),
    global_pca_opts=(False, True),
    save_path="decode_pairs_cv.csv",
):
    nW = len(windows)
    cond_names = ["free0"] + list(inter_blocks.keys())

    # For global PCA
    all_neural = np.concatenate(prep_df[neural_field].values, axis=0)

    # KFold over shared index set {0..nW-1}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(kf.split(np.arange(nW)))

    header_written = False

    for fold, (tr_idx, te_idx) in enumerate(splits):
        win_tr = [windows[i] for i in tr_idx]
        win_te = [windows[i] for i in te_idx]

        for remove_means in remove_means_opts:
            for global_pca in global_pca_opts:

                # Fit PCA basis (global or per-condition later)
                pca_global = None
                if global_pca:
                    pca_global = PCA(n_components=n_components, svd_solver="full").fit(all_neural)

                trained = {}  # cond -> (pca_model, ridge_model)

                # ---- Train decoders for each condition ----
                for train_cond in cond_names:

                    if global_pca:
                        pca_train = pca_global
                    else:
                        # condition-specific PCA fit on TRAIN samples only
                        if train_cond in free_blocks:
                            X_list = []
                            for (t0, t1) in win_tr:
                                i0 = int(np.floor(t0 / Params.BIN_SIZE))
                                i1 = int(np.ceil(t1 / Params.BIN_SIZE))
                                i0 = max(0, min(i0, free0_len))
                                i1 = max(0, min(i1, free0_len))
                                if i1 > i0:
                                    X_list.append(prep_df[neural_field][0][i0:i1])
                            if len(X_list) == 0:
                                continue
                            X_train_raw = np.concatenate(X_list, axis=0)
                        else:
                            it_block = inter_blocks[train_cond]
                            X_list = [it_block.iloc[i][neural_field] for i in tr_idx]
                            X_train_raw = np.concatenate(X_list, axis=0) if len(X_list) else None
                            if X_train_raw is None or X_train_raw.shape[0] == 0:
                                continue

                        pca_train = PCA(n_components=n_components, svd_solver="full").fit(X_train_raw)

                    # Build TRAIN matrices
                    if train_cond == "free0":
                        Xtr, Ytr = collect_free_xy(
                            prep_df, win_tr, neural_field, kin_col, pca_train, lags, remove_means, free0_len=free0_len
                        )
                    else:
                        it_block = inter_blocks[train_cond]
                        Xtr, Ytr = collect_intertrial_xy_block(
                            it_block, tr_idx, neural_field, kin_col, pca_train, lags, remove_means
                        )

                    if Xtr.size == 0:
                        continue

                    ridge = reg.fit_semedo_ridge(Xtr, Ytr)
                    trained[train_cond] = (pca_train, ridge)

                # ---- Evaluate all ordered pairs ----
                for train_cond, (pca_train, ridge) in trained.items():
                    for test_cond in cond_names:

                        if test_cond == "free0":
                            Xte, Yte = collect_free0_xy(
                                prep_df, win_te, neural_field, kin_col, pca_train, lags, remove_means, free0_len=free0_len
                            )
                        else:
                            it_block = inter_blocks[test_cond]
                            Xte, Yte = collect_intertrial_xy_block(
                                it_block, te_idx, neural_field, kin_col, pca_train, lags, remove_means
                            )

                        if Xte.size == 0:
                            continue

                        r2, rmse = eval_decoder(ridge, Xte, Yte)
                        varfrac = decoder_subspace_var_fraction(Xte, ridge)

                        row = {
                            "fold": fold,
                            "train_cond": train_cond,
                            "test_cond": test_cond,
                            "remove_means": bool(remove_means),
                            "global_pca": bool(global_pca),
                            "n_components": int(n_components),
                            "lags": str(tuple(lags)),
                            "n_train_idx": int(len(tr_idx)),
                            "n_test_idx": int(len(te_idx)),
                            "r2": r2,
                            "rmse": rmse,
                            "varfrac": varfrac,
                        }
                        header_written = append_row_csv(row, save_path, header_written)

    df_res = pd.read_csv(save_path)
    df_summary = (
        df_res
        .groupby(["train_cond","test_cond","remove_means","global_pca","n_components","lags"], as_index=False)
        .agg(r2_mean=("r2","mean"), r2_std=("r2","std"),
             rmse_mean=("rmse","mean"), rmse_std=("rmse","std"),
             varfrac_mean=("varfrac","mean"), varfrac_std=("varfrac","std"))
    )
    summary_path = os.path.splitext(save_path)[0] + "_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    return df_res, df_summary
def add_bhv_as(df, bhv_fields, out_col):
    """
    dt.add_bhv writes to column 'bhv'. This helper calls it and renames 'bhv' -> out_col.
    """
    df2, names = dt.add_bhv(df, bhv_fields=bhv_fields)
    if "bhv" not in df2.columns:
        raise ValueError("dt.add_bhv did not create a 'bhv' column as expected.")
    df2 = df2.rename(columns={"bhv": out_col})
    return df2, names


for session_idx in range(len(sessions)):


    prep_df = prep_dfs[session_idx]
    field = "MOp_rates"

    free0_len = get_n_time(prep_df, "free0", field=field)
    free1_len = get_n_time(prep_df, "free1", field=field)

    windows0 = split_free(free0_len * Params.BIN_SIZE, window_lengths_s=[1,3,5], gap_s=1, seed=None, start_offset_s=0)
    windows1 = split_free(free1_len * Params.BIN_SIZE, window_lengths_s=[1,3,5], gap_s=1, seed=None, start_offset_s=0)

    # share folds across all conditions => enforce same number of indices
    nW = min(len(windows0), len(windows1))
    windows0 = windows0[:nW]
    windows1 = windows1[:nW]

    free_blocks = {
        "free0": {"trial_name": "free0", "windows": windows0},
        "free1": {"trial_name": "free1", "windows": windows1},
    }
# free_blocks = {
#     "free0": {"row_idx": 0,  "windows": windows0},
#     "free1": {"row_idx": -1, "windows": windows1},
# }
    # it = pyal.select_trials(prep_df, "trial_name=='intertrial'")
    # print(f"n_windows intertrial (trials) = {len(it)}")

    # if len(it) < nW:
    #     raise ValueError(f"Not enough intertrial trials ({len(it)}) to make a block of size nW={nW}")

#     start = (len(it) - nW) // 2
#     it_middle = it.iloc[start:start + nW].reset_index(drop=True)

#     it_early = it.head(nW).reset_index(drop=True)
#     it_late  = it.tail(nW).reset_index(drop=True)
# # it_early,_ = dt.add_bhv(it_early, bhv_fields=["all_keypoints"])
# # it_late,_ = dt.add_bhv(it_late, bhv_fields=["all_keypoints"])
# # it_middle,_ = dt.add_bhv(it_middle, bhv_fields=["all_keypoints"])

#     inter_blocks = {
#         "inter0": it_early,
#         "inter1": it_middle,
#         "inter2": it_late,
#     }
# inter_blocks = {
#     "inter_early": it_early,
#     "inter_late":  it_late,
# }
    it = pyal.select_trials(prep_df, "trial_name=='intertrial'").reset_index(drop=True)
    print(f"n_windows intertrial (trials) = {len(it)}")

    # nW is your shared number of windows between free0 and free1
    # IMPORTANT: each intertrial block must have >= nW trials because the CV folds index 0..nW-1
    block_size = nW

    # choose overlap: e.g. 50% overlap => step = nW//2
    step = max(1, nW // 4)

    inter_blocks = make_intertrial_sliding_blocks(
        it_df=it,
        block_size=block_size,
        step=step,
        prefix="inter",
        drop_incomplete=True
    )

    print(f"Created {len(inter_blocks)} overlapping intertrial blocks: "
        f"block_size={block_size}, step={step}")



# decoded target only (no matching)
# prep_df, _ = dt.add_bhv(prep_df, bhv_fields=["all_keypoints"])

# save_path = f"/home/il620/earthquake-analysis/notebooks/{sessions[session_idx]}_decode_pairs_cv_MOp.csv"

# df_res, df_summary = run_cv_decode_pairs_incremental(
#     prep_df=prep_df,
#     windows=windows,
#     inter_blocks=inter_blocks,
#     free0_len=free0_len,
#     neural_field="MOp_rates_good",
#     kin_col="bhv",
#     n_splits=5,
#     seed=0,
#     n_components=30,
#     lags=(0,1,2,3),
#     remove_means_opts=(False, True),
#     global_pca_opts=(False, True),
#     save_path=save_path,
# )

# df_summary

# # save_path = f"/home/il620/earthquake-analysis/notebooks/{sessions[session_idx]}_decode_pairs_cv_CP.csv"

# df_res, df_summary = run_cv_decode_pairs_incremental(
#     prep_df=prep_df,
#     windows=windows,
#     inter_blocks=inter_blocks,
#     free0_len=free0_len,
#     neural_field="CP_rates_good",
#     kin_col="bhv",
#     n_splits=5,
#     seed=0,
#     n_components=30,
#     lags=(0,1,2,3),
#     remove_means_opts=(False, True),
#     global_pca_opts=(False, True),
#     save_path=save_path,
# )

# df_summary
# save_path = f"/home/il620/earthquake-analysis/notebooks/{sessions[session_idx]}_decode_pairs_cv_SSp.csv"

# df_res, df_summary = run_cv_decode_pairs_incremental(
#     prep_df=prep_df,
#     windows=windows,
#     inter_blocks=inter_blocks,
#     free0_len=free0_len,
#     neural_field="SSp_rates_good",
#     kin_col="bhv",
#     n_splits=5,
#     seed=0,
#     n_components=30,
#     lags=(0,1,2,3),
#     remove_means_opts=(False, True),
#     global_pca_opts=(False, True),
#     save_path=save_path,
# )

# df_summary
    # prep_df = dt.add_no_mua_field(prep_df)
    save_path = f"/home/il620/earthquake-analysis/notebooks/{sessions[session_idx]}_decode_pairs_cv_SSp.csv"



    df_res, df_summary = run_cv_decode_pairs_incremental_free2(
        prep_df=prep_df,
        free_blocks=free_blocks,
        inter_blocks=inter_blocks,
        neural_field="MOp_rates",
        kin_col="bhv",
        n_splits=5,
        seed=0,
        n_components=30,
        lags=(0,1,2,3,4),
        remove_means_opts=(True,),
        global_pca_opts=(False,),
        save_path=save_path,
    )


# import os
# import numpy as np
# import pandas as pd

# from sklearn.model_selection import KFold
# from sklearn.decomposition import PCA
# from sklearn.metrics import r2_score, mean_squared_error

# from tools.params import Params
# from tools.decoding import regression as reg


# # ----------------------------
# # helpers
# # ----------------------------

# def append_row_csv(row_dict, csv_path, header_written_flag):
#     row_df = pd.DataFrame([row_dict])
#     file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
#     if (not file_exists) and (not header_written_flag):
#         row_df.to_csv(csv_path, index=False, mode="w")
#         return True
#     row_df.to_csv(csv_path, index=False, mode="a", header=False)
#     return True


# def build_lagged_xy(Xw, Yw, lags, remove_means=False):
#     max_lag = int(np.max(lags))
#     if Yw.ndim == 1:
#         Yw = Yw[:, None]
#     if remove_means:
#         Yw = Yw - np.mean(Yw, axis=0, keepdims=True)

#     T = min(len(Xw), len(Yw))
#     Xw, Yw = Xw[:T], Yw[:T]
#     if T <= max_lag:
#         return None, None

#     t = np.arange(max_lag, T)
#     Xdelays = np.concatenate([Xw[t - lag, :] for lag in lags], axis=1)
#     Yv = Yw[t, :]
#     return Xdelays, Yv


# def eval_decoder(model, X_test, Y_test):
#     pred = model.predict(X_test)
#     r2 = r2_score(Y_test, pred, multioutput="variance_weighted")
#     rmse = np.sqrt(mean_squared_error(Y_test, pred))
#     return float(r2), float(rmse)


# def collect_intertrial_xy_block(it_block_df, idx_list, neural_field, kin_col, pca_model, lags, remove_means):
#     Xs, Ys = [], []
#     for idx in idx_list:
#         Xw = it_block_df.iloc[idx][neural_field]
#         Xw = pca_model.transform(Xw)
#         Yw = it_block_df.iloc[idx][kin_col]
#         Xd, Yv = build_lagged_xy(Xw, Yw, lags, remove_means=remove_means)
#         if Xd is None:
#             continue
#         Xs.append(Xd); Ys.append(Yv)
#     if len(Xs) == 0:
#         return np.zeros((0, 0)), np.zeros((0, 0))
#     return np.vstack(Xs), np.vstack(Ys)


# def collect_free_row_xy(prep_df, row_idx, windows_list, neural_field, kin_col,
#                         pca_model, lags, remove_means, bin_size=Params.BIN_SIZE):
#     Xs, Ys = [], []
#     X_free = prep_df.iloc[row_idx][neural_field]
#     Y_free = prep_df.iloc[row_idx][kin_col]
#     T = min(X_free.shape[0], Y_free.shape[0])

#     for (t0, t1) in windows_list:
#         i0 = int(np.floor(t0 / bin_size))
#         i1 = int(np.ceil(t1 / bin_size))
#         i0 = max(0, min(i0, T))
#         i1 = max(0, min(i1, T))
#         if i1 <= i0:
#             continue

#         Xw = pca_model.transform(X_free[i0:i1])
#         Yw = Y_free[i0:i1]
#         Xd, Yv = build_lagged_xy(Xw, Yw, lags, remove_means=remove_means)
#         if Xd is None:
#             continue

#         Xs.append(Xd); Ys.append(Yv)

#     if len(Xs) == 0:
#         return np.zeros((0, 0)), np.zeros((0, 0))
#     return np.vstack(Xs), np.vstack(Ys)


# def pca_fit_free_train(prep_df, row_idx, windows_train, neural_field, n_components):
#     X_free = prep_df.iloc[row_idx][neural_field]
#     T = X_free.shape[0]
#     X_list = []
#     for (t0, t1) in windows_train:
#         i0 = int(np.floor(t0 / Params.BIN_SIZE))
#         i1 = int(np.ceil(t1 / Params.BIN_SIZE))
#         i0 = max(0, min(i0, T))
#         i1 = max(0, min(i1, T))
#         if i1 > i0:
#             X_list.append(X_free[i0:i1])
#     if len(X_list) == 0:
#         return None
#     X_train_raw = np.concatenate(X_list, axis=0)
#     if X_train_raw.shape[0] == 0:
#         return None
#     return PCA(n_components=n_components, svd_solver="full").fit(X_train_raw)


# def pca_fit_inter_train(it_block_df, tr_idx, neural_field, n_components):
#     X_list = [it_block_df.iloc[i][neural_field] for i in tr_idx]
#     if len(X_list) == 0:
#         return None
#     X_train_raw = np.concatenate(X_list, axis=0)
#     if X_train_raw.shape[0] == 0:
#         return None
#     return PCA(n_components=n_components, svd_solver="full").fit(X_train_raw)


# # ------------------------------------------
# # main: within-condition lag sweep (dict API)
# # ------------------------------------------

# def run_within_lag_sweep_incremental(
#     prep_df,
#     free_blocks,     # dict: name -> {"row_idx": int, "windows": list[(t0,t1)]}
#     inter_blocks,    # dict: name -> DataFrame (len >= nW)
#     neural_field="MOp_rates_good",
#     kin_col="bhv",
#     n_splits=5,
#     seed=0,
#     n_components=30,
#     max_lag=12,
#     remove_means_opts=(False, True),
#     global_pca_opts=(True, False),
#     save_path="within_lag_sweep.csv",
# ):
#     """
#     Within-condition only:
#       For each condition (free_* and inter_*), and each max_lag L:
#         train decoder on TRAIN split of that condition
#         test on TEST split of same condition
#       Save fold-level rows incrementally, then write summary + ΔR² vs lag0.
#     """

#     # ---- shared index universe (same folds for all conditions) ----
#     nW_free = [len(v["windows"]) for v in free_blocks.values()]
#     nW_inter = [len(df) for df in inter_blocks.values()] if len(inter_blocks) else [np.inf]
#     nW = int(min(min(nW_free), min(nW_inter)))

#     # truncate windows/trials to nW
#     for k in free_blocks:
#         free_blocks[k] = dict(free_blocks[k])  # shallow copy
#         free_blocks[k]["windows"] = free_blocks[k]["windows"][:nW]
#     for k in list(inter_blocks.keys()):
#         inter_blocks[k] = inter_blocks[k].iloc[:nW].reset_index(drop=True)

#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     splits = list(kf.split(np.arange(nW)))

#     lag_sets = [tuple(range(L + 1)) for L in range(max_lag + 1)]
#     cond_names = list(free_blocks.keys()) + list(inter_blocks.keys())

#     # global PCA cache if requested
#     all_neural = np.concatenate(prep_df[neural_field].values, axis=0)

#     header_written = False

#     # Optional overwrite:
#     # if os.path.exists(save_path): os.remove(save_path)

#     for global_pca in global_pca_opts:
#         pca_global = None
#         if global_pca:
#             pca_global = PCA(n_components=n_components, svd_solver="full").fit(all_neural)

#         for remove_means in remove_means_opts:
#             for L, lags in enumerate(lag_sets):
#                 for fold, (tr_idx, te_idx) in enumerate(splits):

#                     for cond in cond_names:

#                         # ---- fit PCA ----
#                         if global_pca:
#                             pca = pca_global
#                         else:
#                             if cond in free_blocks:
#                                 row_idx = free_blocks[cond]["row_idx"]
#                                 win_tr = [free_blocks[cond]["windows"][i] for i in tr_idx]
#                                 pca = pca_fit_free_train(prep_df, row_idx, win_tr, neural_field, n_components)
#                             else:
#                                 pca = pca_fit_inter_train(inter_blocks[cond], tr_idx, neural_field, n_components)

#                             if pca is None:
#                                 continue

#                         # ---- train/test within condition ----
#                         if cond in free_blocks:
#                             row_idx = free_blocks[cond]["row_idx"]
#                             win_tr = [free_blocks[cond]["windows"][i] for i in tr_idx]
#                             win_te = [free_blocks[cond]["windows"][i] for i in te_idx]

#                             Xtr, Ytr = collect_free_row_xy(
#                                 prep_df, row_idx, win_tr, neural_field, kin_col, pca, lags, remove_means
#                             )
#                             Xte, Yte = collect_free_row_xy(
#                                 prep_df, row_idx, win_te, neural_field, kin_col, pca, lags, remove_means
#                             )
#                         else:
#                             Xtr, Ytr = collect_intertrial_xy_block(
#                                 inter_blocks[cond], tr_idx, neural_field, kin_col, pca, lags, remove_means
#                             )
#                             Xte, Yte = collect_intertrial_xy_block(
#                                 inter_blocks[cond], te_idx, neural_field, kin_col, pca, lags, remove_means
#                             )

#                         if Xtr.size == 0 or Xte.size == 0:
#                             continue

#                         model = reg.fit_semedo_ridge(Xtr, Ytr)
#                         r2, rmse = eval_decoder(model, Xte, Yte)

#                         row = {
#                             "cond": cond,
#                             "fold": int(fold),
#                             "max_lag": int(L),
#                             "lags": str(lags),
#                             "r2": r2,
#                             "rmse": rmse,
#                             "global_pca": bool(global_pca),
#                             "remove_means": bool(remove_means),
#                             "n_components": int(n_components),
#                             "nW": int(nW),
#                             "n_train_idx": int(len(tr_idx)),
#                             "n_test_idx": int(len(te_idx)),
#                         }
#                         header_written = append_row_csv(row, save_path, header_written)

#     # ---- summary from disk ----
#     df_fold = pd.read_csv(save_path)

#     df_sum = (
#         df_fold
#         .groupby(["cond", "max_lag", "global_pca", "remove_means", "n_components"], as_index=False)
#         .agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
#              rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"))
#         .sort_values(["cond", "global_pca", "remove_means", "max_lag"])
#     )

#     # ΔR² vs no-lag baseline within each (cond,global_pca,remove_means,n_components)
#     base = (
#         df_sum[df_sum["max_lag"] == 0][["cond","global_pca","remove_means","n_components","r2_mean"]]
#         .rename(columns={"r2_mean": "r2_lag0"})
#     )
#     df_sum = df_sum.merge(base, on=["cond","global_pca","remove_means","n_components"], how="left")
#     df_sum["delta_r2_vs_lag0"] = df_sum["r2_mean"] - df_sum["r2_lag0"]
#     df_sum["delta_r2_increment"] = df_sum.groupby(["cond","global_pca","remove_means","n_components"])["r2_mean"].diff()

#     summary_path = os.path.splitext(save_path)[0] + "_summary.csv"
#     df_sum.to_csv(summary_path, index=False)

#     return df_fold, df_sum, summary_path


# save_path = f"/home/il620/earthquake-analysis/notebooks/{sessions[session_idx]}_within_lag_sweep_VAL.csv"

# df_fold, df_sum, summary_path = run_within_lag_sweep_incremental(
#     prep_df=prep_df,
#     free_blocks=free_blocks,
#     inter_blocks=inter_blocks,
#     neural_field="VAL_rates_good",
#     kin_col="left_paw",
#     n_splits=5,
#     seed=0,
#     n_components=30,
#     max_lag=5,
#     remove_means_opts=(True,),
#     global_pca_opts=(False,),
#     save_path=save_path,
# )

# save_path = f"/home/il620/earthquake-analysis/notebooks/{sessions[session_idx]}_within_lag_sweep_SSp.csv"

# df_fold, df_sum, summary_path = run_within_lag_sweep_incremental(
#     prep_df=prep_df,
#     free_blocks=free_blocks,
#     inter_blocks=inter_blocks,
#     neural_field="SSp_rates",
#     kin_col="left_paw",
#     n_splits=5,
#     seed=0,
#     n_components=30,
#     max_lag=5,
#     remove_means_opts=(True,),
#     global_pca_opts=(False,),
#     save_path=save_path,
# )