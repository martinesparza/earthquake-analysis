# %%
%load_ext autoreload
%autoreload 2

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

# %%
import logging
logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

# %%
import logging
logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

# %%


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


# %% [markdown]
# start here

# %%
sessions = [
    # 'M062_2025_03_19_14_00',
    # 'M062_2025_03_20_14_00',
    'M062_2025_03_21_14_00',
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

# %%
from sklearn.model_selection import KFold

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
# it = pyal.select_trials(prep_df,"trial_name=='intertrial'")
n_splits = 10
shuffle = True
random_state = None
kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
windows_indices = np.arange(len(windows))
for fold, (train, test) in enumerate(kf.split(windows_indices)):
        windows_train = [windows[tr] for tr in train]
        windows_test = [windows[te] for te in test]
        
        break

inter_indices = np.arange(len(it))
for fold, (train, test) in enumerate(kf.split(inter_indices)):
        inter_idx_train = train
        inter_idx_test = test
        
        break

# %%
len(windows_train)

# %%
field = "bhv"
means, used_bins = window_metric(
    prep_df=prep_df,
    field=field,
    windows=windows_train,
    trial_name="free0",
    metric = np.nanmean
)
print(means.shape)
it_means = np.vstack(it[field].apply(lambda a: np.nanmean(a, axis=0)))[inter_idx_train,:]
all_windows_mean = np.vstack((means, it_means))



# %%
n_comp = 10
model = PCA(n_components=n_comp, svd_solver='full')
model.fit(all_windows_mean)
print(np.cumsum(model.explained_variance_ratio_))
components = model.components_

pca_inter = model.transform(it_means)
pca_free0 = model.transform(means)
pca_inter = pca_inter[:,0].reshape(-1, 1)
pca_free0 = pca_free0[:,0].reshape(-1, 1)
# pca_inter = pca_inter[:,:2]
# pca_free0 = pca_free0[:,:2]
# plot cum var explained
plt.figure()
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA on keypoints vel window means")
plt.show()
# plot feature loadings for the first 10 pcs
n_pcs_to_plot = 2
plt.figure(figsize=(15, 30))
for pc_idx in range(n_pcs_to_plot):
    plt.subplot(n_pcs_to_plot, 1, pc_idx + 1)
    plt.bar(range(components.shape[1]), components[pc_idx])
    plt.title(f"PC {pc_idx + 1}")
    # for x axis put bhv_names
    plt.xticks(range(len(bhv_names)), bhv_names, rotation=90)
    plt.xlabel("Behavioral feature")
    plt.ylabel("Loading")
plt.tight_layout()
plt.show()
plt.figure()
plt.hist(pca_inter[:,0], bins=30, alpha=0.5, label="Intertrial")
plt.hist(pca_free0[:,0], bins=30, alpha=0.5, label="Free0")
plt.legend()
plt.xlabel(f"Mean pca per window")
plt.ylabel("Count")
plt.title("Distribution of window means")
plt.show()

# %%
# plt.figure()
# plt.hist(pca_inter[:,1], bins=30, alpha=0.5, label="Intertrial")
# plt.hist(pca_free0[:,1], bins=30, alpha=0.5, label="Free0")
# plt.legend()
# plt.xlabel(f"Mean pca per window")
# plt.ylabel("Count")
# plt.title("Distribution of window means")
# plt.show()

# %%
from sklearn.neighbors import NearestNeighbors
idx_free = np.arange(len(windows_train))
# it = pyal.select_trials(prep_df,"trial_name=='intertrial'")
idx_it = inter_idx_train


A, A_idx = pca_free0, idx_free
B, B_idx = pca_inter, idx_it
A_is_free = True

nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(B)
dists, nbrs = nn.kneighbors(A, return_distance=True)
dists = dists[:, 0]
nbrs  = nbrs[:, 0]

# Auto caliper: e.g. keep the best 80% of matches OR use a fixed SD-like threshold
caliper = np.quantile(dists, 1)
ok = dists <= caliper

A_idx_ok = A_idx[ok]
B_idx_ok = B_idx[nbrs[ok]]

# Enforce 1:1 without replacement on B (greedy)
used = set()
pairs = []
for a_i, b_i, d in sorted(zip(A_idx_ok, B_idx_ok, dists[ok]), key=lambda t: t[2]):
    if b_i in used:
        continue
    used.add(b_i)
    pairs.append((a_i, b_i))


A_kept = np.array([p[0] for p in pairs], dtype=int)
B_kept = np.array([p[1] for p in pairs], dtype=int)

if A_is_free:
    kept_free_idx = A_kept
    kept_it_idx   = B_kept
else:
    kept_it_idx   = A_kept
    kept_free_idx = B_kept

# %%
idx_free.shape

# %%
# plt.figure()
# plt.hist(pca_inter[kept_it_idx,0], bins=50, alpha=0.5, label="Intertrial")
# plt.hist(pca_free0[kept_free_idx,0], bins=50, alpha=0.5, label="Free0")
# plt.legend()
# plt.xlabel(f"Mean pca per window")
# plt.ylabel("Count")
# plt.title("Distribution of window means")
# plt.show()

# %%
# plt.figure()
# plt.hist(pca_inter[kept_it_idx,1], bins=50, alpha=0.5, label="Intertrial")
# plt.hist(pca_free0[kept_free_idx,1], bins=50, alpha=0.5, label="Free0")
# plt.legend()
# plt.xlabel(f"Mean pca per window")
# plt.ylabel("Count")
# plt.title("Distribution of window means")
# plt.show()

# %%
print(f"inter train time: {np.sum([it.trial_length.values[intertrial] for intertrial in kept_it_idx])/100/60}")
print(f"free train time: {np.sum([windows_train[idx][1]-windows_train[idx][0] for idx in kept_free_idx])/60}")

# %%
# up to here all the same no matter the predictor or predicted variable 

# %%
# 

# %%
prep_df,bhv_names = dt.add_bhv(prep_df, bhv_fields=["all_keypoints_vel"])
it = pyal.select_trials(prep_df,"trial_name=='intertrial'")
neural_field = "SSp_rates"
kin_field = ["bhv"]
train_rates_inter = np.concatenate(it[neural_field].values[kept_it_idx],axis = 0)
train_kin_inter = np.concatenate(it[kin_field].values[kept_it_idx,0],axis = 0)
print(train_rates_inter.shape,train_kin_inter.shape)
test_rates_inter = np.concatenate(it[neural_field].values[inter_idx_test],axis = 0)
test_kin_inter = np.concatenate(it[kin_field].values[inter_idx_test,0],axis = 0)

# %%
idx_list = []
for t0, t1 in windows_train:
    i0 = int(np.floor(t0 / Params.BIN_SIZE))
    i1 = int(np.ceil(t1 / Params.BIN_SIZE))   # end-exclusive-ish
    if free0_len is not None:
        i0 = max(0, min(i0, free0_len))
        i1 = max(0, min(i1, free0_len))
    if i1 > i0:
        idx_list.append(np.arange(i0, i1, dtype=int))

timebins_free = np.unique(np.concatenate(idx_list))

idx_list = []
for t0, t1 in windows_test:
    i0 = int(np.floor(t0 / Params.BIN_SIZE))
    i1 = int(np.ceil(t1 / Params.BIN_SIZE))   # end-exclusive-ish
    if free0_len is not None:
        i0 = max(0, min(i0, free0_len))
        i1 = max(0, min(i1, free0_len))
    if i1 > i0:
        idx_list.append(np.arange(i0, i1, dtype=int))

timebins_free_test = np.unique(np.concatenate(idx_list))

train_rates_free0 = prep_df[neural_field][0][timebins_free]
train_kin_free0 = prep_df.iloc[0][kin_field][0][timebins_free]

test_rates = prep_df[neural_field][0][timebins_free_test]
test_kin= prep_df.iloc[0][kin_field][0][timebins_free_test]

# %%
n_components = 30
model_free0 = PCA(n_components)
pca_free0 = model_free0.fit_transform(train_rates_free0)
model_inter = PCA(n_components)
pca_inter = model_inter.fit_transform(train_rates_inter)

# %%
lags = [0, 1, 2, 3]
max_lag = int(np.max(lags))
valid_intertrial = []
valid_intertrial_Y = []
for idx in kept_it_idx:
    Xw = it.iloc[idx][neural_field]
    Xw = model_inter.transform(Xw)
    Yw = it.iloc[idx][kin_field][0]
    if Yw.ndim == 1:
        Yw = Yw[:, None]
    T = min(len(Xw), len(Yw))
    Xw, Yw = Xw[:T], Yw[:T]
    t = np.arange(max_lag, T)  # valid target times within this intertrial
    Xdelays = np.concatenate([Xw[t - lag, :] for lag in lags], axis=1)  # [X_t, X_{t-1}, ...]
    Yv = Yw[t, :]
    valid_intertrial_Y.append(Yv)
    valid_intertrial.append(Xdelays)

valid_intertrial_Y = np.vstack(valid_intertrial_Y) if valid_intertrial_Y else np.zeros((0, 0))
valid_intertrial_X = np.vstack(valid_intertrial)

# %%
valid_free0_Y_train = []
valid_free0_train = []
for w_idx in kept_free_idx:
    t0, t1 = windows[int(w_idx)]
    i0 = int(np.floor(t0 / Params.BIN_SIZE))
    i1 = int(np.ceil(t1 / Params.BIN_SIZE))
    i0 = max(0, min(i0, free0_len))
    i1 = max(0, min(i1, free0_len))
    Xw = prep_df[neural_field][0][i0:i1,:]
    Xw = model_free0.transform(Xw)
    Yw = prep_df.iloc[0][kin_field][0][i0:i1,:]
    if Yw.ndim == 1:
        Yw = Yw[:, None]
    T = min(len(Xw), len(Yw))
    Xw, Yw = Xw[:T], Yw[:T]
    t = np.arange(max_lag, T)  # valid target times within this intertrial
    Xdelays = np.concatenate([Xw[t - lag, :] for lag in lags], axis=1)  # [X_t, X_{t-1}, ...]
    Yv = Yw[t, :]
    valid_free0_Y_train.append(Yv)
    valid_free0_train.append(Xdelays)
    

valid_free0_Y_train = np.vstack(valid_free0_Y_train) if valid_free0_Y_train else np.zeros((0, 0))
valid_free0_train = np.vstack(valid_free0_train)


# %%
from tools.decoding import regression as reg
ridge_inter = reg.fit_semedo_ridge(valid_intertrial_X, valid_intertrial_Y)
ridge_free0 = reg.fit_semedo_ridge(valid_free0_train, valid_free0_Y_train)


# %%
class Decoder:
    def __init__(self, pca, ridge):
        self.pca = pca
        self.ridge = ridge
    def apply_pca(self,X):
        return self.pca.transform(X)
    def predict_ridge(self,X):
        return self.ridge.predict(X)

# %%
decoder_inter = Decoder(model_inter, ridge_inter)
decoder_free0 = Decoder(model_free0, ridge_free0)

# %%
from sklearn.metrics import r2_score, root_mean_squared_error
decoder = decoder_free0
valid_intertrial_test = []
valid_intertrial_Y_test = []
for idx in inter_idx_test:
    Xw = it.iloc[idx][neural_field]
    Xw = decoder.apply_pca(Xw)
    Yw = it.iloc[idx][kin_field][0]
    # print(Yw.shape)
    if Yw.ndim == 1:
        Yw = Yw[:, None]
    T = min(len(Xw), len(Yw))
    Xw, Yw = Xw[:T], Yw[:T]
    t = np.arange(max_lag, T)  # valid target times within this intertrial
    Xdelays = np.concatenate([Xw[t - lag, :] for lag in lags], axis=1)  # [X_t, X_{t-1}, ...]
    Yv = Yw[t, :]
    valid_intertrial_Y_test.append(Yv)
    valid_intertrial_test.append(Xdelays)

valid_intertrial_Y_test = np.vstack(valid_intertrial_Y_test) if valid_intertrial_Y_test else np.zeros((0, 0))
valid_intertrial_X_test = np.vstack(valid_intertrial_test)
pred = decoder.predict_ridge(valid_intertrial_X_test)
r2_fi= r2_score(valid_intertrial_Y_test,pred, multioutput='variance_weighted')
rmse= root_mean_squared_error(valid_intertrial_Y_test,pred)
print(f"Free on intertrial: r2 = {r2_fi}, rmse = {rmse}")

valid_free0_Y_test = []
valid_free0_test = []
for window in windows_test:
    t0, t1 = window
    i0 = int(np.floor(t0 / Params.BIN_SIZE))
    i1 = int(np.ceil(t1 / Params.BIN_SIZE))
    i0 = max(0, min(i0, free0_len))
    i1 = max(0, min(i1, free0_len))
    Xw = prep_df[neural_field][0][i0:i1,:]
    Xw = decoder.apply_pca(Xw)
    Yw = prep_df.iloc[0][kin_field][0][i0:i1,:]
    if Yw.ndim == 1:
        Yw = Yw[:, None]
    T = min(len(Xw), len(Yw))
    Xw, Yw = Xw[:T], Yw[:T]
    t = np.arange(max_lag, T)  # valid target times within this intertrial
    Xdelays = np.concatenate([Xw[t - lag, :] for lag in lags], axis=1)  # [X_t, X_{t-1}, ...]
    Yv = Yw[t, :]
    valid_free0_Y_test.append(Yv)
    valid_free0_test.append(Xdelays)
    

valid_free0_Y_test = np.vstack(valid_free0_Y_test) if valid_free0_Y_test else np.zeros((0, 0))
valid_free0_test = np.vstack(valid_free0_test)

from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
free_on_free_pred = decoder.predict_ridge(valid_free0_test)
r2_ff = r2_score(valid_free0_Y_test,free_on_free_pred, multioutput='variance_weighted' )
rmse_free_on_free = root_mean_squared_error(valid_free0_Y_test,free_on_free_pred )
print(f"Free on free: r2 = {r2_ff}, rmse = {rmse_free_on_free}")

# %%
decoder = decoder_inter
valid_intertrial_test = []
valid_intertrial_Y_test = []
for idx in inter_idx_test:
    Xw = it.iloc[idx][neural_field]
    Xw = decoder.apply_pca(Xw)
    Yw = it.iloc[idx][kin_field][0]
    if Yw.ndim == 1:
        Yw = Yw[:, None]
    T = min(len(Xw), len(Yw))
    Xw, Yw = Xw[:T], Yw[:T]
    t = np.arange(max_lag, T)  # valid target times within this intertrial
    Xdelays = np.concatenate([Xw[t - lag, :] for lag in lags], axis=1)  # [X_t, X_{t-1}, ...]
    Yv = Yw[t, :]
    valid_intertrial_Y_test.append(Yv)
    valid_intertrial_test.append(Xdelays)

valid_intertrial_Y_test = np.vstack(valid_intertrial_Y_test) if valid_intertrial_Y_test else np.zeros((0, 0))
valid_intertrial_X_test = np.vstack(valid_intertrial_test)
pred = decoder.predict_ridge(valid_intertrial_X_test)
r2_ii= r2_score(valid_intertrial_Y_test,pred, multioutput='variance_weighted' )
rmse= root_mean_squared_error(valid_intertrial_Y_test,pred)
print(f"Intertrial on intertrial: r2 = {r2_ii}, rmse = {rmse}")

valid_free0_Y_test = []
valid_free0_test = []
for window in windows_test:
    t0, t1 = window
    i0 = int(np.floor(t0 / Params.BIN_SIZE))
    i1 = int(np.ceil(t1 / Params.BIN_SIZE))
    i0 = max(0, min(i0, free0_len))
    i1 = max(0, min(i1, free0_len))
    Xw = prep_df[neural_field][0][i0:i1,:]
    Xw = decoder.apply_pca(Xw)
    Yw = prep_df.iloc[0][kin_field][0][i0:i1,:]
    if Yw.ndim == 1:
        Yw = Yw[:, None]
    T = min(len(Xw), len(Yw))
    Xw, Yw = Xw[:T], Yw[:T]
    t = np.arange(max_lag, T)  # valid target times within this intertrial
    Xdelays = np.concatenate([Xw[t - lag, :] for lag in lags], axis=1)  # [X_t, X_{t-1}, ...]
    Yv = Yw[t, :]
    valid_free0_Y_test.append(Yv)
    valid_free0_test.append(Xdelays)
    

valid_free0_Y_test = np.vstack(valid_free0_Y_test) if valid_free0_Y_test else np.zeros((0, 0))
valid_free0_test = np.vstack(valid_free0_test)

from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
free_on_free_pred = decoder.predict_ridge(valid_free0_test)
r2_if = r2_score(valid_free0_Y_test,free_on_free_pred, multioutput='variance_weighted')
rmse_free_on_free = root_mean_squared_error(valid_free0_Y_test,free_on_free_pred)
print(f"Intertrial on free: r2 = {r2_if}, rmse = {rmse_free_on_free}")

# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_transfer_dumbbell(score_ii, score_fi, score_ff, score_if, metric="r2_pooled"):
    """
    score_ii: inter->inter
    score_fi: free->inter
    score_ff: free->free
    score_if: inter->free
    metric: "r2_pooled" or "rmse_pooled"
    """
    # Two paired comparisons by test set
    # Test inter: (inter->inter) vs (free->inter)
    # Test free : (free->free)  vs (inter->free)
    vals = {
        "Test: intertrial": (score_ii, score_fi),
        "Test: free0":      (score_ff, score_if),
    }

    labels = list(vals.keys())
    a = np.array([vals[k][0] for k in labels])  # "matched" model for that test set
    b = np.array([vals[k][1] for k in labels])  # "transfer" model for that test set

    y = np.arange(len(labels))

    plt.figure(figsize=(7, 3.5))
    # connecting lines show pairing (same test data)
    for i in range(len(labels)):
        plt.plot([a[i], b[i]], [y[i], y[i]], linewidth=2)

    plt.scatter(a, y, s=80, label="trained on same condition")
    plt.scatter(b, y, s=80, label="trained on other condition")

    plt.yticks(y, labels)
    plt.xlabel(metric)
    plt.title(f"Within-condition vs cross-condition decoding (paired by test set)")
    plt.legend()
    plt.tight_layout()

    plt.show()

# Example call:
plot_transfer_dumbbell(r2_ii, r2_fi, r2_ff, r2_if, metric="r2_pooled")
# plot_transfer_dumbbell(sc_ii, sc_fi, sc_ff, sc_if, metric="rmse_pooled")


# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_transfer_heatmap(score_ii, score_if, score_fi, score_ff,title, vmin, vmax):
    # rows = train, cols = test
    M = np.array([
        [score_ii, score_if],  # train inter -> test inter/free
        [score_fi, score_ff],  # train free  -> test inter/free
    ], dtype=float)

    plt.figure(figsize=(4.2, 3.8))
    im = plt.imshow(M, aspect="auto", vmin=vmin, vmax=vmax)
    plt.xticks([0, 1], ["Test inter", "Test free"])
    plt.yticks([0, 1], ["Train inter", "Train free"])
    plt.title(f"r2 {title}")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    for (i, j), v in np.ndenumerate(M):
        plt.text(j, i, f"{v:.3f}", ha="center", va="center")

    plt.tight_layout()
    plt.grid(False)
    plt.show()

# call:
plot_transfer_heatmap(r2_ii, r2_if, r2_fi, r2_ff, title = f"{neural_field} -> all keypoints vel", vmin=0, vmax=0.8)


# %%



