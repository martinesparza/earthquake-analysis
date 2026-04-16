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

# %%
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
area = "MOp"
rates_field = f"{area}_rates"
bhv_fields = ["left_paw", "left_foot", "left_ankle"]
delays_bins = [0, -1, -2,-3]
n_pcs = 30

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

# %%
def get_n_time(df, trial_name, field = "MOp_rates"):
    return np.concatenate(df[df['trial_name'] == trial_name][field].values, axis=0).shape[0]

# %%
free0_len = get_n_time(prep_df, "free0", field=field)
windows = split_free(
        free_duration_s=free0_len*Params.BIN_SIZE,
        window_lengths_s=np.unique(df[df['trial_name'] == "intertrial"]["trial_length"].values)*0.01,
        gap_s=1,
        seed=None,
        start_offset_s=0
    )
print(f"n_windows = {len(windows)}")
covered = sum(e - s for s, e in windows)
print(f"total window time = {covered:.1f}s ({covered/60:.2f} min)")

# %%
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


# %%
prep_df =dt.add_velocity_fields(prep_df)

# %%
prep_df,bhv_names = dt.add_bhv(prep_df, bhv_fields=["all_keypoints_vel"])

# %%
field = "left_elbow_vel"
means, used_bins = window_metric(
    prep_df=prep_df,
    field=field,
    windows=windows,
    trial_name="free0"
)




# %%
field = "left_elbow_vel"
means, used_bins = window_metric(
    prep_df=prep_df,
    field=field,
    windows=windows,
    trial_name="free0"
)
it = prep_df[prep_df["trial_name"] == "intertrial"].copy()

it_means = it[field].apply(lambda a: np.nanmean(np.asarray(a, dtype=float))).to_numpy()

plt.figure()
plt.hist(it_means[~np.isnan(it_means)], bins=50, alpha=0.5, label="Intertrial")
plt.hist(means, bins=50, alpha=0.5, label="Free0")
plt.xlabel(f"Mean {field} per window")
plt.ylabel("Count")
plt.legend()
plt.xlabel(f"Mean {field} per intertrial (row)")
plt.ylabel("Count")
plt.title("Distribution of window means")
plt.show()

# %%

field = "left_elbow_angle_vel"
means, used_bins = window_metric(
    prep_df=prep_df,
    field=field,
    windows=windows,
    trial_name="free0",
    metric = np.std
)
it = prep_df[prep_df["trial_name"] == "intertrial"].copy()

it_means = it[field].apply(lambda a: np.nanmean(np.asarray(a, dtype=float))).to_numpy()

plt.figure()
plt.hist(it_means[~np.isnan(it_means)], bins=50, alpha=0.5, label="Intertrial")
plt.hist(means, bins=50, alpha=0.5, label="Free0")
plt.xlabel(f"std {field} per window")
plt.ylabel("Count")
plt.legend()
plt.xlabel(f"std {field} ")
plt.ylabel("Count")
plt.title("Distribution of window means")
plt.show()

# %%
free0_len = get_n_time(prep_df, "free0", field=field)
windows = split_free(
        free_duration_s=free0_len*Params.BIN_SIZE,
        window_lengths_s=np.unique(df[df['trial_name'] == "intertrial"]["trial_length"].values)*0.01,
        gap_s=2,
        seed=None,
        start_offset_s=0
    )
print(f"n_windows = {len(windows)}")
covered = sum(e - s for s, e in windows)
print(f"total window time = {covered:.1f}s ({covered/60:.2f} min)")
field = "hip_center_vel"
means, used_bins = window_means(
    prep_df=prep_df,
    field=field,
    windows=windows,
    trial_name="free0"
)
it = prep_df[prep_df["trial_name"] == "intertrial"].copy()

it_means = it[field].apply(lambda a: np.nanmean(np.asarray(a, dtype=float))).to_numpy()

plt.figure()
plt.hist(it_means[~np.isnan(it_means)], bins=50, alpha=0.5, label="Intertrial")
plt.hist(means, bins=50, alpha=0.5, label="Free0")
plt.xlabel(f"Mean {field} per window")
plt.ylabel("Count")
plt.legend()
plt.xlabel(f"Mean {field} per intertrial (row)")
plt.ylabel("Count")
plt.title("Distribution of window means")
plt.show()

# %%
field = "bhv"
means, used_bins = window_metric(
    prep_df=prep_df,
    field=field,
    windows=windows,
    trial_name="free0",
    metric = np.nanmean
)
it = prep_df[prep_df["trial_name"] == "intertrial"].copy()

it_means = np.vstack(it[field].apply(lambda a: np.nanmean(a, axis=0)))

# %%
field = "bhv"
metric = np.nanvar
vars, used_bins = window_metric(
    prep_df=prep_df,
    field=field,
    windows=windows,
    trial_name="free0",
    metric = metric
)
it = prep_df[prep_df["trial_name"] == "intertrial"].copy()

it_vars = np.vstack(it[field].apply(lambda a: metric(a, axis=0)))

# %%
all_windows_mean = np.vstack((means, it_means))

# %%
all_windows_var = np.vstack((vars, it_vars))

# %%
all_windows = np.hstack((all_windows_mean,all_windows_var))
bhv_names = bhv_names + [f"{bhv}_var" for bhv in bhv_names]

# %%
bhv_names = bhv_names + [f"{bhv}_var" for bhv in bhv_names]

# %%
all_windows.shape

# %%
def prep_bhv(bhv_data):
    """
    Normalize bhv data 
    Assumes bhv_data is a 2D numpy array where each row is a timepoint and each column is a behavioral feature.
    Returns the normalized behavioral data.
    """
    # Normalize each feature (column) to have zero mean and unit variance
    means = np.nanmean(bhv_data, axis=0)
    stds = np.nanstd(bhv_data, axis=0)
    normalized_data = (bhv_data - means) / stds
    return normalized_data
all_windows_normalized = prep_bhv(all_windows)

# %%


# %%
all_windows.shape
model = PCA(n_components=None, svd_solver='full')
model.fit(all_windows_mean)
model.explained_variance_ratio_
components = model.components_
# plot cum var explained
plt.figure()
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA on raw bhv window means")
plt.show()
# plot feature loadings for the first 10 pcs
n_pcs_to_plot = 10
plt.figure(figsize=(12, 6))
for pc_idx in range(n_pcs_to_plot):
    plt.subplot(2, 5, pc_idx + 1)
    plt.bar(range(components.shape[1]), components[pc_idx])
    plt.title(f"PC {pc_idx + 1}")
    plt.xlabel("Feature index")
    plt.ylabel("Loading")
plt.tight_layout()
plt.show()

# %%
n_comp = 10
model = PCA(n_components=n_comp, svd_solver='full')
model.fit(all_windows_mean)
print(np.cumsum(model.explained_variance_ratio_))
components = model.components_
# plot cum var explained
plt.figure()
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA on z-scored bhv window means")
plt.show()
# plot feature loadings for the first 10 pcs
n_pcs_to_plot = 10
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

# %%
n_comp = 10
model = PCA(n_components=n_comp, svd_solver='full')
model.fit(all_windows_normalized)
print(np.cumsum(model.explained_variance_ratio_))
components = model.components_
# plot cum var explained
plt.figure()
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA on z-scored bhv window means")
plt.show()
# plot feature loadings for the first 10 pcs
n_pcs_to_plot = 10
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

# %%
pca_inter = model.transform(it_means)
pca_free0 = model.transform(means)

# %%
pca_inter.shape

# %%
plt.figure()
plt.hist(pca_inter[:,0], bins=30, alpha=0.5, label="Intertrial")
plt.hist(pca_free0[:,0], bins=30, alpha=0.5, label="Free0")
plt.legend()
plt.xlabel(f"Mean pca per window")
plt.ylabel("Count")
plt.title("Distribution of window means")
plt.show()

# %%
from sklearn.linear_model import LogisticRegression
Z = model.transform(all_windows_mean)
y = np.hstack([np.zeros(pca_free0.shape[0], dtype=int), np.ones(pca_inter.shape[0], dtype=int)])
clf = LogisticRegression(max_iter=2000, solver="lbfgs")
clf.fit(Z, y)
p = clf.predict_proba(Z)[:, 1]  # P(intertrial | kinematics)
p_free = p[:pca_free0.shape[0]]
p_it = p[pca_free0.shape[0]:]

# lo, hi = (0.05, 0.95)
# keep = (p_it >= lo) & (p_it <= hi)

# keep_free = keep[:pca_free0.shape[0]]
# keep_it   = keep[pca_free0.shape[0]:]

# Z_free_cs = pca_free0[keep_free]
# Z_it_cs   = pca_inter[keep_it]

# %%
trims = [(0.01,0.99), (0.02,0.98), (0.05,0.95), (0.10,0.90), (0.20,0.80)]

free_keep_fracs = []
it_keep_fracs = []
labels = []
def retention_for_trim(p_free, p_it, trim):
    lo, hi = trim
    keep_free = (p_free >= lo) & (p_free <= hi)
    keep_it = (p_it >= lo) & (p_it <= hi)
    return keep_free, keep_it
for lo, hi in trims:
    keep_free, keep_it = retention_for_trim(p_free, p_it, (lo, hi))
    free_keep_fracs.append(keep_free.mean())
    it_keep_fracs.append(keep_it.mean())
    labels.append(f"[{lo:.2f},{hi:.2f}]")

x = np.arange(len(trims))
width = 0.4

plt.figure()
plt.bar(x - width/2, free_keep_fracs, width, label="Free0 retained")
plt.bar(x + width/2, it_keep_fracs, width, label="Intertrial retained")
plt.xticks(x, labels, rotation=30, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("Fraction retained after prop_trim")
plt.title("Retention vs prop_trim")
plt.legend()
plt.tight_layout()
plt.show()

# %%
plt.figure()
plt.hist(p_free, bins=30, alpha=0.6, label="Free0: P(intertrial|k)")
plt.hist(p_it, bins=30, alpha=0.6, label="Intertrial: P(intertrial|k)")
plt.xlabel("Propensity score")
plt.ylabel("Count")
plt.title("Propensity score distributions")
plt.legend()
plt.show()

# %%
lo, hi = (0.05,0.95)
bins = 50
plt.figure()
plt.hist(p_free, bins=bins, alpha=0.5, label="Free0")
plt.hist(p_it, bins=bins, alpha=0.5, label="Intertrial")
plt.axvline(lo, linestyle="--")
plt.axvline(hi, linestyle="--")
plt.xlabel("Propensity score")
plt.ylabel("Count")
plt.title(f"Propensity distributions with trim lines {lo, hi }")
plt.legend()
plt.show()

# %%
from sklearn.neighbors import NearestNeighbors
idx_free = np.arange(len(windows))
it = pyal.select_trials(prep_df,"trial_name=='intertrial'")
idx_it = it.index.to_numpy()


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
kept_it_idx

# %%
idx_free

# %%
plt.figure()
plt.hist(pca_inter[kept_it_idx,0], bins=50, alpha=0.5, label="Intertrial")
plt.hist(pca_free0[kept_free_idx,0], bins=50, alpha=0.5, label="Free0")
plt.legend()
plt.xlabel(f"Mean pca per window")
plt.ylabel("Count")
plt.title("Distribution of window means")
plt.show()

# %%
len(kept_it_idx)

# %%
len(kept_free_idx)

# %%
# 10 fold cv - split windows

# %%
# check how many keypoints I have in each
np.sum([windows[idx][1]-windows[idx][0] for idx in kept_free_idx])

# %%
np.sum([it["trial_length"].values[intertrial] for intertrial in kept_it_idx])/100

# %%
len(keep_free)

# %%
len(kept_it_idx)

# %%
it["trial_length"]

# %%
kept_it_idx

# %%
plt.figure()
plt.hist(pca_inter[kept_it_idx,0], bins=50, alpha=0.5, label="Intertrial")
plt.hist(pca_free0[kept_free_idx,0], bins=50, alpha=0.5, label="Free0")
plt.legend()
plt.xlabel(f"Mean pca per window")
plt.ylabel("Count")
plt.title("Distribution of window means")
plt.show()

# %%
plt.figure()
plt.hist(pca_inter[:,4], bins=50, alpha=0.5, label="Intertrial")
plt.hist(pca_free0[:,4], bins=50, alpha=0.5, label="Free0")
plt.legend()
plt.xlabel(f"Mean pca per window")
plt.ylabel("Count")
plt.title("Distribution of window means")
plt.show()

# %%
# plot some of the data

# %%
# split test / train idx for each;
# do pca on the training; keep model
# add delays
# do regression to kinematics

# apply the train PCA model on test window
# add delays
# do regression to kinematics
# do rmse and r2 but also save all the predictions to dothe r2 on all the data pooled to see if there s a difference

# repeat for folds;

#  also test on out of distribution data?

# then train on all the data from one condition and test on all the data from the other;


# %%
# kept_it_idx, kept_free_idx;
# also another quick check: plot the distribution from above for each window length;
# number of windows of each duration: 1,3, to have a quick check;




