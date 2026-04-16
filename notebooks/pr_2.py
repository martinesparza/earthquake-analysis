
import os
import sys
sys.path.append("../")


import pyaldata as pyal
import pandas as pd
import numpy as np
import json

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
from tqdm.asyncio import tqdm
def compute_pr(S):
    "Participation ratio based on singular values S"
    # compute the participation ratio
    pr = (S**2).sum()**2 / (S**4).sum()
    return pr
def compute_pr_eig(expl_var):
    "Participation ratio based on eigenvalues expl_var"
    # compute the participation ratio
    pr = (expl_var).sum()**2 / (expl_var**2).sum()
    return pr
def compute_explained_variance(X):
    "Explained variance using sklearn PCA"
    pca = PCA(svd_solver='full')
    pca.fit(X)
    return pca.explained_variance_
def svd_torch(X,center = True):
    "torch batched SVD"
    "X_batch of shape (batch_size, n_neurons, n_timepoints)"
    X_torch = torch.tensor(X)
    if center:
        X_centered = X_torch - torch.mean(X_torch, dim=0)
    else:
        X_centered = X_torch
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    return U, S, Vh
def batched_svd(X_torch, center = True):
    "torch batched SVD"
    "X_batch of shape (batch_size, n_timepoints, n_neurons)"
    if center:
        X_centered = X_torch - torch.mean(X_torch, dim=1, keepdim=True)
    else:
        X_centered = X_torch
    _, S_batched, _ = torch.linalg.svd(X_centered, full_matrices=False)
    return S_batched
def batched_pr(S_batched):
    "compute pr for each batch and return the mean and"
    pr_batched = (S_batched**2).sum(dim=1)**2 / (S_batched**4).sum(dim=1)
    return pr_batched



import numpy as np
import pandas as pd
import torch
from typing import Any, Callable, Optional, Sequence, Tuple


def _as_time_by_features_np(arr: Any) -> Optional[np.ndarray]:
    if arr is None:
        return None
    A = np.asarray(arr)
    if A.size == 0:
        return None
    if A.ndim == 1:
        return A.astype(float)[:, None]
    if A.ndim == 2:
        return A.astype(float)
    return A.reshape(A.shape[0], -1).astype(float)


def add_trial_metric_field(
    prep_df: pd.DataFrame,
    input_field: str,
    metric_fn: Callable[[np.ndarray], Any],
    new_field: str,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Compute one metric per row/trial from input_field and add it as new_field.

    metric_fn receives a (T, F) numpy array and should return scalar/array-like.
    """
    out_df = prep_df if inplace else prep_df.copy()
    values = []

    for _, row in out_df.iterrows():
        A = _as_time_by_features_np(row.get(input_field, None))
        if A is None:
            values.append(None)
            continue
        v = metric_fn(A)
        if v is None:
            values.append(None)
            continue
        v_arr = np.asarray(v)
        if v_arr.size == 0:
            values.append(None)
        elif v_arr.ndim == 0:
            values.append(float(v_arr))
        else:
            values.append(v_arr.reshape(-1).astype(float))

    out_df[new_field] = values
    return out_df


def build_window_tensor_sliding_session(
    prep_df: pd.DataFrame,
    window_size_s: float,
    field: str,
    step_size_s: Optional[float] = None,
    overlap_pct: Optional[float] = None,
    bin_size_col: str = "bin_size",
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """
    Concatenate all rows in prep_df in row order and build overlapping/sliding windows.

    step_size_s and overlap_pct are both supported. If both are provided, step_size_s is used.
    overlap_pct is a fraction in [0, 1), where 0.5 means 50% overlap.

    Returns:
      meta_df: one row per window (session-level sample indices)
      X: (n_windows, w, F)
    """
    if len(prep_df) == 0:
        return pd.DataFrame(), torch.empty((0, 0, 0), dtype=torch.float32)

    dt0 = float(prep_df[bin_size_col].iloc[0])

    w = int(round(window_size_s / dt0))
    if w <= 0:
        raise ValueError("window_size_s is too small relative to bin_size.")

    if step_size_s is not None:
        step = int(round(step_size_s / dt0))
        overlap_pct_used = 1.0 - (step / w)
    else:
        if overlap_pct is None:
            raise ValueError("Provide either step_size_s or overlap_pct.")
        if not (0 <= overlap_pct < 1):
            raise ValueError("overlap_pct must be in [0, 1).")
        step = int(round(w * (1.0 - overlap_pct)))
        overlap_pct_used = overlap_pct

    if step <= 0:
        raise ValueError("Derived step size is <= 0. Use lower overlap_pct or larger step_size_s.")

    As = []
    F_ref = None
    for _, row in prep_df.iterrows():
        A = _as_time_by_features_np(row.get(field, None))
        if A is None:
            continue
        if F_ref is None:
            F_ref = A.shape[1]
        elif A.shape[1] != F_ref:
            raise ValueError(f"Feature dim mismatch: expected {F_ref}, got {A.shape[1]}.")
        As.append(A)

    if not As:
        return pd.DataFrame(), torch.empty((0, w, 0), dtype=torch.float32)

    A_cat = np.concatenate(As, axis=0)
    T, F = A_cat.shape
    if T < w:
        return pd.DataFrame(), torch.empty((0, w, F), dtype=torch.float32)

    starts = np.arange(0, T - w + 1, step, dtype=int)
    Xw = np.stack([A_cat[s:s + w, :] for s in starts], axis=0)

    meta_df = pd.DataFrame({
        "window_index": np.arange(len(starts), dtype=int),
        "start_frame": starts,
        "end_frame": starts + w - 1,
        "bin_size": dt0,
        "window_size_samples": w,
        "step_size_samples": step,
        "window_size_s": w * dt0,
        "step_size_s": step * dt0,
        "overlap_pct": overlap_pct_used,
    })
    X = torch.from_numpy(Xw).to(dtype=torch.float32)
    return meta_df, X
def get_stable_neuron_mask(
    session_name,field
) -> torch.BoolTensor:
    """
    Get a boolean mask of neurons that are stable (not silent for long periods).

    Parameters
    ----------
    X :             (nW, w, N) window tensor
    threshold :     firing rate threshold in Hz
    window_size_s : duration of each window in seconds
    min_silent_s :  minimum consecutive silent duration to flag (default 120s)

    Returns
    -------
    stable_mask : BoolTensor of shape (N,) where True indicates a stable neuron
    """
    import pickle
    if not os.path.exists(f"stable_neurons{session_name}_{field}.pkl"):
        return None
    with open(f"stable_neurons{session_name}_{field}.pkl", "rb") as f:
        df_silent = pickle.load(f)  

    return df_silent["type"] == "stable"

# stable_mask_ = get_stable_neuron_mask(sessions[session_idx],field)
def plot_transition_times(
    df_silent: pd.DataFrame,
    X: torch.Tensor,
    window_size_s: float,
    meta_df: pd.DataFrame,
    trial_type_col: str = "trial_type",
    figsize: tuple = (14, 10),
    title: str = "Neuron activity transitions and firing rate raster",
):
    """
    Plot distribution of transition times and a firing rate raster
    sorted by neuron type (stable at top, transition neurons at bottom).

    Parameters
    ----------
    df_silent :     output of detect_silent_neurons
    X :             (nW, w, N) window tensor
    window_size_s : window size in seconds
    meta_df :       window metadata
    trial_type_col: column in meta_df identifying trial type
    """
    X_mean = X.mean(dim=1).cpu().numpy()           # (nW, N)
    nW, N = X_mean.shape
    time_minutes = np.arange(nW) * window_size_s / 60.0
    session_duration_min = time_minutes[-1]
    bins = np.linspace(0, session_duration_min, 30)

    # Epoch boundaries — preserve first-appearance order
    types_in_order = list(pd.unique(meta_df[trial_type_col]))
    epoch_boundaries = {}
    cumulative = 0
    for name in types_in_order:
        n_windows = (meta_df[trial_type_col] == name).sum()
        start = cumulative * window_size_s / 60.0
        cumulative += n_windows
        end = cumulative * window_size_s / 60.0
        epoch_boundaries[name] = (start, end)

    type_colors = {
        "stable":        "steelblue",
        "dropout":       "tomato",
        "emerge":        "mediumseagreen",
        "transient":     "darkorange",
        "always_silent": "lightgrey",
    }
    epoch_cmap = plt.cm.Pastel1(np.linspace(0, 1, len(epoch_boundaries)))

    def add_epoch_shading(ax):
        for (name, (t0, t1)), c in zip(epoch_boundaries.items(), epoch_cmap):
            ax.axvspan(t0, t1, alpha=0.25, color=c, label=name, zorder=0)

    def to_minutes(series):
        return series.dropna().astype(float) * window_size_s / 60.0

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35,
                          height_ratios=[1, 1.8])

    # ── 1. Dropout times ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    times = to_minutes(
        df_silent.loc[df_silent["type"].isin(["dropout", "transient"]),
                      "first_silent_window"]
    )
    add_epoch_shading(ax)
    if len(times) > 0:
        ax.hist(times, bins=bins, color=type_colors["dropout"],
                alpha=0.85, edgecolor="white", zorder=3)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("N neurons")
    ax.set_title(f"Dropout onset (n={len(times)})")
    ax.set_xlim(0, session_duration_min)
    ax.legend(fontsize=7)

    # ── 2. Recovery / emergence times ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    times = to_minutes(
        df_silent.loc[df_silent["type"].isin(["emerge", "transient"]),
                      "first_active_after_silence"]
    )
    add_epoch_shading(ax)
    if len(times) > 0:
        ax.hist(times, bins=bins, color=type_colors["emerge"],
                alpha=0.85, edgecolor="white", zorder=3)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("N neurons")
    ax.set_title(f"Emergence (n={len(times)})")
    ax.set_xlim(0, session_duration_min)
    ax.legend(fontsize=7)

    # ── 3. Transient neurons ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    transient_df = df_silent[df_silent["type"] == "transient"]
    drop_t  = to_minutes(transient_df["first_silent_window"])
    recov_t = to_minutes(transient_df["first_active_after_silence"])
    add_epoch_shading(ax)
    if len(drop_t) > 0:
        ax.hist(drop_t, bins=bins, color=type_colors["dropout"],
                alpha=0.7, edgecolor="white", label="dropout", zorder=3)
    if len(recov_t) > 0:
        ax.hist(recov_t, bins=bins, color=type_colors["emerge"],
                alpha=0.7, edgecolor="white", label="recovery", zorder=3)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("N neurons")
    ax.set_title(f"Transient (n={len(transient_df)})")
    ax.set_xlim(0, session_duration_min)
    ax.legend(fontsize=7)

    # ── 4. Firing rate raster ─────────────────────────────────────────────
    ax_raster = fig.add_subplot(gs[1, :])

    type_order = ["stable", "always_silent", "emerge", "dropout", "transient"]

    def sort_key(row):
        base = type_order.index(row["type"]) if row["type"] in type_order else len(type_order)
        t = row["first_silent_window"] or row["first_active_after_silence"] or 0
        return (base, t)

    sorted_df = df_silent.copy()
    sorted_df["_sort"] = sorted_df.apply(sort_key, axis=1)
    sorted_df = sorted_df.sort_values("_sort").reset_index(drop=True)
    sorted_idxs = sorted_df["neuron_idx"].values

    X_raster = X_mean[:, sorted_idxs].T                   # (N, nW)

    im = ax_raster.imshow(
        X_raster, aspect="auto", cmap="viridis",
        extent=[time_minutes[0], time_minutes[-1], N, 0],
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax_raster, label="Mean firing rate (Hz)", fraction=0.02)

    # Epoch boundary lines
    for (name, (t0, t1)), c in zip(epoch_boundaries.items(), epoch_cmap):
        ax_raster.axvline(t0, color=c, lw=0.8, ls="--", alpha=0.7)
        ax_raster.text((t0 + t1) / 2, N * 1.01, name,
                       fontsize=7, ha="center", va="bottom", color=c)

    # Horizontal lines between neuron types + labels
    cumulative = 0
    for ntype in type_order:
        n_type = (sorted_df["type"] == ntype).sum()
        if n_type == 0:
            continue
        cumulative += n_type
        ax_raster.axhline(cumulative, color="white", lw=0.8,
                          ls="--", alpha=0.7)
        ax_raster.text(
            time_minutes[-1] * 1.001, cumulative - n_type / 2,
            f"{ntype}\n(n={n_type})",
            fontsize=7, va="center",
            color=type_colors.get(ntype, "grey"),
        )
    ax_raster.grid(False)
    ax_raster.set_xlabel("Time (minutes)")
    ax_raster.set_ylabel("Neuron")
    ax_raster.set_title("Firing rate raster")
    ax_raster.set_xlim(0, session_duration_min)

    fig.suptitle(f"{title}  |  window={window_size_s}s", fontsize=11)
    plt.tight_layout()
    plt.show()
def detect_silent_neurons(
    X: torch.Tensor,
    threshold: float,
    window_size_s: float,
    min_silent_s: float = 120.0,
) -> pd.DataFrame:
    """
    Detect neurons that are silent (mean rate < threshold) for at least
    min_silent_s seconds, using the already-windowed tensor X.

    Each window's mean firing rate is compared to threshold. A neuron is
    flagged if it has min_silent_windows consecutive windows below threshold.

    Parameters
    ----------
    X :             (nW, w, N) window tensor — mean over w gives rate per window
    threshold :     firing rate threshold in Hz
    window_size_s : duration of each window in seconds
    min_silent_s :  minimum consecutive silent duration to flag (default 120s)

    Returns
    -------
    df : DataFrame with one row per neuron:
         - neuron_idx
         - is_silent_period : bool — True if any silent period >= min_silent_s
         - first_silent_window : int or None — window index where silence starts
         - last_active_window : int or None — last window before silence
         - first_active_after_silence : int or None — window where it recovers (if any)
         - type : 'stable' | 'dropout' | 'emerge' | 'transient' | 'always_silent'
    """
    min_silent_windows = int(np.ceil(min_silent_s / window_size_s))

    # Mean over time bins within each window -> (nW, N)
    X_mean = X.mean(dim=1).cpu().numpy()        # (nW, N)
    nW, N = X_mean.shape

    below = X_mean < threshold                  # (nW, N) bool

    # For each neuron find runs of consecutive True in below
    # Use cumsum trick: pad with False on both ends, find diff
    # This is vectorised across neurons

    # Pad: (nW+2, N)
    pad = np.zeros((nW + 2, N), dtype=bool)
    pad[1:-1] = below

    diff = np.diff(pad.astype(np.int8), axis=0)    # (nW+1, N)
    # diff == +1: start of silent run (below goes True)
    # diff == -1: end of silent run (below goes False)

    records = []

    for n in range(N):
        starts = np.where(diff[:, n] == 1)[0]      # window indices where silence starts
        ends   = np.where(diff[:, n] == -1)[0]     # window indices where silence ends

        # run lengths
        run_lengths = ends - starts                 # number of consecutive silent windows

        # keep only runs >= min_silent_windows
        long_runs = run_lengths >= min_silent_windows
        long_starts = starts[long_runs]
        long_ends   = ends[long_runs]

        starts_silent = below[:min_silent_windows, n].all() if nW >= min_silent_windows else False
        ends_silent   = below[-min_silent_windows:, n].all() if nW >= min_silent_windows else False

        if len(long_starts) == 0:
            neuron_type = "always_silent" if below[:, n].all() else "stable"
            first_silent  = None
            last_active   = None
            first_recover = None

        elif ends_silent and not starts_silent:
            # Active at start, silent at end — dropout
            neuron_type   = "dropout"
            first_silent  = int(long_starts[0])
            last_active   = int(long_starts[0]) - 1 if long_starts[0] > 0 else None
            first_recover = None

        elif starts_silent and not ends_silent:
            # Silent at start, active at end — emerge
            neuron_type   = "emerge"
            first_silent  = 0
            last_active   = None
            first_recover = int(long_ends[0])

        else:
            # Has long silent period(s) but active at both ends — transient
            neuron_type   = "transient"
            first_silent  = int(long_starts[0])
            last_active   = int(long_starts[0]) - 1 if long_starts[0] > 0 else None
            first_recover = int(long_ends[0]) if len(long_ends) > 0 else None

        records.append({
            "neuron_idx":                n,
            "type":                      neuron_type,
            "is_silent_period":          len(long_starts) > 0,
            "first_silent_window":       first_silent,
            "last_active_window":        last_active,
            "first_active_after_silence": first_recover,
            "n_long_silent_runs":        int(long_runs.sum()),
        })

    df = pd.DataFrame(records)
    print(df["type"].value_counts().to_string())
    return df
def get_stable_neurons_windows(prep_df, field, window_size_s):
    prep_df_no_trials = pyal.select_trials(prep_df,"trial_name!='trial'" )
    meta_df, X = build_window_tensor_concat_by_trial_type(
        prep_df_no_trials,
        window_size_s=window_size_s,
        field=field,
        trial_type_col="trial_name",   # your type column
        bin_size_col="bin_size",
    )
    # print(f" windows: {X.shape[0]} , bins per window: {X.shape[1]}, features: {X.shape[2]} ")
    # print(meta_df["trial_type"].value_counts())
    meta_df = meta_df.reset_index(drop=True) 
    # load neurons_to_keep from json   
    with open("neurons_to_keep.json", "r") as f:
        neurons_to_keep_dict = json.load(f)
    n_neurons_to_keep = neurons_to_keep_dict.get(session, {}).get(field, None)
    if field == "CP_rates":
        X = X[:,:,n_neurons_to_keep:] if n_neurons_to_keep is not None else X
    else:
        X = X[:,:,:n_neurons_to_keep] if n_neurons_to_keep is not None else X
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)

    stable_mask = get_stable_neuron_mask(prep_df.session[0],field)
    if stable_mask is None:
        df_silent = detect_silent_neurons(
            X=X,                        # (nW, w, N) already on device or cpu
            threshold=0.5,
            window_size_s=window_size_s,
            min_silent_s=120.0,
        )

        # Stable neuron mask
        stable_mask = df_silent["type"] == "stable"
    # print(f"Stable: {stable_mask.sum()} / {len(stable_mask)}")
    # keep only stable neurons in X
    X_stable = X[:, :, stable_mask] 
    return meta_df, X_stable

    

def build_window_tensor_concat_by_trial_type(
    prep_df: pd.DataFrame,
    window_size_s: float,
    field: str,
    trial_type_col: str = "trial_name",
    bin_size_col: str = "bin_size",
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """
    Concatenate all rows within each trial type, window within that type only (no crossing types).

    Returns:
      meta_df: one row per window, only column is 'trial_type'
      X: (nW_total, w, F)
    """
    if len(prep_df) == 0:
        return pd.DataFrame(columns=["trial_type"]), torch.empty((0, 0, 0), dtype=torch.float32)

    dt0 = float(prep_df[bin_size_col].iloc[0])
    if not np.allclose(prep_df[bin_size_col].astype(float).to_numpy(), dt0):
        raise ValueError("bin_size varies across rows; enforce constant bin_size or adapt w per row.")

    w = int(round(window_size_s / dt0))
    if w <= 0:
        raise ValueError("window_size_s is too small relative to bin_size.")

    # Preserve order of first appearance of types
    types_in_order = list(pd.unique(prep_df[trial_type_col]))

    meta_parts = []
    X_parts = []
    F_global = None

    for t in types_in_order:
        sub = prep_df[prep_df[trial_type_col] == t]

        As = []
        F_ref = None
        for _, row in sub.iterrows():
            A = _as_time_by_features_np(row.get(field, None))
            if A is None:
                continue
            if F_ref is None:
                F_ref = A.shape[1]
            elif A.shape[1] != F_ref:
                raise ValueError(f"Feature dim mismatch within type '{t}': expected {F_ref}, got {A.shape[1]}.")
            As.append(A)

        if not As:
            continue

        A_cat = np.concatenate(As, axis=0)  # (Tcat, F)
        Tcat, F = A_cat.shape
        nW = Tcat // w
        if nW <= 0:
            continue

        A_cat = A_cat[: nW * w, :]
        Xw = A_cat.reshape(nW, w, F)
        Xt = torch.from_numpy(Xw).to(dtype=torch.float32)

        if F_global is None:
            F_global = F
        elif F != F_global:
            raise ValueError(f"Feature dim mismatch across types: expected {F_global}, got {F} for type '{t}'.")

        X_parts.append(Xt)
        meta_parts.append(pd.DataFrame({"trial_type": [t] * nW}))

    meta_df = pd.concat(meta_parts, ignore_index=True) if meta_parts else pd.DataFrame(columns=["trial_type"])
    if X_parts:
        X = torch.cat(X_parts, dim=0)
    else:
        # If nothing yielded windows, keep shape consistent with requested w
        X = torch.empty((0, w, 0), dtype=torch.float32)

    return meta_df, X


import numpy as np
import pandas as pd
from typing import Sequence, Optional, List
def trial_change_boundaries(meta_df: pd.DataFrame, col: str = "trial_name") -> List[int]:
    """
    Returns boundary indices k such that a change occurs between k-1 and k.
    These are suitable for drawing a line at x = k-0.5 on an imshow heatmap.
    """
    
    names = meta_df[col].astype(str).to_numpy()
    if names.size == 0:
        return []
    change = names[1:] != names[:-1]
    # boundary index is the index of the first element of the new block
    return (np.where(change)[0] + 1).astype(int).tolist()



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
    # "M078_2025_08_08_10_30",# muscimol session
    # "M086_2025_12_09_16_00",
    # "M086_2025_12_10_15_00",
    # "M086_2025_12_11_15_00"
    # "M066_2025_04_09_15_45" # bci session
    # "M103_2026_02_17_14_00" # control session
    "M103_2026_02_17_14_00",# control session
    "M103_2026_02_18_15_30",
    "M103_2026_02_19_15_30",
    "M103_2026_02_20_16_00",# muscimol session
    "M106_2026_02_24_15_00",# control session
    "M106_2026_02_25_15_00",
    "M106_2026_02_26_16_00",
    "M106_2026_02_27_16_00"# muscimol session





]
window_sizes = [180]
n_neurons_increment = 1
n_iterations = 500


for session in sessions:
    prep_dfs = dt.load_sessions([session], prep = True, only_trials = False)

    pr_subsampled = {}
    trial_type_changes = {}

    pr_subsampled[session] = {}
    prep_df_no_trials = pyal.select_trials(prep_dfs[0],"trial_name!='trial'" )

    for field in ["MOp_rates", "SSp_rates", "CP_rates", "VAL_rates","MOs_rates"]:
        if field not in prep_df_no_trials.columns:
            continue
        pr_subsampled[session][field] = {}
        # Build windows by concatenating within each trial type (e.g., trial vs intertrial)
        for window_size_s in window_sizes:
            meta_df, X = get_stable_neurons_windows(prep_df_no_trials, field, window_size_s)
            
            # Get total number of neurons available
            n_neurons_total = X.shape[2]
            
            pr_subsampled[session][field][window_size_s] = {}
            
            # Iterate over different numbers of neurons
            for n_neurons in range(n_neurons_increment, n_neurons_total + 1, n_neurons_increment):
                pr_results = []
                
                # Run multiple iterations with random subsampling
                for iteration in tqdm(range(n_iterations)):
                    # Randomly select neuron indices
                    neuron_indices = np.random.choice(n_neurons_total, size=n_neurons, replace=False)
                    
                    # Subsample neurons
                    X_subsampled = X[:, :, neuron_indices]
                    
                    # Compute SVD and PR on subsampled data
                    S_batched = batched_svd(X_subsampled, center=True)
                    pr_values = batched_pr(S_batched)
                    
                    # Store results for this iteration
                    pr_results.append(pr_values.cpu().numpy())
                
                # Stack results: (n_iterations, n_windows)
                pr_subsampled[session][field][window_size_s][n_neurons] = np.vstack(pr_results)
            
            # Store trial type change indices for later plotting
            trial_type_changes[session] = trial_change_boundaries(meta_df, col="trial_type")
        
        # Save pr_subsampled and trial_type_changes to disk
        with open(f"/data/pr/pr_subsampled_{session}_{field}.pkl", "wb") as f:
            pickle.dump(pr_subsampled[session][field], f)
    
    # Save trial type changes
    with open(f"/data/pr/trial_type_changes_{session}.pkl", "wb") as f:
        pickle.dump(trial_type_changes[session], f)
    del prep_dfs