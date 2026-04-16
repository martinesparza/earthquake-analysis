
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
    "M103_2026_02_17_14_00" # control session
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
n_neurons_increment = 10
n_iterations = 1000


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
            meta_df, X = build_window_tensor_concat_by_trial_type(
                prep_df_no_trials,
                window_size_s=window_size_s,
                field=field,
                trial_type_col="trial_name",   # your type column
                bin_size_col="bin_size",
            )
        
            meta_df = meta_df.reset_index(drop=True)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            X = X.to(device)
            
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
        with open(f"pr_subsampled_{session}_{field}.pkl", "wb") as f:
            pickle.dump(pr_subsampled[session][field], f)
    
    # Save trial type changes
    with open(f"trial_type_changes_{session}.pkl", "wb") as f:
        pickle.dump(trial_type_changes[session], f)
    del prep_dfs