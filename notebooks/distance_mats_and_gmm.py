#!/usr/bin/env python3
import ast
import gc
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyaldata as pyal
import torch
from sklearn.mixture import GaussianMixture

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools import dataTools as dt  # noqa: E402


AREAS = ("MOp", "MOs", "SSp", "CP", "VAL")
N_PCS_OPTIONS = (5, 10, 30)
WINDOW_SIZES_S = (30.0, 60.0)
OVERLAP_PCT = 0.5
GMM_CLUSTER_OPTIONS = (2, 3)
OUT_DIR = ROOT / "notebooks" / "figures" / "windows" / "exports_dist_gmm"
WINDOWS_NOTEBOOK = ROOT / "notebooks" / "windows.ipynb"



def as_time_by_features_np(arr: Any) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.size == 0:
        return None
    if a.ndim == 1:
        return a.astype(float)[:, None]
    if a.ndim == 2:
        return a.astype(float)
    return a.reshape(a.shape[0], -1).astype(float)


def build_window_tensor_concat_by_trial_type(
    prep_df: pd.DataFrame,
    window_size_s: float,
    field: str,
    trial_type_col: str = "trial_name",
    bin_size_col: str = "bin_size",
) -> Tuple[pd.DataFrame, np.ndarray]:
    if len(prep_df) == 0:
        return pd.DataFrame(columns=["trial_type"]), np.empty((0, 0, 0), dtype=np.float32)

    dt0 = float(prep_df[bin_size_col].iloc[0])
    if not np.allclose(prep_df[bin_size_col].astype(float).to_numpy(), dt0):
        raise ValueError("bin_size varies across rows.")

    w = int(round(window_size_s / dt0))
    if w <= 0:
        raise ValueError("window_size_s is too small relative to bin_size.")

    types_in_order = list(pd.unique(prep_df[trial_type_col]))
    meta_parts: List[pd.DataFrame] = []
    x_parts: List[np.ndarray] = []

    for t in types_in_order:
        sub = prep_df[prep_df[trial_type_col] == t]
        mats = []
        f_ref = None
        for _, row in sub.iterrows():
            a = as_time_by_features_np(row.get(field, None))
            if a is None:
                continue
            if f_ref is None:
                f_ref = a.shape[1]
            elif a.shape[1] != f_ref:
                raise ValueError(f"Feature mismatch in trial type {t}.")
            mats.append(a)
        if not mats:
            continue

        cat = np.concatenate(mats, axis=0)
        tcat, f = cat.shape
        n_w = tcat // w
        if n_w <= 0:
            continue

        cat = cat[: n_w * w, :]
        xw = cat.reshape(n_w, w, f).astype(np.float32)
        x_parts.append(xw)
        meta_parts.append(pd.DataFrame({"trial_type": [t] * n_w}))

    meta_df = pd.concat(meta_parts, ignore_index=True) if meta_parts else pd.DataFrame(columns=["trial_type"])
    x = np.concatenate(x_parts, axis=0) if x_parts else np.empty((0, w, 0), dtype=np.float32)
    return meta_df, x


def build_window_tensor_sliding_session_with_labels(
    prep_df: pd.DataFrame,
    window_size_s: float,
    field: str,
    step_size_s: Optional[float] = None,
    overlap_pct: Optional[float] = 0.5,
    trial_name_col: str = "trial_name",
    bin_size_col: str = "bin_size",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if len(prep_df) == 0:
        return pd.DataFrame(), np.empty((0, 0, 0), dtype=np.float32), np.array([], dtype=str)

    dt0 = float(prep_df[bin_size_col].iloc[0])
    if not np.allclose(prep_df[bin_size_col].astype(float).to_numpy(), dt0):
        raise ValueError("bin_size varies across rows.")

    w = int(round(window_size_s / dt0))
    if w <= 0:
        raise ValueError("window_size_s is too small relative to bin_size.")

    if step_size_s is not None:
        step = int(round(step_size_s / dt0))
    else:
        if overlap_pct is None:
            raise ValueError("Provide step_size_s or overlap_pct.")
        if not (0 <= overlap_pct < 1):
            raise ValueError("overlap_pct must be in [0,1).")
        step = int(round(w * (1.0 - overlap_pct)))
    if step <= 0:
        raise ValueError("Derived step size <= 0.")

    mats = []
    sample_labels = []
    f_ref = None
    for _, row in prep_df.iterrows():
        a = as_time_by_features_np(row.get(field, None))
        if a is None:
            continue
        if f_ref is None:
            f_ref = a.shape[1]
        elif a.shape[1] != f_ref:
            raise ValueError("Feature mismatch across rows.")
        mats.append(a)
        sample_labels.extend([str(row.get(trial_name_col, ""))] * a.shape[0])

    if not mats:
        return pd.DataFrame(), np.empty((0, w, 0), dtype=np.float32), np.array([], dtype=str)

    cat = np.concatenate(mats, axis=0)
    labels = np.asarray(sample_labels, dtype=str)
    t_total, f = cat.shape
    if t_total < w:
        return pd.DataFrame(), np.empty((0, w, f), dtype=np.float32), np.array([], dtype=str)

    starts = np.arange(0, t_total - w + 1, step, dtype=int)
    xw = np.stack([cat[s : s + w, :] for s in starts], axis=0).astype(np.float32)

    w_labels = []
    for s in starts:
        chunk = labels[s : s + w]
        vals, cnt = np.unique(chunk, return_counts=True)
        w_labels.append(vals[int(np.argmax(cnt))] if len(vals) > 0 else "")
    w_labels = np.asarray(w_labels, dtype=str)

    meta_df = pd.DataFrame(
        {
            "window_index": np.arange(len(starts), dtype=int),
            "start_frame": starts,
            "end_frame": starts + w - 1,
            "window_size_s": float(w * dt0),
            "step_size_s": float(step * dt0),
        }
    )
    return meta_df, xw, w_labels


def pairwise_euclidean_on_window_centroids(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        y = x.mean(axis=1)
    elif x.ndim == 2:
        y = x
    else:
        raise ValueError("x must be 2D or 3D.")
    if len(y) == 0:
        return np.empty((0, 0), dtype=np.float32)
    d = y[:, None, :] - y[None, :, :]
    sq = np.einsum("ijk,ijk->ij", d, d)
    return np.sqrt(np.maximum(sq, 0)).astype(np.float32)


def pca_bases_batched_svd_torch(
    x: np.ndarray,  # (nW, T, F)
    n_components: int = 10,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Returns Q: (nW, F, k) where k=min(n_components, F, T-1).
    Torch batched SVD implementation (as in reg.ipynb style).
    """
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    if xt.ndim != 3:
        raise ValueError("x must be (nW, T, F).")
    n_w, t, f = xt.shape
    max_k = min(int(f), int(t) - 1)
    k = min(int(n_components), max_k)
    if k <= 0:
        return torch.empty((n_w, f, 0), device=xt.device, dtype=xt.dtype)

    mu = torch.nanmean(xt, dim=1, keepdim=True)
    xc = xt - mu
    col_means = torch.nanmean(xc, dim=1, keepdim=True)
    xc = torch.where(torch.isnan(xc), col_means, xc)
    xc = torch.nan_to_num(xc, nan=0.0, posinf=0.0, neginf=0.0)

    _, _, vh = torch.linalg.svd(xc, full_matrices=False)  # (nW, r, F)
    q = vh[:, :k, :].transpose(1, 2).contiguous()  # (nW, F, k)
    return q


def mean_principal_cosine_matrix_torch(
    q: torch.Tensor,  # (nW, F, k)
    block: int = 64,
) -> torch.Tensor:
    """
    Returns C: (nW, nW) where C[i,j] is mean cos(theta) across principal angles.
    Uses batched SVD values over Q_i^T Q_j; keeps cosine directly (no arccos).
    """
    if q.ndim != 3:
        raise ValueError("q must be (nW, F, k).")
    n_w, _, k = q.shape
    if k == 0:
        return torch.full((n_w, n_w), torch.nan, device=q.device, dtype=q.dtype)

    qt = q.transpose(1, 2).contiguous()  # (nW, k, F)
    c = torch.empty((n_w, n_w), device=q.device, dtype=q.dtype)

    for i0 in range(0, n_w, block):
        i1 = min(i0 + block, n_w)
        qt_blk = qt[i0:i1]  # (B, k, F)
        m = torch.einsum("bkf,jfl->bjkl", qt_blk, q)  # (B, nW, k, k)
        s = torch.linalg.svdvals(m.reshape(-1, k, k))  # (B*nW, k)
        s = torch.clamp(s, -1.0, 1.0)
        sim = s.reshape(i1 - i0, n_w, k).mean(dim=2)
        c[i0:i1, :] = sim

    return c


def principal_cosine_similarity_on_windows(
    x: np.ndarray,  # (nW, T, F)
    n_components: int = 10,
    block: int = 64,
    use_cuda: bool = True,
) -> np.ndarray:
    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    q = pca_bases_batched_svd_torch(x, n_components=n_components, device=device)
    c = mean_principal_cosine_matrix_torch(q, block=block)
    return c.detach().cpu().numpy().astype(np.float32, copy=False)


def free_boundaries_from_labels(labels: np.ndarray) -> Tuple[int, int]:
    if labels.size == 0:
        return -1, -1
    free0 = np.where(labels == "free0")[0]
    free1 = np.where(labels == "free1")[0]
    free0_end = int(free0.max()) if free0.size else -1
    free1_start = int(free1.min()) if free1.size else -1
    return free0_end, free1_start


def gmm_session_summary(
    labels_by_window: np.ndarray,
    x: np.ndarray,
    session: str,
    area: str,
    window_size_s: float,
    n_clusters: int,
    random_state: int = 0,
) -> Dict[str, Any]:
    if x.ndim == 3:
        y = x.mean(axis=1)
    elif x.ndim == 2:
        y = x
    else:
        raise ValueError("x must be 2D or 3D.")

    if len(y) == 0:
        return {
            "session": session,
            "animal": session.split("_")[0],
            "area": area,
            "window_size_s": float(window_size_s),
            "n_clusters": int(n_clusters),
            "n_windows": 0,
            "cluster_mean_distance_mean": np.nan,
            "cluster_mean_distance_min": np.nan,
            "cluster_mean_distance_max": np.nan,
            "spread_mean_dist_to_center": np.nan,
            "fraction_intertrials": np.nan,
            "intertrial_dominant_cluster_fraction": np.nan,
            "transition_after_intertrial_start_s": np.nan,
            "transition_duration_s": np.nan,
            "bic": np.nan,
            "aic": np.nan,
        }

    gmm = GaussianMixture(
        n_components=int(n_clusters),
        covariance_type="full",
        random_state=random_state,
        n_init=10,
    )
    gmm.fit(y)
    cluster = gmm.predict(y)
    post = gmm.predict_proba(y)
    conf = post.max(axis=1)

    means = gmm.means_
    if len(means) > 1:
        dm = means[:, None, :] - means[None, :, :]
        d_means = np.sqrt(np.maximum(np.einsum("ijk,ijk->ij", dm, dm), 0))
        offdiag = d_means[~np.eye(d_means.shape[0], dtype=bool)]
        cdist_mean = float(np.mean(offdiag))
        cdist_min = float(np.min(offdiag))
        cdist_max = float(np.max(offdiag))
    else:
        cdist_mean = np.nan
        cdist_min = np.nan
        cdist_max = np.nan

    d_to_center = np.linalg.norm(y - means[cluster], axis=1)
    spread = float(np.mean(d_to_center)) if len(d_to_center) else np.nan

    labels_by_window = np.asarray(labels_by_window, dtype=str)
    inter_mask = labels_by_window == "intertrial" if len(labels_by_window) else np.array([], dtype=bool)
    frac_inter = float(inter_mask.mean()) if inter_mask.size else np.nan

    dom_inter = np.nan
    if inter_mask.size and np.any(inter_mask):
        inter_clusters = cluster[inter_mask]
        uniq, cnt = np.unique(inter_clusters, return_counts=True)
        dom_inter = float(cnt.max() / cnt.sum()) if len(cnt) else np.nan

    transition_after = np.nan
    transition_dur = np.nan
    if len(cluster) > 1 and inter_mask.size and np.any(inter_mask):
        inter_start = int(np.where(inter_mask)[0][0])
        cps = np.where(cluster[1:] != cluster[:-1])[0] + 1
        cps = cps[cps >= inter_start]
        if len(cps):
            cp = int(cps[0])
            transition_after = float((cp - inter_start) * window_size_s)

            l = cp
            while l - 1 >= 0 and conf[l - 1] < 0.75:
                l -= 1
            r = cp
            while r < len(conf) and conf[r] < 0.75:
                r += 1
            dur_w = max(1, r - l)
            transition_dur = float(dur_w * window_size_s)

    return {
        "session": session,
        "animal": session.split("_")[0],
        "area": area,
        "window_size_s": float(window_size_s),
        "n_clusters": int(n_clusters),
        "n_windows": int(len(y)),
        "cluster_mean_distance_mean": cdist_mean,
        "cluster_mean_distance_min": cdist_min,
        "cluster_mean_distance_max": cdist_max,
        "spread_mean_dist_to_center": spread,
        "fraction_intertrials": frac_inter,
        "intertrial_dominant_cluster_fraction": dom_inter,
        "transition_after_intertrial_start_s": transition_after,
        "transition_duration_s": transition_dur,
        "bic": float(gmm.bic(y)),
        "aic": float(gmm.aic(y)),
    }


def append_row_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    sessions = [
    'M062_2025_03_19_14_00',
    'M062_2025_03_20_14_00',
    'M062_2025_03_21_14_00',
    'M061_2025_03_04_10_00',
    'M061_2025_03_05_14_00',
    "M061_2025_03_06_14_00",
    "M063_2025_03_12_14_00",  
    "M063_2025_03_13_14_00",  
    "M063_2025_03_14_15_30",
    "M078_2025_08_05_15_30",
    "M078_2025_08_06_15_00",
    "M078_2025_08_07_13_30",
    "M078_2025_08_08_10_30",
    "M086_2025_12_09_16_00",
    "M086_2025_12_10_15_00",
    # "M086_2025_12_11_15_00"
    # "M066_2025_04_09_15_45" # bci session

]

    out_root = OUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    boundary_index_csv = out_root / "window_label_boundaries_index.csv"
    # gmm_summary_csv = out_root / "gmm_summary_all_sessions.csv"

    # Start each run with fresh summary files, then append as we compute.
    for p in (boundary_index_csv,):
        if p.exists():
            p.unlink()

    for sidx, session in enumerate(sessions):
        animal = session.split("_")[0]
        print(f"[{sidx+1}/{len(sessions)}] {session}")
        prep_df_list = dt.load_sessions([session], prep=True, only_trials=False)
        if len(prep_df_list) == 0:
            print("  - skip session: dt.load_sessions returned empty list")
            continue
        prep_df = prep_df_list[0]

        sess_out = out_root / animal / session
        sess_out.mkdir(parents=True, exist_ok=True)

        prep_no_trial = pyal.select_trials(prep_df, "trial_name!='trial'")

        for area in AREAS:
            rates_field = f"{area}_rates"

            if rates_field not in prep_df.columns:
                print(f"  - skip {area}: missing {rates_field}")
                continue

            try:
                alldata = np.concatenate(prep_no_trial[rates_field].values, axis=0)
            except Exception:
                print(f"  - skip {area}: cannot concatenate {rates_field}")
                continue

            if alldata.ndim != 2 or alldata.shape[0] == 0:
                print(f"  - skip {area}: invalid concatenated shape for {rates_field}: {alldata.shape}")
                continue

            n_neurons = int(alldata.shape[1])
            for ws in WINDOW_SIZES_S:
                # Build windows directly from raw {area}_rates.
                meta_sl, x_sl, labels_sl = build_window_tensor_sliding_session_with_labels(
                    prep_df,
                    window_size_s=ws,
                    field=rates_field,
                    overlap_pct=OVERLAP_PCT,
                    trial_name_col="trial_name",
                    bin_size_col="bin_size",
                )
                free0_end_sl, free1_start_sl = free_boundaries_from_labels(labels_sl)

                meta_bt, x_bt = build_window_tensor_concat_by_trial_type(
                    prep_no_trial,
                    window_size_s=ws,
                    field=rates_field,
                    trial_type_col="trial_name",
                    bin_size_col="bin_size",
                )
                labels_bt = meta_bt["trial_type"].astype(str).to_numpy() if len(meta_bt) else np.array([], dtype=str)
                free0_end_bt, free1_start_bt = free_boundaries_from_labels(labels_bt)

                for n_pcs in N_PCS_OPTIONS:
                    if n_pcs > n_neurons:
                        print(f"  - skip {area}: n_pcs={n_pcs} > n_neurons={n_neurons} for this session")
                        continue

                    mode_dir = sess_out / area / f"npc{int(n_pcs)}" / f"ws{int(ws)}"
                    mode_dir.mkdir(parents=True, exist_ok=True)

                    c_sl = principal_cosine_similarity_on_windows(
                        x_sl,
                        n_components=n_pcs,
                        block=64,
                    )
                    np.savez_compressed(
                        mode_dir / "principal_cosine_sliding.npz",
                        cosine_similarity=c_sl.astype(np.float32),
                        labels=labels_sl.astype(str),
                        free0_end=np.int32(free0_end_sl),
                        free1_start=np.int32(free1_start_sl),
                        window_size_s=np.float32(ws),
                        step_size_s=np.float32(meta_sl["step_size_s"].iloc[0]) if len(meta_sl) else np.float32(np.nan),
                        subspace_components=np.int32(n_pcs),
                        source_field=np.array(rates_field),
                    )

                    boundary_row_sliding = {
                        "session": session,
                        "animal": animal,
                        "area": area,
                        "n_pcs": int(n_pcs),
                        "window_size_s": ws,
                        "mode": "sliding",
                        "n_windows": int(len(labels_sl)),
                        "free0_end": free0_end_sl,
                        "free1_start": free1_start_sl,
                    }
                    append_row_csv(boundary_index_csv, boundary_row_sliding)

                    c_bt = principal_cosine_similarity_on_windows(
                        x_bt,
                        n_components=n_pcs,
                        block=64,
                    )
                    np.savez_compressed(
                        mode_dir / "principal_cosine_by_trial_type.npz",
                        cosine_similarity=c_bt.astype(np.float32),
                        labels=labels_bt.astype(str),
                        free0_end=np.int32(free0_end_bt),
                        free1_start=np.int32(free1_start_bt),
                        window_size_s=np.float32(ws),
                        subspace_components=np.int32(n_pcs),
                        source_field=np.array(rates_field),
                    )

                    boundary_row_bytype = {
                        "session": session,
                        "animal": animal,
                        "area": area,
                        "n_pcs": int(n_pcs),
                        "window_size_s": ws,
                        "mode": "by_trial_type",
                        "n_windows": int(len(labels_bt)),
                        "free0_end": free0_end_bt,
                        "free1_start": free1_start_bt,
                    }
                    append_row_csv(boundary_index_csv, boundary_row_bytype)

                    # GMM summary already saved in previous runs.
                    # for k in GMM_CLUSTER_OPTIONS:
                    #     row = gmm_session_summary(
                    #         labels_by_window=labels_bt,
                    #         x=x_bt,
                    #         session=session,
                    #         area=area,
                    #         window_size_s=ws,
                    #         n_clusters=k,
                    #         random_state=sidx + k,
                    #     )
                    #     row["n_pcs"] = int(n_pcs)
                    #     append_row_csv(gmm_summary_csv, row)

        # Release per-session memory before loading next session.
        del prep_no_trial
        del prep_df
        del prep_df_list
        gc.collect()

    config = {
        "sessions": sessions,
        "areas": list(AREAS),
        "n_pcs_options": list(N_PCS_OPTIONS),
        "window_sizes_s": list(WINDOW_SIZES_S),
        "overlap_pct_sliding": OVERLAP_PCT,
        # "gmm_cluster_options": list(GMM_CLUSTER_OPTIONS),
        "principal_cosine_subspace_components": "equals n_pcs for each saved folder",
        "principal_cosine_source_field": "{area}_rates",
    }
    (out_root / "export_config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved outputs to: {out_root}")


if __name__ == "__main__":
    main()
