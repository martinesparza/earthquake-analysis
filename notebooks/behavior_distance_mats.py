#!/usr/bin/env python3
import gc
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyaldata as pyal

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools import dataTools as dt  # noqa: E402


SESSIONS = [
    # "M062_2025_03_19_14_00",
    # "M062_2025_03_20_14_00",
    # "M062_2025_03_21_14_00",
    # "M061_2025_03_04_10_00",
    # "M061_2025_03_05_14_00",
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
    # "M106_2026_02_24_15_00",
    # "M103_2026_02_17_14_00"
    # "M103_2026_02_17_14_00" # control session
    "M103_2026_02_17_14_00",# control session
    "M103_2026_02_18_15_30",
    "M103_2026_02_19_15_30",
    "M103_2026_02_20_16_00",# muscimol session
    "M106_2026_02_24_15_00",# control session
    "M106_2026_02_25_15_00",
    "M106_2026_02_26_16_00",
    "M106_2026_02_27_16_00",# muscimol session
    "M081_2026_01_28_16_00"
]

WINDOW_SIZES_S = (10.0, 30.0, 60.0)
OUT_ROOT = ROOT / "notebooks" / "figures" / "windows" / "exports_dist_gmm"


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


def scalar_series(arr: Any) -> Optional[np.ndarray]:
    a = as_time_by_features_np(arr)
    if a is None:
        return None
    if a.shape[1] == 1:
        return a[:, 0].astype(float)
    return np.linalg.norm(a.astype(float), axis=1)


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
                raise ValueError(f"Feature mismatch in trial type '{t}' for field {field}.")
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


def pairwise_euclidean_from_vector(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=float).reshape(-1, 1)
    if len(x) == 0:
        return np.empty((0, 0), dtype=np.float32)
    d = x[:, None, :] - x[None, :, :]
    sq = np.einsum("ijk,ijk->ij", d, d)
    return np.sqrt(np.maximum(sq, 0)).astype(np.float32)


def free_boundaries_from_labels(labels: np.ndarray) -> Tuple[int, int]:
    if labels.size == 0:
        return -1, -1
    free0 = np.where(labels == "free0")[0]
    free1 = np.where(labels == "free1")[0]
    free0_end = int(free0.max()) if free0.size else -1
    free1_start = int(free1.min()) if free1.size else -1
    return free0_end, free1_start


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def pair_distance_ts(row: pd.Series, left_field: str, right_field: str) -> Optional[np.ndarray]:
    a = as_time_by_features_np(row.get(left_field, None))
    b = as_time_by_features_np(row.get(right_field, None))
    if a is None or b is None:
        return None
    t = min(len(a), len(b))
    if t == 0:
        return None
    a = a[:t, :3] if a.shape[1] >= 3 else a[:t, :]
    b = b[:t, :3] if b.shape[1] >= 3 else b[:t, :]
    return np.linalg.norm(a.astype(float) - b.astype(float), axis=1)


def mean_group_vel_ts(row: pd.Series, vel_fields: Sequence[str]) -> Optional[np.ndarray]:
    vals = []
    for f in vel_fields:
        s = scalar_series(row.get(f, None))
        if s is None or len(s) == 0:
            continue
        vals.append(s)
    if not vals:
        return None
    t = min(len(v) for v in vals)
    if t == 0:
        return None
    return np.vstack([v[:t] for v in vals]).mean(axis=0).astype(float)


def hip_center_xz_angle(row: pd.Series) -> Optional[np.ndarray]:
    """Compute the x-z plane angle of hip center position (arctan2(z, x))."""
    pos = as_time_by_features_np(row.get("hip_center", None))
    if pos is None:
        return None
    if pos.shape[1] < 3:
        return None
    x = pos[:, 0].astype(float)
    z = pos[:, 2].astype(float)
    angle = np.arctan2(z, x).astype(float)
    return angle


def add_behavior_metric_fields(prep_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = prep_df.copy()
    metric_fields: List[str] = []

    def add_col(name: str, vals: List[Optional[np.ndarray]]) -> None:
        out[name] = vals
        metric_fields.append(name)

    # Requested grouped velocity metrics
    add_col(
        "left_forelimb_vel_mean",
        [mean_group_vel_ts(r, ["left_paw_vel", "left_elbow_vel", "left_shoulder_vel"]) for _, r in out.iterrows()],
    )
    add_col(
        "right_forelimb_vel_mean",
        [mean_group_vel_ts(r, ["right_paw_vel", "right_elbow_vel", "right_shoulder_vel"]) for _, r in out.iterrows()],
    )
    add_col(
        "left_hindlimb_vel_mean",
        [mean_group_vel_ts(r, ["left_foot_vel", "left_ankle_vel", "left_knee_vel"]) for _, r in out.iterrows()],
    )
    add_col(
        "right_hindlimb_vel_mean",
        [mean_group_vel_ts(r, ["right_foot_vel", "right_ankle_vel", "right_knee_vel"]) for _, r in out.iterrows()],
    )

    # Requested distances
    add_col("lr_paw_dist", [pair_distance_ts(r, "left_paw", "right_paw") for _, r in out.iterrows()])
    add_col("lr_hind_knee_dist", [pair_distance_ts(r, "left_knee", "right_knee") for _, r in out.iterrows()])
    add_col("lr_hind_ankle_dist", [pair_distance_ts(r, "left_ankle", "right_ankle") for _, r in out.iterrows()])
    add_col("lr_hind_foot_dist", [pair_distance_ts(r, "left_foot", "right_foot") for _, r in out.iterrows()])

    # Hip center x-z angle
    add_col("hip_center_xz_angle", [hip_center_xz_angle(r) for _, r in out.iterrows()])

    # Requested angles: each *_angle (excluding *_angle_vel)
    angle_fields = [c for c in out.columns if ("_angle" in c and not c.endswith("_angle_vel"))]
    for f in angle_fields:
        add_col(f, [scalar_series(r.get(f, None)) for _, r in out.iterrows()])

    return out, metric_fields


def append_row_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    out_root = OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    index_csv = out_root / "behavior_distance_index.csv"
    if index_csv.exists():
        index_csv.unlink()

    for sidx, session in enumerate(SESSIONS):
        animal = session.split("_")[0]
        print(f"[{sidx+1}/{len(SESSIONS)}] {session}")
        loaded = dt.load_sessions([session], prep=True, only_trials=False)
        if len(loaded) == 0:
            print("  - skip: no prep_df returned")
            continue
        prep_df = loaded[0]
        prep_df = pyal.select_trials(prep_df, "trial_name!='trial'")
        prep_df = dt.add_velocity_fields(prep_df)
        prep_bhv, metric_fields = add_behavior_metric_fields(prep_df)
        if not metric_fields:
            print("  - skip: no behavior metric fields built")
            continue

        for ws in WINDOW_SIZES_S:
            out_dir = out_root / animal / session / "behavior" / f"ws{int(ws)}"
            out_dir.mkdir(parents=True, exist_ok=True)

            for metric in metric_fields:
                meta_df, x = build_window_tensor_concat_by_trial_type(
                    prep_bhv,
                    window_size_s=ws,
                    field=metric,
                    trial_type_col="trial_name",
                    bin_size_col="bin_size",
                )
                if x.shape[0] == 0:
                    continue

                labels = meta_df["trial_type"].astype(str).to_numpy() if len(meta_df) else np.array([], dtype=str)
                free0_end, free1_start = free_boundaries_from_labels(labels)

                # x: (nW, T, F). These metrics are expected scalar per timepoint (F=1),
                # but average across F if needed.
                x2 = x.mean(axis=2)
                med = np.nanmedian(x2, axis=1)
                mad = np.nanmedian(np.abs(x2 - med[:, None]), axis=1)

                d_med = pairwise_euclidean_from_vector(med)
                d_mad = pairwise_euclidean_from_vector(mad)

                metric_tag = sanitize_name(metric)
                np.savez_compressed(
                    out_dir / f"behavior_median_{metric_tag}.npz",
                    distance=d_med.astype(np.float32),
                    values=med.astype(np.float32),
                    labels=labels.astype(str),
                    metric=np.array(metric),
                    stat=np.array("median"),
                    window_size_s=np.float32(ws),
                    free0_end=np.int32(free0_end),
                    free1_start=np.int32(free1_start),
                )
                np.savez_compressed(
                    out_dir / f"behavior_mad_{metric_tag}.npz",
                    distance=d_mad.astype(np.float32),
                    values=mad.astype(np.float32),
                    labels=labels.astype(str),
                    metric=np.array(metric),
                    stat=np.array("mad"),
                    window_size_s=np.float32(ws),
                    free0_end=np.int32(free0_end),
                    free1_start=np.int32(free1_start),
                )

                append_row_csv(
                    index_csv,
                    {
                        "session": session,
                        "animal": animal,
                        "window_size_s": ws,
                        "metric": metric,
                        "n_windows": int(len(labels)),
                        "free0_end": int(free0_end),
                        "free1_start": int(free1_start),
                        "median_file": str(out_dir / f"behavior_median_{metric_tag}.npz"),
                        "mad_file": str(out_dir / f"behavior_mad_{metric_tag}.npz"),
                    },
                )

        del prep_bhv
        del prep_df
        del loaded
        gc.collect()

    config = {
        "sessions": SESSIONS,
        "window_sizes_s": list(WINDOW_SIZES_S),
        "metrics": [
            "angles: all *_angle except *_angle_vel",
            "left_forelimb_vel_mean",
            "right_forelimb_vel_mean",
            "left_hindlimb_vel_mean",
            "right_hindlimb_vel_mean",
            "lr_paw_dist",
            "lr_hind_knee_dist",
            "lr_hind_ankle_dist",
            "lr_hind_foot_dist",
        ],
        "distance": "euclidean on per-window scalar summaries",
        "summaries": ["median", "mad"],
    }
    (out_root / "behavior_distance_config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved behavior distance outputs to: {out_root}")


if __name__ == "__main__":
    main()
