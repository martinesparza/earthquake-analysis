import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = ROOT / "notebooks" / "figures" / "windows" / "exports_dist_gmm"
OUT_CSV = EXPORT_ROOT / "rsa_results.csv"
OUT_CFG = EXPORT_ROOT / "rsa_results_config.json"

AREAS = ("MOp", "MOs", "SSp", "CP", "VAL")
N_PCS_OPTIONS = (5, 10, 30)
WINDOW_SIZES_S = (10.0, 30.0, 60.0)


def upper_tri_values(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 2:
        return np.array([], dtype=float)
    idx = np.triu_indices(m.shape[0], k=1)
    return m[idx]


def labels_to_window_keys(labels: Sequence[str]) -> List[Tuple[str, int]]:
    """
    Create stable keys even with repeated labels:
      ["free0","free0","intertrial"] -> [("free0",0),("free0",1),("intertrial",0)]
    """
    counts: Dict[str, int] = {}
    keys: List[Tuple[str, int]] = []
    for lab in labels:
        s = str(lab)
        c = counts.get(s, 0)
        keys.append((s, c))
        counts[s] = c + 1
    return keys


def align_matrices_by_labels(
    mat_a: np.ndarray,
    labels_a: Sequence[str],
    mat_b: np.ndarray,
    labels_b: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, int]:
    keys_a = labels_to_window_keys(labels_a)
    keys_b = labels_to_window_keys(labels_b)

    map_b = {k: i for i, k in enumerate(keys_b)}
    idx_a, idx_b = [], []
    for ia, k in enumerate(keys_a):
        ib = map_b.get(k, None)
        if ib is not None:
            idx_a.append(ia)
            idx_b.append(ib)

    if len(idx_a) == 0:
        return np.empty((0, 0)), np.empty((0, 0)), 0

    a_aligned = np.asarray(mat_a)[np.ix_(idx_a, idx_a)]
    b_aligned = np.asarray(mat_b)[np.ix_(idx_b, idx_b)]
    return a_aligned, b_aligned, len(idx_a)


def rsa_pearson(mat_a: np.ndarray, mat_b: np.ndarray) -> Tuple[float, float, int]:
    va = upper_tri_values(mat_a)
    vb = upper_tri_values(mat_b)
    mask = np.isfinite(va) & np.isfinite(vb)
    va = va[mask]
    vb = vb[mask]
    if len(va) < 3:
        return np.nan, np.nan, int(len(va))
    rho, p = pearsonr(va, vb)
    return float(rho), float(p), int(len(va))


def to_dissimilarity(mat: np.ndarray, matrix_type: str) -> np.ndarray:
    """
    Convert matrix to dissimilarity space for RSA.
    - distance -> unchanged
    - cosine similarity -> 1 - cosine
    """
    m = np.asarray(mat, dtype=float)
    if matrix_type == "cosine":
        return 1.0 - m
    return m


def load_npz(path: Path, matrix_key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    if matrix_key not in z or "labels" not in z:
        return None
    mat = np.asarray(z[matrix_key], dtype=float)
    labels = np.asarray(z["labels"]).astype(str)
    return mat, labels


def load_neural_matrix(
    session: str,
    area: str,
    n_pcs: int,
    window_size_s: float,
    matrix_type: str,  # "distance" | "cosine"
) -> Optional[Tuple[np.ndarray, np.ndarray, Path]]:
    animal = session.split("_")[0]
    ws_tag = f"ws{int(window_size_s)}"
    npc_tag = f"npc{int(n_pcs)}"

    # Prefer new folder structure with npc level.
    base_new = EXPORT_ROOT / animal / session / area / npc_tag / ws_tag
    # Legacy fallback (older runs without npc subfolder).
    base_old = EXPORT_ROOT / animal / session / area / ws_tag

    if matrix_type == "distance":
        fname = "distance_by_trial_type.npz"
        key = "distance"
    elif matrix_type == "cosine":
        fname = "principal_cosine_by_trial_type.npz"
        key = "cosine_similarity"
    else:
        raise ValueError(f"Unknown matrix_type: {matrix_type}")

    for base in (base_new, base_old):
        p = base / fname
        loaded = load_npz(p, key)
        if loaded is not None:
            mat, labels = loaded
            return mat, labels, p
    return None


def load_behavior_index() -> pd.DataFrame:
    idx_csv = EXPORT_ROOT / "behavior_distance_index.csv"
    if not idx_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(idx_csv)
    for col in ("median_file", "mad_file"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def load_behavior_matrix_from_row(row: pd.Series, stat: str) -> Optional[Tuple[np.ndarray, np.ndarray, Path]]:
    file_col = "median_file" if stat == "median" else "mad_file"
    if file_col not in row or pd.isna(row[file_col]):
        return None
    p = Path(str(row[file_col]))
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.exists():
        return None
    loaded = load_npz(p, "distance")
    if loaded is None:
        return None
    mat, labels = loaded
    return mat, labels, p


def append_row_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    behavior_index = load_behavior_index()

    # Derive session list from folders already exported.
    sessions = sorted({p.name for p in EXPORT_ROOT.glob("M*/M*_*_*_*_*") if p.is_dir()})
    if not sessions:
        print(f"No exported sessions found under {EXPORT_ROOT}")
        return

    for session in sessions:
        animal = session.split("_")[0]
        print(f"Processing RSA: {session}")
        for ws in WINDOW_SIZES_S:
            # 1) Area-area RSA for each neural matrix type separately.
            for n_pcs in N_PCS_OPTIONS:
                for matrix_type in ("distance", "cosine"):
                    loaded_area: Dict[str, Tuple[np.ndarray, np.ndarray, Path]] = {}
                    for area in AREAS:
                        got = load_neural_matrix(
                            session=session,
                            area=area,
                            n_pcs=n_pcs,
                            window_size_s=ws,
                            matrix_type=matrix_type,
                        )
                        if got is not None:
                            loaded_area[area] = got

                    for a1, a2 in combinations(sorted(loaded_area.keys()), 2):
                        m1, l1, p1 = loaded_area[a1]
                        m2, l2, p2 = loaded_area[a2]
                        ma, mb, n_common = align_matrices_by_labels(m1, l1, m2, l2)
                        ma_d = to_dissimilarity(ma, matrix_type)
                        mb_d = to_dissimilarity(mb, matrix_type)
                        rho, pval, n_pairs = rsa_pearson(ma_d, mb_d)
                        append_row_csv(
                            OUT_CSV,
                            {
                                "session": session,
                                "animal": animal,
                                "window_size_s": ws,
                                "n_pcs": n_pcs,
                                "rsa_kind": "area_area",
                                "matrix_type_a": matrix_type,
                                "matrix_type_b": matrix_type,
                                "matrix_transform_a": "1-cosine" if matrix_type == "cosine" else "identity",
                                "matrix_transform_b": "1-cosine" if matrix_type == "cosine" else "identity",
                                "entity_a": a1,
                                "entity_b": a2,
                                "behavior_stat": "",
                                "behavior_metric": "",
                                "n_common_windows": n_common,
                                "n_pairs": n_pairs,
                                "rho_pearson": rho,
                                "p_value": pval,
                                "path_a": str(p1),
                                "path_b": str(p2),
                            },
                        )

            # 2) Area-behavior RSA.
            if ("session" not in behavior_index.columns) or ("window_size_s" not in behavior_index.columns):
                sub_b = pd.DataFrame()
            else:
                sub_b = behavior_index[
                    (behavior_index["session"] == session)
                    & (behavior_index["window_size_s"].astype(float) == float(ws))
                ].copy()
            if len(sub_b) == 0:
                continue

            for n_pcs in N_PCS_OPTIONS:
                for area in AREAS:
                    for neural_type in ("distance", "cosine"):
                        nload = load_neural_matrix(
                            session=session,
                            area=area,
                            n_pcs=n_pcs,
                            window_size_s=ws,
                            matrix_type=neural_type,
                        )
                        if nload is None:
                            continue
                        nm, nl, npath = nload

                        for _, brow in sub_b.iterrows():
                            metric = str(brow.get("metric", ""))
                            for stat in ("median", "mad"):
                                bload = load_behavior_matrix_from_row(brow, stat=stat)
                                if bload is None:
                                    continue
                                bm, bl, bpath = bload
                                ma, mb, n_common = align_matrices_by_labels(nm, nl, bm, bl)
                                ma_d = to_dissimilarity(ma, neural_type)
                                # behavior matrices are already distances
                                mb_d = to_dissimilarity(mb, "distance")
                                rho, pval, n_pairs = rsa_pearson(ma_d, mb_d)
                                append_row_csv(
                                    OUT_CSV,
                                    {
                                        "session": session,
                                        "animal": animal,
                                        "window_size_s": ws,
                                        "n_pcs": n_pcs,
                                        "rsa_kind": "area_behavior",
                                        "matrix_type_a": neural_type,
                                        "matrix_type_b": "behavior_distance",
                                        "matrix_transform_a": "1-cosine" if neural_type == "cosine" else "identity",
                                        "matrix_transform_b": "identity",
                                        "entity_a": area,
                                        "entity_b": "behavior",
                                        "behavior_stat": stat,
                                        "behavior_metric": metric,
                                        "n_common_windows": n_common,
                                        "n_pairs": n_pairs,
                                        "rho_pearson": rho,
                                        "p_value": pval,
                                        "path_a": str(npath),
                                        "path_b": str(bpath),
                                    },
                                )

    cfg = {
        "export_root": str(EXPORT_ROOT),
        "areas": list(AREAS),
        "n_pcs_options": list(N_PCS_OPTIONS),
        "window_sizes_s": list(WINDOW_SIZES_S),
        "rsa": "pearson over upper-triangle aligned matrix entries",
        "rsa_dissimilarity_transform": {
            "distance": "identity",
            "cosine": "1-cosine",
            "behavior_distance": "identity",
        },
        "neural_matrix_types": ["distance_by_trial_type", "principal_cosine_by_trial_type"],
        "behavior_stats": ["median", "mad"],
    }
    OUT_CFG.write_text(json.dumps(cfg, indent=2))
    print(f"Saved RSA rows to: {OUT_CSV}")


if __name__ == "__main__":
    main()
