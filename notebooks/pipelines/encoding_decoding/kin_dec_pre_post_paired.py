from __future__ import annotations

import argparse
import pickle
import sys

sys.path.append("../../")

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

import tools.dsp as dsp

# ---------------------------------------------------------------------------
# All available sessions, keypoints, and areas
# ---------------------------------------------------------------------------
ALL_SESSIONS = [
    # "M061_2025_03_05_14_00",
    "M061_2025_03_06_14_00",
    # "M063_2025_03_13_14_00",
    # "M062_2025_03_20_14_00",
    "M062_2025_03_21_14_00",
    "M078_2025_08_06_15_00",
    "M103_2026_02_18_15_30",
    "M106_2026_02_25_15_00",
]

ALL_KEYPOINTS = [
    "hip_center",
    "left_ankle",
    "left_elbow",
    "left_foot",
    "left_knee",
    "left_paw",
    "left_shoulder",
    "left_wrist",
    "right_ankle",
    "right_elbow",
    "right_foot",
    "right_knee",
    "right_paw",
    "right_shoulder",
    "right_wrist",
    "shoulder_center",
    "tail_base",
    "tail_middle",
    "tail_tip",
]

ALL_AREAS = ["MOp", "SSp", "CP", "VAL"]

# Default number of PCs per area. None = use all available.
DEFAULT_PC_DICT = {
    "MOp": 30,
    "SSp": 20,
    "CP": 30,
    "VAL": 50,
}

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Pre/post perturbation kinematics decoding — paired alpha, all keypoints, all areas"
)
parser.add_argument(
    "--sessions",
    nargs="+",
    default=ALL_SESSIONS,
    metavar="SESSION",
    help="Session ID(s) to process (default: all sessions)",
)
parser.add_argument(
    "--areas",
    nargs="+",
    default=ALL_AREAS,
    metavar="AREA",
    help="Brain area(s) to decode from (default: all areas)",
)
parser.add_argument(
    "--keypoints",
    nargs="+",
    default=ALL_KEYPOINTS,
    metavar="KP",
    help="Keypoint(s) to decode (default: all keypoints)",
)
parser.add_argument(
    "--pre-start",
    type=float,
    default=-1.5,
    metavar="S",
    help="Pre-perturbation window start in seconds (default: -1.0)",
)
parser.add_argument(
    "--pre-end",
    type=float,
    default=-0.5,
    metavar="S",
    help="Pre-perturbation window end in seconds (default: 0.0)",
)
parser.add_argument(
    "--post-start",
    type=float,
    default=0.0,
    metavar="S",
    help="Post-perturbation window start in seconds (default: 0.0)",
)
parser.add_argument(
    "--post-end",
    type=float,
    default=1.0,
    metavar="S",
    help="Post-perturbation window end in seconds (default: 1.0)",
)
parser.add_argument(
    "--pc-dict",
    nargs="+",
    default=None,
    metavar="AREA:N",
    help=(
        "Per-area PC counts as AREA:N pairs, e.g. --pc-dict MOp:25 CP:30 SSp:20 VAL:15. "
        "Areas omitted use the value from DEFAULT_PC_DICT; set N=0 to use all available."
    ),
)
parser.add_argument(
    "--n-splits",
    type=int,
    default=5,
    metavar="K",
    help="Number of GroupKFold CV splits (default: 5)",
)
parser.add_argument(
    "--results-dir",
    default="/data/equake_results/encoding_decoding_v2/",
    metavar="DIR",
    help="Output directory for pickle files",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Pipeline parameters (resolved from CLI)
# ---------------------------------------------------------------------------
SESSIONS = args.sessions
AREAS = args.areas
KEYPOINTS = args.keypoints
RESULTS_DIR = args.results_dir
N_SPLITS = args.n_splits

# Build PC dict: start from defaults, then apply any CLI overrides
PC_DICT = dict(DEFAULT_PC_DICT)
if args.pc_dict is not None:
    for item in args.pc_dict:
        area, n = item.split(":")
        PC_DICT[area] = int(n) if int(n) > 0 else None  # 0 → use all

BIN_SIZE = 0.01  # s per bin (10 ms) — fixed
REL_START = -200  # bins before perturbation onset in loaded data
PERTURB_BIN = -REL_START  # bin index of perturbation onset = 200

# Window definitions in seconds → bin slices
WINDOWS_S = {
    "pre": (args.pre_start, args.pre_end),
    "post": (args.post_start, args.post_end),
}
WINDOWS_BINS = {
    name: (
        int(round(t0 / BIN_SIZE)) + PERTURB_BIN,
        int(round(t1 / BIN_SIZE)) + PERTURB_BIN,
    )
    for name, (t0, t1) in WINDOWS_S.items()
}

RIDGE_ALPHAS = np.logspace(-4, 2, 100)

print(f"Sessions   : {SESSIONS}")
print(f"Areas      : {AREAS}")
print(f"Keypoints  : {KEYPOINTS}")
print(f"PC dict    : {PC_DICT}")
print(f"Windows (s): {WINDOWS_S}")
print(f"Windows (bins): {WINDOWS_BINS}")
print(f"N splits   : {N_SPLITS}")
print(f"Results dir: {RESULTS_DIR}")

# ---------------------------------------------------------------------------
# Core decoding function
# ---------------------------------------------------------------------------


def decode_windows_paired(neural_arr, kin_arr, windows_bins, n_splits):
    """Decode all windows with a shared alpha selected per fold.

    For each CV fold:
      1. Fit RidgeCV on combined training data from all windows → shared alpha.
      2. For each window: fit Ridge(alpha) on that window's training data,
         score on that window's test data.

    This ensures identical model capacity for pre and post → fair comparison.
    GroupKFold on trial indices prevents temporal leakage across trial boundaries.

    Parameters
    ----------
    neural_arr   : (n_trials, n_time, n_pcs)
    kin_arr      : (n_trials, n_time, n_xyz)
    windows_bins : dict {name: (t0, t1)}
    n_splits     : int

    Returns
    -------
    scores : dict {name: np.ndarray (n_splits,)}
    """
    n_trials = neural_arr.shape[0]
    trial_idx = np.arange(n_trials)
    cv = GroupKFold(n_splits=n_splits)
    fold_scores = {name: [] for name in windows_bins}

    for train_idx, test_idx in cv.split(trial_idx, groups=trial_idx):
        X_train_parts, y_train_parts = [], []
        window_arrays = {}

        for name, (t0, t1) in windows_bins.items():
            Xtr = neural_arr[train_idx][:, t0:t1, :].reshape(-1, neural_arr.shape[-1])
            ytr = kin_arr[train_idx][:, t0:t1, :].reshape(-1, kin_arr.shape[-1])
            Xte = neural_arr[test_idx][:, t0:t1, :].reshape(-1, neural_arr.shape[-1])
            yte = kin_arr[test_idx][:, t0:t1, :].reshape(-1, kin_arr.shape[-1])
            X_train_parts.append(Xtr)
            y_train_parts.append(ytr)
            window_arrays[name] = (Xtr, ytr, Xte, yte)

        # Shared alpha from combined training data across all windows
        rcv = RidgeCV(alphas=RIDGE_ALPHAS)
        rcv.fit(np.vstack(X_train_parts), np.vstack(y_train_parts))
        alpha = rcv.alpha_

        # Score each window independently with the shared alpha
        for name, (Xtr, ytr, Xte, yte) in window_arrays.items():
            model = Ridge(alpha=alpha)
            model.fit(Xtr, ytr)
            score = r2_score(yte, model.predict(Xte), multioutput="variance_weighted")
            fold_scores[name].append(score)

    return {name: np.array(scores) for name, scores in fold_scores.items()}


# ---------------------------------------------------------------------------
# Load sessions
# ---------------------------------------------------------------------------
all_session_processed = dsp.load_sessions_for_trial_analyses(
    SESSIONS, use_sem_dropping=True, rates=True, std_dropping=False, std=0.03
)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for session, val in all_session_processed.items():
    if val is None:
        print(f"\nSkipping {session} (failed to load)")
        continue

    perturb_td = val["td"]
    td_shuff = perturb_td.sample(frac=1, random_state=42).reset_index(drop=False)

    for area in AREAS:
        pca_col = f"{area}_rates_pca"
        if pca_col not in td_shuff.columns:
            print(f"  Skipping {area} — {pca_col} not found")
            continue

        neural_arr = np.stack(td_shuff[pca_col].values)  # (n_trials, n_time, n_pcs)
        n_available = neural_arr.shape[-1]
        n_pcs_area = PC_DICT.get(area, None)
        n_use = min(n_pcs_area, n_available) if n_pcs_area is not None else n_available
        neural_arr = neural_arr[:, :, :n_use]
        print(f"\n{session} | {area} | {n_use}/{n_available} PCs | {len(td_shuff)} trials")

        results_area = {}  # keypoint → {window_name → (n_splits,)}

        for kp in tqdm(KEYPOINTS, desc=f"{area} | keypoints"):
            if kp not in td_shuff.columns:
                print(f"  Skipping {kp} — not found in dataframe")
                continue

            kin_arr = np.stack(td_shuff[kp].values)  # (n_trials, n_time, 3)
            if kin_arr.ndim == 2:
                kin_arr = kin_arr[:, :, np.newaxis]

            if np.any(np.isnan(kin_arr)):
                print(f"  Skipping {kp} — NaNs found")
                continue

            scores = decode_windows_paired(neural_arr, kin_arr, WINDOWS_BINS, N_SPLITS)
            results_area[kp] = scores

            pre_m = scores["pre"].mean()
            post_m = scores["post"].mean()
            print(
                f"  {kp:20s}  pre={pre_m:.3f}  post={post_m:.3f}  delta={post_m - pre_m:+.3f}"
            )

        out_path = f"{RESULTS_DIR}kin_dec_paired_{area}_{session}_npcs{n_use}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "_meta": {
                        "windows_s": WINDOWS_S,
                        "windows_bins": WINDOWS_BINS,
                        "bin_size": BIN_SIZE,
                        "n_pcs": n_use,
                        "n_splits": N_SPLITS,
                        "area": area,
                        "session": session,
                        "keypoints": list(results_area.keys()),
                    },
                    **results_area,
                },
                f,
            )
        print(f"Saved → {out_path}")
