from __future__ import annotations

import argparse
import pickle
import sys

sys.path.append("../../")

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold, cross_validate
from tqdm import tqdm

import pyaldata as pyal
import tools.dataTools as dt
import tools.dsp as dsp

# ---------------------------------------------------------------------------
# All available sessions, keypoints, and areas
# ---------------------------------------------------------------------------
ALL_SESSIONS = [
    # "M061_2025_03_04_10_00",
    "M061_2025_03_05_14_00",
    "M061_2025_03_06_14_00",
    "M063_2025_03_13_14_00",
    "M063_2025_03_14_15_30",
    "M062_2025_03_20_14_00",
    "M062_2025_03_21_14_00",
    "M078_2025_08_06_15_00",
    "M086_2025_12_10_15_00",
    "M103_2026_02_18_15_30",
    # "M103_2026_02_19_15_30",
    "M106_2026_02_25_15_00",
    # "M106_2026_02_26_16_00",
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

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Pre/post perturbation kinematics decoding — no lags, all keypoints, all areas"
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
    help="Pre-perturbation window start in seconds (default: -1.5)",
)
parser.add_argument(
    "--pre-end",
    type=float,
    default=-0.5,
    metavar="S",
    help="Pre-perturbation window end in seconds (default: -0.5)",
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
    "--results-dir",
    default="/data/equake_results/encoding_decoding_v3/",
    metavar="DIR",
    help="Output directory for pickle files",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Pipeline parameters (resolved from CLI)
# ---------------------------------------------------------------------------
sessions = args.sessions
AREAS = args.areas
KEYPOINTS = args.keypoints
RESULTS_DIR = args.results_dir

BIN_SIZE = 0.01  # s per bin (10 ms) — fixed
REL_START = -200  # bins before perturbation onset in loaded data

# Window definitions in seconds relative to perturbation onset
WINDOWS_S = {
    "pre": (args.pre_start, args.pre_end),
    "post": (args.post_start, args.post_end),
}

# Convert to bin slices (offset by REL_START so bin 0 = perturbation onset)
_perturb_bin = -REL_START
WINDOWS_BINS = {
    name: (
        int(round(t0 / BIN_SIZE)) + _perturb_bin,
        int(round(t1 / BIN_SIZE)) + _perturb_bin,
    )
    for name, (t0, t1) in WINDOWS_S.items()
}

pc_dict = {"MOp": 30, "CP": 30, "SSp": 20, "VAL": 50}

# Per-session area exclusions
EXCLUDE_AREAS = {
    "M062_2025_03_20_14_00": ["VAL"],
    "M062_2025_03_21_14_00": ["VAL"],
    "M086_2025_12_10_15_00": ["VAL"],
}

print(f"Sessions   : {sessions}")
print(f"Areas      : {AREAS}")
print(f"Keypoints  : {KEYPOINTS}")
print(f"N PCs      : {pc_dict}")
print(f"Windows (s): {WINDOWS_S}")
print(f"Results dir: {RESULTS_DIR}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
vw_r2_scorer = make_scorer(r2_score, multioutput="variance_weighted")


def opt_ridge_alpha(x_cv, y_cv):
    alphas = np.logspace(-4, 2, 100)
    ridge_cv = RidgeCV(alphas=alphas)
    ridge_cv.fit(x_cv, y_cv)
    return ridge_cv.alpha_


def score_window(neural_arr, kin_arr, t_start, t_end, model, cv):
    """
    Cross-validated R² for decoding kin_arr[:, t_start:t_end, :] from
    neural_arr[:, t_start:t_end, :].  Both arrays are (trials, time, features).
    Time bins are concatenated across the window before fitting.
    """
    X = neural_arr[:, t_start:t_end, :].reshape(-1, neural_arr.shape[-1])
    y = kin_arr[:, t_start:t_end, :].reshape(-1, kin_arr.shape[-1])
    cv_out = cross_validate(model, X, y, cv=cv, scoring=vw_r2_scorer, n_jobs=1)
    return cv_out["test_score"]  # (n_folds,)


# ---------------------------------------------------------------------------
# Load sessions
# ---------------------------------------------------------------------------
all_session_processed = dsp.load_sessions_for_trial_analyses(
    sessions, use_sem_dropping=True, rates=True, std=0.03
)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
cv = KFold(n_splits=5, shuffle=False)

for session, val in all_session_processed.items():
    if val is None:
        print(f"Skipping {session} (failed to load)")
        continue

    perturb_td = val["td"]
    perturb_td_shuff = perturb_td.sample(frac=1, random_state=42).reset_index(drop=False)

    for area in AREAS:
        if area in EXCLUDE_AREAS.get(session, []):
            print(f"  Skipping {area} — excluded for {session}")
            continue

        pca_col = f"{area}_rates_pca"
        if pca_col not in perturb_td_shuff.columns:
            print(f"  Skipping {area} — {pca_col} not found")
            continue

        n_pcs = perturb_td_shuff[pca_col].values[0].shape[-1]
        neural_data = np.stack(perturb_td_shuff[pca_col].values)
        print(f"\n{session} | {area} | {n_pcs} PCs")

        results_area = {}  # keypoint → {window_name → (n_folds,)}

        for keypoint in tqdm(KEYPOINTS, desc=f"{area} | keypoints"):
            if keypoint not in perturb_td_shuff.columns:
                print(f"  Skipping {keypoint} — not found in dataframe")
                continue

            power_ = np.stack(perturb_td_shuff[keypoint].values)
            alpha = opt_ridge_alpha(
                neural_data.reshape(-1, neural_data.shape[-1]),
                power_.reshape(-1, power_.shape[-1]),
            )
            print(f"  {keypoint}: alpha = {alpha:.2f}")
            model = Ridge(alpha=alpha)
            results_kp = {}
            for win_name, (t0, t1) in WINDOWS_BINS.items():
                scores = score_window(neural_data, power_, t0, t1, model, cv)
                results_kp[win_name] = scores  # (n_folds,)

            results_area[keypoint] = results_kp

        out_path = f"{RESULTS_DIR}kin_dec_pre_post_{area}_{session}_npcs{n_pcs}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "_meta": {
                        "windows_s": WINDOWS_S,
                        "windows_bins": WINDOWS_BINS,
                        "bin_size": BIN_SIZE,
                        "n_pcs": n_pcs,
                        "area": area,
                        "session": session,
                        "keypoints": list(results_area.keys()),
                    },
                    **results_area,
                },
                f,
            )
        print(f"Saved → {out_path}")
