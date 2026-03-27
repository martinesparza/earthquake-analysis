from __future__ import annotations

import argparse
import pickle
import sys

sys.path.append("../../")

import numpy as np
from joblib import Parallel, delayed
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
    # "M061_2025_03_05_14_00",
    # "M061_2025_03_06_14_00",
    # "M063_2025_03_13_14_00",
    # "M063_2025_03_14_15_30",
    "M062_2025_03_20_14_00",
    "M062_2025_03_21_14_00",
    "M078_2025_08_06_15_00",
    "M086_2025_12_10_15_00",
    # "M103_2026_02_18_15_30",
    # "M103_2026_02_19_15_30",
    # "M106_2026_02_25_15_00",
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
    description="Lagged kinematics decoding — all keypoints, all areas"
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
    "--n-pcs",
    type=int,
    default=None,
    metavar="N",
    help="Number of PCs to use per area (default: all available)",
)
parser.add_argument(
    "--window",
    type=float,
    default=0.20,
    metavar="S",
    help="Decode window width in seconds (default: 0.20)",
)
parser.add_argument(
    "--lag-min",
    type=float,
    default=-0.15,
    metavar="S",
    help="Minimum lag in seconds (default: -0.25)",
)
parser.add_argument(
    "--lag-max",
    type=float,
    default=0.16,
    metavar="S",
    help="Maximum lag in seconds (default: 0.26)",
)
parser.add_argument(
    "--results-dir",
    default="/data/equake_results/lagged_kin_dec_all_kp/",
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
N_PCS = args.n_pcs  # None → use all available PCs
WINDOW_S = args.window
RESULTS_DIR = args.results_dir

BIN_SIZE = 0.01  # s per bin (10 ms) — not exposed, always fixed
REL_START = -200
LAGS_S = np.arange(args.lag_min, args.lag_max, 0.01)
TIME_BINS_S = np.arange(-1.0, 1.1, 0.03)

print(f"Sessions   : {sessions}")
print(f"Areas      : {AREAS}")
print(f"Keypoints  : {KEYPOINTS}")
print(f"N PCs      : {'all available' if N_PCS is None else N_PCS}")
print(f"Window     : {WINDOW_S} s")
print(f"Lags       : {LAGS_S[0]:.2f} → {LAGS_S[-1]:.2f} s")
print(f"Results dir: {RESULTS_DIR}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
vw_r2_scorer = make_scorer(r2_score, multioutput="variance_weighted")


def opt_ridge_alpha(x_cv, y_cv):
    alphas = np.logspace(-4, 6, 100)
    ridge_cv = RidgeCV(alphas=alphas)
    ridge_cv.fit(x_cv, y_cv)
    return ridge_cv.alpha_


def score_one_lag(arr_, t, lag, y, model, cv, window_bins):
    arr = dt.shift_time_no_wrap(arr_, lag)
    arr = arr[:, t : t + window_bins, :]
    X = arr.reshape(-1, arr.shape[-1])
    cv_out = cross_validate(model, X, y, cv=cv, scoring=vw_r2_scorer, n_jobs=1)
    return cv_out["test_score"]  # (n_folds,)


# ---------------------------------------------------------------------------
# Load sessions
# ---------------------------------------------------------------------------
all_session_processed = dsp.load_sessions_for_trial_analyses(sessions, use_sem_dropping=True)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
_perturb_bin = -REL_START
_window_bins = int(round(WINDOW_S / BIN_SIZE))
_lags_bins = np.round(LAGS_S / BIN_SIZE).astype(int)
_time_bins = (np.round(TIME_BINS_S / BIN_SIZE) + _perturb_bin).astype(int)

cv = KFold(n_splits=5, shuffle=False)

for session, val in all_session_processed.items():
    if val is None:
        print(f"Skipping {session} (failed to load)")
        continue

    perturb_td = val["td"]
    perturb_td_shuff = perturb_td.sample(frac=1, random_state=42).reset_index(drop=False)

    for area in AREAS:
        pca_col = f"{area}_rates_pca"
        if pca_col not in perturb_td_shuff.columns:
            print(f"  Skipping {area} — {pca_col} not found")
            continue

        n_pcs = perturb_td_shuff[pca_col].iloc[0].shape[-1]
        if N_PCS is not None:
            n_pcs = min(N_PCS, n_pcs)

        neural_data = np.stack(perturb_td_shuff[pca_col].values)[:, :, :n_pcs]
        print(f"\n{session} | {area} | {n_pcs} PCs")

        results_session = {}

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
            for t_s, t in zip(TIME_BINS_S, _time_bins):
                y = power_[:, t : t + _window_bins, :].reshape(-1, power_.shape[-1])
                per_lag = Parallel(n_jobs=-1, prefer="processes")(
                    delayed(score_one_lag)(neural_data, t, lag, y, model, cv, _window_bins)
                    for lag in _lags_bins
                )
                results_kp[t_s + WINDOW_S] = np.stack(per_lag, axis=0)  # (n_lags, n_folds)

            results_session[keypoint] = results_kp

        out_path = f"{RESULTS_DIR}lagged_dec_{area}_{session}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "_meta": {
                        "lags_s": LAGS_S,
                        "bin_size": BIN_SIZE,
                        "window_s": WINDOW_S,
                        "time_step_s": TIME_BINS_S[1] - TIME_BINS_S[0],
                        "n_pcs": n_pcs,
                        "area": area,
                        "session": session,
                    },
                    **results_session,
                },
                f,
            )
        print(f"Saved → {out_path}")
