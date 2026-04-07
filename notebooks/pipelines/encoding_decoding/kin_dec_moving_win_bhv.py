from __future__ import annotations

import argparse
import pickle
import sys

sys.path.append("../../")

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold, cross_validate

import tools.dataTools as dt
import tools.dsp as dsp

# ---------------------------------------------------------------------------
# All available sessions and areas
# ---------------------------------------------------------------------------
ALL_SESSIONS = [
    # "M061_2025_03_04_10_00",
    "M061_2025_03_05_14_00",
    "M061_2025_03_06_14_00",
    # "M063_2025_03_13_14_00",
    # "M063_2025_03_14_15_30",
    "M062_2025_03_20_14_00",
    "M062_2025_03_21_14_00",
    "M078_2025_08_06_15_00",
    # "M086_2025_12_10_15_00",
    "M103_2026_02_18_15_30",
    # "M103_2026_02_19_15_30",
    "M106_2026_02_25_15_00",
    # "M106_2026_02_26_16_00",
]

ALL_AREAS = ["MOp", "SSp", "CP", "VAL"]

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Moving-window bhv-PCA decoding — neural PCs → top K bhv PCs"
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
    "--window",
    type=float,
    default=0.2,
    metavar="S",
    help="Decode window width in seconds (default: 0.20)",
)
parser.add_argument(
    "--step",
    type=float,
    default=0.02,
    metavar="S",
    help="Step size between window centres in seconds (default: 0.02)",
)
parser.add_argument(
    "--time-min",
    type=float,
    default=-1.0,
    metavar="S",
    help="Start of analysis epoch relative to perturbation onset (default: -1.0)",
)
parser.add_argument(
    "--time-max",
    type=float,
    default=1.0,
    metavar="S",
    help="End of analysis epoch relative to perturbation onset (default: 1.0)",
)
parser.add_argument(
    "--n-bhv-pcs",
    type=int,
    default=15,
    metavar="K",
    help="Number of bhv PCs to decode (default: 15)",
)
parser.add_argument(
    "--results-dir",
    default="/data/equake_results/encoding_decoding_v4/",
    metavar="DIR",
    help="Output directory for pickle files",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------
SESSIONS = args.sessions
AREAS = args.areas
WINDOW_S = args.window
STEP_S = args.step
N_BHV_PCS = args.n_bhv_pcs
RESULTS_DIR = args.results_dir

BIN_SIZE = 0.01  # s per bin (10 ms) — fixed
REL_START = -200
PERTURB_BIN = -REL_START

_window_bins = int(round(WINDOW_S / BIN_SIZE))
_step_bins = int(round(STEP_S / BIN_SIZE))

TIME_BINS_S = np.arange(args.time_min, args.time_max, STEP_S)
_time_bins = (np.round(TIME_BINS_S / BIN_SIZE) + PERTURB_BIN).astype(int)

neural_pc_dict = {"MOp": 25, "CP": 30, "SSp": 15, "VAL": 50}

EXCLUDE_AREAS = {
    "M062_2025_03_20_14_00": ["VAL"],
    "M062_2025_03_21_14_00": ["VAL"],
}

print(f"Sessions   : {SESSIONS}")
print(f"Areas      : {AREAS}")
print(f"Neural PCs : {neural_pc_dict}")
print(f"Bhv PCs    : {N_BHV_PCS}")
print(f"Window     : {WINDOW_S} s  ({_window_bins} bins)")
print(f"Step       : {STEP_S} s  ({_step_bins} bins)")
print(f"Epoch      : {TIME_BINS_S[0]:.2f} → {TIME_BINS_S[-1] + WINDOW_S:.2f} s")
print(f"Results dir: {RESULTS_DIR}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
vw_r2_scorer = make_scorer(r2_score, multioutput="variance_weighted")

cv = KFold(n_splits=5, shuffle=False)


def opt_ridge_alpha(X, y):
    alphas = np.logspace(-4, 3, 100)
    return RidgeCV(alphas=alphas).fit(X, y).alpha_


def score_window(neural_arr, bhv_arr, t, model):
    """Cross-validated variance-weighted R² for one time window.

    neural_arr : (n_trials, n_time, n_neural_pcs)
    bhv_arr    : (n_trials, n_time, n_bhv_pcs)
    t          : window start bin (absolute index into loaded array)
    """
    X = neural_arr[:, t : t + _window_bins, :].reshape(-1, neural_arr.shape[-1])
    y = bhv_arr[:, t : t + _window_bins, :].reshape(-1, bhv_arr.shape[-1])
    return cross_validate(model, X, y, cv=cv, scoring=vw_r2_scorer, n_jobs=1)["test_score"]


# ---------------------------------------------------------------------------
# Load sessions
# ---------------------------------------------------------------------------
all_session_processed = dsp.load_sessions_for_trial_analyses(
    SESSIONS, use_sem_dropping=True, rates=True, std=0.03, std_dropping=True
)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for session, val in all_session_processed.items():
    if val is None:
        print(f"\nSkipping {session} (failed to load)")
        continue

    perturb_td = val["td"]

    # ── Build bhv field (xyz + angles for all keypoints) and fit bhv PCA ──
    perturb_td = dt.add_bhv(perturb_td, bhv_fields=["all"])
    perturb_td = dt.add_pca_field(perturb_td, "bhv", N_BHV_PCS)
    n_bhv_dims = perturb_td["bhv_pca"].iloc[0].shape[-1]
    print(f"\n{session} | bhv dims: {perturb_td['bhv'].iloc[0].shape[-1]} → {n_bhv_dims} PCs")

    perturb_td_shuff = perturb_td.sample(frac=1, random_state=42).reset_index(drop=False)
    bhv_arr = np.stack(perturb_td_shuff["bhv_pca"].values)  # (n_trials, n_time, n_bhv_pcs)

    for area in AREAS:
        if area in EXCLUDE_AREAS.get(session, []):
            print(f"  Skipping {area} — excluded for {session}")
            continue

        pca_col = f"{area}_rates_pca"
        if pca_col not in perturb_td_shuff.columns:
            print(f"  Skipping {area} — {pca_col} not found")
            continue

        n_pcs = min(neural_pc_dict[area], perturb_td_shuff[pca_col].iloc[0].shape[-1])
        neural_arr = np.stack(perturb_td_shuff[pca_col].values)[:, :, :n_pcs]
        print(f"  {area} | {n_pcs} neural PCs | {len(perturb_td_shuff)} trials")

        alpha = opt_ridge_alpha(
            neural_arr.reshape(-1, neural_arr.shape[-1]),
            bhv_arr.reshape(-1, bhv_arr.shape[-1]),
        )
        model = Ridge(alpha=alpha)
        print(f"  {area} | alpha = {alpha:.4f}")

        results = {}
        for t_s, t in zip(TIME_BINS_S, _time_bins):
            scores = score_window(neural_arr, bhv_arr, t, model)
            t_centre = t_s + WINDOW_S / 2
            results[t_centre] = scores  # (n_folds,)

        out_path = (
            f"{RESULTS_DIR}moving_win_bhv_{area}_{session}"
            f"_npcs{n_pcs}_nbhv{n_bhv_dims}.pkl"
        )
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "_meta": {
                        "window_s": WINDOW_S,
                        "step_s": STEP_S,
                        "time_bins_s": TIME_BINS_S,
                        "bin_size": BIN_SIZE,
                        "n_neural_pcs": n_pcs,
                        "n_bhv_pcs": n_bhv_dims,
                        "area": area,
                        "session": session,
                    },
                    "scores": results,  # {t_centre: (n_folds,)}
                },
                f,
            )
        print(f"  Saved → {out_path}")
