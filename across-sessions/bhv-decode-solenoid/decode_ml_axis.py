"""Decode 90 deg vs 270 deg -- the pure left/right push -- from each body-frame dimension.

WHY THIS TEST. `decode_solenoid_per_dim.py` found `ml` best for `contra_ipsi` and `angle`, but
those targets mix several stimulus axes together, so "ml wins" could be an average over
heterogeneous contrasts rather than a statement about a specific one. 90 deg and 270 deg are
antipodal on a single axis: the only thing separating them is which side the animal was pushed
from. If the mediolateral channel is what carries push direction, this is where it should win
outright -- and if it does not win here, the earlier result was not really about laterality.

Upper and lower are POOLED (both solenoid codes contributing to each angle are kept), so `level`
cannot leak into the discrimination:
    90 deg  <- codes 4, 10
    270 deg <- codes 7, 11

MATCHED PRE-ONSET CONTROL. The same decode is run on an equal-length window ending at onset
(bins 50:200 = -1.5..0 s), which contains no perturbation. A dimension that decodes the push
direction just as well BEFORE the solenoid fires is reporting posture or ongoing state, not the
response -- a trap this project has hit before. Only the POST-minus-PRE excess is evidence.

Reuses the shared interim cache and the exact trial selection of `decode_solenoid_per_dim.py`
(imported from it, so the two cannot drift apart).

Saves one .npz per session to `data/mlaxis_<session>.npz`:
    <window>_<dim>_scores   (n_folds,)  CV accuracy per fold, window in {post, pre}
    n_trials, n_90, n_270   int         trial counts entering the decode
    chance                  float       0.5 (balanced 2-class by construction of the contrast)
    session                 str

Usage:
    uv run python across-sessions/bhv-decode-solenoid/decode_ml_axis.py
    uv run python across-sessions/bhv-decode-solenoid/decode_ml_axis.py --sessions M061_2025_03_06_14_00
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.append(str(Path(__file__).resolve().parents[1]))  # across-sessions/, for common_utils
sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root, for tools
sys.path.append(str(Path(__file__).resolve().parent))  # this folder, for the per-dim script
import common_utils as cu  # noqa: E402
from decode_solenoid_per_dim import (  # noqa: E402
    CV_FOLDS, DIMS, FS, OUT_DIR, POST, dim_subsets, load_from_cache, load_from_raw,
)
from tools.params import Params  # noqa: E402

PAIR = (90.0, 270.0)  # the antipodal left/right axis
PRE = (-1.5, 0.0)  # matched-length control window, ends exactly at onset


def run(session, data_dir=cu.DATA_DIR, std=cu.STD, use_cache=True):
    print(f"\n{'=' * 72}\n### {session}", flush=True)

    cached = (Params.ACROSS_SESSION_RESULTS_DIR / "cache" / f"{session}.npz").exists()
    if use_cache and cached:
        X_all, sol_dir, onset, _ = load_from_cache(session)
    else:
        X_all, sol_dir, onset, _ = load_from_raw(session, data_dir, std)

    # map code -> angle, then keep only the two antipodal angles (upper AND lower codes for each)
    angle = np.array([Params.sol_dir_to_angle[int(s)] for s in sol_dir], float)
    keep = np.isin(angle, PAIR)
    y = angle[keep]
    n_90, n_270 = int((y == PAIR[0]).sum()), int((y == PAIR[1]).sum())
    codes_used = sorted(set(int(s) for s in np.asarray(sol_dir)[keep]))
    print(f"  {keep.sum()} trials on the {PAIR[0]:.0f}/{PAIR[1]:.0f} axis "
          f"(codes {codes_used}): {n_90} vs {n_270}")

    if min(n_90, n_270) < CV_FOLDS:
        print(f"  !! smallest class {min(n_90, n_270)} < {CV_FOLDS} -- skipping session")
        return

    windows = {
        "post": slice(onset + int(POST[0] * FS), onset + int(POST[1] * FS)),
        "pre": slice(onset + int(PRE[0] * FS), onset + int(PRE[1] * FS)),
    }
    out = dict(session=session, n_trials=int(keep.sum()), n_90=n_90, n_270=n_270, chance=0.5,
               codes_used=np.array(codes_used))
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=0)

    for wname, wsl in windows.items():
        dims = dim_subsets(X_all[keep][:, wsl, :])
        print(f"  [{wname}] window {wsl.start}:{wsl.stop}")
        for dname, Xd in dims.items():
            scores = cross_val_score(
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                X=Xd.reshape(len(Xd), -1), y=y, cv=cv, scoring="accuracy",
            )
            out[f"{wname}_{dname}_scores"] = scores
            print(f"    {dname:<4s}: acc {scores.mean():.3f} +- {scores.std():.3f}", flush=True)

    best = max(DIMS, key=lambda d: out[f"post_{d}_scores"].mean())
    excess = {d: out[f"post_{d}_scores"].mean() - out[f"pre_{d}_scores"].mean() for d in DIMS}
    best_excess = max(DIMS, key=lambda d: excess[d])
    print(f"  best POST dim: {best};  best POST-minus-PRE excess: {best_excess} "
          f"({ {d: round(v, 3) for d, v in excess.items()} })")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"mlaxis_{session}.npz"
    np.savez_compressed(out_path, **out)
    print(f"  -> {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", default=cu.ALL_SESSIONS, metavar="SESSION")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    for session in args.sessions:
        try:
            run(session, use_cache=not args.no_cache)
        except Exception as e:
            print(f"  !! FAILED {session}: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
