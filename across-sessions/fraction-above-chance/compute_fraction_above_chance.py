"""Compute the fraction of kinematic features with a detectable perturbation response, per
session.

Mirrors the "## 1. Prove sessions have responses" analysis in
notebooks/behaviour/kinematics_001.ipynb (`pta_and_null`, run on the RAW x/y/z `bhv` -- this is
the pre-rotation analysis, unlike trial-correlations/ which uses body-relative rc/vt/ml): for
each of the n_keypoints * 3 (x/y/z) features, the peak |baseline-subtracted trial-average| in the
post-perturbation window is compared to the 95th percentile of a null built from pseudo-onsets
drawn from the pre-perturbation baseline. `ratio = peak / null95`; a feature counts as "above
chance" when ratio > 1. This script saves that per-feature ratio and the resulting fraction above
chance for every session, so the fraction can be compared across sessions.

Pipeline, per session:
    1. common_utils.load_session(session)                    -- load, preprocess, drop unperturbed
    2. dt.add_bhv(td, bhv_fields=["pos_keypoints"])           -- stack keypoints into `bhv` (raw
       camera x/y/z, NOT rotated -- rotation happens later in the notebook's section 3)
    3. pyal.restrict_to_interval(idx_sol_on, -200..200 bins)  -- +/-2s around perturbation onset
    4. pta_and_null(X, onset, base_sl, post_sl)               -- per-feature peak-triggered average
       and its pseudo-onset null distribution
    5. ratio = peak / null95; fraction_above_chance = mean(ratio > 1)

Saves one .npz per session to `data/<session>.npz`:
    ratio                 (n_features,)  peak / null95 per feature
    fraction_above_chance scalar         mean(ratio > 1)
    n_above               int            count(ratio > 1)
    n_features            int            total feature count (n_keypoints * 3)
    feature_labels        (n_features,)  "<keypoint>_<x|y|z>"
    n_trials              int
    session                str

Usage:
    uv run python across-sessions/fraction-above-chance/compute_fraction_above_chance.py
    uv run python across-sessions/fraction-above-chance/compute_fraction_above_chance.py --sessions M061_2025_03_06_14_00
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pyaldata as pyal

sys.path.append(str(Path(__file__).resolve().parents[1]))  # across-sessions/, for common_utils
sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root, for tools
import common_utils as cu  # noqa: E402
import tools.dataTools as dt  # noqa: E402
from tools.params import Params  # noqa: E402

FS = 100
BASE = (-1.6, -0.1)  # baseline window (s, rel. onset) -- pseudo-onsets are drawn from here
POST = (0.0, 1.5)  # response window (s, rel. onset) -- peak is measured here
N_SHUFFLE = 200
REL_START, REL_END = -200, 200  # bins around idx_sol_on -> +/-2s epoch

OUT_DIR = Path(__file__).resolve().parent / "data"


def pta_and_null(X, onset, base_sl, post_sl, n_shuffle, rng):
    """X: (n_trials, T, F). Returns PTA (n_post, F) and null peak distribution (n_shuffle, F)."""
    base = np.nanmean(X[:, base_sl, :], axis=1, keepdims=True)  # per-trial baseline
    Xc = X - base
    pta = np.nanmean(Xc[:, post_sl, :], axis=0)  # (n_post, F)

    n_post = post_sl.stop - post_sl.start
    lo, hi = 10, onset - n_post - 5
    null = np.full((n_shuffle, X.shape[2]), np.nan)
    if hi > lo:
        for s in range(n_shuffle):
            o = rng.integers(lo, hi)
            b = np.nanmean(X[:, max(0, o - 60) : max(1, o - 10), :], axis=1, keepdims=True)
            seg = X[:, o : o + n_post, :] - b
            null[s] = np.nanmax(np.abs(np.nanmean(seg, axis=0)), axis=0)
    return pta, null


def run(session, data_dir=cu.DATA_DIR, std=cu.STD):
    print(f"\n{'='*72}\n### {session}", flush=True)

    td = cu.load_session(session, data_dir=data_dir, std=std, run_pca=False)
    td = dt.add_bhv(td, bhv_fields=["pos_keypoints"])
    td = pyal.restrict_to_interval(
        td, start_point_name="idx_sol_on", rel_start=REL_START, rel_end=REL_END
    )
    if len(td) < 2:
        print(f"  !! only {len(td)} trials left, skipping")
        return

    onset = int(td.idx_sol_on.values[0])
    base_sl = slice(onset + int(BASE[0] * FS), onset + int(BASE[1] * FS))
    post_sl = slice(onset + int(POST[0] * FS), onset + int(POST[1] * FS))

    X = np.stack(td.bhv.values[:])
    rng = np.random.default_rng(0)
    pta, null = pta_and_null(X, onset, base_sl, post_sl, N_SHUFFLE, rng)

    peak = np.nanmax(np.abs(pta), axis=0)
    null95 = np.nanpercentile(null, 95, axis=0)
    ratio = peak / null95

    kp_dims = [np.asarray(td[kp].values[0]).shape[1] for kp in Params.pos_keypoints]
    dim_labels = ["x", "y", "z"]
    feature_labels = np.array(
        [
            f"{kp}_{dim_labels[j] if j < len(dim_labels) else f'dim{j}'}"
            for kp, d in zip(Params.pos_keypoints, kp_dims)
            for j in range(d)
        ]
    )

    n_above = int(np.nansum(ratio > 1))
    n_features = int(np.sum(~np.isnan(ratio)))
    frac_above = n_above / n_features if n_features else np.nan
    print(f"  {len(td)} trials -- {n_above}/{n_features} features above chance ({frac_above:.1%})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{session}.npz"
    np.savez_compressed(
        out,
        ratio=ratio,
        fraction_above_chance=frac_above,
        n_above=n_above,
        n_features=n_features,
        feature_labels=feature_labels,
        n_trials=len(td),
        session=session,
    )
    print(f"  -> {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", default=cu.ALL_SESSIONS, metavar="SESSION")
    args = parser.parse_args()

    for session in args.sessions:
        try:
            run(session)
        except Exception as e:
            print(f"  !! FAILED {session}: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
