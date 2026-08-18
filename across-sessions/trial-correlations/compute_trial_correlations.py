"""Compute trial-by-trial correlation, per keypoint x body-relative (rc/vt/ml) dimension, across
sessions.

Mirrors the trial-correlation analysis in the "## 3. Rotate axes" section of
notebooks/behaviour/kinematics_001.ipynb (`trial_corr_vs_null`, run on `td.bhv_rot` rather than
the raw x/y/z `bhv` used in section 2): for each keypoint's rc/vt/ml component, the mean pairwise
trial-to-trial correlation in the post-perturbation window (POST) is compared against a null built
from pseudo-onsets drawn from the pre-perturbation baseline -- so the null reflects the
correlation ongoing rhythmic gait alone would produce, and a positive margin (observed > null)
means trials are more stereotyped than gait alone predicts.

Pipeline, per session:
    1. common_utils.load_session(session)                    -- load, preprocess, drop unperturbed
    2. dt.add_bhv(td, bhv_fields=["pos_keypoints"])           -- stack keypoints into `bhv`
    3. pyal.restrict_to_interval(idx_sol_on, -200..200 bins)  -- +/-2s around perturbation onset
    4. kin.rotate_bhv_td(td)                                  -- rotate into body frame (rc/vt/ml)
    5. trial_corr_vs_null on every (keypoint, dim) pair's `bhv_rot` column, POST window

Saves one .npz per session to `data/<session>.npz`:
    obs_corr    (n_keypoints, 3)  observed mean pairwise trial correlation, POST window
    null_mean   (n_keypoints, 3)  mean of the null distribution (pseudo-onset baseline)
    null95      (n_keypoints, 3)  95th percentile of the null distribution
    keypoints   (n_keypoints,)    Params.pos_keypoints, for indexing axis 0
    dim_labels  (3,)              ["rc", "vt", "ml"]
    n_trials    int
    session     str

Usage:
    uv run python across-sessions/trial-correlations/compute_trial_correlations.py
    uv run python across-sessions/trial-correlations/compute_trial_correlations.py --sessions M061_2025_03_06_14_00
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
import tools.kinematics as kin  # noqa: E402
from tools.params import Params  # noqa: E402

FS = 100
BASE = (-1.6, -0.1)  # baseline window (s, rel. onset) -- pseudo-onsets are drawn from here
POST = (0.0, 1.5)  # response window (s, rel. onset) -- correlation is computed here
N_SHUFFLE = 200
REL_START, REL_END = -200, 200  # bins around idx_sol_on -> +/-2s epoch

DIM_LABELS = ["rc", "vt", "ml"]
OUT_DIR = Path(__file__).resolve().parent / "data"


def _mean_offdiag(mat):
    iu = np.triu_indices_from(mat, k=1)
    return np.nanmean(mat[iu])


def trial_corr_vs_null(signal, onset, post_sl, n_shuffle, rng):
    """signal: (n_trials, T). Returns the observed mean pairwise trial correlation in post_sl,
    and a null distribution of the same statistic from pseudo-onsets drawn from the
    pre-perturbation period (same window length)."""
    obs_stat = _mean_offdiag(np.corrcoef(signal[:, post_sl]))

    n_post = post_sl.stop - post_sl.start
    lo, hi = 10, onset - n_post - 5
    null_stats = np.full(n_shuffle, np.nan)
    if hi > lo:
        for s in range(n_shuffle):
            o = rng.integers(lo, hi)
            null_stats[s] = _mean_offdiag(np.corrcoef(signal[:, o : o + n_post]))
    return obs_stat, null_stats


def run(session, data_dir=cu.DATA_DIR, std=cu.STD):
    print(f"\n{'='*72}\n### {session}", flush=True)

    td = cu.load_session(session, data_dir=data_dir, std=std, run_pca=False)
    td = dt.add_bhv(td, bhv_fields=["pos_keypoints"])
    td = pyal.restrict_to_interval(
        td, start_point_name="idx_sol_on", rel_start=REL_START, rel_end=REL_END
    )
    if len(td) < 2:
        print(f"  !! only {len(td)} trials left, skipping (need >= 2 for a correlation)")
        return

    onset = int(td.idx_sol_on.values[0])
    post_sl = slice(onset + int(POST[0] * FS), onset + int(POST[1] * FS))

    td, R, yaw = kin.rotate_bhv_td(td, verbose=False)
    X = np.stack(td.bhv_rot.values)

    keypoints = Params.pos_keypoints
    kp_dims = [np.asarray(td[kp].values[0]).shape[1] for kp in keypoints]
    offsets = np.concatenate(([0], np.cumsum(kp_dims)))

    n_kp = len(keypoints)
    obs_corr = np.full((n_kp, len(DIM_LABELS)), np.nan)
    null_mean = np.full((n_kp, len(DIM_LABELS)), np.nan)
    null95 = np.full((n_kp, len(DIM_LABELS)), np.nan)

    rng = np.random.default_rng(0)
    for k in range(n_kp):
        start_, end_ = offsets[k], offsets[k + 1]
        for j in range(min(end_ - start_, len(DIM_LABELS))):
            f = start_ + j
            obs, null = trial_corr_vs_null(X[:, :, f], onset, post_sl, N_SHUFFLE, rng)
            obs_corr[k, j] = obs
            null_mean[k, j] = np.nanmean(null)
            null95[k, j] = np.nanpercentile(null, 95)

    print(
        f"  {len(td)} trials -- mean obs corr {np.nanmean(obs_corr):.3f}, "
        f"mean null corr {np.nanmean(null_mean):.3f}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{session}.npz"
    np.savez_compressed(
        out,
        obs_corr=obs_corr,
        null_mean=null_mean,
        null95=null95,
        keypoints=np.array(keypoints),
        dim_labels=np.array(DIM_LABELS),
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
