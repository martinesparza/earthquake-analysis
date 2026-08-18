"""Decode solenoid direction (0-11) from rotated (rc/vt/ml) keypoint kinematics, across sessions.

Two decoders, both cross-validated:
  1. FIXED post-perturbation window (POST, LDA, 5-fold accuracy) -- the single-number summary.
  2. MOVING window over the full loaded epoch (tools.decoding.moving_window_decoding, GaussianNB,
     5-fold accuracy) -- the time course, so the reader can see when direction becomes decodable
     relative to perturbation onset (t=0).

Pipeline (mirrors the "Decode solenoid ID" section of notebooks/behaviour/kinematics_001.ipynb,
but run across `common_utils.ALL_SESSIONS` and saved instead of only plotted):
    1. common_utils.load_session(session)                    -- load, preprocess, drop unperturbed
    2. dt.add_bhv(td, bhv_fields=["pos_keypoints"])           -- stack keypoints into `bhv`
    3. pyal.restrict_to_interval(idx_sol_on, -200..200 bins)  -- +/-2s around perturbation onset
    4. kin.rotate_bhv_td(td)                                  -- rotate into body frame (rc/vt/ml)
    5. decode `values_Sol_direction` from `bhv_rot`

Saves one .npz per session to `data/<session>.npz`:
    fixed_scores   (5,)         LDA accuracy per CV fold, POST window only
    moving_scores  (n_win, 5)   GaussianNB accuracy per window x fold, full epoch
    moving_time_s  (n_win,)     window centre time (s), relative to perturbation onset
    n_trials       int          trials entering the decoder (post unperturbed/stopped drop)
    n_classes      int          number of distinct solenoid directions present
    chance         float        1 / n_classes
    post_window_s  (2,)         POST window bounds (s, rel. onset), for reference when plotting

Usage:
    uv run python across-sessions/bhv-decode-solenoid/decode_solenoid_direction.py
    uv run python across-sessions/bhv-decode-solenoid/decode_solenoid_direction.py --sessions M061_2025_03_06_14_00
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pyaldata as pyal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score

sys.path.append(str(Path(__file__).resolve().parents[1]))  # across-sessions/, for common_utils
sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root, for tools
import common_utils as cu  # noqa: E402
import tools.dataTools as dt  # noqa: E402
import tools.decoding as decode  # noqa: E402
import tools.kinematics as kin  # noqa: E402

FS = 100  # bhv stays at 10 ms bins regardless of Params.BIN_SIZE
REL_START, REL_END = -200, 200  # bins around idx_sol_on -> +/-2s epoch
POST = (0.0, 1.5)  # fixed decoding window (s, rel. onset)
CV_FOLDS = 5

OUT_DIR = Path(__file__).resolve().parent / "data"


def run(session, data_dir=cu.DATA_DIR, std=cu.STD):
    print(f"\n{'='*72}\n### {session}", flush=True)

    td = cu.load_session(session, data_dir=data_dir, std=std, run_pca=False)
    td = dt.add_bhv(td, bhv_fields=["pos_keypoints"])
    td = pyal.restrict_to_interval(
        td, start_point_name="idx_sol_on", rel_start=REL_START, rel_end=REL_END
    )
    if len(td) < CV_FOLDS:
        print(f"  !! only {len(td)} trials left, skipping (need >= {CV_FOLDS} for CV)")
        return

    onset = int(td.idx_sol_on.values[0])
    post_sl = slice(onset + int(POST[0] * FS), onset + int(POST[1] * FS))

    td, R, yaw = kin.rotate_bhv_td(td, verbose=False)
    X_ = np.stack(td.bhv_rot.values[:])
    y = td.values_Sol_direction.values.tolist()
    n_classes = len(set(y))
    print(f"  {len(td)} trials, {n_classes} solenoid directions, X_ shape {X_.shape}")

    # 1. Fixed post-perturbation window, LDA
    X_post = X_[:, post_sl, :]
    n_trials, n_time, n_features = X_post.shape
    fixed_scores = cross_val_score(
        LinearDiscriminantAnalysis(),
        X=X_post.reshape(n_trials, -1),
        y=y,
        cv=KFold(CV_FOLDS),
        scoring="accuracy",
    )
    print(f"  fixed window {POST} s: accuracy {fixed_scores.mean():.3f} +- {fixed_scores.std():.3f}")

    # 2. Moving window over the full epoch, GaussianNB (repo defaults: 0.5s window, 0.05s step)
    moving_scores, moving_time = decode.moving_window_decoding(X_, y, cv=CV_FOLDS, bin_size=1 / FS)
    onset_offset_s = -REL_START / FS  # seconds from window start to perturbation onset
    moving_time_s = moving_time - onset_offset_s
    print(
        f"  moving window: {moving_scores.shape[0]} windows, "
        f"peak accuracy {moving_scores.mean(axis=1).max():.3f}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{session}.npz"
    np.savez_compressed(
        out,
        fixed_scores=fixed_scores,
        moving_scores=moving_scores,
        moving_time_s=moving_time_s,
        n_trials=n_trials,
        n_classes=n_classes,
        chance=1.0 / n_classes,
        post_window_s=np.array(POST),
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
