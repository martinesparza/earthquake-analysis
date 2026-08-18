"""Compute per-trial running direction (yaw) across sessions.

Mirrors the "3. Rotate axes" section of notebooks/behaviour/kinematics_001.ipynb: fits the
per-trial body frame (tools.kinematics.body_frame.build_body_frame_td), which also yields `yaw`
-- the animal's heading (rc axis) in the horizontal camera plane, in degrees -- and validates the
frame is anatomically sane (validate_body_frame_td) before trusting it.

Pipeline, per session:
    1. common_utils.load_session(session)                    -- load, preprocess, drop unperturbed
    2. pyal.restrict_to_interval(idx_sol_on, -200..200 bins)  -- +/-2s around perturbation onset,
       so idx_sol_on is well-defined and the baseline window (-1.5, -0.1s) sits inside every trial
    3. kin.build_body_frame_td(td)                            -- fit R, yaw per trial
    4. kin.validate_body_frame_td(td, R, yaw)                 -- raise if the frame fails QC
       (left/right separates on ml, rc dominates stride amplitude, limbs sit below hip on vt)

`yaw` is circular (wraps at +-180 deg): saved as raw per-trial degrees so any pooling (across
trials, across a session's trials, across an animal's sessions) uses circular statistics at
aggregation time rather than baking in one choice here.

Saves one .npz per session to `data/<session>.npz`:
    yaw_deg        (n_trials,)  per-trial heading, degrees
    circ_mean_deg  scalar       circular mean heading (Rayleigh formula)
    circ_std_deg   scalar       circular std
    resultant_len  scalar       0-1, concentration of headings (1 = all trials face the same way)
    session        str
    animal         str          session[:4]
    n_trials       int

Usage:
    uv run python across-sessions/running-direction/compute_running_direction.py
    uv run python across-sessions/running-direction/compute_running_direction.py --sessions M061_2025_03_06_14_00
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
import tools.kinematics as kin  # noqa: E402

REL_START, REL_END = -200, 200  # bins around idx_sol_on -> +/-2s epoch

OUT_DIR = Path(__file__).resolve().parent / "data"


def circular_stats(yaw_deg):
    yaw_rad = np.radians(yaw_deg)
    resultant = np.nanmean(np.exp(1j * yaw_rad))
    R_len = np.abs(resultant)
    circ_mean = np.angle(resultant)
    circ_std = np.sqrt(-2 * np.log(R_len)) if R_len > 0 else np.nan
    return np.degrees(circ_mean), np.degrees(circ_std), R_len


def run(session, data_dir=cu.DATA_DIR, std=cu.STD):
    print(f"\n{'='*72}\n### {session}", flush=True)

    td = cu.load_session(session, data_dir=data_dir, std=std, run_pca=False)
    td = pyal.restrict_to_interval(
        td, start_point_name="idx_sol_on", rel_start=REL_START, rel_end=REL_END
    )
    if len(td) == 0:
        print("  !! no trials left, skipping")
        return

    R, yaw = kin.build_body_frame_td(td)
    kin.validate_body_frame_td(td, R, yaw, verbose=True)

    circ_mean, circ_std, R_len = circular_stats(yaw)
    print(
        f"  {len(td)} trials -- yaw circular mean {circ_mean:+.1f} +- {circ_std:.1f} deg "
        f"(R={R_len:.3f})"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{session}.npz"
    np.savez_compressed(
        out,
        yaw_deg=yaw,
        circ_mean_deg=circ_mean,
        circ_std_deg=circ_std,
        resultant_len=R_len,
        session=session,
        animal=session[:4],
        n_trials=len(td),
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
