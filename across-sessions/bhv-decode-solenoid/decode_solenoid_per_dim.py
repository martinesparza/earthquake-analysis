"""Decode every solenoid label encoding from each body-frame kinematic dimension separately,
across sessions -- extends `decode_solenoid_direction.py` (which only ever decoded the raw 12-way
`values_Sol_direction` from all 57 features at once).

Four label encodings, all derived from the same raw `values_Sol_direction` (0-11) via the repo's
canonical maps in `tools.params.Params` (see the "Data model" section of the repo CLAUDE.md):
    contra_ipsi   2-way   Params.sol_dir_to_contra_ipsi
    level         2-way   Params.sol_dir_to_level      ("upper vs lower")
    angle         6-way   Params.sol_dir_to_angle       (the axis/direction after collapsing level)
    direction    12-way   values_Sol_direction itself, unmapped ("everything")

Three body-frame dimensions, decoded SEPARATELY (mirrors the "Decode solenoid ID" per-dimension
loop in notebooks/behaviour/kinematics_001.ipynb, generalised to all four label encodings). No
"all features together" option here -- see the note in `dim_subsets` for why (it hung for >2h on
the 6-class target before being cut); `decode_solenoid_direction.py` already covers "all features,
12-way direction" separately.
    rc / vt / ml   19 features each (one keypoint's rc, vt, or ml column)

Pipeline, per session (matches decode_solenoid_direction.py through step 4):
    1. common_utils.load_session(session)                    -- load, preprocess, drop unperturbed
    2. dt.add_bhv(td, bhv_fields=["pos_keypoints"])           -- stack keypoints into `bhv`
    3. pyal.restrict_to_interval(idx_sol_on, -200..200 bins)  -- +/-2s around perturbation onset
    4. kin.rotate_bhv_td(td)                                  -- rotate into body frame (rc/vt/ml)
    5. slice the fixed POST window (0.0-1.5s), flatten time within each of the 3 dim subsets
    6. decode each of the 4 label encodings from each of the 3 dim subsets (12 decodes/session)

BY DEFAULT steps 1-4 are NOT recomputed: they are read from the shared interim cache at
`Params.ACROSS_SESSION_RESULTS_DIR/cache/<session>.npz`, which already stores the epoched,
body-frame-rotated tensor. That turns a ~10 min/session raw load into a fraction of a second. The
one thing the cache lacks is the unperturbed/stopped-trial drop, which `load_from_cache` reproduces
exactly from the cached per-keypoint `ep_perturb_score` (verified identical trial counts to the raw
path -- see that function's docstring). Pass `--no-cache` to force the raw pipeline instead.

LDA regularisation and CV, fixed relative to the notebook cell this generalises: a single dim
subset flattened over the POST window is 19 features x 150 timepoints = 2850 features against a
few hundred trials -- badly underdetermined for default LDA (solver='svd', no shrinkage). Uses
`solver='lsqr', shrinkage='auto'` (Ledoit-Wolf) throughout, and `StratifiedKFold(shuffle=True)`
rather than plain `KFold` -- important here specifically because the 12-way `direction` target has
few trials per class, so an unstratified, unshuffled split risks a class being absent from a
training fold entirely.

Saves one .npz per session to `data/perdim_<session>.npz` (kept separate from
decode_solenoid_direction.py's `data/<session>.npz` -- different analysis, same folder):
    <target>_<dim>_scores   (n_folds,)  CV accuracy per fold, or absent if skipped (see below)
    <target>_<dim>_chance   float       1 / n_classes actually present for that target/session
    <target>_n_classes      int         classes present in this session (can be < the max, e.g. a
                                         session missing one solenoid direction entirely)
    <target>_min_class_n    int         smallest per-class trial count (diagnostic: low values are
                                         why a target/dim combo may have been skipped)
    n_trials                int
    session                 str
    targets, dims            the label names / dim names actually attempted, for the loader

A (target, dim) combo is skipped, with a warning printed and no `_scores` key written, if the
smallest class has fewer trials than CV_FOLDS -- StratifiedKFold cannot form that many folds
otherwise, and this happens most often for `direction` (12-way) in the smaller sessions.

Usage:
    uv run python across-sessions/bhv-decode-solenoid/decode_solenoid_per_dim.py
    uv run python across-sessions/bhv-decode-solenoid/decode_solenoid_per_dim.py --sessions M061_2025_03_06_14_00
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pyaldata as pyal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.append(str(Path(__file__).resolve().parents[1]))  # across-sessions/, for common_utils
sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root, for tools
import common_utils as cu  # noqa: E402
import tools.dataTools as dt  # noqa: E402
import tools.kinematics as kin  # noqa: E402
from tools.params import Params  # noqa: E402

FS = 100  # bhv stays at 10 ms bins regardless of Params.BIN_SIZE
REL_START, REL_END = -200, 200  # bins around idx_sol_on -> +/-2s epoch
POST = (0.0, 1.5)  # fixed decoding window (s, rel. onset), matches decode_solenoid_direction.py
CV_FOLDS = 5
DIMS = ["rc", "vt", "ml"]  # column order add_bhv/rotate_bhv_td lay features out in

OUT_DIR = Params.ACROSS_SESSION_RESULTS_DIR / "bhv-decode-solenoid" / "data"
CACHE_DIR = Params.ACROSS_SESSION_RESULTS_DIR / "cache"


def load_from_cache(session):
    """Load one session's epoched, body-frame-rotated kinematics from the shared interim cache.

    The cache (`Params.ACROSS_SESSION_RESULTS_DIR/cache/<session>.npz`) already holds exactly the
    product of steps 1-4 of the raw pipeline -- `ep_X` is (n_trials, 401, 57) rotated into the body
    frame, epoched -200..+200 bins around onset, with `labels` in `<keypoint>_<rc|vt|ml>` order --
    so using it skips the ~10 min per-session raw `.mat` load and preprocess entirely.

    ONE THING IT DOES NOT DO: the cache is written BEFORE
    `dsp.drop_unperturbed_or_stopped_trials`, so it holds every perturbed trial, not the subset the
    rest of this folder's analyses use. That drop is reproduced here from the cached
    `ep_perturb_score` / `ep_perturb_score_mean` using the identical criterion (keep a trial only
    when mean + sem < 0 AND perturb_score_mean > -2, `stat="sem"`, `thresh_val=-2`, matching
    `dsp.drop_unperturbed_or_stopped_trials`'s defaults as called by `common_utils.load_session`).
    Verified to reproduce the raw pipeline's trial counts exactly -- e.g. M061_2025_03_06_14_00
    244 -> 208 and M061_2025_03_04_10_00 243 -> 203, both identical to the raw-path run.
    """
    d = np.load(CACHE_DIR / f"{session}.npz", allow_pickle=True)
    ps, psm = d["ep_perturb_score"], d["ep_perturb_score_mean"]
    keep = ((ps.mean(1) + ps.std(1) / np.sqrt(ps.shape[1])) < 0) & (psm > -2)
    print(f"  cache: drop_unperturbed_or_stopped reproduced -> kept {keep.sum()}/{len(keep)} "
          f"({100 * keep.sum() / len(keep):.1f}%)")
    return (
        np.asarray(d["ep_X"], float)[keep],
        np.asarray(d["ep_code"], float)[keep],
        int(d["ep_onset"]),
        np.asarray(d["labels"]),
    )


def load_from_raw(session, data_dir, std):
    """Original path: load the raw session and redo steps 1-4. Slow (~10 min/session)."""
    td = cu.load_session(session, data_dir=data_dir, std=std, run_pca=False)
    td = dt.add_bhv(td, bhv_fields=["pos_keypoints"])
    td = pyal.restrict_to_interval(
        td, start_point_name="idx_sol_on", rel_start=REL_START, rel_end=REL_END
    )
    td, _, _ = kin.rotate_bhv_td(td, verbose=False)
    labels = np.array([f"{kp}_{DIMS[j]}" for kp in Params.pos_keypoints for j in range(3)])
    return (
        np.stack(td.bhv_rot.values[:]),
        td.values_Sol_direction.values,
        int(td.idx_sol_on.values[0]),
        labels,
    )


def label_maps(sol_dir):
    """The four label encodings, all derived from the same raw 0-11 code."""
    sol_dir = np.asarray(sol_dir, int)
    return {
        "contra_ipsi": np.array([Params.sol_dir_to_contra_ipsi[int(s)] for s in sol_dir]),
        "level": np.array([Params.sol_dir_to_level[int(s)] for s in sol_dir]),
        "angle": np.array([Params.sol_dir_to_angle[int(s)] for s in sol_dir]),
        "direction": sol_dir,  # "everything" -- the raw, unmapped 12-way code
    }


def dim_subsets(X):
    """X: (n_trials, T, 57) -> {'rc': (n,T,19), 'vt': ..., 'ml': ..., 'all': (n,T,57)}.

    Features cycle rc/vt/ml every 3 columns (one triplet per keypoint, `Params.pos_keypoints`
    order) -- `X[:, :, i::3]` for i in 0,1,2 is exactly "every keypoint's rc/vt/ml column".

    Deliberately does NOT include an "all" (57-feature) option: sklearn's LDA with
    `shrinkage='auto'` estimates the Ledoit-Wolf shrinkage separately PER CLASS before pooling, so
    its cost scales with n_classes x n_features**2. At 8550 flattened features (57 x 150) that is
    fine for a 2-class target (contra_ipsi, level both finished in minutes) but explodes for the
    6-class `angle` target -- observed >2h with no result on a single session before being killed.
    `decode_solenoid_direction.py` already covers "all features, 12-way direction" separately.
    """
    return {DIMS[i]: X[:, :, i::3] for i in range(3)}


def run(session, data_dir=cu.DATA_DIR, std=cu.STD, use_cache=True):
    print(f"\n{'=' * 72}\n### {session}", flush=True)

    cached = (CACHE_DIR / f"{session}.npz").exists()
    if use_cache and cached:
        X_all, sol_dir, onset, feat_labels = load_from_cache(session)
    else:
        if use_cache and not cached:
            print(f"  no cache entry, falling back to the raw pipeline (slow)")
        X_all, sol_dir, onset, feat_labels = load_from_raw(session, data_dir, std)

    if len(X_all) < CV_FOLDS:
        print(f"  !! only {len(X_all)} trials left, skipping (need >= {CV_FOLDS} for CV)")
        return

    post_sl = slice(onset + int(POST[0] * FS), onset + int(POST[1] * FS))
    X_post = X_all[:, post_sl, :]
    n_trials = X_post.shape[0]
    print(f"  {n_trials} trials, X_post shape {X_post.shape}", flush=True)

    labels = label_maps(sol_dir)
    dims = dim_subsets(X_post)

    out = dict(n_trials=n_trials, session=session, targets=np.array(list(labels)),
               dims=np.array(list(dims)), feat_labels=feat_labels,
               from_cache=bool(use_cache and cached))
    for tname, y in labels.items():
        classes, counts = np.unique(y, return_counts=True)
        n_classes, min_n = len(classes), int(counts.min())
        out[f"{tname}_n_classes"] = n_classes
        out[f"{tname}_min_class_n"] = min_n
        print(f"\n  [{tname}] {n_classes} classes present, smallest = {min_n} trials")
        if min_n < CV_FOLDS:
            print(f"    !! smallest class has {min_n} < {CV_FOLDS} trials -- skipping all dims "
                  f"for this target")
            continue

        cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=0)
        for dname, Xd in dims.items():
            Xf = Xd.reshape(n_trials, -1)
            scores = cross_val_score(
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                X=Xf, y=y, cv=cv, scoring="accuracy",
            )
            out[f"{tname}_{dname}_scores"] = scores
            out[f"{tname}_{dname}_chance"] = 1.0 / n_classes
            print(f"    {dname:<4s} ({Xf.shape[1]:>4d} feat): "
                  f"acc {scores.mean():.3f} +- {scores.std():.3f}  "
                  f"(chance {1.0 / n_classes:.3f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"perdim_{session}.npz"
    np.savez_compressed(out_path, **out)
    print(f"\n  -> {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", default=cu.ALL_SESSIONS, metavar="SESSION")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="ignore the shared interim cache and reload/preprocess each session from raw "
             "(~10 min/session instead of seconds; use to regenerate if the cache is stale)",
    )
    args = parser.parse_args()

    for session in args.sessions:
        try:
            run(session, use_cache=not args.no_cache)
        except Exception as e:
            print(f"  !! FAILED {session}: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
