"""Shared session-loading pipeline for across-session analyses.

Every analysis under `across-sessions/` should load sessions through `load_session` below
instead of re-deriving the preprocessing pipeline, so they can't silently drift apart. Wraps
the repo's canonical functions (`tools/dsp/preprocessing.py`) rather than reimplementing them:

    td = load_session(session)

which runs, in order:
    1. dsp.load_and_preprocess_trials_from_sess(session, data_dir=DATA_DIR, std=STD,
       run_pca=run_pca)  -- load, preprocess, drop unsteady running trials, compute
       perturb_score. run_pca=False by default since kinematics-only analyses don't need
       `<area>_rates_pca`; pass run_pca=True for analyses that also decode/encode from neural
       PCs.
    2. dsp.drop_unperturbed_or_stopped_trials(td)  -- drop trials without a genuine
       perturbation response (repo defaults: thresh_val=-2, stat="sem").

Returns the full trial-level DataFrame (all trial types, minus dropped trials); callers still
need their own `pyal.restrict_to_interval` / `dt.add_bhv` / etc. for analysis-specific shaping.

Usage:
    sys.path.append(str(Path(__file__).resolve().parents[1]))  # from a script one level down
    import common_utils as cu
    td = cu.load_session("M061_2025_03_06_14_00")
"""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # repo root, for `tools`
import tools.dsp as dsp  # noqa: E402

DATA_DIR = "/mnt/mouse-data-london/proc/"
STD = 0.03

# Session list -- must match the "active" rows of the Sessions table in .claude/CLAUDE.md,
# which is the source of truth. Commented-out entries are "excluded" there; the muscimol sessions
# (M103_2026_02_20_16_00, M106_2026_02_27_16_00) are deliberately not listed and are only used for
# muscimol comparisons. Change the table first, then this list.
ALL_SESSIONS = [
    "M061_2025_03_04_10_00",
    "M061_2025_03_05_14_00",
    "M061_2025_03_06_14_00",
    # "M063_2025_03_13_14_00",
    # "M063_2025_03_14_15_30",
    "M062_2025_03_20_14_00",
    "M062_2025_03_21_14_00",
    "M078_2025_08_06_15_00",
    "M086_2025_12_10_15_00",
    "M103_2026_02_18_15_30",
    # "M103_2026_02_19_15_30",  # less trials
    "M106_2026_02_25_15_00",
    # "M106_2026_02_26_16_00",  # less trials
]


def load_session(session, data_dir=DATA_DIR, std=STD, run_pca=False):
    """Load, preprocess, and drop unperturbed/stopped trials for one session.

    Returns
    -------
    td : pd.DataFrame  trial table (all trial types minus dropped trials), with
         `perturb_score` / `perturb_score_mean` columns and, if run_pca=True,
         `<area>_rates_pca` columns.
    """
    td = dsp.load_and_preprocess_trials_from_sess(
        session, data_dir=data_dir, std=std, run_pca=run_pca
    )
    td = dsp.drop_unperturbed_or_stopped_trials(td)
    return td
