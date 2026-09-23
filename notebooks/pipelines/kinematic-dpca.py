"""Label-shuffle noise floor for rebalanced dPCA of the kinematic response, across sessions.

Per marginalization (`lt`, `at`, `lat`, ...), tests whether the variance dPCA assigns to it
exceeds what unstructured trial noise produces at this N and per-condition K. The reference is
an empirical floor: permute the condition labels across trials (killing every condition effect
while keeping each trial's kinematics and the per-condition trial counts), refit the whole
rebalanced-dPCA pipeline, re-score. The refit-on-shuffle null already carries both the
finite-trial averaging noise and the reduced-rank in-sample fitting bias, so no analytic
correction is layered on; `signal_frac` (Kobak 2016 Fig 3-6d) is reported as the effect-size
point estimate only.

Per session: load <session>.npz from equake-interim-results/cache/ (baseline z-score `ep_X`,
drop trials by `ep_perturb_score`, slice the RESPONSE window -- as in
notebooks/behaviour/response-anatomy.ipynb) -> fit -> n_shuffles permutations refit in
parallel -> p = (1 + #{null >= observed}) / (n_shuffles + 1), z = (obs - null.mean) / null.std.
With regularizer='auto' the plain-ridge CV lambda search runs once on the real data and the
null reuses it (never re-tuned per shuffle).

Outputs to OUT_DIR / <run-tag> / (run-tag = "<labels>_r<reg>_k<ncomp>", or --tag):
    <session>.npz         observed, null (n_shuffles x n_marg), null_median/p95, p, z,
                          signal_frac, keys, n_trials, cell_counts, regularizer_used
    <session>_fit.joblib  {dpca, X_psth, trialX, codes, cond_values, config}
    summary.csv           per-marginalization aggregate. Headline: per-session z pooled
                          with animal as the unit (two-stage clustered t) --
                          animal_mean_z +/- 95% CI (animal_z_ci_lo/hi), animal_z_p
                          (one-sided, H1 > 0); sd_animal_z + n_animals_pos = consistency.
                          A CI excluding 0 => real signal for that marginalization.
and, at OUT_DIR level (shared across run-tags):
    regularizer_auto.json  {"<labels>_k<ncomp>": {session: lambda}} -- the auto-resolved
                           lambda per session, so later runs skip the CV search (--refit-reg
                           re-searches).

    uv run python notebooks/pipelines/kinematic-dpca.py \\
        [--labels ladt --factors level axis direction] [--regularizer auto] \\
        [--n-components 15] [--n-shuffles 1000] [--sessions M061_...] [--aggregate-only --tag ...]
"""

import os

# dPCA._marginalize uses numexpr; pin it (and BLAS, via joblib inner_max_num_threads)
# to 1 thread per worker so N parallel processes don't oversubscribe the cores.
# Must run before numexpr is first imported (i.e. before importing dPCA).
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, dump
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root -> tools

import tools.dimensionality.dpca as dim  # noqa: E402
from tools.params import Params  # noqa: E402

# --- defaults (overridable on the CLI) ------------------------------------
# The canonical 11 sessions -- mirrors across-sessions/common_utils.ALL_SESSIONS
# (M063 once, M103/M106 as listed). Also the cache filenames minus ".npz".
SESSIONS = [
    "M061_2025_03_04_10_00",
    "M061_2025_03_05_14_00",
    "M061_2025_03_06_14_00",
    "M062_2025_03_20_14_00",
    "M062_2025_03_21_14_00",
    "M063_2025_03_13_14_00",
    "M078_2025_08_06_15_00",
    "M086_2025_12_10_15_00",
    "M103_2026_02_18_15_30",
    "M103_2026_02_19_15_30",
    "M106_2026_02_25_15_00",
]
FACTORS = ["level", "angle"]  # dPCA non-time factors, names from dim.CONDITION_FACTORS
LABELS = "lat"  # dPCA labels string; must end in the time label, len == len(FACTORS)+1
REGULARIZER = "auto"  # "auto" -> plain-ridge CV lambda search, run once per session; or a fixed float
N_COMPONENTS = 10
N_SHUFFLES = 1000
SEED = 0

CACHE_DIR = Params.ACROSS_SESSION_RESULTS_DIR / "cache"  # one <session>.npz per session
FS = 100  # Hz, cache `fs`
BASE = (-1.6, -0.1)  # baseline window (s rel. onset) for the per-trial z-score
RESPONSE = (-0.5, 1.5)  # response window (s rel. onset) fed to dPCA
PERTURB_THRESH = -2.0  # drop_by_perturb_score thresh_val
OUT_DIR = Params.ACROSS_SESSION_RESULTS_DIR / "kinematic-dpca-noisefloor"


def _regularizer_arg(s: str):
    """argparse type: 'auto' -> str, otherwise a float."""
    return "auto" if s == "auto" else float(s)


def _run_tag(labels: str, regularizer, n_components: int) -> str:
    reg = "auto" if regularizer == "auto" else f"{regularizer:g}"
    return f"{labels}_r{reg}_k{n_components}"


# --- cache of the auto-resolved lambda, shared across run-tags -----------
# The plain-ridge CV lambda search depends on the data (session, BASE/RESPONSE windows,
# perturb-score cut) and the model (labels, n_components); all of that goes into the key so
# a changed window auto-invalidates. Cached so later runs (any --tag / --n-shuffles /
# --seed) skip the search; --refit-reg forces a re-search.
REG_CACHE = OUT_DIR / "regularizer_auto.json"


def _reg_cache_key(labels: str, n_components: int) -> str:
    win = f"base{BASE[0]}_{BASE[1]}_resp{RESPONSE[0]}_{RESPONSE[1]}_pt{PERTURB_THRESH}"
    return f"{labels}_k{n_components}_{win}"


def _reg_cache_get(session: str, labels: str, n_components: int):
    if not REG_CACHE.exists():
        return None
    cache = json.loads(REG_CACHE.read_text())
    return cache.get(_reg_cache_key(labels, n_components), {}).get(session)


def _reg_cache_put(session: str, labels: str, n_components: int, value: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = json.loads(REG_CACHE.read_text()) if REG_CACHE.exists() else {}
    cache.setdefault(_reg_cache_key(labels, n_components), {})[session] = float(value)
    REG_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))


# --- loading, from the interim-results cache ------------------------------
def drop_by_perturb_score(
    perturb_score, thresh_val=PERTURB_THRESH, stat="sem", verbose=True
):
    """Keep genuinely-perturbed, non-stopped trials -- verbatim from
    notebooks/behaviour/response-anatomy.ipynb. `perturb_score` is
    ``ep_perturb_score`` (n_trials, 12)."""
    a = np.asarray(perturb_score, float)
    mean = a.mean(axis=1)
    spread = a.std(axis=1) / (np.sqrt(a.shape[1]) if stat == "sem" else 1.0)
    with np.errstate(invalid="ignore"):
        by_spread = (mean + spread) < 0  # (1) unperturbed -> drop
        by_value = mean > thresh_val  # (2) stopped -> drop
    keep = by_spread & by_value
    if verbose:
        print(
            f"  drop_by_perturb_score ({stat}, thresh={thresh_val}): "
            f"dropped by {stat}: {(~by_spread).sum()}  |  dropped by value: {(~by_value).sum()}  |  "
            f"kept: {keep.sum()}/{len(keep)} ({100 * keep.sum() / len(keep):.1f}%)",
            flush=True,
        )
    return keep


def load_session_tensor(session) -> tuple[np.ndarray, np.ndarray]:
    """Load one session's epoched body-frame kinematics from the interim-results
    cache and shape it for dPCA -- mirrors the loading in
    notebooks/behaviour/response-anatomy.ipynb.

    Returns
    -------
    X : ndarray (n_trials, n_features, n_time)   RESPONSE window, baseline z-scored
    codes : ndarray (n_trials,)                  raw solenoid codes 0-11
    """
    d = np.load(CACHE_DIR / f"{session}.npz", allow_pickle=True)
    onset = int(d["ep_onset"])
    base_sl = slice(onset + int(BASE[0] * FS), onset + int(BASE[1] * FS))
    resp_sl = slice(onset + int(RESPONSE[0] * FS), onset + int(RESPONSE[1] * FS))

    X_ = d["ep_X"].astype(float)  # (n_ep, n_time_full, 57) body-frame keypoints
    bl_std = X_[:, base_sl, :].std(axis=1, keepdims=True)  # (n_ep, 1, 57)
    Xz = X_ / bl_std  # per-trial baseline z-score

    keep = drop_by_perturb_score(d["ep_perturb_score"])
    X = Xz[keep].transpose(0, 2, 1)[:, :, resp_sl]  # -> (n_trials, 57, n_time_resp)
    codes = d["ep_code"][keep].astype(int)
    return X, codes


# --- generic dPCA noise-floor (reusable for neural data too) --------------
def _shuffle_expl_var(perm_codes, X, factors, labels, regularizer, n_components):
    """One label-shuffle iteration: rebuild the tensor, refit the rebalanced dPCA,
    return per-marginalization explained variance (margVar summed over components).
    Module-level so joblib can pickle it. `regularizer` here is always a resolved
    float -- the CV search, if any, happens once on the real data."""
    conds = dim.conditions_from_codes(perm_codes, factors)
    tX, Xp, *_ = dim.build_unbalanced_trial_tensor(X, conds)
    m = dim.RebalancedDPCA(
        labels=labels,
        regularizer=regularizer,
        join=dim.default_join(labels),
        n_components=n_components,
    )
    m.noise_var, m.protect = dim.rebalanced_noise_variance(tX), [labels[-1]]
    m.fit(Xp, tX)
    return dim.get_var_split(m, Xp)[1].sum(axis=1)


def noise_floor_session(
    X,
    codes,
    *,
    factors=FACTORS,
    labels=LABELS,
    regularizer=REGULARIZER,
    n_components=N_COMPONENTS,
    n_shuffles=N_SHUFFLES,
    seed=SEED,
    n_jobs=-1,
) -> tuple[dict, dict]:
    """Observed vs label-shuffle null for every marginalization of one session.

    Returns
    -------
    stats : dict   keys / observed / signal_frac / null / null_median / null_p95 / p / z /
                   n_trials / cell_counts / regularizer_used  (goes to the .npz)
    fit : dict     dpca / X_psth / trialX / codes / cond_values / config  (goes to .joblib)
    """
    conds = dim.conditions_from_codes(codes, factors)
    trialX, Xpsth, cond_values, counts = dim.build_unbalanced_trial_tensor(X, conds)

    m = dim.RebalancedDPCA(
        labels=labels,
        regularizer=regularizer,
        join=dim.default_join(labels),
        n_components=n_components,
    )
    m.noise_var, m.protect = dim.rebalanced_noise_variance(trialX), [labels[-1]]
    m.fit(Xpsth, trialX)  # regularizer='auto' -> CV search runs here, sets m.regularizer
    reg_used = float(m.regularizer)

    keys = list(m.marginalizations)
    observed = dim.get_var_split(m, Xpsth)[1].sum(axis=1)
    _, signal_frac = dim.noise_corrected_marg_var(
        m, Xpsth, trialX
    )  # Kobak Fig 3-6d point estimate

    rng = np.random.default_rng(seed)
    perms = [
        rng.permutation(codes) for _ in range(n_shuffles)
    ]  # fixed -> scheduling-independent
    null = np.array(
        Parallel(n_jobs=n_jobs, inner_max_num_threads=1)(
            delayed(_shuffle_expl_var)(p, X, factors, labels, reg_used, n_components)
            for p in perms
        )
    )

    p = (1 + (null >= observed).sum(axis=0)) / (
        n_shuffles + 1
    )  # +1 permutation p, never 0
    z = (observed - null.mean(axis=0)) / null.std(axis=0)

    stats_out = dict(
        keys=np.array(keys),
        observed=observed,
        signal_frac=signal_frac,
        null=null,  # full (n_shuffles, n_marginalizations) array
        null_median=np.median(null, axis=0),
        null_p95=np.percentile(null, 95, axis=0),
        p=p,
        z=z,
        n_trials=len(codes),
        cell_counts=np.array(sorted(counts.values())),
        regularizer_used=reg_used,
    )
    fit_out = dict(
        dpca=m,
        X_psth=Xpsth,
        trialX=trialX,
        codes=np.asarray(codes),
        cond_values=cond_values,
        config=dict(
            factors=list(factors),
            labels=labels,
            regularizer=regularizer,
            regularizer_used=reg_used,
            n_components=n_components,
            n_shuffles=n_shuffles,
            seed=seed,
        ),
    )
    return stats_out, fit_out


def run_one(session, args, run_dir):
    t0 = time.time()
    print(f"[{session}] loading ...", flush=True)
    X, codes = load_session_tensor(session)

    reg = args.regularizer
    cached = None
    if reg == "auto" and not args.refit_reg:
        cached = _reg_cache_get(session, args.labels, args.n_components)
        if cached is not None:
            reg = cached
            print(
                f"[{session}] cached auto lambda = {reg:.3g}  ({REG_CACHE.name})",
                flush=True,
            )

    print(
        f"[{session}] X {X.shape}  n_trials {len(codes)}  labels={args.labels} "
        f"reg={'auto' if reg == 'auto' else f'{reg:.3g}'} k={args.n_components} "
        f"-> {args.n_shuffles} shuffles",
        flush=True,
    )

    stats_out, fit_out = noise_floor_session(
        X,
        codes,
        factors=args.factors,
        labels=args.labels,
        regularizer=reg,
        n_components=args.n_components,
        n_shuffles=args.n_shuffles,
        seed=args.seed,
        n_jobs=args.jobs,
    )

    if args.regularizer == "auto" and cached is None:
        _reg_cache_put(
            session, args.labels, args.n_components, stats_out["regularizer_used"]
        )
        print(f"[{session}] cached auto lambda -> {REG_CACHE}", flush=True)

    npz = run_dir / f"{session}.npz"
    joblib_path = run_dir / f"{session}_fit.joblib"
    np.savez(npz, session=session, **stats_out)
    dump(fit_out, joblib_path, compress=3)

    print(
        f"[{session}] done in {time.time() - t0:.0f}s  reg_used={stats_out['regularizer_used']:.2g}\n"
        f"    -> {npz}\n    -> {joblib_path}",
        flush=True,
    )
    for k, pk, zk in zip(stats_out["keys"], stats_out["p"], stats_out["z"]):
        print(f"    {str(k):>5}  p={pk:.4f}  z={zk:+.2f}", flush=True)


# --- across-session aggregate -------------------------------------------
def _pool_across_animals(z: np.ndarray, animals: list[str]) -> dict:
    """Pool per-session z (obs vs its shuffle null) with animal as the unit.

    Two-stage clustered t: average z within each animal -> one number per animal
    -> one-sample t across those (H1: pooled effect > 0). Chosen for this sample
    size (~10 sessions from ~6-7 animals): it uses the right denominator df
    (n_animals - 1), an animal that ran more sessions can't dominate, and it
    needs neither large-cluster asymptotics (mixed-model Wald p) nor a discrete
    floor (sign test).

    Returns
    -------
    dict with:
      animal_mean_z         pooled standardized effect (mean of the per-animal mean z)
      animal_z_ci_lo/hi     two-sided 95% CI on it (t, n_animals-1 df)
      animal_z_p            one-sided p, H1: animal_mean_z > 0
      sd_animal_z           SD of the per-animal means -- between-animal spread
      n_animals_pos         how many per-animal means are > 0
    """
    z = np.asarray(z, float)
    by_animal: dict[str, list] = {}
    for a, zv in zip(animals, z):
        by_animal.setdefault(a, []).append(zv)
    zbar = np.array([np.mean(v) for v in by_animal.values()])  # one per animal
    n = len(zbar)

    mean = float(zbar.mean())
    sd = float(zbar.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 and sd > 0 else float("nan")
    tcrit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else float("nan")
    tstat = mean / se if np.isfinite(se) and se > 0 else float("nan")
    return dict(
        animal_mean_z=mean,
        animal_z_ci_lo=mean - tcrit * se if np.isfinite(se) else float("nan"),
        animal_z_ci_hi=mean + tcrit * se if np.isfinite(se) else float("nan"),
        animal_z_p=float(stats.t.sf(tstat, df=n - 1)) if np.isfinite(tstat) else float("nan"),
        sd_animal_z=sd,
        n_animals_pos=int((zbar > 0).sum()),
    )


def aggregate(run_dir):
    """Combine per-session npz in `run_dir` into one per-marginalization table.

    Headline stat (`_pool_across_animals`): the per-session z pooled with animal
    as the unit -- `animal_mean_z` +/- `animal_z_ci_hi/lo` (95% CI), `animal_z_p`
    (one-sided, H1 > 0). Read a marginalization as carrying real signal if the CI
    excludes 0; `n_animals_pos` and `sd_animal_z` say whether that is consistent
    across animals or driven by a subset. `n_sig_p05` / `median_z` /
    `median_signal_frac` are descriptive per-session summaries.
    """
    files = sorted(f for f in run_dir.glob("*.npz"))
    if not files:
        print(f"no per-session .npz in {run_dir}", flush=True)
        return

    per_marg: dict[str, list] = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        animal = f.stem.split("_")[0]
        for i, k in enumerate(d["keys"]):
            per_marg.setdefault(str(k), []).append(
                (animal, float(d["p"][i]), float(d["z"][i]), float(d["signal_frac"][i]))
            )

    rows = []
    for k, recs in per_marg.items():
        ps = np.array([r[1] for r in recs])
        zs = np.array([r[2] for r in recs])
        animals = [r[0] for r in recs]
        rows.append(
            dict(
                marg=k,
                n_sessions=len(recs),
                n_animals=len(set(animals)),
                n_sig_sessions_p05=int((ps < 0.05).sum()),  # raw, uncorrected -- diagnostic
                median_session_z=float(np.median(zs)),
                median_signal_frac=float(np.median([r[3] for r in recs])),
                **_pool_across_animals(zs, animals),
            )
        )

    import csv

    out_csv = run_dir / "summary.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}", flush=True)
    for r in rows:
        print(
            f"  {r['marg']:>5}  {r['n_animals_pos']}/{r['n_animals']} animals+  "
            f"z={r['animal_mean_z']:+.2f} [{r['animal_z_ci_lo']:+.2f},{r['animal_z_ci_hi']:+.2f}]  "
            f"p={r['animal_z_p']:.2e}  ({r['n_sig_sessions_p05']}/{r['n_sessions']} sessions p<.05)",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sessions",
        nargs="+",
        default=SESSIONS,
        help="session names (default: the canonical 11, module-level SESSIONS)",
    )
    ap.add_argument(
        "--labels", default=LABELS, help="dPCA labels string, ends in the time label"
    )
    ap.add_argument(
        "--factors",
        nargs="+",
        default=FACTORS,
        help=f"non-time factors, from {sorted(dim.CONDITION_FACTORS)}; len == len(labels)-1",
    )
    ap.add_argument(
        "--regularizer",
        type=_regularizer_arg,
        default=REGULARIZER,
        help="'auto' for the CV lambda search (cached per session in regularizer_auto.json), "
        "or a fixed float",
    )
    ap.add_argument(
        "--refit-reg",
        action="store_true",
        help="with regularizer=auto, ignore the cached per-session lambda and re-search",
    )
    ap.add_argument("--n-components", type=int, default=N_COMPONENTS)
    ap.add_argument("--n-shuffles", type=int, default=N_SHUFFLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument(
        "--tag",
        default=None,
        help="output subfolder name (default: <labels>_r<reg>_k<ncomp>)",
    )
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if len(args.factors) != len(args.labels) - 1:
        ap.error(
            f"--factors ({args.factors}) must have len(labels)-1 = {len(args.labels) - 1} entries"
        )
    bad = [f for f in args.factors if f not in dim.CONDITION_FACTORS]
    if bad:
        ap.error(f"unknown --factors {bad}; choose from {sorted(dim.CONDITION_FACTORS)}")

    tag = args.tag or _run_tag(args.labels, args.regularizer, args.n_components)
    run_dir = OUT_DIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run-tag: {tag}  ->  {run_dir}", flush=True)

    if not args.aggregate_only:
        for session in args.sessions:
            try:
                run_one(session, args, run_dir)
            except Exception:
                print(f"[{session}] FAILED\n{traceback.format_exc()}", flush=True)

    aggregate(run_dir)


if __name__ == "__main__":
    main()
