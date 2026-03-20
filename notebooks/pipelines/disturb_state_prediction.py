"""
notebooks/pipelines/disturb_state_prediction.py
================================================
Does the pre-perturbation kinematic state — and its interaction with
perturbation properties — explain disturbance score out-of-sample?

Design
------
Sessions are pooled across all animals.  Session identity is included
as a nuisance covariate in every model (C(session) in OLS / dummy
columns in sklearn) so that R² gains reflect within-session variance
explained, not between-session differences.

Predictor groups compared
-------------------------
 1. session_only          — baseline
 2. perturb               — direction + duration + level + laterality
 3. phase                 — gait phase sin/cos × 12 KPs at perturbation onset
 4. amplitude             — mean pre-perturbation log-power × 12 KPs
 5. speed                 — mean + std of running speed pre-perturbation
 6. phase + perturb       — additive combination (no interaction)
 7. phase × dir           — phase + direction + phase:direction interactions
 8. phase × dur           — phase + duration  + phase:duration  interactions
 9. phase × dir × dur     — full 3-way (all lower-order terms included)

For each group:
  - statsmodels OLS  → in-sample adjusted R²
  - sklearn 5-fold CV → out-of-sample mean R²

A permutation test (n=1000) is run on the best-CV model.

Usage
-----
    poetry run python notebooks/pipelines/disturb_state_prediction.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from tqdm import tqdm

from statsmodels.formula.api import ols as sm_ols
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import pyaldata as pyal
import tools.dsp as dsp
from tools.params import Params
from tools.viz.rc_style import apply_rc

# ── Parameters ────────────────────────────────────────────────────────────────

DATA_DIR = "/data/bnd-data/raw/"

sessions = [
    # 'M061_2025_03_04_10_00',
    # 'M061_2025_03_05_14_00',
    "M061_2025_03_06_14_00",
    # 'M063_2025_03_13_14_00',
    # 'M063_2025_03_14_15_30',
    "M062_2025_03_20_14_00",
    # 'M062_2025_03_21_14_00',
    "M078_2025_08_06_15_00",
    # 'M086_2025_12_10_15_00',
    # 'M103_2026_02_18_15_30',
    # 'M106_2026_02_25_15_00'
]

N_CV_FOLDS = 5
N_PERM = 1000
PRE_WIN_BINS = 100  # bins before perturbation onset used for amplitude/speed (= 1 s)

apply_rc()


# ── Feature extraction ────────────────────────────────────────────────────────


def extract_features(df_tr: pd.DataFrame, session_name: str) -> pd.DataFrame:
    """
    Extract per-trial predictor features.

    Returns one row per trial with columns:
      - disturb_sum                         target
      - session                             session label
      - direction, duration, level, contra_ipsi   perturbation properties
      - phase_{i}_sin/cos  (i=0..11)        gait phase sin/cos at perturbation onset
      - amplitude_{i}      (i=0..11)        mean pre-perturb log-power per keypoint
      - speed_mean, speed_std               pre-perturbation running speed
    """
    rows = []
    for _, row in df_tr.iterrows():
        feat = {"session": session_name, "disturb_sum": row["disturb_sum"]}

        # ── perturbation properties ──────────────────────────────────────────
        feat["direction"] = float(row["values_Sol_direction"])
        feat["level"] = float(row.get("sol_level_id", np.nan))
        feat["contra_ipsi"] = float(row.get("sol_contra_ipsi", np.nan))

        # duration — may be absent in some sessions
        raw_dur = row.get("values_Sol_duration", np.nan)
        if hasattr(raw_dur, "__len__"):
            feat["duration"] = float(raw_dur[0]) if len(raw_dur) > 0 else np.nan
        else:
            feat["duration"] = float(raw_dur) if raw_dur is not None else np.nan

        # ── gait phase (sin/cos at perturbation onset) ───────────────────────
        phases_ok = isinstance(row.get("phases"), np.ndarray)
        if phases_ok:
            t0 = int(row["concat_perturb_time"])
            phase_vec = row["phases"][t0, :]  # (12,)
            for i, ph in enumerate(phase_vec):
                feat[f"phase_{i}_sin"] = float(np.sin(ph))
                feat[f"phase_{i}_cos"] = float(np.cos(ph))
        else:
            for i in range(12):
                feat[f"phase_{i}_sin"] = np.nan
                feat[f"phase_{i}_cos"] = np.nan

        # ── gait amplitude (mean log-power pre-perturbation) ─────────────────
        power_ok = isinstance(row.get("power"), np.ndarray)
        if power_ok:
            t0 = int(row["concat_perturb_time"])
            pre_power = row["power"][max(0, t0 - PRE_WIN_BINS) : t0, :]  # (<=100, 12)
            for i in range(pre_power.shape[1]):
                feat[f"amplitude_{i}"] = float(pre_power[:, i].mean())
        else:
            for i in range(12):
                feat[f"amplitude_{i}"] = np.nan

        # ── running speed (from bhv, pre-perturbation) ───────────────────────
        if isinstance(row.get("bhv"), np.ndarray):
            t0_bhv = int(row["idx_sol_on"])
            bhv_pre = row["bhv"][max(0, t0_bhv - PRE_WIN_BINS) : t0_bhv, :]
            vel = np.gradient(bhv_pre, axis=0)
            speed = np.linalg.norm(vel, axis=1)
            feat["speed_mean"] = float(speed.mean())
            feat["speed_std"] = float(speed.std())
        else:
            feat["speed_mean"] = np.nan
            feat["speed_std"] = np.nan

        rows.append(feat)

    return pd.DataFrame(rows)


# ── Session loading ───────────────────────────────────────────────────────────

print("Loading sessions...")
all_frames = []

for sess in sessions:
    print(f"  {sess}")
    try:
        df_tr, dstrb_idx = dsp.load_and_process_session(sess, data_dir=DATA_DIR)
        df_tr = dsp.drop_unperturbed_trials(df_tr, dstrb_idx)
        feats = extract_features(df_tr, sess)
        all_frames.append(feats)
        print(f"    → {len(feats)} trials")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"    FAILED: {e}")

df = pd.concat(all_frames, ignore_index=True)

# Report NaN counts before dropping
print("\nNaN counts per column:")
print(df.isna().sum()[df.isna().sum() > 0].to_string())

df = df.dropna()
print(f"\nTotal trials after pooling + dropna: {len(df)}")
print(f"Sessions: {df['session'].value_counts().to_dict()}")


# ── Predictor column lists ─────────────────────────────────────────────────────

PHASE_COLS = [f"phase_{i}_sin" for i in range(12)] + [f"phase_{i}_cos" for i in range(12)]
AMPLITUDE_COLS = [f"amplitude_{i}" for i in range(12)]
SPEED_COLS = ["speed_mean", "speed_std"]
PERTURB_COLS = ["direction", "duration", "level", "contra_ipsi"]

# Interaction columns — computed below once df is ready
# phase_i_sin/cos × direction (24 cols)
PHASE_X_DIR_COLS = [f"phase_{i}_{trig}_x_dir" for i in range(12) for trig in ("sin", "cos")]
# phase_i_sin/cos × duration (24 cols)
PHASE_X_DUR_COLS = [f"phase_{i}_{trig}_x_dur" for i in range(12) for trig in ("sin", "cos")]
# phase_i_sin/cos × direction × duration (24 cols)
PHASE_X_DIR_X_DUR_COLS = [f"phase_{i}_{trig}_x_dir_x_dur" for i in range(12) for trig in ("sin", "cos")]


def add_interaction_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i in range(12):
        for trig in ("sin", "cos"):
            ph = df[f"phase_{i}_{trig}"]
            df[f"phase_{i}_{trig}_x_dir"] = ph * df["direction"]
            df[f"phase_{i}_{trig}_x_dur"] = ph * df["duration"]
            df[f"phase_{i}_{trig}_x_dir_x_dur"] = ph * df["direction"] * df["duration"]
    return df


df = add_interaction_cols(df)
print("Interaction columns added.")

# ── CV helper ─────────────────────────────────────────────────────────────────

session_dummies = pd.get_dummies(df["session"], drop_first=True).astype(float)
y = df["disturb_sum"].values
cv = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)


def cv_r2(extra_cols):
    if extra_cols:
        X = pd.concat([session_dummies, df[extra_cols].astype(float)], axis=1).values
    else:
        X = session_dummies.values
    scores = cross_val_score(
        make_pipeline(StandardScaler(), LinearRegression()), X, y, cv=cv, scoring="r2"
    )
    return scores.mean(), scores.std()


# ── Model definitions ─────────────────────────────────────────────────────────

# For each model: (OLS formula extra terms, CV extra cols list)
# OLS formulas handle interactions with * notation;
# CV uses pre-computed columns.

SESSION_TERM = "C(session)"

models_spec = {
    "session only":    ([], []),
    "perturb":         (PERTURB_COLS, PERTURB_COLS),
    "phase":           (PHASE_COLS, PHASE_COLS),
    "amplitude":       (AMPLITUDE_COLS, AMPLITUDE_COLS),
    "speed":           (SPEED_COLS, SPEED_COLS),
    "phase + perturb": (PHASE_COLS + PERTURB_COLS, PHASE_COLS + PERTURB_COLS),
    # phase × direction: main effects + explicit interaction cols
    "phase × dir": (
        # OLS: use * to get main effects + interaction in one shot
        # but statsmodels * with many terms is fragile — use pre-computed cols
        PHASE_COLS + ["direction"] + PHASE_X_DIR_COLS,
        PHASE_COLS + ["direction"] + PHASE_X_DIR_COLS,
    ),
    # phase × duration
    "phase × dur": (
        PHASE_COLS + ["duration"] + PHASE_X_DUR_COLS,
        PHASE_COLS + ["duration"] + PHASE_X_DUR_COLS,
    ),
    # full 3-way: phase + dir + dur + all two-way + three-way
    "phase × dir × dur": (
        PHASE_COLS + ["direction", "duration"] + PHASE_X_DIR_COLS + PHASE_X_DUR_COLS + PHASE_X_DIR_X_DUR_COLS,
        PHASE_COLS + ["direction", "duration"] + PHASE_X_DIR_COLS + PHASE_X_DUR_COLS + PHASE_X_DIR_X_DUR_COLS,
    ),
}


def make_formula(extra_cols):
    terms = [SESSION_TERM] + extra_cols
    return "disturb_sum ~ " + " + ".join(terms)


# ── OLS (in-sample) ───────────────────────────────────────────────────────────

print("\nFitting OLS models...")
ols_results = {}

for name, (ols_cols, _) in models_spec.items():
    formula = make_formula(ols_cols)
    m = sm_ols(formula, data=df).fit()
    ols_results[name] = {
        "adj_r2": m.rsquared_adj,
        "r2": m.rsquared,
        "p": m.f_pvalue,
        "n_params": int(m.df_model),
    }
    print(
        f"  {name:<22s}  adj-R²={m.rsquared_adj:.3f}  p={m.f_pvalue:.2e}  "
        f"n_params={int(m.df_model)}"
    )


# ── Cross-validated R² (out-of-sample) ────────────────────────────────────────

print("\nRunning 5-fold CV...")
cv_results = {}

for name, (_, cv_cols) in models_spec.items():
    mean_r2, std_r2 = cv_r2(cv_cols)
    cv_results[name] = {"mean": mean_r2, "std": std_r2}
    print(f"  {name:<22s}  CV R²={mean_r2:.3f} ± {std_r2:.3f}")


# ── Permutation test on best CV model ─────────────────────────────────────────

best_name = max(cv_results, key=lambda k: cv_results[k]["mean"])
print(f"\nPermutation test on best CV model: '{best_name}' (n={N_PERM})")

best_cv_cols = models_spec[best_name][1]
X_best = (
    pd.concat([session_dummies, df[best_cv_cols].astype(float)], axis=1).values
    if best_cv_cols
    else session_dummies.values
)

pipe = make_pipeline(StandardScaler(), LinearRegression())
observed_cv_r2 = cv_results[best_name]["mean"]

null_r2s = []
for _ in tqdm(range(N_PERM)):
    y_perm = np.random.permutation(y)
    null_r2s.append(cross_val_score(pipe, X_best, y_perm, cv=cv, scoring="r2").mean())

null_r2s = np.array(null_r2s)
perm_p = np.mean(null_r2s >= observed_cv_r2)
print(
    f"  Observed CV R²={observed_cv_r2:.3f}  |  "
    f"null median={np.median(null_r2s):.3f}  |  p={perm_p:.4f}"
)


# ── Summary table ─────────────────────────────────────────────────────────────

names = list(models_spec.keys())
adj_r2s = [ols_results[n]["adj_r2"] for n in names]
cv_means = [cv_results[n]["mean"] for n in names]
cv_stds = [cv_results[n]["std"] for n in names]

baseline_adj = adj_r2s[0]
baseline_cv = cv_means[0]

print(
    f"\n{'Model':<22}  {'adj-R²':>7}  {'CV R²':>7}  {'CV std':>7}  "
    f"{'Δ adj-R²':>9}  {'Δ CV R²':>9}  {'n_params':>9}  {'p (OLS)':>10}"
)
print("-" * 100)
for name in names:
    o = ols_results[name]
    c = cv_results[name]
    d_adj = o["adj_r2"] - baseline_adj
    d_cv = c["mean"] - baseline_cv
    print(
        f"{name:<22}  {o['adj_r2']:>7.3f}  {c['mean']:>7.3f}  {c['std']:>7.3f}  "
        f"{d_adj:>+9.3f}  {d_cv:>+9.3f}  {o['n_params']:>9d}  {o['p']:>10.2e}"
    )


# ── Plot ───────────────────────────────────────────────────────────────────────

y_pos = np.arange(len(names))
bar_h = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

ax.barh(y_pos + bar_h / 2, adj_r2s, height=bar_h, color="steelblue", label="In-sample adj. R²")
ax.barh(y_pos - bar_h / 2, cv_means, height=bar_h, color="tomato", alpha=0.85, label="CV R² (5-fold)")
ax.errorbar(cv_means, y_pos - bar_h / 2, xerr=cv_stds,
            fmt="none", color="black", capsize=3, linewidth=1)

ax.axvline(0, color="k", linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.set_xlabel("R²")
ax.set_title(
    f"Disturbance score prediction — pooled {len(sessions)} sessions  (n={len(df)} trials)\n"
    f"Perm p={perm_p:.3f} for best CV model: '{best_name}'"
)
ax.legend(fontsize=9)

for i, name in enumerate(names[1:], start=1):
    d_adj = adj_r2s[i] - baseline_adj
    d_cv = cv_means[i] - baseline_cv
    ax.text(
        max(adj_r2s[i], cv_means[i], 0) + 0.003,
        y_pos[i],
        f"Δ={d_adj:+.3f} / {d_cv:+.3f}",
        va="center", fontsize=7, color="gray",
    )

plt.tight_layout()
plt.savefig("results/disturb_state_prediction.png", dpi=150)
plt.show()
