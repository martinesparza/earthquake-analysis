"""Summarise and plot the 90 deg vs 270 deg (pure left/right) decode from `decode_ml_axis.py`.

Reports, per body-frame dimension:
  - POST accuracy across sessions
  - the matched PRE-onset control on the same trials and the same window length
  - POST minus PRE, which is the only quantity that is evidence about the RESPONSE rather than
    about posture or ongoing state
  - how often each dimension is the winner, counted per session and per animal, since a mean over
    sessions can be carried by one animal with several sessions
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root, for tools
from tools.params import Params  # noqa: E402

DIMS = ["rc", "vt", "ml"]
DATA_DIR = Params.ACROSS_SESSION_RESULTS_DIR / "bhv-decode-solenoid" / "data"
FIG_DIR = Params.ACROSS_SESSION_RESULTS_DIR / "bhv-decode-solenoid" / "figures"
CO, CB, CG = "#D55E00", "#0072B2", "#009E73"


def load():
    rows = []
    for f in sorted(DATA_DIR.glob("mlaxis_*.npz")):
        d = np.load(f, allow_pickle=True)
        r = dict(session=str(d["session"]), animal=str(d["session"])[:4],
                 n=int(d["n_trials"]), n90=int(d["n_90"]), n270=int(d["n_270"]))
        for w in ("post", "pre"):
            for dim in DIMS:
                r[f"{w}_{dim}"] = float(d[f"{w}_{dim}_scores"].mean())
                r[f"{w}_{dim}_folds"] = d[f"{w}_{dim}_scores"]
        rows.append(r)
    return rows


def main():
    rows = load()
    if not rows:
        print("no mlaxis_*.npz found")
        return
    animals = sorted({r["animal"] for r in rows})
    print(f"{len(rows)} sessions, {len(animals)} animals, "
          f"{sum(r['n'] for r in rows)} trials on the 90/270 axis\n")

    print(f"{'session':<24}{'n':>5}{'90':>5}{'270':>5}" +
          "".join(f"{d + ' post':>11}" for d in DIMS) +
          "".join(f"{d + ' pre':>10}" for d in DIMS) + f"{'best':>7}")
    for r in rows:
        best = max(DIMS, key=lambda d: r[f"post_{d}"] - r[f"pre_{d}"])
        print(f"{r['session']:<24}{r['n']:>5}{r['n90']:>5}{r['n270']:>5}"
              + "".join(f"{r[f'post_{d}']:>11.3f}" for d in DIMS)
              + "".join(f"{r[f'pre_{d}']:>10.3f}" for d in DIMS) + f"{best:>7}")

    print()
    for d in DIMS:
        post = np.array([r[f"post_{d}"] for r in rows])
        pre = np.array([r[f"pre_{d}"] for r in rows])
        exc = post - pre
        w = stats.wilcoxon(post, pre) if len(rows) > 5 else None
        print(f"  {d:<3s} POST {post.mean():.3f} +- {post.std():.3f}   "
              f"PRE {pre.mean():.3f} +- {pre.std():.3f}   "
              f"excess {exc.mean():+.3f}  ({int((exc > 0).sum())}/{len(rows)} sessions positive"
              + (f", Wilcoxon p={w.pvalue:.4f})" if w else ")"))

    exc_all = {d: np.array([r[f"post_{d}"] - r[f"pre_{d}"] for r in rows]) for d in DIMS}
    win = [max(DIMS, key=lambda d: r[f"post_{d}"] - r[f"pre_{d}"]) for r in rows]
    print(f"\n  best dim by POST-PRE excess, per session: "
          f"{ {d: win.count(d) for d in DIMS} }")
    by_animal = {a: [w for w, r in zip(win, rows) if r["animal"] == a] for a in animals}
    print(f"  per animal: " + ", ".join(
        f"{a}={max(set(v), key=v.count)}" for a, v in by_animal.items()))
    ml_vs_best_other = exc_all["ml"] - np.maximum(exc_all["rc"], exc_all["vt"])
    print(f"  ml excess minus best-other excess: median {np.median(ml_vs_best_other):+.3f}, "
          f"{int((ml_vs_best_other > 0).sum())}/{len(rows)} sessions positive")

    # ---------------------------------------------------------------- figure
    acolor = {a: plt.cm.tab10(i % 10) for i, a in enumerate(animals)}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    ax = axes[0]
    for x, d in enumerate(DIMS):
        for w, off, c, lab in [("post", -0.16, CO, "POST 0-1.5s"), ("pre", 0.16, CB, "PRE -1.5-0s")]:
            v = np.array([r[f"{w}_{d}"] for r in rows])
            ax.errorbar([x + off], [v.mean()], yerr=[v.std()], fmt="o", color=c, capsize=4,
                        markersize=8, zorder=3, label=lab if x == 0 else None)
            j = rng.normal(0, 0.04, size=len(v))
            for r, jj, vv in zip(rows, j, v):
                ax.scatter(x + off + jj, vv, color=acolor[r["animal"]], s=16, alpha=0.75,
                           zorder=2, edgecolors="none")
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels(DIMS)
    ax.set_ylabel("accuracy (5-fold CV)")
    ax.set_title("90° vs 270°: response window vs\nmatched pre-onset control", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for x, d in enumerate(DIMS):
        v = exc_all[d]
        ax.errorbar([x], [v.mean()], yerr=[v.std()], fmt="o", color=CG, capsize=4,
                    markersize=8, zorder=3)
        j = rng.normal(0, 0.05, size=len(v))
        for r, jj, vv in zip(rows, j, v):
            ax.scatter(x + jj, vv, color=acolor[r["animal"]], s=16, alpha=0.75, zorder=2,
                       edgecolors="none")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels(DIMS)
    ax.set_ylabel("POST − PRE accuracy")
    ax.set_title("above-control excess\n(the response-specific part)", fontsize=9)

    ax = axes[2]
    ax.hist(ml_vs_best_other, bins=np.linspace(-0.3, 0.3, 16), color="0.75",
            edgecolor="k", linewidth=0.5)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvline(float(np.median(ml_vs_best_other)), color=CO, lw=2, label="median")
    ax.set_xlabel("ml excess − best of (rc, vt)")
    ax.set_ylabel("sessions")
    ax.set_title(f"is ml the winner?\n{int((ml_vs_best_other > 0).sum())}/{len(rows)} sessions > 0",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=a,
                          markersize=7) for a, c in acolor.items()]
    axes[1].legend(handles=handles, fontsize=6, frameon=False, ncol=2, loc="lower center")
    fig.suptitle(f"Pure left/right axis (90° vs 270°, upper+lower pooled) decoded per body-frame "
                 f"dimension — {len(rows)} sessions, {len(animals)} animals", fontsize=10)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "mlaxis_90v270_per_dim.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
