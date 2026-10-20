"""Single-trial behaviour plots, the shared colour palette, and the dPCA / decoding helpers for bhv-001 / bhv-002."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sgnl
from matplotlib.colors import ListedColormap, to_hex, to_rgb

import tools.dimensionality as dim
import tools.dsp as dsp
import tools.viz.utilityTools as vizutils
from tools.params import Params

SUBGROUPS = {
    "forelimb": [
        "left_elbow",
        "left_wrist",
        "left_paw",
        "right_elbow",
        "right_wrist",
        "right_paw",
    ],
    "hindlimb": [
        "left_knee",
        "left_ankle",
        "left_foot",
        "right_knee",
        "right_ankle",
        "right_foot",
    ],
    "shoulders": ["shoulder_center", "left_shoulder", "right_shoulder"],
    "hip": ["hip_center"],
    "tail": ["tail_base", "tail_middle", "tail_tip"],
}

# Colour palette. Angle -> PiYG wrapped into a circle (hue = side, lightness =
# front -> back); level -> line style; factors -> grey / pink / hatching.
# Explained and previewed in the "Colour palette" section of bhv-001.ipynb.
ANGLES = [30, 90, 150, 210, 270, 330]
BRIDGE = 30  # degrees of the dark-green -> dark-magenta bridge, opposite the white point

_M = np.array(
    [
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ]
)
_W = _M.sum(axis=1)
_piyg = plt.get_cmap("PiYG")


def _lin(c):
    """Convert sRGB components in [0, 1] to linear light.

    Parameters
    ----------
    c : array_like of float
        sRGB components.
    """
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _unlin(c):
    """Convert linear-light components to sRGB, clipped to [0, 1].

    Parameters
    ----------
    c : array_like of float
        Linear-light components.
    """
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _f(x):
    """Apply the forward CIELAB nonlinearity.

    Parameters
    ----------
    x : array_like of float
        XYZ components divided by the white point.
    """
    return np.where(
        x > (6 / 29) ** 3, np.cbrt(x), x / (3 * (6 / 29) ** 2) + 4 / 29
    )


def _finv(x):
    """Apply the inverse CIELAB nonlinearity.

    Parameters
    ----------
    x : array_like of float
        Values returned by ``_f``.
    """
    return np.where(x > 6 / 29, x**3, 3 * (6 / 29) ** 2 * (x - 4 / 29))


def to_lab(c):
    """Convert a matplotlib colour to CIELAB (D65).

    Parameters
    ----------
    c : color
        Any colour matplotlib accepts (hex string, RGB tuple, ...).

    Returns
    -------
    numpy.ndarray
        L*, a*, b*, shape (3,).
    """
    x, y, z = _f((_M @ _lin(to_rgb(c))) / _W)
    return np.array([116 * y - 16, 500 * (x - y), 200 * (y - z)])


def from_lab(lab):
    """Convert CIELAB (D65) to sRGB, clipped to [0, 1].

    Parameters
    ----------
    lab : array_like of float
        L*, a*, b*, shape (3,).

    Returns
    -------
    tuple of float
        sRGB components.
    """
    y = (lab[0] + 16) / 116
    xyz = _W * _finv(np.array([y + lab[1] / 500, y, y - lab[2] / 200]))
    return tuple(_unlin(np.linalg.solve(_M, xyz)))


def piyg_cyclic_at(theta):
    """Sample PiYG wrapped into a circle, white at the back, magenta right, green left.

    Parameters
    ----------
    theta : float
        Angle in degrees, 0 = nose, 90 = animal's right.

    Returns
    -------
    tuple of float
        sRGB components.
    """
    th = (
        (theta + 180) % 360 + 180
    ) % 360 - 180  # (-180, 180], 0 = white point
    half = 180 - BRIDGE / 2
    if abs(th) <= half:
        return _piyg(0.5 + 0.5 * th / half)[:3]
    s_ = (
        abs(th) - half
    ) / BRIDGE  # 0..0.5 from the nearer PiYG end into the bridge
    g, m = to_lab(_piyg(1.0)), to_lab(_piyg(0.0))
    return from_lab(
        g + (m - g) * (0.5 + s_) if th > 0 else m + (g - m) * (0.5 + s_)
    )


def squeeze_lightness(pal, lo=30, hi=74):
    """Remap the L* of sampled colours into [lo, hi], keeping hue.

    Parameters
    ----------
    pal : dict
        Key -> colour.
    lo : float, optional
        Lowest L* after the remap (default is 30).
    hi : float, optional
        Highest L* after the remap (default is 74).

    Returns
    -------
    dict
        Same keys, hex colours.
    """
    labs = {k: to_lab(c) for k, c in pal.items()}
    Ls = np.array([v[0] for v in labs.values()])
    return {
        k: to_hex(
            from_lab(
                np.r_[lo + (v[0] - Ls.min()) / np.ptp(Ls) * (hi - lo), v[1:]]
            )
        )
        for k, v in labs.items()
    }


piyg_cyclic = ListedColormap(
    [piyg_cyclic_at(x) for x in np.linspace(0, 360, 512, endpoint=False)],
    name="PiYG_cyclic",
)
ANGLE_COLOR = squeeze_lightness({a: piyg_cyclic_at(a) for a in ANGLES})
LEVEL_STYLE = {  # 0 = top, 1 = bottom
    0: dict(color="#5E5E5E", ls="-"),
    1: dict(color="#262626", ls="--"),
}

FACTOR_COLOR = {
    "time": "#999999",
    "angle": to_hex(_piyg(0.2)),
    "level": "#262626",
}
HATCH = "///"
FACTOR_BAR = {
    "time": dict(color=FACTOR_COLOR["time"]),
    "level": dict(
        facecolor="white", edgecolor=FACTOR_COLOR["level"], hatch=HATCH
    ),
    "angle": dict(color=FACTOR_COLOR["angle"]),
    "level×angle": dict(
        facecolor=FACTOR_COLOR["angle"],
        edgecolor=FACTOR_COLOR["level"],
        hatch=HATCH,
    ),
}
FACTOR_LINE = {
    "time": dict(color=FACTOR_COLOR["time"], ls="-"),
    "level": dict(color=FACTOR_COLOR["level"], ls="--"),
    "angle": dict(color=FACTOR_COLOR["angle"], ls="-"),
    "level×angle": dict(color=FACTOR_COLOR["angle"], ls="--"),
}

# Other plot colours, from the PiYG ends and the palette's dark grey.
DIM_COLOR = ["#b04d86", "#828282", "#68925e"]  # x / rc, y / vt, z / ml
GROUP_COLOR = {
    "forelimb": "#c41a7c",
    "hindlimb": "#4c9121",
    "axial": "#969696",
}
NEUTRAL, ACCENT = "#575757", "#c41a7c"

# Highlight for the example session in per-session plots. Level is never a colour: it is
# always the hatched / dashed style (FACTOR_BAR, FACTOR_LINE, LEVEL_STYLE).
EXAMPLE_COLOR = to_hex(_piyg(0.8))


# Session highlighted in every per-session plot.
EXAMPLE = "M061_2025_03_06_14_00"

# dPCA marginalization keys (lpt labels) -> display name, and -> key of FACTOR_BAR / FACTOR_LINE.
MARG_NAME = {"t": "time", "lt": "level", "pt": "angle", "lpt": "level × angle"}
MARG_STYLE = {"t": "time", "lt": "level", "pt": "angle", "lpt": "level×angle"}

# Body-frame dimensions in channel order, and the body regions used for the limb projections.
DIMS = ["rc", "vt", "ml"]
REGIONS = {
    "forelimb": SUBGROUPS["forelimb"],
    "hindlimb": SUBGROUPS["hindlimb"],
    "axial": SUBGROUPS["shoulders"] + SUBGROUPS["hip"] + SUBGROUPS["tail"],
}


def bar_style(key):
    """Return bar / wedge keyword arguments, with a neutral edge when hatched.

    Parameters
    ----------
    key : str
        Marginalization key, one of ``MARG_STYLE``.

    Returns
    -------
    dict
        Keyword arguments for ``Axes.bar`` / ``Wedge.set``.
    """
    style = FACTOR_BAR[MARG_STYLE[key]]
    return {**style, "edgecolor": NEUTRAL} if "hatch" in style else style


def load_bhv_cache(sessions, cache_dir, fields=None):
    """Load the per-session bhv caches.

    Parameters
    ----------
    sessions : list of str
        Session names, each needing ``<cache_dir>/<session>.npz``.
    cache_dir : pathlib.Path
        Folder holding the caches.
    fields : list of str or None, optional
        Arrays to load; None loads all (default is None).

    Returns
    -------
    dict
        Session -> dict of field -> array.

    Raises
    ------
    FileNotFoundError
        If any session has no cache.
    """
    cache_dir = Path(cache_dir)
    missing = [s for s in sessions if not (cache_dir / f"{s}.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"No cache for {missing} in {cache_dir} -- build it first"
        )
    out = {}
    for sess in sessions:
        with np.load(cache_dir / f"{sess}.npz", allow_pickle=False) as z:
            keys = fields if fields is not None else z.files
            out[sess] = {k: z[k] for k in keys}
    return out


def animal_means(values, sessions):
    """Average a per-session value within each animal (first four characters of the session name).

    Parameters
    ----------
    values : numpy.ndarray
        One value per session, shape (sessions,).
    sessions : list of str
        Session names, same order as ``values``.

    Returns
    -------
    numpy.ndarray
        One mean per animal in sorted order, shape (animals,).
    """
    animals = sorted({s[:4] for s in sessions})
    return np.array(
        [
            values[[s.startswith(a) for s in sessions]].mean()
            for a in animals
        ]
    )


def channel_regions(keypoints, regions=REGIONS, dims=DIMS):
    """Label keypoint-major channels (kp0_rc, kp0_vt, ...) with their dimension and body region.

    Parameters
    ----------
    keypoints : sequence of str
        Keypoint names along the keypoint axis of ``bhv_rot``.
    regions : dict of {str: list of str}, optional
        Region -> keypoint names (default is ``REGIONS``).
    dims : list of str, optional
        Dimension names in channel order (default is ``DIMS``).

    Returns
    -------
    dim_of : numpy.ndarray
        Dimension name of each channel, shape (channels,).
    region_mask : dict of {str: numpy.ndarray}
        Region -> boolean mask over channels.
    """
    kps = [str(k) for k in keypoints]
    kp_of, dim_of = np.repeat(kps, len(dims)), np.tile(dims, len(kps))
    region_mask = {g: np.isin(kp_of, kp_list) for g, kp_list in regions.items()}
    return dim_of, region_mask


def region_dim_table(model, X, dim_of, region_mask, rows, dims=DIMS):
    """Split each marginalization's variance over body region and dimension as v_c * F[i, c] ** 2 per channel.

    Parameters
    ----------
    model : RebalancedDPCA
        Fitted model.
    X : numpy.ndarray
        Centred condition means the model was fit on.
    dim_of : numpy.ndarray
        Dimension name of each channel, from ``channel_regions``.
    region_mask : dict of {str: numpy.ndarray}
        Region -> boolean channel mask, from ``channel_regions``.
    rows : list of str
        Marginalization keys to tabulate.
    dims : list of str, optional
        Dimension names (default is ``DIMS``).

    Returns
    -------
    numpy.ndarray
        Shape (rows, regions, dims), in % of the total variance.
    """
    component_var, _, which_marg, order = dim.get_var_split(model, X)
    F = np.hstack([model.P[k] for k in model.marginalizations])[:, order]  # encoders, in component_var's order
    table = np.zeros((len(rows), len(region_mask), len(dims)))
    for ri, marg in enumerate(rows):
        cols = which_marg == marg
        for ci, mask in enumerate(region_mask.values()):
            for di, d in enumerate(dims):
                table[ri, ci, di] = (
                    F[mask & (dim_of == d)][:, cols] ** 2 * component_var[cols]
                ).sum()
    return table


def channel_energy(model, Z, marg):
    """Return each channel's squared, condition-averaged projection of one marginalization over time.

    Parameters
    ----------
    model : RebalancedDPCA
        Fitted model.
    Z : dict
        ``model.transform(X)``, marginalization -> (components, levels, angles, time).
    marg : str
        Marginalization key, e.g. "lt".

    Returns
    -------
    numpy.ndarray
        Shape (channels, time), in baseline-z squared.
    """
    z = Z[marg].reshape(Z[marg].shape[0], -1, Z[marg].shape[-1])  # (components, conditions, time)
    w = np.tensordot(model.P[marg], z, axes=([1], [0]))  # (channels, conditions, time)
    return (w**2).mean(axis=1)


def grid_cross(t, y, level, rising=True):
    """Return the grid time where a curve crosses a level, without interpolation.

    Parameters
    ----------
    t : numpy.ndarray
        Window times in seconds.
    y : numpy.ndarray
        Curve values, same length as ``t``.
    level : float
        Threshold.
    rising : bool, optional
        First upward crossing if True, else last downward (default is True).

    Returns
    -------
    float
        Time of that sample, NaN if the curve never crosses.
    """
    steps = np.diff((y >= level).astype(int))
    idx = np.where(steps == (1 if rising else -1))[0]
    if idx.size == 0:
        return np.nan
    return t[idx[0] + 1] if rising else t[idx[-1]]


def timing_metrics_grid(t, curve, frac=0.5):
    """Return rise, peak and fall times and width of a moving-window decoding curve.

    Parameters
    ----------
    t : numpy.ndarray
        Window right-edge times in seconds from onset.
    curve : numpy.ndarray
        Accuracy above chance.
    frac : float, optional
        Fraction of the curve's peak that counts as the edge (default is 0.5).

    Returns
    -------
    dict
        ``t_rise50``, ``t_peak``, ``t_fall50`` and ``fwhm``, in seconds.
    """
    level = frac * curve.max()
    t_rise = grid_cross(t, curve, level, rising=True)
    t_fall = grid_cross(t, curve, level, rising=False)
    return dict(
        t_rise50=t_rise,
        t_peak=t[int(np.argmax(curve))],
        t_fall50=t_fall,
        fwhm=t_fall - t_rise,
    )


FS, ON, WIN = 100, 200, 30
MAX_DELAY = 80


def rel_position(X, limb, ref=None, ch=0, keypoints=None):
    """Return one limb's position along one body-frame channel, relative to a reference keypoint.

    Parameters
    ----------
    X : numpy.ndarray
        Body-frame data with keypoint-major channels, shape (..., time, 57).
    limb : str
        Keypoint name.
    ref : str or None, optional
        Reference keypoint subtracted from the limb; None keeps the raw position (default is None).
    ch : int, optional
        Channel within a keypoint: 0 = rc, 1 = vt, 2 = ml (default is 0).
    keypoints : sequence of str or None, optional
        Keypoint order of ``X``; None uses ``Params.pos_keypoints`` (default is None).

    Returns
    -------
    numpy.ndarray
        Shape (..., time).
    """
    kps = list(Params.pos_keypoints) if keypoints is None else [str(k) for k in keypoints]
    out = X[..., kps.index(limb) * 3 + ch]
    return out if ref is None else out - X[..., kps.index(ref) * 3 + ch]


def stride_frequency(x, on=ON):
    """Return the stride frequency of one trial as the Welch peak of its pre-push samples.

    Parameters
    ----------
    x : numpy.ndarray
        Position trace of one trial, shape (time,).
    on : int, optional
        Push sample; only samples before it are used (default is ``ON``).

    Returns
    -------
    float
        Frequency in Hz.
    """
    pre = x[:on]
    fr, pw = sgnl.welch(pre - pre.mean(), fs=FS, nperseg=on, nfft=1024)
    return fr[pw.argmax()]


def phase_at(x, end, f0):
    """Return the sine-fit phase at the last sample of the WIN samples ending before ``end``.

    Parameters
    ----------
    x : numpy.ndarray
        Position trace of one trial, shape (time,).
    end : int
        Index one past the last sample of the fit window.
    f0 : float
        Centre of the frequency search in Hz.

    Returns
    -------
    float
        Phase in radians in (-pi, pi]; 0 = peak of the fitted cosine.
    """
    return dsp.sinefit(x[end - WIN:end], f0, fs=FS)[0]


def phase_at_push(rc):
    """Return each trial's sine-fit phase from the window that ends at the push.

    Parameters
    ----------
    rc : numpy.ndarray
        Position traces, shape (trials, time).

    Returns
    -------
    numpy.ndarray
        Phase in radians, shape (trials,).
    """
    return np.array([phase_at(tr, ON, stride_frequency(tr)) for tr in rc])


def circular_r2(phase, y):
    """Return the share of the variance of y explained by cos(phase) and sin(phase).

    Parameters
    ----------
    phase : numpy.ndarray
        Phase in radians, shape (trials,).
    y : numpy.ndarray
        Score per trial, shape (trials,).

    Returns
    -------
    float
        R squared of the least-squares fit y ~ 1 + cos + sin, in [0, 1].
    """
    A = np.column_stack([np.ones_like(phase), np.cos(phase), np.sin(phase)])
    resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return 1 - resid.var() / y.var()


def circular_r2_test(phase, y, codes, n_perm, rng):
    """Return the circular R2 and its null from shuffling phases among trials of the same solenoid code.

    Parameters
    ----------
    phase : numpy.ndarray
        Phase in radians, shape (trials,).
    y : numpy.ndarray
        Score per trial, shape (trials,).
    codes : numpy.ndarray
        Solenoid code per trial, shape (trials,); shuffles stay within a code.
    n_perm : int
        Number of shuffles.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    dict
        r2, null_mean, null_std, null95 and the one-sided permutation p.
    """
    null = []
    for _ in range(n_perm):
        perm = np.arange(len(phase))
        for c in np.unique(codes):
            m = np.flatnonzero(codes == c)
            perm[m] = rng.permutation(m)
        null.append(circular_r2(phase[perm], y))
    r2 = circular_r2(phase, y)
    return dict(
        r2=r2,
        null_mean=np.mean(null),
        null_std=np.std(null, ddof=1),
        null95=np.percentile(null, 95),
        p=(1 + np.sum(np.array(null) >= r2)) / (n_perm + 1),
    )


def touchdowns(fa, on, subsample=False, prom_frac=0.25):
    """Return the touchdown samples, the forward peaks of a fore-aft trace.

    Parameters
    ----------
    fa : numpy.ndarray
        Fore-aft position of one trial, shape (time,).
    on : int
        Push sample; the pre-push 5-95% range sets the prominence threshold.
    subsample : bool, optional
        Refine each peak with a parabola through three samples (default is False).
    prom_frac : float, optional
        Minimum prominence as a fraction of the pre-push range (default is 0.25).

    Returns
    -------
    numpy.ndarray
        Peak positions in samples.
    """
    lo, hi = np.percentile(fa[:on], [5, 95])
    pk, _ = sgnl.find_peaks(fa, distance=6, prominence=prom_frac * (hi - lo))
    if subsample:
        a, b, c = fa[pk - 1], fa[pk], fa[pk + 1]
        den = a - 2 * b + c
        pk = pk + np.where(den != 0, 0.5 * (a - c) / np.where(den != 0, den, 1), 0.0)
    return pk


def delay(tds, anchor, end, max_delay=MAX_DELAY):
    """Return how late the first touchdown after an anchor lands compared with the pre-anchor schedule.

    Parameters
    ----------
    tds : numpy.ndarray
        Touchdown positions in samples.
    anchor : int
        Anchor sample; the schedule is the last touchdown at or before it plus the median of the last 3 intervals.
    end : int
        Only touchdowns before this sample count as the next one.
    max_delay : float, optional
        Delays larger than this in ms are dropped as stalls or missed touchdowns (default is ``MAX_DELAY``).

    Returns
    -------
    float
        Delay in ms; NaN with fewer than 4 touchdowns before the anchor, none after, or beyond ``max_delay``.
    """
    before, after = tds[tds <= anchor], tds[(tds > anchor) & (tds < end)]
    if len(before) < 4 or len(after) == 0:
        return np.nan
    stride = np.median(np.diff(before[-4:]))
    dly = (after[0] - (before[-1] + stride)) * (1000 / FS)
    return dly if abs(dly) <= max_delay else np.nan


def delay_table(trials, limb, ref, keypoints=None, subsample=True):
    """Return each trial's phase and touchdown delay at the push and at a control anchor 1 s earlier.

    Parameters
    ----------
    trials : numpy.ndarray
        Body-frame data of one session, shape (trials, time, 57).
    limb : str
        Keypoint whose touchdowns are timed.
    ref : str
        Reference keypoint subtracted from the limb's rc position.
    keypoints : sequence of str or None, optional
        Keypoint order of ``trials``; None uses ``Params.pos_keypoints`` (default is None).
    subsample : bool, optional
        Refine touchdown times below the sampling grid (default is True).

    Returns
    -------
    pandas.DataFrame
        One row per trial with phase_push, delay_push, phase_run and delay_run.
    """
    rows = []
    for tr in trials:
        rc = rel_position(tr, limb, ref, keypoints=keypoints)
        tds = touchdowns(rc, ON, subsample=subsample)
        f0 = stride_frequency(rc)
        rows.append(
            dict(
                phase_push=phase_at(rc, ON, f0),
                delay_push=delay(tds, ON, len(rc)),
                phase_run=phase_at(rc, ON - 100, f0),
                delay_run=delay(tds, ON - 100, ON),
            )
        )
    return pd.DataFrame(rows)


def median_se(v, rng, n_boot=500):
    """Return the standard error of the median by bootstrapping the trials of one bin.

    Parameters
    ----------
    v : numpy.ndarray
        Delays of the trials in one phase bin.
    rng : numpy.random.Generator
        Random generator.
    n_boot : int, optional
        Number of bootstrap resamples (default is 500).

    Returns
    -------
    float
        SD of the median over the resamples; NaN with fewer than 3 trials.
    """
    if len(v) < 3:
        return np.nan
    return np.std([np.median(rng.choice(v, len(v))) for _ in range(n_boot)])


def get_mean_sess(d_all, sess, bins):
    """Return the mean touchdown delay in each phase bin for one session.

    Parameters
    ----------
    d_all : pandas.DataFrame
        One row per trial with session, phase_push, delay_push, phase_run and delay_run.
    sess : str
        Session name.
    bins : int
        Number of equal phase bins between -pi and pi.

    Returns
    -------
    dict of {str: numpy.ndarray}
        "push" and "run", each shape (bins,); NaN where a bin has no delays.
    """
    dd = d_all[d_all["session"] == sess]
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    out = {}
    for which in ("push", "run"):
        idx = np.digitize(dd[f"phase_{which}"], edges) - 1
        out[which] = np.array([dd[f"delay_{which}"][idx == b].mean() for b in range(bins)])
    return out


def baseline_normalise(X, t, zscore=False):
    """Subtract the pre-onset (t < 0) mean, and optionally divide by its std, per keypoint and dimension.

    Parameters
    ----------
    X : numpy.ndarray
        Positions of one trial, shape (time, keypoints, 3).
    t : numpy.ndarray
        Time from onset in seconds, shape (time,).
    zscore : bool, optional
        Also divide by the pre-onset std (default is False).

    Returns
    -------
    numpy.ndarray
        Same shape as ``X``.
    """
    base = X[t < 0]
    out = X - base.mean(axis=0)
    if zscore:
        out = out / base.std(axis=0)
    return out


def plot_trial_heatmap(
    X,
    t,
    kps,
    subgroups=SUBGROUPS,
    zscore=False,
    vmax=None,
    cmap="PiYG",
    xlim=(-1, 1.5),
    figsize=(10, 12),
    hspace=0.075,
    show=True,
):
    """Plot a baseline-normalised heatmap of one trial, one stacked panel (x, -y, z) per dimension.

    Parameters
    ----------
    X : numpy.ndarray
        Positions of one trial, shape (time, keypoints, 3).
    t : numpy.ndarray
        Time from onset in seconds, shape (time,).
    kps : list of str
        Keypoint names along axis 1 of ``X``.
    subgroups : dict of {str: list of str}, optional
        Group name -> keypoint names, drawn in order (default is ``SUBGROUPS``).
    zscore : bool, optional
        Divide by the pre-onset std too (default is False).
    vmax : float or None, optional
        Symmetric colour limit; None uses the largest absolute value (default is None).
    cmap : str, optional
        Diverging colormap name (default is "PiYG").
    xlim : tuple of float, optional
        Time axis limits in seconds (default is (-1, 1.5)).
    figsize : tuple of float, optional
        Figure size in inches (default is (10, 12)).
    hspace : float, optional
        Space between panels (default is 0.075).
    show : bool, optional
        Call ``plt.show()`` inside the style context (default is True).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axs : numpy.ndarray
        The three panel axes, shape (3,).
    """
    Xd = baseline_normalise(X, t, zscore=zscore)
    names = [kp for kp_list in subgroups.values() for kp in kp_list]
    idx = [kps.index(kp) for kp in names]
    n = len(names)
    if vmax is None:
        vmax = np.abs(Xd[:, idx]).max()

    ticks, bounds, start = [], [], 0
    for kp_list in subgroups.values():
        end = start + len(kp_list)
        ticks.append((start + end - 1) / 2)
        if end < n:
            bounds.append(end - 0.5)
        start = end

    with vizutils.apply_plot_style("talk"):
        fig, axs = plt.subplots(
            3,
            1,
            figsize=figsize,
            sharex=True,
            sharey=True,
            gridspec_kw=dict(hspace=hspace),
        )
        for d, ax in enumerate(axs):
            im = ax.pcolormesh(
                t,
                np.arange(n),
                Xd[:, idx, d].T,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                shading="nearest",
            )
            ax.axvline(0, color="k", ls=":")
            for b in bounds:
                ax.axhline(b, color="grey")
            ax.set_ylabel(
                ["x", "-y", "z"][d],
                rotation=0,
                ha="right",
                va="center",
                labelpad=10,
            )
        axs[0].set_yticks(ticks, list(subgroups))
        axs[-1].set_xlabel("time from onset (s)")
        for ax in axs[:-1]:
            ax.tick_params(bottom=False)
        axs[0].invert_yaxis()
        axs[0].set_xlim(*xlim)
        fig.colorbar(
            im,
            ax=axs,
            shrink=0.6,
            label=(
                "z-score vs baseline" if zscore else "Δ position from baseline"
            ),
        )
        if show:
            plt.show()
    return fig, axs
