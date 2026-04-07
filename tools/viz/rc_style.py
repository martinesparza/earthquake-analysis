"""
tools/viz/rc_style.py
=====================
Centralised matplotlib rcParams for reproducible figures across the
earthquake-analysis codebase.

Usage
-----
Apply once at notebook start::

    from tools.viz.rc_style import apply_rc

    apply_rc()          # mutates global rcParams (persistent for the session)

Or use as a context manager for a single figure::

    from tools.viz.rc_style import rc_context

    with rc_context():
        fig, ax = plt.subplots(...)

The ``apply_plot_style`` context manager in ``utilityTools.py`` calls
``apply_rc()`` automatically when ``use_rc=True`` (the default), so most
notebooks get this for free.
"""

from __future__ import annotations

import glob as _glob
import matplotlib as mpl
import matplotlib.font_manager as _fm
from contextlib import contextmanager


def _register_arial() -> None:
    """Register Arial TTFs from the msttcorefonts directory if not yet known."""
    arial_files = _glob.glob(
        "/usr/share/fonts/truetype/msttcorefonts/Arial*.ttf"
    ) + _glob.glob("/usr/share/fonts/truetype/msttcorefonts/arial*.ttf")
    known = {f.fname for f in _fm.fontManager.ttflist}
    for path in arial_files:
        if path not in known:
            _fm.fontManager.addfont(path)


_register_arial()


# ---------------------------------------------------------------------------
# Master parameter dict
# ---------------------------------------------------------------------------

RC_PARAMS: dict[str, object] = {
    # --- Figure ---
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    # --- Font ---
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    # --- Lines & markers ---
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "patch.linewidth": 0.8,
    # --- Axes ---
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    # --- Ticks ---
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # --- Legend ---
    "legend.frameon": False,
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.3,
    # --- Colour cycle (matches area palette order used in params.py) ---
    "axes.prop_cycle": mpl.cycler(
        color=[
            "#4878CF",  # MOp  blue
            "#D65F5F",  # SSp  red
            "#6ACC65",  # CP   green
            "#B47CC7",  # VAL  purple
            "#C4AD66",  # GPe  ochre
            "#77BEDB",  # MOs  light-blue
        ]
    ),
    # --- Misc ---
    "image.interpolation": "none",
    "pdf.fonttype": 42,  # editable text in Illustrator / Inkscape
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


RC_PARAMS_TALK: dict[str, object] = {
    **RC_PARAMS,
    # --- Font ---
    "font.size": 16,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "legend.title_fontsize": 16,
    # --- Lines & markers ---
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "patch.linewidth": 1.5,
    # --- Axes ---
    "axes.linewidth": 1.5,
    # --- Ticks ---
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    # --- Figure ---
    "figure.dpi": 150,
}


def apply_rc() -> None:
    """Mutate ``matplotlib.rcParams`` with the project defaults."""
    mpl.rcParams.update(RC_PARAMS)


def apply_rc_talk() -> None:
    """Mutate ``matplotlib.rcParams`` with talk-optimised defaults (bigger fonts/lines)."""
    mpl.rcParams.update(RC_PARAMS_TALK)


@contextmanager
def rc_context():
    """Context manager: apply project rcParams, restore originals on exit."""
    with mpl.rc_context(RC_PARAMS):
        yield


@contextmanager
def rc_context_talk():
    """Context manager: apply talk-optimised rcParams, restore originals on exit."""
    with mpl.rc_context(RC_PARAMS_TALK):
        yield
