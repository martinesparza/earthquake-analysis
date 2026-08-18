"""
Rotate keypoint kinematics from raw camera axes into an anatomical body frame, and validate that
the fitted frame is anatomically sane.
"""

import numpy as np
import pandas as pd

from tools.params import Params

ML_DOMINANCE_MIN = 2.0
# how many times larger the left/right ml separation must be than the largest off-axis (rc/vt)
# separation. A ratio (not an absolute tolerance) so it stays scale-invariant across animals of
# different size/camera distance -- see gaitlib.py for the M061/M062 vs M078 case that motivated
# this. A swapped ml/rc axis lands near 0.4, a swapped ml/vt axis near 0.2, both well below this.


def build_body_frame_td(
    td,
    base_win=(-1.5, -0.1),
    fs=100,
    hip_field="hip_center",
    shoulder_field="shoulder_center",
):
    """
    Fit a per-trial rotation into an anatomical (rc, vt, ml) body frame, from the
    `hip_field`/`shoulder_field` keypoints.

    The frame is fitted once per trial, on the pre-perturbation baseline window only, so a
    perturbation-induced heading change cannot rotate the frame mid-response:

        rc   rostrocaudal   + = rostral (forward)   baseline hip -> shoulder direction,
                                                      projected into the horizontal (x/z) plane
        vt   vertical       + = dorsal (up)          = -y (camera y increases downward)
        ml   mediolateral   + = animal's left        ml = vt x rc

    Parameters
    ----------
    td             : pd.DataFrame  pyalData trial table. `hip_field`/`shoulder_field` columns
                     hold a (T, 3) array per trial, and an `idx_sol_on` column gives the
                     perturbation-onset sample index within each trial.
    base_win       : tuple(float, float)  baseline window (s, relative to `idx_sol_on`) the frame
                     is fitted on.
    fs             : float  sampling rate (Hz) of `hip_field`/`shoulder_field` (raw keypoints are
                     10 ms bins = 100 Hz).
    hip_field      : str  column name of the hip keypoint.
    shoulder_field : str  column name of the shoulder keypoint.

    Returns
    -------
    R   : np.ndarray  (n_trials, 3, 3)  rows = [rc, vt, ml], expressed in the keypoints' own
          (x, y, z) axes. `R @ v` rotates a camera-frame vector `v` into (rc, vt, ml).
    yaw : np.ndarray  (n_trials,)  body-axis heading in the horizontal plane, in degrees (QC
          only -- not used to build R). This is a circular quantity (wraps at +-180 deg): use
          circular statistics, not a plain mean/std, when summarising it across trials.
    """
    trial_cols = td[[hip_field, shoulder_field, "idx_sol_on"]]

    n = len(td)
    R = np.full((n, 3, 3), np.nan)
    yaw = np.full(n, np.nan)

    for i, (_, hip, sho, onset) in enumerate(trial_cols.itertuples()):
        hip = np.asarray(hip)
        sho = np.asarray(sho)
        if hip.ndim != 2 or hip.shape[1] != 3:
            raise ValueError(
                f"trial {i}: {hip_field!r} has shape {hip.shape}, expected (T, 3). If you just "
                "renamed/edited tools.kinematics, restart the kernel -- %autoreload can get out "
                "of sync across a module rename."
            )
        if sho.ndim != 2 or sho.shape[1] != 3:
            raise ValueError(
                f"trial {i}: {shoulder_field!r} has shape {sho.shape}, expected (T, 3). If you "
                "just renamed/edited tools.kinematics, restart the kernel -- %autoreload can get "
                "out of sync across a module rename."
            )
        onset = int(onset)
        base_sl = slice(
            onset + int(round(base_win[0] * fs)),
            onset + int(round(base_win[1] * fs)),
        )

        rc = np.nanmean(sho[base_sl] - hip[base_sl], axis=0)
        rc[1] = 0.0  # project into horizontal plane
        rc = rc / np.linalg.norm(rc)

        vt = np.array([0.0, -1.0, 0.0])  # camera -y is dorsal (up)
        ml = np.cross(vt, rc)
        ml = ml / np.linalg.norm(ml)

        R[i] = np.stack([rc, vt, ml], axis=0)
        yaw[i] = np.degrees(np.arctan2(rc[0], rc[2]))

    return R, yaw


def _rotate_field_baseline(td, R, field, hip_field, base_win, fs):
    """Rotate one keypoint's baseline-window trace, relative to `hip_field`, into (rc, vt, ml)
    for every trial. Returns (n_trials, T_base, 3); T_base is constant across trials since it only
    depends on `base_win`/`fs`, not on each trial's own onset."""
    trial_cols = td[[field, hip_field, "idx_sol_on"]]

    out = None
    for i, (_, kp, hip, onset) in enumerate(trial_cols.itertuples()):
        kp = np.asarray(kp)
        hip = np.asarray(hip)
        onset = int(onset)
        base_sl = slice(
            onset + int(round(base_win[0] * fs)),
            onset + int(round(base_win[1] * fs)),
        )
        rel = kp[base_sl] - hip[base_sl]  # (T_base, 3), camera axes
        rotated = rel @ R[i].T  # (T_base, 3), columns = rc, vt, ml
        if out is None:
            out = np.full((len(td), rotated.shape[0], 3), np.nan)
        out[i] = rotated
    return out


def validate_body_frame_td(
    td,
    R,
    yaw,
    base_win=(-1.5, -0.1),
    fs=100,
    hip_field="hip_center",
    left_paw_field="left_paw",
    right_paw_field="right_paw",
    left_foot_field="left_foot",
    right_foot_field="right_foot",
    ml_dominance_min=ML_DOMINANCE_MIN,
    verbose=True,
):
    """
    Assert that a fitted body frame (`R`, `yaw` from `build_body_frame_td`) is anatomically
    correct. Mirrors `gaitlib.validate_frame`'s three checks, each of which must hold for any
    correct frame:

      1. left/right limb pairs separate on ml, and ml DOMINATES the other two components
      2. fore-aft (rc) dominates each limb's baseline stride oscillation amplitude
      3. paws/feet sit below hip_center on vt (i.e. vt sign is up)

    plus a circular summary of `yaw` (mean/std/resultant length) and a wrap flag, reported but
    not asserted on.

    Raises AssertionError on the first failing check. Returns a dict of the check values
    otherwise.
    """
    out = {}

    seps = []
    for l_field, r_field, name in [
        (left_paw_field, right_paw_field, "paw"),
        (left_foot_field, right_foot_field, "foot"),
    ]:
        l_mean = np.nanmean(
            _rotate_field_baseline(td, R, l_field, hip_field, base_win, fs),
            axis=(0, 1),
        )
        r_mean = np.nanmean(
            _rotate_field_baseline(td, R, r_field, hip_field, base_win, fs),
            axis=(0, 1),
        )
        sep = l_mean - r_mean
        s = {"rc": float(sep[0]), "vt": float(sep[1]), "ml": float(sep[2])}
        off = max(abs(s["rc"]), abs(s["vt"]))
        s["ml_dominance"] = (
            float(abs(s["ml"]) / off) if off > 0 else float("inf")
        )
        seps.append(s)
        assert s["ml"] > 0, (
            f"frame check 1 failed for {name} ({l_field}/{r_field}): "
            f"+ml should be the animal's LEFT, got {s}"
        )
        assert s["ml_dominance"] > ml_dominance_min, (
            f"frame check 1 failed for {name} ({l_field}/{r_field}): ml does not dominate "
            f"(ratio {s['ml_dominance']:.1f} <= {ml_dominance_min}): {s}"
        )
    out["lr_separation"] = seps

    limb_fields = (
        left_paw_field,
        right_paw_field,
        left_foot_field,
        right_foot_field,
    )
    amps = {}
    below = {}
    for field in limb_fields:
        rot = _rotate_field_baseline(td, R, field, hip_field, base_win, fs)
        sd = np.nanstd(
            rot - np.nanmean(rot, axis=1, keepdims=True), axis=(0, 1)
        )
        amps[field] = [float(v) for v in sd]
        assert (
            sd[0] > sd[1] and sd[0] > sd[2]
        ), f"frame check 2 failed for {field}: rc not dominant (sd [rc,vt,ml] = {sd})"
        below[field] = float(np.nanmean(rot[..., 1]))  # vt, relative to hip
    out["stride_amplitude"] = amps

    assert all(
        v < 0 for v in below.values()
    ), f"frame check 3 failed (vt sign): {below}"
    out["limb_vt_vs_hip"] = below

    # Yaw is CIRCULAR: an arithmetic mean/std is only valid while no trial straddles the +-180 deg
    # wrap -- see gaitlib.py's M078 case, where it inflated the arithmetic sd to 38.5 deg vs a
    # circular sd of 9.9 deg. The frame itself never uses yaw; this is QC only.
    yaw_rad = np.radians(yaw)
    Rres = np.abs(np.nanmean(np.exp(1j * yaw_rad)))
    circ_mean = float(np.degrees(np.angle(np.nanmean(np.exp(1j * yaw_rad)))))
    circ_std = (
        float(np.degrees(np.sqrt(-2 * np.log(Rres))))
        if Rres > 0
        else float("nan")
    )
    out["yaw_deg"] = [circ_mean, circ_std]
    out["yaw_resultant"] = float(Rres)
    out["yaw_wraps"] = bool((np.abs(yaw) > 170).mean() > 0.005)

    if verbose:
        print(
            f"  [frame] L/R ml separation: "
            f"{seps[0]['ml']:+.2f} (paws), {seps[1]['ml']:+.2f} (feet); "
            f"ml dominance {seps[0]['ml_dominance']:.1f}x / {seps[1]['ml_dominance']:.1f}x "
            f"(needs > {ml_dominance_min})"
        )
        print(
            f"  [frame] stride sd [rc,vt,ml] "
            f"{left_paw_field}={np.round(amps[left_paw_field], 2).tolist()} "
            f"{right_foot_field}={np.round(amps[right_foot_field], 2).tolist()}  "
            f"(rc dominant, as required)"
        )
        print(
            f"  [frame] yaw {circ_mean:+.1f} +- {circ_std:.1f} deg "
            f"(circular; R={Rres:.3f}{', WRAPS +-180' if out['yaw_wraps'] else ''}); "
            f"limbs below hip on vt: {all(v < 0 for v in below.values())}"
        )

    return out


def _field_width(td, field):
    """Number of columns `field` contributes to a `bhv`-style design matrix."""
    arr = np.asarray(td[field].values[0])
    return 1 if arr.ndim == 1 else arr.shape[1]


def rotate_bhv_td(
    td,
    bhv_fields=None,
    base_win=(-1.5, -0.1),
    fs=100,
    hip_field="hip_center",
    shoulder_field="shoulder_center",
    left_paw_field="left_paw",
    right_paw_field="right_paw",
    left_foot_field="left_foot",
    right_foot_field="right_foot",
    ml_dominance_min=ML_DOMINANCE_MIN,
    validate=True,
    verbose=True,
    bhv_col="bhv",
    out_col="bhv_rot",
):
    """
    Fit the body frame, validate it, and rotate `bhv_col` into it -- a one-call wrapper around
    `build_body_frame_td`, `validate_body_frame_td`, and the rotation itself.

    `bhv_fields` defaults to `Params.pos_keypoints` -- the field list used throughout this repo to
    build `bhv_col` (`dt.add_bhv(td, bhv_fields=["pos_keypoints"])`) -- so the common case needs no
    extra argument. Pass it explicitly only if `bhv_col` was built from a different field list; it
    must match that list, in that order, since each field is assumed to lay out contiguously in
    `bhv_col` (the same assumption `get_keypoint_dim_indices` makes elsewhere in this codebase).
    Only 3-column (x, y, z) blocks are rotated; scalar fields (e.g. `*_angle`) are not vectors and
    are copied through unchanged.

    Rotation is ALLOCENTRIC. each 3-column block is referenced to
    its own  baseline-window mean (a constant per trial), not to `hip_field`'s per-timepoint
    position -- so whole-body translation survives the rotation. This is the right choice for "did
    the animal move in the direction it was pushed", which a hip-relative frame cannot answer by
    construction (hip-relative rotation subtracts the hip's own translation from every keypoint,
    including itself). `hip_field`/`shoulder_field` are still used to FIT the rotation `R` (via
    `build_body_frame_td`) and for the `validate_body_frame_td` checks, just not to re-reference
    `bhv_col` before rotating.

    Parameters
    ----------
    td               : pd.DataFrame  pyalData trial table with a `bhv_col` column already built,
                       plus `hip_field`/`shoulder_field`/limb columns and `idx_sol_on`.
    bhv_fields       : list[str] or None  field names `bhv_col` was built from, in order; defaults
                       to `Params.pos_keypoints` if not given.
    base_win, fs, hip_field, shoulder_field
                     : passed through to `build_body_frame_td`; `base_win`/`fs` are also used to
                       locate each field's own baseline window for the allocentric re-referencing.
    left_paw_field, right_paw_field, left_foot_field, right_foot_field, ml_dominance_min, verbose
                     : passed through to `validate_body_frame_td` (only used if `validate=True`).
    validate         : bool  if True (default), run `validate_body_frame_td` and raise on a
                       failing check before rotating -- catches a bad frame fit before it silently
                       propagates into `bhv_rot`.
    bhv_col          : str  column to rotate.
    out_col          : str  column to write the rotated result to.

    Returns
    -------
    td  : pd.DataFrame  copy of the input with `out_col` added.
    R   : np.ndarray  (n_trials, 3, 3)  the fitted rotation, as from `build_body_frame_td`.
    yaw : np.ndarray  (n_trials,)  body-axis heading in degrees (QC only).

    Examples
    --------
    >>> td = dt.add_bhv(td, bhv_fields=["pos_keypoints"])
    >>> td, R, yaw = kin.rotate_bhv_td(td)
    >>> td.bhv_rot.iloc[0].shape == td.bhv.iloc[0].shape
    True

    To skip the anatomical sanity checks (e.g. once a session is already known-good, for speed):

    >>> td, R, yaw = kin.rotate_bhv_td(td, validate=False)
    """
    if bhv_fields is None:
        bhv_fields = Params.pos_keypoints

    R, yaw = build_body_frame_td(
        td,
        base_win=base_win,
        fs=fs,
        hip_field=hip_field,
        shoulder_field=shoulder_field,
    )

    if validate:
        validate_body_frame_td(
            td,
            R,
            yaw,
            base_win=base_win,
            fs=fs,
            hip_field=hip_field,
            left_paw_field=left_paw_field,
            right_paw_field=right_paw_field,
            left_foot_field=left_foot_field,
            right_foot_field=right_foot_field,
            ml_dominance_min=ml_dominance_min,
            verbose=verbose,
        )

    widths = [_field_width(td, f) for f in bhv_fields]
    offsets = np.concatenate(([0], np.cumsum(widths)))

    onset_vals = td["idx_sol_on"].values
    bhv_vals = td[bhv_col].values

    rotated_col = [None] * len(td)
    for i in range(len(td)):
        bhv = np.asarray(bhv_vals[i])  # (T, D)
        onset = int(onset_vals[i])
        base_sl = slice(
            onset + int(round(base_win[0] * fs)),
            onset + int(round(base_win[1] * fs)),
        )
        out = bhv.copy()
        for field, start, end in zip(bhv_fields, offsets[:-1], offsets[1:]):
            if end - start != 3:
                continue  # scalar field (e.g. an angle), not a rotatable vector
            block = bhv[:, start:end]  # (T, 3), camera axes
            block = block - np.nanmean(
                block[base_sl], axis=0, keepdims=True
            )  # allocentric: own baseline mean, not hip's per-timepoint position
            out[:, start:end] = block @ R[i].T  # (T, 3), now (rc, vt, ml)
        rotated_col[i] = out

    td = td.copy()
    td[out_col] = pd.Series(rotated_col, index=td.index)
    return td, R, yaw
