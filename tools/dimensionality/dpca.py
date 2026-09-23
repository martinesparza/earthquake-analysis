"""Run dPCA on the earthquake data with the "rebalanced" ridge fit.

dPCA decomposes trial-averaged neural activity into parts due to time,
stimulus, decision, etc., and for each part finds a low-dimensional
projection (like PCA) that captures as much of that part's variance as
possible while keeping the sources demixed.

This module wraps the ``dPCA`` python package with the ridge fit actually
used by the reference MATLAB implementation (``dpca.m`` +
``dpca_getNoiseCovariance.m``, ``type='averaged'``), which the pip package
does not reproduce: the pip package regularizes the closed-form solve with a
plain ``lambda^2 * I``, whereas ``dpca.m`` uses
``Omega = Cnoise + (lambda*totalVar)^2 * I``, where ``Cnoise`` is a per-neuron
noise covariance estimated from the single-trial data and re-weighted so that
every condition counts equally regardless of trial count ("averaged" branch).
This matters here because conditions are rarely trial-balanced (12 solenoid
directions, unequal counts after trial rejection): the plain pip-package fit
lets over-represented conditions dominate the noise estimate and leak into
the "signal" components.

Notes
-----
`fit_dpca_rebalanced` and `rebalanced_noise_variance` were checked
line-by-line against the MATLAB source in Octave -- axis directions, leakage,
and the noise estimate all matched -- and `get_var_split` reproduces
``dpca_explainedVariance.m``'s formulas verbatim. See
``notebooks/behaviour/response-anatomy.ipynb`` for the original exploration.

References
----------
Kobak et al. 2016, eLife, https://elifesciences.org/articles/10989.
MATLAB source: https://github.com/machenslab/dPCA (``dpca.m``,
``dpca_getNoiseCovariance.m``, ``dpca_explainedVariance.m``).
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Sequence

import numpy as np
from dPCA import dPCA
from sklearn.utils.extmath import randomized_svd

from tools.params import Params


def _codes_to_angle(codes: Sequence[int] | np.ndarray) -> np.ndarray:
    """Map solenoid-direction codes (0-11) to their angle in degrees."""
    return np.array([Params.sol_dir_to_angle[int(c)] for c in codes])


# Solenoid-direction code (0-11, `values_Sol_direction` -- see CLAUDE.md, there is
# no separate `sol_dir` column) -> condition-factor lookups, backed by the Params
# tables so callers never re-derive or hardcode this mapping themselves.
CONDITION_FACTORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "level": lambda codes: np.array([Params.sol_dir_to_level[int(c)] for c in codes]),
    "angle": _codes_to_angle,
    "contra_ipsi": lambda codes: np.array(
        [Params.sol_dir_to_contra_ipsi[int(c)] for c in codes]
    ),
    # axis (the 30/90/150 deg line through the animal, independent of which end the
    # perturbation came from) and direction (which of that axis's two opposite ends,
    # e.g. 30 vs 210 deg) split `angle` in two: angle = axis + 180*direction, valid
    # because the six sol_dir_to_angle values are exactly {30,90,150} and their
    # +180 deg opposites {210,270,330}. Use these instead of "angle" when you want
    # the 3-way/2-way split as separate dPCA factors (e.g. labels='ladt').
    "axis": lambda codes: _codes_to_angle(codes) % 180,
    "direction": lambda codes: (_codes_to_angle(codes) // 180).astype(int),
}


def conditions_from_codes(
    codes: Sequence[int] | np.ndarray,
    factors: Sequence[str],
) -> list[np.ndarray]:
    """Map raw solenoid-direction codes to one condition array per factor.

    Parameters
    ----------
    codes : array-like of int, shape (n_trials,)
        Raw ``values_Sol_direction`` codes (0-11).
    factors : sequence of str
        Names from `CONDITION_FACTORS` (``"level"``, ``"angle"``, ``"axis"``,
        ``"direction"``, ``"contra_ipsi"``), one per non-time factor, in the
        order they will be passed to `fit_rebalanced_dpca`. Position, not the
        letters of dPCA's ``labels`` string, is what is matched.

    Returns
    -------
    list of ndarray
        One ``(n_trials,)`` array per name in `factors`, in the same order.

    Raises
    ------
    ValueError
        If any name in `factors` is not a key of `CONDITION_FACTORS`.
    """
    codes = np.asarray(codes)
    unknown = [f for f in factors if f not in CONDITION_FACTORS]
    if unknown:
        raise ValueError(
            f"unknown condition factor(s) {unknown}; choose from {sorted(CONDITION_FACTORS)}"
        )
    return [CONDITION_FACTORS[f](codes) for f in factors]


def default_join(labels: str, time_label: str = "t") -> dict[str, list[str]]:
    """Fold every non-time marginalization into its "+time" counterpart.

    Parameters
    ----------
    labels : str
        dPCA ``labels`` string, e.g. ``'lat'`` or ``'ladt'``. Must end in
        `time_label`, which must not appear anywhere else in the string.
    time_label : str, optional
        Single letter used for the time factor, by default ``"t"``.

    Returns
    -------
    dict of {str : list of str}
        Suitable for ``dPCA.dPCA(..., join=...)``: maps each joined key
        (e.g. ``'lat'``) to the marginalizations folded into it
        (``['la', 'lat']``).

    Raises
    ------
    ValueError
        If `labels` does not end in `time_label` exactly once.

    Notes
    -----
    For ``labels='lat'`` every static marginalization (``'l'``, ``'a'``,
    ``'la'``) is joined with its time-varying twin (``'lt'``, ``'at'``,
    ``'lat'``), so the fit returns one time-resolved component per subset of
    the non-time factors and no purely-static marginalization survives on its
    own. This is the join pattern used throughout this project: after
    per-neuron centering the purely-static effect is near zero anyway, and
    every returned component should be plottable as a time course.
    """
    if labels[-1] != time_label or time_label in labels[:-1]:
        raise ValueError(
            f"labels={labels!r} must end in the time label {time_label!r} exactly once"
        )
    factors = labels[:-1]
    join: dict[str, list[str]] = {}
    for r in range(1, len(factors) + 1):
        for combo in itertools.combinations(factors, r):
            key = "".join(combo)
            join[key + time_label] = [key, key + time_label]
    return join


def build_unbalanced_trial_tensor(
    X: np.ndarray,
    conditions: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[list], dict[tuple, int]]:
    """Arrange ``(trial, feature, time)`` data into dPCA's NaN-padded layout.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_features, n_time)
        Per-trial data, already restricted to the fit's time window (e.g. the
        post-perturbation response window). Time must be the last axis.
    conditions : sequence of ndarray
        One ``(n_trials,)`` array per non-time condition factor, e.g.
        ``[level, angle]`` for ``labels='lat'``. Order matches the non-time
        part of ``labels``.

    Returns
    -------
    trialX : ndarray, shape (K, n_features, *cond_dims, n_time)
        Single-trial data. ``K`` is the largest per-cell trial count; cells
        with fewer trials are NaN-padded up to ``K`` and cells with no trials
        are left all-NaN.
    X_psth : ndarray, shape (n_features, *cond_dims, n_time)
        Condition mean (``nanmean`` over the trial axis of `trialX`), then
        centered per feature. This is the array dPCA is fit on.
    cond_values : list of list
        ``cond_values[k]`` is the sorted list of distinct values taken by
        factor ``k``; ``cond_values[k][i]`` is the original label for index
        ``i`` along condition axis ``k`` -- use it to map components back to
        e.g. real angles in degrees.
    counts : dict of {tuple : int}
        Number of trials for each populated condition cell, keyed by the tuple
        of real label values. Inspect before fitting to catch badly
        unbalanced designs.

    Raises
    ------
    ValueError
        If a condition array's length does not match ``n_trials``, or there
        are no trials.

    Warns
    -----
    RuntimeWarning
        If any feature is all-NaN in `X_psth` (dead channel, or every
        condition cell it appears in is empty); its centering is skipped and
        it stays NaN.

    Notes
    -----
    NaN-padding every cell up to ``K`` keeps all trials, unlike subsampling
    every cell down to the smallest.
    """
    conditions = [np.asarray(c) for c in conditions]
    n_trials, n_features, n_time = X.shape
    if any(c.shape[0] != n_trials for c in conditions):
        raise ValueError("each condition array must have length n_trials")

    # 1. Lay out the condition grid. For each factor: its sorted distinct
    #    values, and -- per trial -- which slot (0 .. n_values-1) along that
    #    factor's axis the trial belongs to. Stacking the per-factor slots
    #    gives every trial one coordinate in the grid, e.g. cell (0, 2) =
    #    first level, third angle.
    cond_values: list[list] = []
    slot_per_factor: list[np.ndarray] = []
    for c in conditions:
        values, slots = np.unique(c, return_inverse=True)
        cond_values.append(values.tolist())
        slot_per_factor.append(slots)
    cond_dims = tuple(len(v) for v in cond_values)
    cell_of_trial = list(zip(*slot_per_factor))  # (n_trials,) tuples of ints

    # 2. Group trial indices by grid cell. A cell with no trials simply never
    #    appears here (and stays all-NaN in the tensor below).
    trials_by_cell: dict[tuple, list[int]] = {}
    for trial, cell in enumerate(cell_of_trial):
        trials_by_cell.setdefault(cell, []).append(trial)
    if not trials_by_cell:
        raise ValueError("no trials to fit")

    # 3. Fill the tensor. trialX[:, :, *cell, :] holds every trial for one
    #    cell, stacked on axis 0 and NaN-padded from that cell's own trial
    #    count up to K, the largest cell.
    K = max(len(t) for t in trials_by_cell.values())
    trialX = np.full((K, n_features, *cond_dims, n_time), np.nan)
    for cell, trials in trials_by_cell.items():
        trialX[: len(trials), :, *cell, :] = X[trials]

    counts = {  # keyed by the real label values, not slot indices
        tuple(cond_values[f][s] for f, s in enumerate(cell)): len(trials)
        for cell, trials in trials_by_cell.items()
    }

    # 4. Condition mean, centered per feature. X_psth[feature, *cell, time] is
    #    the mean over that cell's trials. A feature that comes out all-NaN
    #    (dead channel, or every cell it appears in is empty) can't be
    #    centered -- flag it and leave it NaN, rather than let numpy warn
    #    cryptically and push NaN into the fit.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        X_psth = np.nanmean(trialX, axis=0)
        by_feature = X_psth.reshape(n_features, -1)
        dead = np.all(np.isnan(by_feature), axis=1)
        feature_mean = np.zeros(n_features)
        feature_mean[~dead] = np.nanmean(by_feature[~dead], axis=1)
    if dead.any():
        warnings.warn(
            f"{int(dead.sum())}/{n_features} feature(s) are all-NaN in the condition "
            "mean (dead channel, or an empty condition cell -- check `counts`); their "
            "centering is skipped and they stay NaN in the returned X_psth",
            RuntimeWarning,
            stacklevel=2,
        )
    X_psth -= feature_mean.reshape(n_features, *[1] * (X_psth.ndim - 1))

    return trialX, X_psth, cond_values, counts


def rebalanced_noise_variance(trialX: np.ndarray, time_axis: int = -1) -> np.ndarray:
    """Per-neuron noise variance, re-balanced so conditions count equally.

    Parameters
    ----------
    trialX : ndarray
        Single-trial data with the trial axis first, NaN-padded past each
        condition's own trial count (as returned by
        `build_unbalanced_trial_tensor`).
    time_axis : int, optional
        Axis of `trialX` that indexes time, by default ``-1``.

    Returns
    -------
    ndarray, shape (n_features,)
        Per-neuron noise variance.

    Notes
    -----
    Sums squared trial-to-mean deviations over time first, divides by the
    per-cell trial count, then sums (not averages) over the remaining
    condition cells, so every cell contributes equally regardless of how many
    trials it has. This is exactly ``dpca_getNoiseCovariance.m``'s
    ``'averaged'`` branch::

        Cnoise = diag(nansum(SSnoiseSumOverT ./ numOfTrials, 2))

    i.e. ``SQT * Cnoise_tilde`` from the paper; the ``SQT`` and the paper's
    ``1/SQT`` cancel algebraically, so what MATLAB actually computes and uses
    is this unscaled sum.
    """
    X = np.nanmean(trialX, axis=0)  # PSTH
    K = np.sum(~np.isnan(trialX), axis=0).astype(float)  # trials per neuron/cell/time
    K[K == 0] = np.nan
    SS = np.nansum((trialX - X[None, ...]) ** 2, axis=0)  # summed sq. deviations per cell

    t_ax = time_axis if time_axis >= 0 else X.ndim + time_axis
    SS_sumT = np.sum(SS, axis=t_ax).reshape(X.shape[0], -1)  # sum over time first
    K_noT = np.take(K, 0, axis=t_ax).reshape(
        X.shape[0], -1
    )  # trial count is time-invariant

    with np.errstate(invalid="ignore"):
        return np.nansum(SS_sumT / K_noT, axis=1)  # sum, not mean, over cells


def fit_dpca_rebalanced(
    dpca_obj: dPCA.dPCA,
    X: np.ndarray,
    trialX: np.ndarray,
    lam: float | None = None,
    random_state: int | np.random.Generator | None = None,
) -> dPCA.dPCA:
    """Refit ``dpca_obj.D`` / ``dpca_obj.P`` with the re-balanced ridge penalty.

    Parameters
    ----------
    dpca_obj : dPCA.dPCA
        A dPCA model that has already had ``.fit(X, trialX)`` called on it
        (needed to populate ``.marginalizations`` / ``.n_components``). Its
        ``.D`` / ``.P`` are overwritten in place.
    X : ndarray, shape (n_features, *cond_dims, n_time)
        Condition-mean data, as passed to ``dpca_obj.fit``.
    trialX : ndarray
        Single-trial NaN-padded data, as passed to ``dpca_obj.fit``.
    lam : float, optional
        Ridge weight. Defaults to ``dpca_obj.regularizer``.
    random_state : int or numpy.random.Generator, optional
        Seeds the per-marginalization `randomized_svd` calls for a
        reproducible fit. ``None`` reproduces the pip package's own unseeded
        behaviour.

    Returns
    -------
    dPCA.dPCA
        The same `dpca_obj`, mutated in place (also returned, for chaining).

    Notes
    -----
    Replaces the plain ``lambda^2 * I`` the pip ``dPCA`` package uses
    internally with ``Omega = diag(rebalanced_noise_variance(trialX)) +
    lambda^2 * I``, matching ``dpca.m``'s actual closed-form solve::

        C = Xmarg @ X.T @ inv(X @ X.T + Cnoise + (lambda*totalVar)^2 * I)

    `fit_rebalanced_dpca` does the plain fit and this refit in one call.
    """
    n = X.shape[0]
    Xcen = dpca_obj._zero_mean(X)
    rX = Xcen.reshape(n, -1)

    lam = dpca_obj.regularizer if lam is None else lam
    lam2 = (
        (lam * np.sum(rX**2)) ** 2 if lam else 0.0
    )  # matches dPCA._fit's own convention
    Omega = np.diag(rebalanced_noise_variance(trialX)) + lam2 * np.eye(n)

    rng = np.random.default_rng(random_state)
    P: dict[str, np.ndarray] = {}
    D: dict[str, np.ndarray] = {}
    for key, mX in dpca_obj._marginalize(Xcen).items():
        mX = mX.reshape(n, -1)
        C = mX @ rX.T @ np.linalg.inv(rX @ rX.T + Omega)
        # A_RR = X_j X^T (XX^T + Omega)^-1 paper formula
        nc = (
            dpca_obj.n_components[key]
            if isinstance(dpca_obj.n_components, dict)
            else dpca_obj.n_components
        )
        U, _, _ = randomized_svd(
            C @ rX,
            n_components=nc,
            n_iter=dpca_obj.n_iter,
            random_state=int(rng.integers(1_000_000)),
        )
        P[key], D[key] = U, (U.T @ C).T

    dpca_obj.P, dpca_obj.D = P, D
    return dpca_obj


def get_var_split(
    dpca_obj: dPCA.dPCA,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-component and per-marginalization variance explained, in percent.

    Parameters
    ----------
    dpca_obj : dPCA.dPCA
        Already fit (``.D`` / ``.P`` populated -- plain fit or after
        `fit_dpca_rebalanced`).
    X : ndarray, shape (n_features, *cond_dims, n_time)
        The condition-mean data the model was fit on.

    Returns
    -------
    componentVar : ndarray, shape (n_components_total,)
        Percent of total PSTH variance per component, sorted descending.
    margVar : ndarray, shape (n_marginalizations, n_components_total)
        Percent of total variance each component explains within each
        marginalization, in the same component order as `componentVar`.
    whichMarg : ndarray, shape (n_components_total,)
        Marginalization key each component was extracted for, in
        `componentVar`'s order.
    order : ndarray of int, shape (n_components_total,)
        Index into the concatenated (unsorted) encoder/decoder columns that
        produces `componentVar`'s order -- reuse it to reorder ``dpca_obj.D``
        / ``dpca_obj.P`` / ``Z`` consistently with the variance ranking.

    Notes
    -----
    Reproduces ``dpca_explainedVariance.m`` verbatim, with ``F`` / ``D`` the
    stacked encoders (``dpca_obj.P``) / decoders (``dpca_obj.D``) across every
    marginalization, ``Z = D.T @ X``, and ``Xmarg_j`` the j-th marginalization
    of ``X``::

        componentVar[i] = 1 - ||X - outer(F[:,i], Z[i,:])||^2 / totalVar
        margVar[j, i]   = (||Xmarg_j||^2
                           - ||Xmarg_j - outer(F[:,i], D[:,i].T @ Xmarg_j)||^2)
                          / totalVar
    """
    keys = list(dpca_obj.marginalizations.keys())
    D = np.hstack([dpca_obj.D[k] for k in keys])
    F = np.hstack([dpca_obj.P[k] for k in keys])
    whichMarg = np.concatenate([[k] * dpca_obj.D[k].shape[1] for k in keys])

    Xflat = X.reshape(X.shape[0], -1)
    totalVar = np.sum(Xflat**2)
    Z = D.T @ Xflat

    comps = F.shape[1]
    componentVar = (
        np.array(
            [
                1 - np.sum((Xflat - np.outer(F[:, i], Z[i, :])) ** 2) / totalVar
                for i in range(comps)
            ]
        )
        * 100
    )
    order = np.argsort(-componentVar)

    Xmargs = dpca_obj._marginalize(X.copy())
    margVar = np.zeros((len(keys), comps))
    for comp in range(comps):
        for k_, k in enumerate(keys):
            recon = np.outer(F[:, comp], D[:, comp] @ Xmargs[k])
            margVar[k_, comp] = (
                (np.sum(Xmargs[k] ** 2) - np.sum((Xmargs[k] - recon) ** 2))
                / totalVar
                * 100
            )

    return componentVar[order], margVar[:, order], whichMarg[order], order


def noise_corrected_marg_var(
    dpca_obj: dPCA.dPCA,
    X: np.ndarray,
    trialX: np.ndarray,
    time_axis: int = -1,
) -> tuple[list[str], np.ndarray]:
    """Per-marginalization *signal* variance fraction -- Kobak et al. 2016, Fig. 3-6d.

    Each marginalization's full variance ``||X_f||^2`` (the marginalized data,
    not dPCA's reduced-rank reconstruction) has its degrees-of-freedom share of
    the residual finite-trial noise subtracted, then is normalised by the total
    signal variance::

        slice_f = (||X_f||^2 - Q_f) / (||X||^2 - Q)
        Q_f     = K_f / (prod(cond_dims) * n_time - 1) * Q

    ``Q`` is the total residual noise sum of squares carried into ``||X||^2`` by
    averaging a finite number of trials (the dashed "signal variance" line of
    Figs 3-6c). ``K_f`` is marginalization ``f``'s degrees of freedom:
    ``n_time - 1`` for the pure-time marginalization, ``n_time *
    prod_{i in f}(cond_dims[i] - 1)`` otherwise (e.g. for ``labels='lat'`` with
    ``L`` levels, ``A`` angles: ``t`` -> ``T-1``, ``lt`` -> ``T(L-1)``, ``at``
    -> ``T(A-1)``, ``lat`` -> ``T(L-1)(A-1)``). The ``K_f`` sum to
    ``prod(cond_dims) * n_time - 1`` and the returned slices sum to 1.

    Parameters
    ----------
    dpca_obj : dPCA.dPCA
        Fitted model. Only ``labels`` / ``marginalizations`` / ``_marginalize``
        are used (not ``.D`` / ``.P``), so the plain and rebalanced fits give
        the same result.
    X : ndarray, shape (n_features, *cond_dims, n_time)
        Centered condition-mean data.
    trialX : ndarray
        NaN-padded single-trial data, for the noise estimate.
    time_axis : int, optional
        Axis of `trialX` that indexes time, by default ``-1``.

    Returns
    -------
    keys : list of str
        Marginalization keys, in ``dpca_obj.marginalizations`` order.
    signal_frac : ndarray, shape (n_marginalizations,)
        Noise-corrected signal-variance fraction per marginalization (sums to 1).
    """
    keys = list(dpca_obj.marginalizations)
    labels = dpca_obj.labels
    time_label = labels[-1]

    total_var = float(np.sum(X.reshape(X.shape[0], -1) ** 2))

    # Q: residual noise sum of squares present in ||X||^2 from finite-trial
    # averaging -- per-neuron noise variance, then divided by mean trials/cell.
    noise_var = rebalanced_noise_variance(trialX, time_axis=time_axis)
    n_per_cell = np.sum(~np.isnan(trialX), axis=0).astype(float)
    t_ax = time_axis if time_axis >= 0 else X.ndim + time_axis
    kbar = np.take(n_per_cell, 0, axis=t_ax).reshape(X.shape[0], -1).mean(axis=1)
    Q = float(np.sum(noise_var / kbar))

    cond_dims = X.shape[1:-1]
    n_time = X.shape[-1]
    total_df = int(np.prod(cond_dims)) * n_time - 1

    Xmargs = dpca_obj._marginalize(X.copy())
    signal_frac = np.empty(len(keys))
    for i, key in enumerate(keys):
        factor_letters = [c for c in key if c != time_label]
        if factor_letters:
            K_f = n_time * int(
                np.prod([cond_dims[labels.index(c)] - 1 for c in factor_letters])
            )
        else:
            K_f = n_time - 1
        Q_f = K_f / total_df * Q
        signal_frac[i] = (np.sum(Xmargs[key] ** 2) - Q_f) / (total_var - Q)

    return keys, signal_frac


class RebalancedDPCA(dPCA.dPCA):
    """`dPCA.dPCA` whose closed-form solve uses the rebalanced ridge.

    Every internal fit -- including the per-split refits inside
    ``dPCA.dPCA.significance_analysis`` -- replaces the package's plain
    ``lambda^2 * I`` penalty with ``Omega = diag(noise_var) + lambda^2 * I``,
    the same penalty as `fit_dpca_rebalanced`. The package's significance
    machinery is otherwise unchanged (train/test split, nearest-centroid
    classification, label-shuffle null, ``n_consecutive`` denoising).

    Constructor arguments are exactly ``dPCA.dPCA``'s (``labels``, ``join``,
    ``n_components``, ``regularizer``, ``copy``, ``n_iter``). Set the extra
    fields below as plain attributes after construction, the same way you set
    ``dPCA.dPCA.protect``.

    Attributes
    ----------
    noise_var : ndarray, shape (n_features,)
        Per-feature noise variance added to the ridge, e.g.
        ``rebalanced_noise_variance(trialX)`` on the full-session ``trialX``.
        Held fixed across every split and shuffle. Must be assigned before any
        fit or ``_fit`` raises.
    protect : list of str
        ``dPCA.dPCA`` only sets this when ``regularizer='auto'``, so assign it
        yourself (e.g. ``model.protect = ['t']``) before fitting.

    See Also
    --------
    rebalanced_significance : wrapper that builds the tensors, sets
        `noise_var` / `protect`, and calls ``significance_analysis``.

    Notes
    -----
    The explicit ``__init__`` (rather than ``*args``) and the ``self.join``
    mirror are needed only to keep scikit-learn's estimator ``repr`` / cloning
    happy: it forbids varargs in an estimator ``__init__`` and expects an
    attribute matching every constructor argument, but ``dPCA.dPCA`` stores
    ``join`` as ``self._join``.
    """

    noise_var: np.ndarray | None = None

    def __init__(
        self,
        labels: str | int | None = None,
        join: dict | None = None,
        n_components: int | dict = 10,
        regularizer: float | str | None = None,
        copy: bool = True,
        n_iter: int = 0,
    ) -> None:
        super().__init__(
            labels=labels,
            join=join,
            n_components=n_components,
            regularizer=regularizer,
            copy=copy,
            n_iter=n_iter,
        )
        self.join = (
            join  # base stores it as self._join; sklearn get_params wants self.join
        )

    def _fit(
        self, X, trialX=None, mXs=None, center=True, SVD=None, optimize=True
    ) -> None:
        """Rebalanced closed-form solve; overrides ``dPCA.dPCA._fit``.

        Mirrors `fit_dpca_rebalanced`: ``C_key = Xmarg_key @ X.T @ inv(X @ X.T
        + diag(noise_var) + lambda^2 * I)``, then a ``randomized_svd`` per
        marginalization. Signature matches the parent so ``fit`` /
        ``fit_transform`` / ``significance_analysis`` call it unchanged.

        With ``regularizer='auto'`` the parent's cross-validated lambda search
        (``_optimize_regularization``, plain-ridge CV, as ``dpca.m`` does) runs
        once on the first fit, sets ``self.regularizer`` to a float, and every
        later fit reuses it.
        """
        if getattr(self, "opt_regularizer_flag", False) and optimize:
            if trialX is None:
                raise ValueError(
                    "regularizer='auto' needs trialX for the cross-validated lambda search"
                )
            self._optimize_regularization(X, trialX)  # sets self.regularizer to a float
        if self.noise_var is None:
            raise AttributeError(
                "RebalancedDPCA.noise_var must be set (e.g. rebalanced_noise_variance(trialX)) "
                "before fitting"
            )
        n = X.shape[0]
        Xc = self._zero_mean(X) if center else X
        rX = Xc.reshape(n, -1)
        lam2 = (self.regularizer * np.sum(rX**2)) ** 2 if self.regularizer else 0.0
        inv = np.linalg.inv(rX @ rX.T + np.diag(self.noise_var) + lam2 * np.eye(n))

        margs = mXs if mXs is not None else self._marginalize(Xc)
        self.P, self.D = {}, {}
        for key, mX in margs.items():
            mX = mX.reshape(n, -1)
            C = mX @ rX.T @ inv
            nc = (
                self.n_components[key]
                if isinstance(self.n_components, dict)
                else self.n_components
            )
            U, _, _ = randomized_svd(
                C @ rX,
                n_components=nc,
                n_iter=self.n_iter,
                random_state=np.random.randint(100_000),
            )
            self.P[key], self.D[key] = U, (U.T @ C).T
