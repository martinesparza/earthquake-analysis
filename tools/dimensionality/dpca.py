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

There is one fitting path: `RebalancedDPCA`. It overrides the package's
``_fit``, so every internal refit -- the cross-validated lambda search and the
per-split refits of ``significance_analysis`` -- uses the rebalanced ridge
too. (An older helper, ``fit_dpca_rebalanced``, patched ``P``/``D`` onto a
plain-ridge fit after the fact, so the lambda search and significance refits
silently stayed plain-ridge; it was removed.) `fit_dpca` is the usual entry
point: solenoid codes in, fitted model out.

Notes
-----
The rebalanced solve (now `RebalancedDPCA._fit`) and
`rebalanced_noise_variance` were checked line-by-line against the MATLAB
source in Octave -- axis directions, leakage, and the noise estimate all
matched -- and `get_var_split` reproduces ``dpca_explainedVariance.m``'s
formulas verbatim. See ``notebooks/behaviour/response-anatomy.ipynb`` for the
original exploration.

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
from joblib import Parallel, delayed
from sklearn.utils.extmath import randomized_svd

from tools.params import Params


def _lookup(table: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function mapping solenoid-direction codes through a Params table.

    Parameters
    ----------
    table : str
        Name of a code -> value dict on ``Params``, e.g. "sol_dir_to_level".

    Returns
    -------
    callable
        Maps an array of codes to an array of values.
    """
    return lambda codes: np.array(
        [getattr(Params, table)[int(c)] for c in codes]
    )


# Solenoid-direction code (0-11, `values_Sol_direction`; there is no `sol_dir` column) ->
# condition-factor lookups, backed by the Params tables. "axis" (30/90/150 deg) and
# "direction" (which end of that axis) split `angle` in two: angle = axis + 180 * direction,
# since the six angles are {30, 90, 150} and their +180 deg opposites.
CONDITION_FACTORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "level": _lookup("sol_dir_to_level"),
    "angle": _lookup("sol_dir_to_angle"),
    "contra_ipsi": _lookup("sol_dir_to_contra_ipsi"),
    "axis": lambda codes: _lookup("sol_dir_to_angle")(codes) % 180,
    "direction": lambda codes: (
        _lookup("sol_dir_to_angle")(codes) // 180
    ).astype(int),
}


def conditions_from_codes(
    codes: Sequence[int] | np.ndarray,
    factors: Sequence[str],
) -> list[np.ndarray]:
    """Map solenoid-direction codes to one condition array per named factor.

    Parameters
    ----------
    codes : sequence of int or np.ndarray
        Raw ``values_Sol_direction`` codes, shape (n_trials,).
    factors : sequence of str
        Keys of ``CONDITION_FACTORS``, in the order passed to the fit.

    Returns
    -------
    list of np.ndarray
        One (n_trials,) array per factor.

    Raises
    ------
    ValueError
        If a factor name is not in ``CONDITION_FACTORS``.
    """
    codes = np.asarray(codes)
    unknown = [f for f in factors if f not in CONDITION_FACTORS]
    if unknown:
        raise ValueError(
            f"unknown condition factor(s) {unknown}; choose from {sorted(CONDITION_FACTORS)}"
        )
    return [CONDITION_FACTORS[f](codes) for f in factors]


def default_join(labels: str, time_label: str = "t") -> dict[str, list[str]]:
    """Join each non-time marginalization with its time-varying twin, e.g. 'la'
    with 'lat'.

    Parameters
    ----------
    labels : str
        dPCA labels ending in the time label, e.g. "lat".
    time_label : str, optional
        Letter of the time factor (default is "t").

    Returns
    -------
    dict of {str : list of str}
        Joined key mapped to the marginalizations folded into it.

    Raises
    ------
    ValueError
        If ``labels`` does not end in ``time_label`` exactly once.
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
    """Arrange trials into dPCA's NaN-padded condition grid and per-feature-
    centred condition means.

    Parameters
    ----------
    X : np.ndarray
        Trials, shape (n_trials, n_features, n_time).
    conditions : sequence of np.ndarray
        One (n_trials,) array per non-time factor.

    Returns
    -------
    trialX : np.ndarray
        Shape (K, n_features, *cond_dims, n_time); NaN-padded to K trials.
    X_psth : np.ndarray
        Centred condition means, shape (n_features, *cond_dims, n_time).
    cond_values : list of list
        Sorted distinct labels of each factor.
    counts : dict of {tuple : int}
        Trial count per populated condition cell.

    Raises
    ------
    ValueError
        If a condition length is not n_trials, or there are no trials.

    Warns
    -----
    RuntimeWarning
        If a feature is all-NaN in ``X_psth`` (centring skipped).
    """
    conditions = [np.asarray(c) for c in conditions]
    n_trials, n_features, n_time = X.shape
    if any(c.shape[0] != n_trials for c in conditions):
        raise ValueError("each condition array must have length n_trials")

    # Grid coordinates: each trial gets one slot per factor (its rank among that factor's
    # sorted distinct values), so its cell is the tuple of slots, e.g. (0, 2).
    cond_values, slots = [], []
    for c in conditions:
        values, slot = np.unique(c, return_inverse=True)
        cond_values.append(values.tolist())
        slots.append(slot)
    cond_dims = tuple(len(v) for v in cond_values)

    trials_by_cell: dict[tuple, list[int]] = {}
    for trial, cell in enumerate(zip(*slots)):
        trials_by_cell.setdefault(cell, []).append(trial)
    if not trials_by_cell:
        raise ValueError("no trials to fit")

    # Empty cells stay all-NaN; others are NaN-padded up to K, the largest cell.
    K = max(len(t) for t in trials_by_cell.values())
    trialX = np.full((K, n_features, *cond_dims, n_time), np.nan)
    for cell, trials in trials_by_cell.items():
        trialX[: len(trials), :, *cell, :] = X[trials]
    counts = {
        tuple(cond_values[f][s] for f, s in enumerate(cell)): len(trials)
        for cell, trials in trials_by_cell.items()
    }

    # Condition mean, centred per feature. A feature that is all-NaN (dead channel, or only
    # in empty cells) cannot be centred: warn and leave it NaN rather than let numpy fail.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Mean of empty slice", RuntimeWarning
        )
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


def rebalanced_noise_variance(
    trialX: np.ndarray, time_axis: int = -1
) -> np.ndarray:
    """Compute per-neuron noise variance with every condition cell weighted
    equally (dpca.m 'averaged').

    Parameters
    ----------
    trialX : np.ndarray
        NaN-padded single-trial data, trial axis first.
    time_axis : int, optional
        Time axis of ``trialX`` (default is -1).

    Returns
    -------
    np.ndarray
        Noise variance, shape (n_features,).
    """
    X = np.nanmean(trialX, axis=0)  # PSTH
    K = np.sum(~np.isnan(trialX), axis=0).astype(
        float
    )  # trials per neuron/cell/time
    K[K == 0] = np.nan
    SS = np.nansum(
        (trialX - X[None, ...]) ** 2, axis=0
    )  # summed sq. deviations per cell

    t_ax = time_axis if time_axis >= 0 else X.ndim + time_axis
    SS_sumT = np.sum(SS, axis=t_ax).reshape(
        X.shape[0], -1
    )  # sum over time first
    K_noT = np.take(K, 0, axis=t_ax).reshape(
        X.shape[0], -1
    )  # trial count is time-invariant

    with np.errstate(invalid="ignore"):
        return np.nansum(SS_sumT / K_noT, axis=1)  # sum, not mean, over cells


def get_var_split(
    dpca_obj: dPCA.dPCA,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute percent variance explained per component and per
    marginalization, sorted by component.

    Parameters
    ----------
    dpca_obj : dPCA.dPCA
        Fitted model with ``D`` and ``P`` populated.
    X : np.ndarray
        Condition means the model was fit on.

    Returns
    -------
    componentVar : np.ndarray
        Percent of total variance per component, descending.
    margVar : np.ndarray
        Percent per marginalization, shape (n_marg, n_components).
    whichMarg : np.ndarray
        Marginalization key of each component.
    order : np.ndarray of int
        Indices that sort the unsorted component columns.
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
                1
                - np.sum((Xflat - np.outer(F[:, i], Z[i, :])) ** 2) / totalVar
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


def cumulative_explained_variance(
    dpca_obj: dPCA.dPCA,
    X: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the cumulative % of ``X`` explained by the first q dPCA and PCA
    components, as in Kobak et al. 2016 Figs 3-6c.

    Parameters
    ----------
    dpca_obj : dPCA.dPCA
        Fitted model with ``D`` and ``P`` populated.
    X : np.ndarray
        Centred condition means the model was fit on.
    n_components : int, optional
        Largest q (default is every dPCA component).

    Returns
    -------
    dpca_cum : np.ndarray
        % explained by the q strongest dPCA components, q = 1..n.
    pca_cum : np.ndarray
        % explained by the first q principal components, q = 1..n.
    """
    keys = list(dpca_obj.marginalizations)
    F = np.hstack([dpca_obj.P[k] for k in keys])
    D = np.hstack([dpca_obj.D[k] for k in keys])
    Xf = X.reshape(X.shape[0], -1)
    total_var = np.sum(Xf**2)

    # first q components in get_var_split's order, reconstructed jointly
    # (dPCA components are not orthogonal, so this is not a sum of single-component values)
    order = get_var_split(dpca_obj, X)[3]
    n = len(order) if n_components is None else min(n_components, len(order))
    dpca_cum = (
        np.array(
            [
                1
                - np.sum(
                    (Xf - F[:, order[:q]] @ (D[:, order[:q]].T @ Xf)) ** 2
                )
                / total_var
                for q in range(1, n + 1)
            ]
        )
        * 100
    )

    # PCA: the error of the first q PCs is the sum of the remaining squared singular values
    s = np.linalg.svd(Xf, compute_uv=False)
    pca_cum = np.cumsum(s**2)[:n] / total_var * 100
    return dpca_cum, pca_cum


def _residual_noise_ss(
    X: np.ndarray, trialX: np.ndarray, time_axis: int = -1
) -> float:
    """Estimate the noise sum of squares left in ``X`` by finite-trial averaging
    (Kobak et al. 2016's Theta).

    Parameters
    ----------
    X : np.ndarray
        Centred condition means, shape (n_features, *cond_dims, n_time).
    trialX : np.ndarray
        NaN-padded single-trial data, trial axis first.
    time_axis : int, optional
        Time axis of ``X`` (default is -1).

    Returns
    -------
    float
        Theta = sum over features of noise_var / mean trials per cell.
    """
    # Theta = sum_n noise_var_n / K_n; noise_var is already summed over cells and time, so
    # only the division by K_n (mean trials per cell, empty cells counted as 0) remains
    noise_var = rebalanced_noise_variance(trialX, time_axis=time_axis)
    n_per_cell = np.sum(~np.isnan(trialX), axis=0).astype(float)
    t_ax = time_axis if time_axis >= 0 else X.ndim + time_axis
    kbar = (
        np.take(n_per_cell, 0, axis=t_ax).reshape(X.shape[0], -1).mean(axis=1)
    )
    return float(np.sum(noise_var / kbar))


def signal_variance_fraction(
    X: np.ndarray, trialX: np.ndarray, time_axis: int = -1
) -> float:
    """Estimate the fraction of the variance of ``X`` that is signal rather than
    averaging noise, 1 - Theta / ||X||^2 (Kobak et al. 2016 Figs 3-6c, dashed).

    Parameters
    ----------
    X : np.ndarray
        Centred condition means, shape (n_features, *cond_dims, n_time).
    trialX : np.ndarray
        NaN-padded single-trial data for the noise estimate.
    time_axis : int, optional
        Time axis of ``X`` (default is -1).

    Returns
    -------
    float
        Signal fraction of ``||X||^2``, between 0 and 1 (can dip below 0).
    """
    return 1 - _residual_noise_ss(X, trialX, time_axis) / float(np.sum(X**2))


def noise_corrected_marg_var(
    dpca_obj: dPCA.dPCA,
    X: np.ndarray,
    trialX: np.ndarray,
    time_axis: int = -1,
) -> tuple[list[str], np.ndarray]:
    """Compute each marginalization's noise-corrected signal-variance fraction
    (Kobak et al. 2016).

    Parameters
    ----------
    dpca_obj : dPCA.dPCA
        Fitted model; only ``labels`` and marginalizations are used.
    X : np.ndarray
        Centred condition means, shape (n_features, *cond_dims, n_time).
    trialX : np.ndarray
        NaN-padded single-trial data for the noise estimate.
    time_axis : int, optional
        Time axis of ``trialX`` (default is -1).

    Returns
    -------
    keys : list of str
        Marginalization keys.
    signal_frac : np.ndarray
        Signal-variance fraction per key; sums to 1.
    """
    keys = list(dpca_obj.marginalizations)
    labels = dpca_obj.labels
    time_label = labels[-1]

    total_var = float(np.sum(X.reshape(X.shape[0], -1) ** 2))
    Q = _residual_noise_ss(
        X, trialX, time_axis
    )  # Theta: averaging noise left in ||X||^2

    cond_dims = X.shape[1:-1]
    n_time = X.shape[-1]
    total_df = int(np.prod(cond_dims)) * n_time - 1

    Xmargs = dpca_obj._marginalize(X.copy())
    signal_frac = np.empty(len(keys))
    for i, key in enumerate(keys):
        factor_letters = [c for c in key if c != time_label]
        if factor_letters:
            K_f = n_time * int(
                np.prod(
                    [cond_dims[labels.index(c)] - 1 for c in factor_letters]
                )
            )
        else:
            K_f = n_time - 1
        Q_f = K_f / total_df * Q
        signal_frac[i] = (np.sum(Xmargs[key] ** 2) - Q_f) / (total_var - Q)

    return keys, signal_frac


class RebalancedDPCA(dPCA.dPCA):
    """`dPCA.dPCA` whose closed-form solve uses the rebalanced ridge of dpca.m.

    The penalty is ``Omega = diag(noise_var) + (lambda * totalVar)^2 * I``
    instead of the package's plain identity term. Because ``_fit`` itself is
    overridden, every fit the package runs internally -- the cross-validated
    search when ``regularizer='auto'``, and the per-split refits inside
    ``significance_analysis`` -- uses this penalty as well.

    Usage is the package's: ``RebalancedDPCA(labels=..., ...).fit(X, trialX)``.
    The two manual steps the package needs are done for you:

    - ``noise_var`` is recomputed from ``trialX`` whenever a fit is given
      ``trialX`` (i.e. every public ``fit(X, trialX)``). Internal refits get
      no ``trialX`` and reuse it, so it stays fixed at the full-data value
      across CV splits and significance splits -- test trials therefore help
      set the penalty. This matches how dpca.m is normally used.
    - ``protect`` is set to the time label, so CV splits never shuffle time
      bins within a trial (the package leaves it ``None`` for 'auto').

    Attributes
    ----------
    noise_var : ndarray, shape (n_features,)
        Per-feature noise variance from `rebalanced_noise_variance`; set by
        the last fit that was given ``trialX``.
    protect : list of str
        Axes kept intact when splitting trials; ``[labels[-1]]``.

    Notes
    -----
    The explicit ``__init__`` (rather than ``*args``) and the ``self.join`` /
    ``self.random_state`` attributes only keep scikit-learn's estimator
    ``repr`` / cloning happy: it forbids varargs in ``__init__`` and expects an
    attribute per constructor argument, but ``dPCA.dPCA`` stores ``join`` as
    ``self._join``.
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
        random_state: int | None = None,
    ) -> None:
        """Initialise with ``dPCA.dPCA``'s arguments plus an SVD seed.

        Parameters
        ----------
        labels : str, int or None, optional
            Factor letters, time last (default is None).
        join : dict, optional
            Marginalizations to merge (default is None).
        n_components : int or dict, optional
            Components per marginalization (default is 10).
        regularizer : float or str, optional
            Ridge weight, or "auto" (default is None, i.e. 0).
        copy : bool, optional
            Passed to ``dPCA.dPCA`` (default is True).
        n_iter : int, optional
            Randomized-SVD power iterations (default is 0).
        random_state : int, optional
            Seed for the randomized SVD (default is None, unseeded).
        """
        super().__init__(
            labels=labels,
            join=join,
            n_components=n_components,
            regularizer=regularizer,
            copy=copy,
            n_iter=n_iter,
        )
        self.join = join  # base stores it as self._join; sklearn get_params wants self.join
        self.random_state = random_state
        self.protect = [
            self.labels[-1]
        ]  # after super(): it resets protect to None for 'auto'

    def _fit(
        self, X, trialX=None, mXs=None, center=True, SVD=None, optimize=True
    ) -> None:
        """Fit ``P`` and ``D`` with the rebalanced ridge, overriding the package.

        Parameters
        ----------
        X : np.ndarray
            Condition means, shape (n_features, *cond_dims, n_time).
        trialX : np.ndarray, optional
            Single-trial data; refreshes ``noise_var`` when given.
        mXs : dict, optional
            Precomputed marginalizations of ``X`` (default is None).
        center : bool, optional
            Zero-mean ``X`` first (default is True).
        SVD : object, optional
            Unused; kept for the package's signature.
        optimize : bool, optional
            Run the lambda search when "auto" (default is True).

        Raises
        ------
        ValueError
            If no ``noise_var`` exists yet and ``trialX`` is None.
        """
        if trialX is not None:
            self.noise_var = rebalanced_noise_variance(trialX)
        if self.noise_var is None:
            raise ValueError(
                "the first fit needs trialX, to estimate noise_var"
            )
        if self.opt_regularizer_flag and optimize:
            self._optimize_regularization(
                X, trialX
            )  # refits via _fit; sets self.regularizer

        n = X.shape[0]
        Xc = self._zero_mean(X) if center else X
        rX = Xc.reshape(n, -1)
        lam = self.regularizer
        lam2 = (lam * np.sum(rX**2)) ** 2 if lam else 0.0
        inv = np.linalg.inv(
            rX @ rX.T + np.diag(self.noise_var) + lam2 * np.eye(n)
        )

        # one seed per fit, so a seeded model refits identically on every split
        rng = np.random.default_rng(self.random_state)
        self.P, self.D = {}, {}
        for key, mX in (
            mXs if mXs is not None else self._marginalize(Xc)
        ).items():
            C = (
                mX.reshape(n, -1) @ rX.T @ inv
            )  # A_RR = X_j X^T (X X^T + Omega)^-1
            nc = (
                self.n_components[key]
                if isinstance(self.n_components, dict)
                else self.n_components
            )
            U, _, _ = randomized_svd(
                C @ rX,
                n_components=nc,
                n_iter=self.n_iter,
                random_state=int(rng.integers(1_000_000)),
            )
            self.P[key], self.D[key] = U, (U.T @ C).T


def fit_dpca(
    X: np.ndarray,
    codes: Sequence[int] | np.ndarray,
    factors: Sequence[str],
    labels: str,
    regularizer: float | str,
    n_components: int,
    random_state: int | None = None,
) -> tuple[
    RebalancedDPCA, np.ndarray, np.ndarray, list[list], dict[tuple, int]
]:
    """Fit the rebalanced dPCA to trials labelled by solenoid code.

    Parameters
    ----------
    X : np.ndarray
        Trials, shape (n_trials, n_features, n_time).
    codes : sequence of int or np.ndarray
        Solenoid-direction code of each trial, shape (n_trials,).
    factors : sequence of str
        Keys of ``CONDITION_FACTORS``, one per non-time letter of ``labels``.
    labels : str
        dPCA labels, ending in the time letter, e.g. "lpt".
    regularizer : float or str
        Ridge weight, or "auto" to search for it by cross-validation.
    n_components : int
        Components per marginalization.
    random_state : int, optional
        Seed for the randomized SVD (default is None, unseeded).

    Returns
    -------
    model : RebalancedDPCA
        The fitted model.
    X_psth : np.ndarray
        Centred condition means.
    trialX : np.ndarray
        NaN-padded single-trial tensor.
    cond_values : list of list
        Sorted distinct labels of each factor.
    counts : dict of {tuple : int}
        Trial count per populated condition cell.
    """
    trialX, X_psth, cond_values, counts = build_unbalanced_trial_tensor(
        X, conditions_from_codes(codes, factors)
    )
    model = RebalancedDPCA(
        labels=labels,
        regularizer=regularizer,
        join=default_join(labels),
        n_components=n_components,
        random_state=random_state,
    )
    model.fit(X_psth, trialX)
    return model, X_psth, trialX, cond_values, counts


def shuffled_variance(
    X: np.ndarray,
    codes: Sequence[int] | np.ndarray,
    factors: Sequence[str],
    labels: str,
    regularizer: float,
    n_components: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Refit with the given (e.g. shuffled) codes and return each
    marginalization's variance.

    Parameters
    ----------
    X : np.ndarray
        Trials, shape (n_trials, n_features, n_time).
    codes : sequence of int or np.ndarray
        Solenoid-direction codes to fit with, shape (n_trials,).
    factors : sequence of str
        Keys of ``CONDITION_FACTORS``, one per non-time letter of ``labels``.
    labels : str
        dPCA labels, ending in the time letter, e.g. "lpt".
    regularizer : float
        Ridge weight, already resolved (not "auto").
    n_components : int
        Components per marginalization.
    random_state : int, optional
        Seed for the randomized SVD (default is None, unseeded).

    Returns
    -------
    np.ndarray
        Percent of total variance per marginalization, shape (n_marg,).
    """
    model, X_psth, *_ = fit_dpca(
        X, codes, factors, labels, regularizer, n_components, random_state
    )
    return get_var_split(model, X_psth)[1].sum(axis=1)


def noise_floor(
    X: np.ndarray,
    codes: Sequence[int] | np.ndarray,
    factors: Sequence[str],
    labels: str,
    regularizer: float | str = "auto",
    n_components: int = 10,
    n_shuffles: int = 1000,
    seed: int = 0,
    n_jobs: int = -1,
) -> tuple[dict, dict]:
    """Compare each marginalization's variance with a label-shuffle null,
    searching lambda once on the real data.

    Parameters
    ----------
    X : np.ndarray
        Trials, shape (n_trials, n_features, n_time).
    codes : sequence of int or np.ndarray
        Solenoid-direction code of each trial, shape (n_trials,).
    factors : sequence of str
        Keys of ``CONDITION_FACTORS``, one per non-time letter of ``labels``.
    labels : str
        dPCA labels, ending in the time letter, e.g. "lpt".
    regularizer : float or str, optional
        Ridge weight, or "auto" for the CV search (default is "auto").
    n_components : int, optional
        Components per marginalization (default is 10).
    n_shuffles : int, optional
        Number of label permutations (default is 1000).
    seed : int, optional
        Seed for the permutations and the SVD (default is 0).
    n_jobs : int, optional
        Parallel workers for the shuffles (default is -1, all cores).

    Returns
    -------
    stats : dict
        keys, observed, signal_frac, null, null_median, null_p95, p
        (``(1 + #{null >= obs}) / (n_shuffles + 1)``), z
        (``(obs - null.mean) / null.std``), n_trials, cell_counts,
        regularizer_used.
    fit : dict
        dpca, X_psth, trialX, codes, cond_values and config.
    """
    model, X_psth, trialX, cond_values, counts = fit_dpca(
        X, codes, factors, labels, regularizer, n_components, random_state=seed
    )
    reg = float(model.regularizer)  # i will reuse lambda in the null
    observed = get_var_split(model, X_psth)[1].sum(axis=1)

    rng = np.random.default_rng(seed)
    perms = [rng.permutation(codes) for _ in range(n_shuffles)]
    null = np.array(
        Parallel(n_jobs=n_jobs, inner_max_num_threads=1)(
            delayed(shuffled_variance)(
                X, perm, factors, labels, reg, n_components, seed
            )
            for perm in perms
        )
    )

    stats = dict(
        keys=np.array(list(model.marginalizations)),
        observed=observed,
        signal_frac=noise_corrected_marg_var(model, X_psth, trialX)[1],
        null=null,
        null_median=np.median(null, axis=0),
        null_p95=np.percentile(null, 95, axis=0),
        p=(1 + (null >= observed).sum(axis=0)) / (n_shuffles + 1),
        z=(observed - null.mean(axis=0)) / null.std(axis=0),
        n_trials=len(codes),
        cell_counts=np.array(sorted(counts.values())),
        regularizer_used=reg,
    )
    fit = dict(
        dpca=model,
        X_psth=X_psth,
        trialX=trialX,
        codes=np.asarray(codes),
        cond_values=cond_values,
        config=dict(
            factors=list(factors),
            labels=labels,
            regularizer=regularizer,
            regularizer_used=reg,
            n_components=n_components,
            n_shuffles=n_shuffles,
            seed=seed,
        ),
    )
    return stats, fit
