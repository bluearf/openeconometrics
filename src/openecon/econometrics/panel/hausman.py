"""Hausman specification test (Stata's hausman).

For a consistent estimator b_c (fixed effects) and an estimator b_e that is
efficient under the null hypothesis (random effects), over the common
non-constant coefficients,

    d = b_c - b_e,   V = V_c - V_e,   H = d' V^- d ~ chi2(rank V).

V^- is the generalized inverse built from the eigenvalues of V (scaled to a
correlation-like matrix so the cutoff is unit free); eigenvalues below
1e-12 of the largest in magnitude are dropped and rank V is the number kept.
As in Stata the negative eigenvalues of a non-positive-definite V are NOT
discarded, so the statistic can be negative; the result then carries no
p-value and Stata's note.

``sigmamore`` puts both covariances on the efficient model's disturbance
variance: V_c is rescaled by sigma_e^2 / sigma_c^2 (V_e already rests on
sigma_e^2); ``sigmaless`` puts both on the consistent model's variance by
rescaling V_e by sigma_c^2 / sigma_e^2 (help hausman). The disturbance
variance is read as Stata's hausman reads it: ``e(sigma_e)`` (``metrics
['sigma_e']``) after ``xtreg, fe`` and ``xtreg, mle``, ``e(rmse)``
(``metrics['rmse']``) after ``xtreg, re`` and after every other estimator
(``regress``, ``xtreg, be``, first differences, pooled OLS). For fe versus re
``sigmamore`` therefore compares sigma_e^2 (X_w'X_w)^-1 with rmse^2 (X*'X*)^-1
rescaled to the same rmse^2, which makes V_c - V_e positive semidefinite.

The statistic is a chi2 only when both covariances are the conventional
(model-based) ones. As Stata does ("hausman cannot be used with vce(robust),
vce(cluster cvar), or p-weighted data"), fits with a robust, cluster,
Driscoll-Kraay or resampling covariance or with sampling weights are refused
unless ``force=True`` (Stata's ``force`` option), which computes the statistic
as is and says so in the note.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import chi2_sf
from openecon.engines.linalg import symmetrize
from openecon.models import ResultBundle

_TOL = 1e-12
_SIGMA_E_MODELS = {"fe", "mle"}      # xtreg models that store e(sigma_e); re stores e(rmse)
_MODEL_BASED = {"nonrobust", "opg"}  # covariances under which the Hausman statistic is a chi2


def _model_based(bundle: ResultBundle) -> bool:
    """Whether a fit carries a conventional covariance and no sampling weights."""
    return bundle.spec.covariance in _MODEL_BASED and bundle.spec.weight_type != "pweight"


def _sigma2(bundle: ResultBundle, role: str) -> float:
    """The disturbance variance Stata's hausman reads for this fit (see module docstring)."""
    spec = bundle.spec
    if spec.estimator == "xtreg" and spec.options.get("model", "fe") in _SIGMA_E_MODELS:
        keys = ("sigma_e", "rmse")
    else:
        keys = ("rmse", "sigma_e")
    value = next((bundle.metrics.get(key) for key in keys if bundle.metrics.get(key) is not None),
                 None)
    if value is None or value <= 0:
        raise AnalysisError("missing_sigma", f"The {role} model reports neither rmse nor sigma_e; "
                            "sigmamore/sigmaless need the disturbance variance.")
    return value * value


def _block(bundle: ResultBundle, terms: list[str]) -> tuple[Tensor, Tensor]:
    index = {c.term: i for i, c in enumerate(bundle.coefficients)}
    rows = [index[term] for term in terms]
    estimates = torch.tensor([bundle.coefficients[i].estimate for i in rows], dtype=torch.float64)
    covariance = torch.tensor(bundle.covariance_matrix, dtype=torch.float64)[rows][:, rows]
    return estimates, covariance


def _generalized_quadratic(d: Tensor, v: Tensor) -> tuple[float, int, bool]:
    """(d' V^- d, rank V, V has a negative eigenvalue) with a unit-free eigenvalue cutoff."""
    scale = v.diagonal().abs().sqrt()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    try:
        values, vectors = torch.linalg.eigh(symmetrize(v / scale[:, None] / scale))
    except RuntimeError as exc:
        raise AnalysisError("numerical_failure", f"The eigen-decomposition failed: {exc}") from exc
    largest = float(values.abs().max())
    if largest <= 0:
        raise AnalysisError("singular_covariance", "The covariance difference is zero.")
    keep = values.abs() > _TOL * largest
    rotated = vectors[:, keep].T @ (d / scale)
    statistic = float((rotated.square() / values[keep]).sum())
    return statistic, int(keep.sum()), bool((values[keep] < 0).any())


def hausman(consistent: ResultBundle, efficient: ResultBundle, *, sigmamore: bool = False,
            sigmaless: bool = False, alpha: float = 0.05, force: bool = False) -> dict[str, Any]:
    """Hausman test comparing a consistent and an efficient fit (Stata's ``hausman``).

    ``consistent`` is typically ``xtreg(model='fe')`` and ``efficient``
    ``xtreg(model='re')``; any two ResultBundles sharing coefficient names
    qualify, provided both carry their conventional covariance
    (``covariance='nonrobust'``): with a robust, cluster or Driscoll-Kraay
    covariance or sampling weights the statistic is not a chi2 and, like
    Stata, the function raises ``unsupported_covariance`` unless
    ``force=True``. The comparison runs over the common terms other than
    ``Intercept`` and ancillary ``/...`` parameters. ``sigmamore`` and
    ``sigmaless`` (mutually exclusive) put both covariances on one
    disturbance variance, the efficient model's (``sigmamore``: V_c is
    rescaled by sigma_e^2/sigma_c^2) or the consistent model's
    (``sigmaless``: V_e is rescaled by sigma_c^2/sigma_e^2), where sigma^2 is
    ``metrics['sigma_e']`` for ``xtreg`` fe/mle fits (Stata's ``e(sigma_e)``)
    and ``metrics['rmse']`` for ``xtreg, re`` and every other estimator
    (Stata's ``e(rmse)``). Stata recommends ``sigmamore`` for fe-versus-re
    comparisons: the difference matrix is then positive semidefinite.

    Returns a dict with ``statistic``, ``df`` (rank of V_c - V_e), ``p_value``
    (``None`` when the statistic is negative), ``distribution`` ``'chi2'``,
    ``terms``, ``difference`` (b_c - b_e), ``se_difference``
    (sqrt(diag(V_c - V_e)), ``None`` where negative), ``negative_definite``
    (True when V_c - V_e is not positive semidefinite), ``note`` (Stata's
    warning text or ``None``), ``sigma`` (the option used), ``sigma2``
    (the two disturbance variances read), ``forced`` (True when ``force``
    overrode the covariance requirement), ``reject`` at level ``alpha`` and
    the two estimator labels.

    Example::

        fe = oe.xtreg(data=df, y="y", x=["x1", "x2"], panel="id")
        re = oe.xtreg(data=df, y="y", x=["x1", "x2"], panel="id", model="re")
        oe.hausman(fe, re, sigmamore=True)["p_value"]
    """
    for role, bundle in (("consistent", consistent), ("efficient", efficient)):
        if not isinstance(bundle, ResultBundle):
            raise AnalysisError("invalid_spec", f"The {role} argument must be a ResultBundle.")
    if sigmamore and sigmaless:
        raise AnalysisError("invalid_spec", "sigmamore and sigmaless are mutually exclusive.")
    if not (isinstance(alpha, float) and 0 < alpha < 1):
        raise AnalysisError("invalid_spec", "alpha must lie strictly between zero and one.")
    forced = False
    for role, bundle in (("consistent", consistent), ("efficient", efficient)):
        if _model_based(bundle):
            continue
        if not force:
            weights = " with sampling weights" if bundle.spec.weight_type == "pweight" else ""
            raise AnalysisError(
                "unsupported_covariance",
                f"hausman cannot be used with a robust, cluster or Driscoll-Kraay covariance or "
                f"with sampling weights: the {role} fit uses covariance="
                f"'{bundle.spec.covariance}'{weights}. The statistic is a chi2 only for the "
                "conventional covariance of both fits (Stata's hausman refuses these too). Refit "
                "with covariance='nonrobust', or pass force=True to compute it regardless.")
        forced = True
    efficient_terms = {c.term for c in efficient.coefficients}
    terms = [c.term for c in consistent.coefficients
             if c.term in efficient_terms and c.term != "Intercept" and not c.term.startswith("/")]
    if not terms:
        raise AnalysisError("no_common_terms", "The two fits share no non-constant coefficient.")
    b_c, v_c = _block(consistent, terms)
    b_e, v_e = _block(efficient, terms)
    sigma, sigma2 = None, None
    if sigmamore or sigmaless:
        s2_c, s2_e = _sigma2(consistent, "consistent"), _sigma2(efficient, "efficient")
        sigma2 = {"consistent": s2_c, "efficient": s2_e}
        if sigmamore:
            v_c, sigma = v_c * (s2_e / s2_c), "sigmamore"
        else:
            v_e, sigma = v_e * (s2_c / s2_e), "sigmaless"
    difference = b_c - b_e
    v = symmetrize(v_c - v_e)
    statistic, rank, negative = _generalized_quadratic(difference, v)
    p_value = chi2_sf(statistic, rank) if statistic >= 0 else None
    note = None
    if statistic < 0:
        note = ("chi2 < 0: the model fitted on these data fails to meet the asymptotic "
                "assumptions of the Hausman test; try sigmamore=True.")
    elif negative:
        note = ("V_c - V_e is not positive definite; the test is reported as computed. "
                "Consider sigmamore=True.")
    if forced:
        caveat = ("forced: at least one fit does not carry its conventional covariance, so the "
                  "chi2 reference distribution is not justified.")
        note = caveat if note is None else f"{note} {caveat}"
    diagonal = v.diagonal().tolist()
    return {
        "statistic": statistic, "df": rank, "p_value": p_value, "distribution": "chi2",
        "label": "Hausman test", "terms": terms, "difference": difference.tolist(),
        "se_difference": [math.sqrt(d) if d > 0 else None for d in diagonal],
        "negative_definite": negative, "note": note, "sigma": sigma, "sigma2": sigma2,
        "forced": forced,
        "alpha": alpha, "reject": None if p_value is None else bool(p_value < alpha),
        "consistent": f"{consistent.spec.estimator}:{consistent.spec.options.get('model', '')}",
        "efficient": f"{efficient.spec.estimator}:{efficient.spec.options.get('model', '')}",
    }
