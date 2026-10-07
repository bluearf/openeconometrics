"""Stochastic production and cost frontiers by maximum likelihood: Stata's ``frontier``.

Model
-----
``y_i = x_i'b + v_i - s u_i`` with ``s = 1`` for a production frontier and
``s = -1`` for a cost frontier (``cost=True``), noise ``v ~ N(0, sigma_v^2)``
and one-sided inefficiency ``u >= 0`` independent of ``v``:

* ``hnormal``: ``u ~ N+(0, sigma_u^2)``, estimated as ``/lnsig2v``, ``/lnsig2u``;
* ``exponential``: ``u ~ Exp`` with mean ``sigma_u``, same parameters;
* ``tnormal``: ``u ~ N+(mu, sigma_u^2)``, estimated as ``/mu``, ``/lnsigma2``
  (``sigma^2 = sigma_u^2 + sigma_v^2``) and ``/ilgtgamma``
  (``logit(gamma)``, ``gamma = sigma_u^2 / sigma^2``), Stata's parameterization.

Estimation
----------
Newton-Raphson with the analytic score and Hessian (``frontier_kernels``) on
regressors centred at their means (the constant is mapped back exactly),
started at OLS with the method-of-moments ``sigma_u`` from the third moment of
the OLS residuals and the constant shifted by ``s E[u]``. OLS residuals whose
skewness has the wrong sign (positive for a production frontier) mean that
the likelihood is maximized at ``sigma_u = 0``: ``boundary_solution`` is raised,
as is a fit that converges to ``sigma_u^2 / sigma_v^2`` or ``sigma_v^2 / sigma_u^2``
below ``1e-8`` (or ``gamma`` within ``1e-10`` of 0 or 1), and a run that did not
converge while drifting to the edge (ratio or ``gamma`` beyond ``1e-6``, or a
truncated normal with ``mu/sigma_u < -10``, its exponential limit). A constant
outcome raises ``constant_outcome``, an exact OLS fit ``perfect_fit``.

Reported
--------
z inference; ``metrics``: ``log_likelihood``, ``aic``, ``bic``, ``sigma_v``,
``sigma_u``, ``sigma2``, ``lambda`` (``sigma_u / sigma_v``) for hnormal and
exponential, ``sigma2``, ``gamma``, ``sigma_u2``, ``sigma_v2`` for tnormal, with
delta-method standard errors in ``extra['ancillary']``; ``tests['model']`` the
Wald chi2 test of the slopes; ``tests['sigma_u']`` the likelihood-ratio test of
``sigma_u = 0`` against OLS, ``chibar2(01)`` with ``p = Pr(chi2(1) > LR) / 2``
(conventional and OPG covariances, no pweights).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, kernel_call, table, wald_test,
)
from openecon.econometrics.discrete.common import (
    build_spec, centre_regressors, check_pweights, constant_shift, likelihood_covariance,
    likelihood_weights, maximize, optimizer_record, require_observations, resolve_covariance,
)
from openecon.econometrics.discrete.kernels import mills_ratio
from openecon.econometrics.systems.frontier_kernels import (
    ANCILLARIES, FrontierObjective, conditional_inefficiency,
)
from openecon.engines.distributions import chi2_sf, normal_ppf
from openecon.engines.linalg import least_squares
from openecon.models import ModelSpec, ResultBundle

_BOUNDARY = 1e-8
# A run that did not converge with a variance ratio (or gamma) beyond this is drifting to the edge.
_DRIFT = 1e-6
# ... and a truncated normal whose mu/sigma_u fell below this is drifting to the exponential limit.
_MU_DRIFT = -10.0
# sum w e^2 of the OLS start below this fraction of sum w y^2 is rounding: an exact fit.
_EXACT_FIT = 1e-24
_TITLES = {"hnormal": "half-normal", "exponential": "exponential", "tnormal": "truncated-normal"}


def _boundary(message: str) -> AnalysisError:
    return AnalysisError("boundary_solution", message + " The stochastic frontier reduces to a "
                         "model without inefficiency (estimate it by OLS), or the inefficiency "
                         "distribution does not fit these data.")


def _wrong_skew(sign: float) -> str:
    side = "positive" if sign > 0 else "negative"
    frontier = "production" if sign > 0 else "cost"
    return f"The OLS residuals have {side} skewness, the wrong sign for a {frontier} frontier"


def _start(x: Tensor, y: Tensor, weights: Tensor, distribution: str, sign: float,
           intercept: bool) -> tuple[Tensor, float, float, float, bool]:
    """Method-of-moments starting values ``(b, sigma_v^2, sigma_u^2)``, the OLS ML variance
    and whether the OLS residuals are skewed the wrong way."""
    root = weights.sqrt()
    fit = kernel_call(least_squares, x * root[:, None], y * root, drop_collinear=False)
    resid = y - x @ fit.beta
    total = float(weights.sum())
    if bool((y == y[0]).all()):
        raise AnalysisError("constant_outcome", "The outcome has no variation in the estimation "
                            "sample; a frontier cannot be estimated.")
    if float(weights @ resid.square()) <= _EXACT_FIT * float(weights @ y.square()):
        raise AnalysisError("perfect_fit", "The regressors fit the outcome exactly, so neither "
                            "noise nor inefficiency can be estimated. Remove the regressor that "
                            "reproduces the outcome.")
    centre = float(weights @ resid) / total
    m2 = float(weights @ (resid - centre).square()) / total
    m3 = float(weights @ (resid - centre).pow(3)) / total
    skew = -sign * m3
    beta = fit.beta.clone()
    if skew <= 0 and distribution != "tnormal":
        raise _boundary(f"{_wrong_skew(sign)}, so the maximum-likelihood estimate of sigma_u "
                        "is zero.")
    wrong = skew <= 0
    skew = max(skew, 1e-6 * m2 ** 1.5)
    if distribution == "exponential":
        sigma_u = (skew / 2) ** (1 / 3)
        sigma_u2, mean_u = sigma_u ** 2, sigma_u
    else:
        sigma_u = (skew / (math.sqrt(2 / math.pi) * (4 / math.pi - 1))) ** (1 / 3)
        sigma_u2, mean_u = sigma_u ** 2, sigma_u * math.sqrt(2 / math.pi)
    variance_u = sigma_u2 if distribution == "exponential" else (1 - 2 / math.pi) * sigma_u2
    if variance_u >= 0.95 * m2:
        scale = 0.5 * m2 / variance_u
        sigma_u2, mean_u, variance_u = sigma_u2 * scale, mean_u * math.sqrt(scale), 0.5 * m2
    sigma_v2 = m2 - variance_u
    if intercept:
        beta[0] = beta[0] + sign * mean_u
    return beta, sigma_v2, sigma_u2, float(weights @ resid.square()) / total, wrong


def _ancillary_start(distribution: str, sigma_v2: float, sigma_u2: float) -> Tensor:
    if distribution == "tnormal":
        total = sigma_u2 + sigma_v2
        gamma = min(max(sigma_u2 / total, 0.05), 0.95)
        return torch.tensor([0.0, math.log(total), math.log(gamma / (1 - gamma))],
                            dtype=torch.float64)
    return torch.tensor([math.log(sigma_v2), math.log(sigma_u2)], dtype=torch.float64)


def _transforms(distribution: str, anc: Tensor) -> dict[str, tuple[float, Tensor]]:
    """Derived variance parameters and their gradients with respect to the ancillaries."""
    def vector(*values: float) -> Tensor:
        return torch.tensor(values, dtype=torch.float64)

    if distribution == "tnormal":
        t, g = float(anc[1]), float(anc[2])
        sigma2 = math.exp(t)
        gamma = 1 / (1 + math.exp(-g))
        dg = gamma * (1 - gamma)
        return {"sigma2": (sigma2, vector(0.0, sigma2, 0.0)),
                "gamma": (gamma, vector(0.0, 0.0, dg)),
                "sigma_u2": (gamma * sigma2, vector(0.0, gamma * sigma2, dg * sigma2)),
                "sigma_v2": ((1 - gamma) * sigma2,
                             vector(0.0, (1 - gamma) * sigma2, -dg * sigma2))}
    p, q = float(anc[0]), float(anc[1])
    sigma_v, sigma_u = math.exp(p / 2), math.exp(q / 2)
    ratio = sigma_u / sigma_v
    return {"sigma_v": (sigma_v, vector(sigma_v / 2, 0.0)),
            "sigma_u": (sigma_u, vector(0.0, sigma_u / 2)),
            "sigma2": (sigma_v ** 2 + sigma_u ** 2, vector(sigma_v ** 2, sigma_u ** 2)),
            "lambda": (ratio, vector(-ratio / 2, ratio / 2))}


def _check_boundary(distribution: str, anc: Tensor, bound: float = _BOUNDARY) -> None:
    """Refuse variance ratios at the edge of the parameter space.

    ``bound`` is ``1e-8`` for a converged fit (``1e-10`` for gamma) and the looser
    ``_DRIFT`` for a run that stopped while still drifting towards the edge.
    """
    if distribution == "tnormal":
        gamma = float(torch.sigmoid(anc[2]))
        edge = bound * 1e-2 if bound == _BOUNDARY else bound
        if gamma < edge or gamma > 1 - edge:
            raise _boundary(f"The estimate of gamma = sigma_u^2/sigma^2 is at the boundary "
                            f"({gamma:.3g}).")
        return
    log_ratio = float(anc[1] - anc[0])
    if log_ratio < math.log(bound):
        raise _boundary("The estimate of sigma_u^2 is zero relative to sigma_v^2.")
    if log_ratio > -math.log(bound):
        raise _boundary("The estimate of sigma_v^2 is zero relative to sigma_u^2 (a "
                        "deterministic frontier).")


def fit_frontier(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``frontier``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    distribution, cost = frame.option("distribution"), bool(frame.option("cost"))
    sign = -1.0 if cost else 1.0
    weights = likelihood_weights(frame)
    design = frame.drop_collinear(frame.design(), weights.for_screen())
    k = len(design.terms)
    if k == 0:
        raise AnalysisError("no_parameters", "frontier: no regressor is left; include a constant "
                            "or a regressor that varies.")
    m = len(ANCILLARIES[distribution])
    require_observations(weights.nobs, k + m)
    y = frame.numeric(spec.outcome)
    means = centre_regressors(design, weights)
    beta0, sigma_v2, sigma_u2, sigma2_ols, wrong = _start(design.x, y, weights.user,
                                                          distribution, sign, design.intercept)
    objective = FrontierObjective(design.x, y, weights.user, distribution, cost)
    start = torch.cat([beta0, _ancillary_start(distribution, sigma_v2, sigma_u2)])
    result = maximize(objective, start, what="frontier", scale=weights.scale)
    anc = result.theta[k:]
    _check_boundary(distribution, anc)
    if not result.converged:
        _check_boundary(distribution, anc, _DRIFT)
        if distribution == "tnormal":
            sigma_u = math.sqrt(math.exp(float(anc[1])) * float(torch.sigmoid(anc[2])))
            if float(anc[0]) / sigma_u < _MU_DRIFT:
                raise _boundary(f"The estimate of mu is diverging to minus infinity (mu/sigma_u "
                                f"= {float(anc[0]) / sigma_u:.3g}): the truncated-normal "
                                "inefficiency degenerates to an exponential distribution and "
                                "the likelihood has no interior maximum; use "
                                "distribution='exponential'.")
        hint = (f" {_wrong_skew(sign)}: the truncated-normal inefficiency is probably not "
                "identified (the likelihood is flat or maximized at sigma_u = 0), so consider "
                "OLS or another distribution." if wrong else " Check the scale of the outcome "
                "and regressors, or try another distribution.")
        raise AnalysisError("nonconvergence", f"frontier did not converge: "
                            f"{result.diagnostics.get('message')}{hint}")
    centred = result.theta
    covariance, info = likelihood_covariance(frame, result.hessian,
                                             lambda: objective.score_rows(centred), weights)
    jacobian = torch.block_diag(constant_shift(means), torch.eye(m, dtype=torch.float64))
    theta, covariance = jacobian @ centred, jacobian @ covariance @ jacobian.T
    terms = [*design.terms, *ANCILLARIES[distribution]]
    ll = float(result.value)
    nobs = weights.nobs
    metrics: dict[str, Any] = information_criteria(ll, k + m, nobs)
    extra_anc: dict[str, Any] = {}
    z = normal_ppf(1 - spec.alpha / 2)
    block = covariance[k:, k:]
    for name, (value, gradient) in _transforms(distribution, anc).items():
        se = math.sqrt(max(float(gradient @ block @ gradient), 0.0))
        metrics[name] = value
        extra_anc[name] = {"estimate": value, "std_error": se, "ci_low": value - z * se,
                           "ci_high": value + z * se}
    slopes = [i for i, term in enumerate(design.terms) if term != "Intercept"]
    tests: dict[str, Any] = {"model": wald_test(theta, covariance, slopes,
                                                label="Wald chi2 test of the slopes")}
    resid = y - design.x @ centred[:k]
    if spec.covariance in {"nonrobust", "opg"} and spec.weight_type != "pweight":
        total = float(weights.user.sum())
        ll_ols = -0.5 * total * (math.log(2 * math.pi * sigma2_ols) + 1)
        statistic = max(0.0, 2 * (ll - ll_ols))
        tests["sigma_u"] = {"statistic": statistic, "df": 1,
                            "p_value": 0.5 * chi2_sf(statistic, 1) if statistic > 0 else 1.0,
                            "distribution": "chibar2",
                            "label": "LR test of sigma_u = 0 (chibar2(01))",
                            "restricted_log_likelihood": ll_ols}
        metrics["log_likelihood_ols"] = ll_ols
    metrics.update({"df_model": len(slopes)})
    info.update({"nobs": nobs, "distribution": "normal", "inefficiency_distribution": distribution, "frontier": "cost" if cost
                 else "production"})
    extra = {"distribution": distribution, "cost": cost, "ancillary": extra_anc,
             "start": dict(zip(terms, (jacobian @ start).tolist(), strict=True))}
    return build_result(
        frame, terms=terms, params=theta, covariance=covariance,
        title=f"Stochastic {'cost' if cost else 'production'} frontier "
              f"({_TITLES[distribution]})",
        use_t=False, metrics=metrics, fitted=y - resid, nobs=nobs, inference=info, tests=tests,
        extra=extra, categories=design.categories, solver="newton_raphson_analytic_hessian",
        optimizer=optimizer_record(result),
        solver_diagnostics={"converged": True, "centred_regressors": bool(design.intercept)})


# ---- efficiency ----------------------------------------------------------------------------


def _coefficients(result: ResultBundle) -> dict[str, float]:
    return {c.term: c.estimate for c in result.coefficients}


def frontier_efficiency(result: ResultBundle, data: Any) -> Any:
    """Technical (cost) efficiency of every observation after ``oe.frontier``.

    With ``e_i = y_i - x_i'b`` and ``u_i | e_i ~ N+(mu*_i, sigma*^2)``
    (Jondrow, Lovell, Materov and Schmidt 1982) the table reports, per
    estimation-sample observation:

    - ``u``: ``E[u|e] = sigma* (phi(z)/Phi(z) + z)``, ``z = mu*/sigma*`` (JLMS);
    - ``u_mode``: the mode ``M(u|e) = max(mu*, 0)``;
    - ``te``: ``E[exp(-s u)|e]`` (Battese and Coelli 1988), i.e.
      ``exp(-mu* + sigma*^2/2) Phi(z - sigma*) / Phi(z)`` for a production
      frontier and ``exp(mu* + sigma*^2/2) Phi(z + sigma*) / Phi(z)`` for a cost
      frontier (Stata's ``predict, te``).

    ``mu*`` and ``sigma*``: half-normal ``mu* = -s e su2/s2``,
    ``sigma* = su sv/s``; exponential ``mu* = -s e - sv2/su``, ``sigma* = sv``;
    truncated normal ``mu* = (mu sv2 - s e su2)/s2``.

    Parameters
    ----------
    result : the ResultBundle of ``oe.frontier``.
    data : the data the model was fitted on (the estimation sample is rebuilt
        and must match the stored sample positions).

    Returns a table with columns ``row`` (0-based position in ``data``),
    ``residual``, ``u``, ``u_mode`` and ``te``; ``attrs`` holds the
    distribution, the frontier type and the mean efficiency.

    Stata: ``predict u, u``; ``predict m, m``; ``predict te, te``.

    Example::

        fit = oe.frontier(data=df, y="lnoutput", x=["lncapital", "lnlabor"])
        eff = oe.frontier_efficiency(fit, df)
        print(eff["te"].describe())
    """
    if not isinstance(result, ResultBundle) or result.spec.estimator != "frontier":
        raise AnalysisError("invalid_result", "frontier_efficiency needs the result returned by "
                            "oe.frontier.")
    frame = ModelFrame(result.spec, data)
    if frame.positions != result.sample_positions:
        raise AnalysisError("sample_mismatch", "The data do not reproduce the estimation sample "
                            "of this result; pass the data the model was fitted on.")
    coefficients = _coefficients(result)
    design = frame.design()
    kept = [i for i, term in enumerate(design.terms) if term in coefficients]
    design = design.select(kept)
    beta = torch.tensor([coefficients[term] for term in design.terms], dtype=torch.float64)
    y = frame.numeric(result.spec.outcome)
    e = y - design.x @ beta
    distribution, cost = result.extra["distribution"], bool(result.extra["cost"])
    if distribution == "tnormal":
        sigma2 = math.exp(coefficients["/lnsigma2"])
        gamma = 1 / (1 + math.exp(-coefficients["/ilgtgamma"]))
        sigma_u2, sigma_v2, mu = gamma * sigma2, (1 - gamma) * sigma2, coefficients["/mu"]
    else:
        sigma_v2, sigma_u2 = math.exp(coefficients["/lnsig2v"]), math.exp(coefficients["/lnsig2u"])
        mu = 0.0
    centre, spread = conditional_inefficiency(e, distribution, cost, sigma_v2, sigma_u2, mu)
    z = centre / spread
    log_cdf, ratio, _ = mills_ratio(z)
    u = spread * (ratio + z)
    shift = -spread if not cost else spread
    log_shifted, _, _ = mills_ratio(z + shift)
    sign = -1.0 if not cost else 1.0
    te = torch.exp(sign * centre + spread.square() / 2 + log_shifted - log_cdf)
    if not (bool(torch.isfinite(u).all()) and bool(torch.isfinite(te).all())):
        raise AnalysisError("non_finite_result", "The efficiency estimates overflow for some "
                            "observations; check the scale of the outcome.")
    frame_out = table({"row": frame.positions, "residual": e.tolist(), "u": u.tolist(),
                       "u_mode": centre.clamp_min(0).tolist(), "te": te.tolist()},
                      distribution=distribution, frontier="cost" if cost else "production",
                      mean_te=float(te.mean()), mean_u=float(u.mean()),
                      te_definition="E[exp(-s u)|e] (Battese-Coelli)",
                      u_definition="E[u|e] (Jondrow-Lovell-Materov-Schmidt)")
    return frame_out


def frontier(*, data: Any, y: str, x: Sequence[str], distribution: str = "hnormal",
             cost: bool = False, intercept: bool = True, covariance: str | None = None,
             cluster: str | None = None, categorical: Sequence[str] | None = None,
             weights: str | None = None, weight_type: str | None = None,
             missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Stochastic frontier model by maximum likelihood (Stata ``frontier``; EViews and SPSS
    have no built-in equivalent).

    Model: ``y = x'b + v - s u`` with ``s = 1`` (production frontier: output
    below the frontier) or ``s = -1`` (``cost=True``: cost above the frontier),
    ``v ~ N(0, sigma_v^2)`` and inefficiency ``u >= 0``:

    - ``distribution='hnormal'``: ``u ~ N+(0, sigma_u^2)``;
      ``l_i = 1/2 ln(2/pi) - ln sigma + ln Phi(-s e_i lambda/sigma) - e_i^2/(2 sigma^2)``,
      ``sigma^2 = sigma_u^2 + sigma_v^2``, ``lambda = sigma_u/sigma_v``;
    - ``'exponential'``: ``u ~ Exp`` with mean ``sigma_u``;
      ``l_i = -ln sigma_u + sigma_v^2/(2 sigma_u^2) + s e_i/sigma_u
      + ln Phi(-s e_i/sigma_v - sigma_v/sigma_u)``;
    - ``'tnormal'``: ``u ~ N+(mu, sigma_u^2)``;
      ``l_i = -1/2 ln(2 pi) - ln sigma - (s e_i + mu)^2/(2 sigma^2)
      + ln Phi(mu*_i/sigma*) - ln Phi(mu/sigma_u)``.

    Parameters are estimated as Stata does: ``/lnsig2v``, ``/lnsig2u``
    (hnormal, exponential) or ``/mu``, ``/lnsigma2``, ``/ilgtgamma`` (tnormal,
    ``gamma = sigma_u^2/sigma^2``). Newton-Raphson with analytic score and
    Hessian on centred regressors, started at OLS with method-of-moments
    variances. Wrong-skewed OLS residuals, a variance ratio at the boundary,
    or a truncated normal drifting to its exponential limit (``mu -> -inf``)
    raise ``boundary_solution`` instead of reporting a degenerate fit; a
    constant outcome raises ``constant_outcome`` and an exact fit
    ``perfect_fit``.

    Parameters
    ----------
    data : DataFrame, dict of columns or list of records.
    y : outcome (usually log output or log cost). x : regressors.
    distribution : ``'hnormal'`` (default), ``'exponential'`` or ``'tnormal'``.
    cost : cost frontier. intercept : include the constant.
    covariance : ``'nonrobust'`` (observed information, default), ``'opg'``,
        ``'robust'`` (``N/(N-1)`` sandwich) or ``'cluster'`` (``G/(G-1)``).
    weights, weight_type : ``'fweight'``, ``'pweight'`` (robust by default)
        or ``'iweight'``. categorical, missing, alpha : as usual.

    Returns
    -------
    z inference; ``metrics``: ``log_likelihood``, ``aic``, ``bic``,
    ``sigma_v``, ``sigma_u``, ``sigma2``, ``lambda`` (tnormal: ``sigma2``,
    ``gamma``, ``sigma_u2``, ``sigma_v2``), ``log_likelihood_ols``;
    ``extra['ancillary']``: those quantities with delta-method standard errors
    and confidence intervals; ``tests``: ``model`` (Wald chi2 of the slopes)
    and ``sigma_u``: the LR test of ``sigma_u = 0`` against OLS,
    ``chibar2(01)`` with ``p = Pr(chi2(1) > LR)/2``. Efficiency scores:
    ``oe.frontier_efficiency(result, data)``.

    Stata: ``frontier lnoutput lncapital lnlabor, distribution(exponential) cost``.

    Example::

        import openecon as oe
        fit = oe.frontier(data=df, y="lnv", x=["lnk", "lnl"])
        print(fit.summary(), fit.metrics["lambda"])
        eff = oe.frontier_efficiency(fit, df)
    """
    from openecon.analysis import fit

    spec = build_spec(
        "frontier", outcome=y, predictors=column_list(x, "x"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        categorical=column_list(categorical, "categorical"), weights=weights,
        weight_type=weight_type, missing=missing, alpha=alpha,
        options={"distribution": None if distribution == "hnormal" else distribution,
                 "cost": cost or None})
    return fit(spec, data=data)
