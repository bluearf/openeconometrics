"""Shared-frailty Cox regression: Stata's ``stcox, shared()``; R ``coxph(... frailty())``.

Model ``h_ij(t) = z_i h0_s(t) exp(x_ij'b + offset)``: the records of shared group
i (``shared``) have a common multiplicative frailty ``z_i``, gamma with mean 1
and variance theta (``frailty='gamma'``) or log-normal with ``ln z_i ~ N(0,
theta)`` (``frailty='lognormal'``). The baseline hazard is unspecified (per
stratum).

Estimation (``frailty_kernels``). For fixed theta the fixed effects b and the
log frailties w maximize the penalized partial likelihood (Newton-Raphson with
the analytic (k+q)-dimensional Hessian). theta maximizes the integrated log
partial likelihood ``L(theta)`` of that solution: for the gamma frailty the
closed form of R ``survival`` (equivalent to the EM algorithm of Klein 1992 and
Nielsen et al. 1992 with Breslow ties; Therneau, Grambsch and Pankratz 2003),
for the log-normal frailty its Laplace approximation (Ripatti and Palmgren
2000; R ``coxme``). The search over ``ln theta`` evaluates a fixed grid, then
brackets the best grid point and refines it: for the gamma frailty by the
Brent-Dekker root of the analytic derivative of ``L``, for the log-normal by
Brent's derivative-free maximization. ``L(0)`` is the Cox partial likelihood;
an estimate whose integrated likelihood gain over the Cox model is below
``BOUNDARY_GAIN * max(1, |L(0)|)`` is reported as the boundary ``theta = 0``,
and the result is then the Cox model.

Inference. Coefficients are log hazard ratios with z tests. Their covariance is
the b block of the inverse negative Hessian of the penalized partial likelihood
at ``(b, w, theta_hat)`` - conditional on theta_hat, R ``coxph``'s ``var`` (with
``sparse=FALSE``) and ``coxme``'s fixed-effect variance. ``extra['se2']`` gives
R's alternative ``var2 = H^-1 I H^-1`` (I the unpenalized information). The
test of ``theta = 0`` is the likelihood-ratio statistic ``2 (L(theta_hat) - L(0))``
against the boundary mixture ``chibar2(01)`` (half a chi2(1) tail; Self and Liang
1987), as Stata reports for ``shared()``. No standard error of theta is
reported (R reports none; Stata's convention is not validated here).

Domain: no weights, no robust/cluster covariance, no tvc. ``entry`` gives
``(start, stop]`` records; the frailty then multiplies the whole history of the
group, R's convention for counting-process data (not a frailty distribution
selected at a delayed entry). At most ``MAX_GROUPS`` groups and
``frailty_kernels.FRAILTY_CELL_LIMIT`` failure-time x group cells.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import build_result, column_list, kernel_call, wald_test
from openecon.econometrics.discrete.common import build_spec, maximize, optimizer_record
from openecon.econometrics.survival.cox import _hazard_ratios, prepare
from openecon.econometrics.survival.cox_kernels import CoxObjective
from openecon.econometrics.survival.data import FLOAT, safe_exp
from openecon.econometrics.survival.frailty_kernels import (
    FrailtyCox, PenalizedFrailty, brent_maximize, gamma_integrated, lognormal_integrated,
    zeroin,
)
from openecon.engines.distributions import chi2_sf
from openecon.engines.optimize import information_inverse
from openecon.models import ModelSpec, ResultBundle

MAX_GROUPS = 2000
GRID = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
THETA_FLOOR = 1e-8
THETA_CEILING = 1024.0
BOUNDARY_GAIN = 1e-10


def chibar_test(statistic: float, label: str) -> dict[str, Any]:
    """LR test of a variance at the boundary: 50:50 mixture of chi2(0) and chi2(1)."""
    statistic = max(0.0, statistic)
    p_value = 1.0 if statistic == 0 else 0.5 * chi2_sf(statistic, 1)
    return {"statistic": statistic, "df": 1, "p_value": p_value, "distribution": "chibar2",
            "label": label}


class _Profile:
    """Integrated log partial likelihood ``L(ln theta)`` with warm-started inner fits."""

    def __init__(self, frail: FrailtyCox, distribution: str, beta: Tensor, command: str):
        self.frail, self.distribution, self.command = frail, distribution, command
        self.start = torch.cat([beta, torch.zeros(frail.q, dtype=FLOAT)])
        self.cache: dict[float, dict[str, Any]] = {}

    def __call__(self, log_theta: float) -> dict[str, Any]:
        if log_theta in self.cache:
            return self.cache[log_theta]
        theta = math.exp(log_theta)
        objective = PenalizedFrailty(self.frail, self.distribution, theta)
        result = maximize(objective, self.start, what=f"{self.command} penalized partial")
        if not result.converged:
            raise AnalysisError(
                "nonconvergence", f"{self.command}: the penalized partial likelihood did not "
                f"converge at theta = {theta:.6g} ({result.diagnostics.get('message')}). Check "
                "for covariates that separate failures from survivors.")
        params = result.theta
        self.start = params.clone()
        k = self.frail.k
        omega = params[k:]
        partial = float(self.frail.value(params))
        if self.distribution == "gamma":
            value, slope = gamma_integrated(partial, omega, self.frail.group_events, theta)
            # R's reported penalty nu sum(exp(w) - w - 1): zero at w = 0, -nu sum(w) here.
            penalty = float((torch.expm1(omega) - omega).sum()) / theta
        else:
            information = -self.frail(params)[2][k:, k:]
            value = kernel_call(lognormal_integrated, partial, omega, information, theta)
            slope = None
            penalty = float(omega.square().sum()) / (2 * theta)
        record = {"theta": theta, "log_theta": log_theta, "value": value, "slope": slope,
                  "partial": partial, "penalized": partial - penalty, "result": result}
        self.cache[log_theta] = record
        return record

    def history(self) -> list[dict[str, float]]:
        return [{"theta": r["theta"], "log_likelihood": r["value"]}
                for _, r in sorted(self.cache.items())]


def _search(profile: _Profile, cox_ll: float) -> dict[str, Any] | None:
    """The maximizer of the integrated likelihood, or None for the boundary theta = 0."""
    points = [math.log(value) for value in GRID]
    values = [profile(t)["value"] for t in points]
    best = max(range(len(points)), key=values.__getitem__)
    while best == len(points) - 1:
        if math.exp(points[-1]) >= THETA_CEILING:
            raise AnalysisError(
                "boundary_solution", f"{profile.command}: the integrated likelihood keeps "
                f"increasing with the frailty variance (theta > {THETA_CEILING:g}); the frailty "
                "model does not fit these data (a group whose members all fail at once?).")
        points.append(points[-1] + math.log(2.0))
        values.append(profile(points[-1])["value"])
        best = max(range(len(points)), key=values.__getitem__)
    if values[best] <= cox_ll:
        low, high = math.log(THETA_FLOOR), points[0]
    else:
        low = points[best - 1] if best > 0 else math.log(THETA_FLOOR)
        high = points[best + 1]
    if profile.distribution == "gamma":
        f_low, f_high = profile(low)["slope"], profile(high)["slope"]
        if f_low <= 0 and values[best] <= cox_ll:
            return None
        if f_low > 0 > f_high:
            t = zeroin(lambda u: profile(u)["slope"], low, high, f_low, f_high)
        else:
            t = brent_maximize(lambda u: profile(u)["value"], low, high)
    else:
        t = brent_maximize(lambda u: profile(u)["value"], low, high)
    record = profile(t)
    if record["value"] - cox_ll <= BOUNDARY_GAIN * max(1.0, abs(cox_ll)):
        return None
    return record


def _cox_fit(sample, ties: str, command: str):
    objective = CoxObjective(sample.x, sample.risk, None, sample.offset, ties)
    k = len(sample.terms)
    result = maximize(objective, torch.zeros(k, dtype=FLOAT), what=command)
    if not result.converged:
        spread = sample.x.std(0) if sample.x.shape[0] > 1 else torch.ones(k, dtype=FLOAT)
        if bool(((result.theta * spread).abs() > 10).any()):
            raise AnalysisError(
                "separation_detected", f"{command}: the partial likelihood keeps increasing as a "
                "coefficient diverges (a covariate separates failures from survivors).")
        raise AnalysisError("nonconvergence", f"{command}: the Cox model without frailty did "
                            f"not converge: {result.diagnostics.get('message')}")
    return objective, result


def fit_stcox_frailty(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``stcox_frailty``: ``(spec, data) -> ResultBundle`` (module notes)."""
    command = "stcox_frailty"
    sample = prepare(spec, data, command)
    frame = sample.frame
    distribution, ties = frame.option("frailty"), frame.option("ties")
    fixed = frame.option("theta")
    if fixed is not None and not fixed > 0:
        raise AnalysisError("invalid_option", f"{command}: a fixed theta must be positive "
                            "(theta = 0 is the Cox model: use oe.stcox).")
    shared = frame.role("shared")[0]
    groups, q = frame.codes(shared)
    labels = frame.levels(shared)
    if q < 2:
        raise AnalysisError("invalid_groups", f"{command}: the shared frailty needs at least two "
                            f"groups in '{shared}'.")
    if q > MAX_GROUPS:
        raise AnalysisError("resource_limit", f"{command}: {q} shared groups exceed the dense "
                            f"limit of {MAX_GROUPS} (the Hessian has one row per group).")
    k = len(sample.terms)
    cox_objective, cox = _cox_fit(sample, ties, command)
    cox_ll = float(cox.value)
    frail = kernel_call(FrailtyCox, sample.x, groups, q, sample.risk, sample.offset, ties)
    frame.workspace_plan("shared frailty risk-set blocks and dense Hessian", {
        "time_by_group_blocks": 8 * 4 * (sample.risk.n_events + 1) * q,
        "penalized_hessian_and_factor": 8 * 3 * (k + q) ** 2})
    profile = _Profile(frail, distribution, cox.theta, command)
    if fixed is not None:
        record = profile(math.log(fixed))
    else:
        record = _search(profile, cox_ll)
    notes: list[str] = []
    if record is None:
        theta = 0.0
        beta = cox.theta
        covariance = kernel_call(information_inverse, -cox.hessian)
        omega = torch.zeros(q, dtype=FLOAT)
        integrated, penalized, partial = cox_ll, cox_ll, cox_ll
        second = covariance
        solver_result = cox
        notes.append("The frailty variance is estimated at the boundary theta = 0: the fit is "
                     "the Cox model without frailty.")
    else:
        theta = record["theta"]
        result = record["result"]
        beta, omega = result.theta[:k], result.theta[k:]
        full = kernel_call(information_inverse, -result.hessian)
        covariance = full[:k, :k]
        unpenalized = -frail(result.theta)[2]
        second = (full @ unpenalized @ full)[:k, :k]
        integrated, penalized, partial = record["value"], record["penalized"], record["partial"]
        solver_result = result
    tests: dict[str, Any] = {"model": wald_test(beta, covariance, range(k),
                                                label="Wald chi2 test of the coefficients")}
    if fixed is None:
        tests["theta"] = chibar_test(
            2.0 * (integrated - cox_ll),
            f"LR test of theta = 0 ({distribution} frailty; chibar2(01): chi2(1) tail halved)")
    else:
        notes.append(f"theta is fixed at {fixed:g}; no test of theta = 0 is reported.")
    estimated = fixed is None and theta > 0
    parameters = k + (1 if estimated else 0)
    nobs = frame.n
    se2 = second.diagonal().clamp_min(0).sqrt()
    t_exit = frame.numeric(spec.outcome)
    entry_names = frame.role("entry")
    t_entry = frame.numeric(entry_names[0]) if entry_names else torch.zeros_like(t_exit)
    events = sample.failure
    metrics = {
        "log_likelihood": integrated, "log_likelihood_cox": cox_ll,
        "penalized_log_likelihood": penalized, "partial_log_likelihood": partial,
        "theta": theta,
        "aic": -2 * integrated + 2 * parameters, "bic": -2 * integrated + math.log(nobs) * parameters,
        "df_model": k, "n_groups": q, "n_failures": float(events.sum()),
        "time_at_risk": float((t_exit - t_entry).sum()),
    }
    frailties = [{"group": label, "log_frailty": float(w), "frailty": safe_exp(float(w)),
                  "failures": float(d)}
                 for label, w, d in zip(labels, omega.tolist(), frail.group_events.tolist(),
                                        strict=True)]
    integration = ("closed-form gamma integrated partial likelihood (R survival I-likelihood)"
                   if distribution == "gamma" else
                   "Laplace approximation of the log-normal integrated partial likelihood")
    extra = {
        "hazard_ratios": _hazard_ratios(sample.terms, beta, covariance, spec.alpha),
        "frailty": {"distribution": distribution, "shared": shared, "theta": theta,
                    "theta_estimated": fixed is None, "boundary": record is None,
                    "integration": integration, "ties": ties},
        "frailty_estimates": frailties,
        "se2": dict(zip(sample.terms, se2.tolist(), strict=True)),
        "profile": profile.history(),
        "covariance_convention": "b block of the inverse negative Hessian of the penalized "
                                 "partial likelihood at theta_hat (conditional on theta_hat)",
        "strata": frame.role("strata") or None,
    }
    if notes:
        extra["notes"] = notes
    info ={"correction": "inverse negative penalized Hessian (b block), conditional on theta",
            "covariance": "nonrobust"}
    provenance = {"ties": ties, "frailty": distribution, "theta_search":
                  "fixed" if fixed is not None else
                  ("grid + Brent-Dekker root of dL/dln(theta)" if distribution == "gamma"
                   else "grid + Brent maximization over ln(theta)"),
                  "bic_n": "number of observations (records)"}
    return build_result(
        frame, terms=sample.terms, params=beta, covariance=covariance, use_t=False,
        metrics=metrics, nobs=nobs, inference=info, tests=tests, extra=extra,
        categories=sample.design.categories, provenance=provenance,
        title=f"Cox regression with shared {distribution} frailty",
        solver="newton_penalized_partial_likelihood",
        solver_diagnostics={"converged": True, "iterations": solver_result.iterations,
                            "profile_evaluations": len(profile.cache)},
        optimizer=optimizer_record(solver_result))


def stcox_frailty(*, data: Any, time: str, x: Sequence[str], shared: str,
                  failure: str | None = None, entry: str | None = None,
                  strata: str | Sequence[str] | None = None, frailty: str = "gamma",
                  ties: str = "breslow", theta: float | None = None, offset: str | None = None,
                  categorical: Sequence[str] | None = None, missing: str = "raise",
                  alpha: float = 0.05) -> ResultBundle:
    """Cox regression with a shared gamma or log-normal frailty (Stata ``stcox, shared()``).

    Model
        ``h_ij(t) = z_i h0_s(t) exp(x_ij'b + offset)`` with a frailty ``z_i`` shared by the
        records of group i: gamma with mean 1 and variance theta, or log-normal with
        ``ln z_i ~ N(0, theta)``.

    Estimator
        Penalized partial likelihood in (b, ln z) for fixed theta (Newton-Raphson);
        theta maximizes the integrated log partial likelihood: closed form for the gamma
        frailty (R ``coxph(... frailty(g))`` I-likelihood, the EM solution of Klein 1992
        with Breslow ties), Laplace approximation for the log-normal (R ``coxme``).

    Parameters
        data, time, failure, entry, strata, offset, categorical, missing, alpha: as in
            ``oe.stcox`` (``entry`` gives (start, stop] records; the frailty multiplies the
            whole history of a group).
        x: covariates (log hazard ratios).
        shared: the group column of the shared frailty (Stata ``shared(varname)``).
        frailty: ``'gamma'`` (default) or ``'lognormal'``.
        ties: ``'breslow'`` (default) or ``'efron'``.
        theta: fix the frailty variance instead of estimating it (positive).

    Result
        Coefficients with z tests; covariance: b block of the inverse negative penalized
        Hessian at theta_hat (R ``coxph`` ``var``). ``metrics``: log_likelihood
        (integrated), log_likelihood_cox (theta = 0), penalized_log_likelihood, theta,
        aic/bic (k + 1 parameters when theta is estimated), n_groups, n_failures.
        ``tests``: ``model`` (Wald chi2), ``theta`` (LR chibar2(01) test of theta = 0).
        ``extra``: hazard_ratios, frailty, frailty_estimates (log frailty per group),
        se2 (R ``var2`` standard errors), profile (evaluated ``L(theta)``).

    Stata / R
        ``stcox x1 x2, shared(g)`` is ``oe.stcox_frailty(data=df, time='t', failure='d',
        x=['x1', 'x2'], shared='g')``; R: ``coxph(Surv(t, d) ~ x1 + x2 + frailty(g),
        ties='breslow')``; log-normal: ``coxme(Surv(t, d) ~ x1 + x2 + (1 | g))``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(1)
        >>> g = np.repeat(np.arange(40), 5); z = rng.gamma(2, 0.5, 40)[g]
        >>> x = rng.normal(size=200); t = rng.exponential(1 / (z * np.exp(0.5 * x)))
        >>> df = pd.DataFrame({"t": t, "d": 1, "x": x, "g": g})
        >>> print(oe.stcox_frailty(data=df, time="t", failure="d", x=["x"], shared="g").summary())
    """
    spec = build_spec(
        "stcox_frailty", outcome=time, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=False,
        covariance="nonrobust", missing=missing, alpha=alpha,
        columns={"failure": failure, "entry": entry, "shared": shared,
                 "strata": column_list([strata] if isinstance(strata, str) else strata,
                                       "strata"),
                 "offset": offset},
        options={"frailty": frailty, "ties": ties, "theta": theta})
    return fit(spec, data=data)
