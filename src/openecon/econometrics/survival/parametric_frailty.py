"""Parametric shared-frailty survival regression: Stata's ``streg, shared() frailty()``.

Model: conditional on the frailty ``z_i`` shared by the records of group i,
the hazard is ``z_i h(t | x)`` with the exponential or Weibull ``h`` of
``streg`` (PH or AFT metric, ``survival.parametric``). ``z_i`` is gamma or
inverse Gaussian with mean 1 and variance theta. Integrating ``z_i`` out gives
the closed-form marginal log likelihood (``frailty_kernels``)

    l_i = sum_j delta_ij ln h(t_ij) + ln M_{D_i}(sum_j H(t_ij)),
    M_m(s) = E[Z^m exp(-s Z)],

with ``D_i`` the failures of group i: ``nu ln nu + ln Gamma(nu + m) - ln
Gamma(nu) - (nu + m) ln(nu + s)`` for the gamma (``nu = 1/theta``) and the
half-integer Bessel closed form for the inverse Gaussian. Parameters
``(b, /ln_p, /ln_the)`` are estimated jointly by Newton-Raphson with analytic
score and Hessian; regressors are centred for the iteration and mapped back
exactly. The reported log likelihood is on Stata's scale (``+ sum delta ln t``,
as ``streg``).

Boundary. The model without frailty (``theta = 0``) is fitted first; Newton
runs for the frailty model start from it with ``theta`` in ``STARTS``. When no
run reaches an interior maximum whose log likelihood exceeds that of the model
without frailty by more than ``BOUNDARY_GAIN max(1, |ll|)`` (runs drifting to
``ln theta < LOG_THETA_FLOOR`` count as reaching the boundary), the estimate is
``theta = 0`` and the result is the model without frailty. ``tests['theta']``
is the likelihood-ratio test of ``theta = 0`` against ``chibar2(01)``.

Covariance: observed information of ``(b, /ln_p, /ln_the)`` (Stata's OIM);
theta's interval is the exp transform of the ``/ln_the`` interval. Domain: no
delayed entry, weights, robust or cluster covariance; at most ``MAX_EVENTS``
failures in one group for the inverse Gaussian (its Bessel sums are dense).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, kernel_call, wald_test
from openecon.econometrics.discrete.common import (
    Weights, build_spec, centre_regressors, constant_shift, maximize, optimizer_record,
)
from openecon.econometrics.survival.cox import _survival_columns
from openecon.econometrics.survival.data import FLOAT, check_times, end_before_entry_note, safe_exp
from openecon.econometrics.survival.frailty import chibar_test
from openecon.econometrics.survival.frailty_kernels import (
    gamma_log_moment, inverse_gaussian_log_moment,
)
from openecon.econometrics.survival.streg import resolve_metric
from openecon.engines.inference import critical_value
from openecon.engines.optimize import information_inverse
from openecon.models import ModelSpec, ResultBundle

STARTS = (0.5, 0.05, 2.0)
LOG_THETA_FLOOR = -18.0
BOUNDARY_GAIN = 1e-10
MAX_EVENTS = 2000


class _Boundary(Exception):
    """A Newton run whose frailty variance drifts to the boundary theta = 0."""


class FrailtyLikelihood:
    """Marginal log likelihood of the exponential/Weibull shared-frailty model.

    ``params = (b [k], a (Weibull only), ln theta (frailty only))``; without a
    frailty the likelihood is the ``streg`` time-scale likelihood.
    """

    def __init__(self, distribution: str, metric: str, x: Tensor, time: Tensor,
                 failure: Tensor, groups: Tensor, q: int, offset: Tensor | None,
                 frailty: str | None):
        self.distribution, self.metric, self.frailty = distribution, metric, frailty
        self.x, self.delta, self.g, self.q, self.offset = x, failure, groups, q, offset
        self.log_t = torch.log(time)
        self.k = x.shape[1]
        self.weibull = distribution == "weibull"
        self.size = self.k + int(self.weibull) + int(frailty is not None)
        self.events = torch.zeros(q, dtype=FLOAT).index_add_(0, groups, failure)
        self.order = int(self.events.max()) + 3

    def _observation(self, params: Tensor):
        """ln h, H and their derivatives in (mu, a) per record."""
        mu = self.x @ params[:self.k]
        if self.offset is not None:
            mu = mu + self.offset
        lt = self.log_t
        zero = torch.zeros_like(mu)
        if not self.weibull:
            if self.metric == "ph":
                big_h = torch.exp(mu + lt)
                return mu, big_h, (1.0 + zero, zero, None, None, None), (big_h, big_h, None, None, None)
            big_h = torch.exp(lt - mu)
            return -mu, big_h, (-1.0 + zero, zero, None, None, None), (-big_h, big_h, None, None, None)
        a = params[self.k]
        p = torch.exp(a)
        if self.metric == "ph":
            plt = p * lt
            big_h = torch.exp(mu + plt)
            log_h = a + mu + (p - 1) * lt
            dh = (1.0 + zero, zero, 1 + plt, plt, zero)              # mu, mumu, a, aa, mua
            dH = (big_h, big_h, big_h * plt, big_h * (plt + plt * plt), big_h * plt)
            return log_h, big_h, dh, dH
        z = lt - mu
        pz = p * z
        big_h = torch.exp(pz)
        log_h = a + pz - lt
        dh = (-p + zero, zero, 1 + pz, pz, -p + zero)
        dH = (-p * big_h, p * p * big_h, pz * big_h, (pz + pz * pz) * big_h,
              -p * big_h * (1 + pz))
        return log_h, big_h, dh, dH

    def _group(self, exposure: Tensor, params: Tensor):
        if self.frailty is None:
            ones = torch.ones_like(exposure)
            return -exposure, -ones, 0 * ones, None, None, None
        log_theta = float(params[-1])
        if self.frailty == "gamma":
            return gamma_log_moment(self.events, exposure, log_theta)
        return inverse_gaussian_log_moment(self.events, exposure, log_theta, self.order)

    def _sums(self, params: Tensor):
        log_h, big_h, dh, dH = self._observation(params)
        exposure = torch.zeros(self.q, dtype=FLOAT).index_add_(0, self.g, big_h)
        group = self._group(exposure, params)
        return log_h, big_h, dh, dH, group

    def value(self, params: Tensor) -> Tensor:
        log_h, _, _, _, group = self._sums(params)
        return (self.delta * log_h).sum() + group[0].sum()

    def __call__(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x, g, q, k, delta = self.x, self.g, self.q, self.k, self.delta
        log_h, _, dh, dH, group = self._sums(params)
        value = (delta * log_h).sum() + group[0].sum()
        l_s, l_ss = group[1], group[2]
        k1 = l_s[g]
        gradient = torch.zeros(self.size, dtype=FLOAT)
        hessian = torch.zeros((self.size, self.size), dtype=FLOAT)
        w_mu = delta * dh[0] + k1 * dH[0]
        gradient[:k] = x.T @ w_mu
        g_mu = torch.zeros((q, k), dtype=FLOAT).index_add_(0, g, dH[0][:, None] * x)
        w_mumu = delta * dh[1] + k1 * dH[1]
        hessian[:k, :k] = (x * w_mumu[:, None]).T @ x + (g_mu * l_ss[:, None]).T @ g_mu
        column = k
        if self.weibull:
            g_a = torch.zeros(q, dtype=FLOAT).index_add_(0, g, dH[2])
            gradient[k] = (delta * dh[2] + k1 * dH[2]).sum()
            cross = x.T @ (delta * dh[4] + k1 * dH[4]) + g_mu.T @ (l_ss * g_a)
            hessian[:k, k] = hessian[k, :k] = cross
            hessian[k, k] = (delta * dh[3] + k1 * dH[3]).sum() + (l_ss * g_a.square()).sum()
            column = k + 1
        if self.frailty is not None:
            l_t, l_tt, l_st = group[3], group[4], group[5]
            gradient[column] = l_t.sum()
            hessian[column, column] = l_tt.sum()
            hessian[:k, column] = hessian[column, :k] = g_mu.T @ l_st
            if self.weibull:
                hessian[k, column] = hessian[column, k] = (l_st * g_a).sum()
        return value, gradient, (hessian + hessian.T) / 2

    def boundary_score(self, params: Tensor) -> float:
        """Score of theta at 0: ``sum_i ((D_i - A_i)^2 - D_i) / 2`` (any mean-1 frailty)."""
        _, big_h, _, _ = self._observation(params)
        exposure = torch.zeros(self.q, dtype=FLOAT).index_add_(0, self.g, big_h)
        return float((((self.events - exposure).square() - self.events) / 2).sum())


def _start(distribution: str, metric: str, time: Tensor, failure: Tensor) -> tuple[float, float]:
    events, exposure = float(failure.sum()), float(time.sum())
    rate = math.log(events / exposure)
    return (rate if metric == "ph" else -rate), 0.0


def _newton(objective, start: Tensor, command: str, *, watch: bool = False):
    callback = None
    if watch:
        def callback(iteration: int, theta: Tensor, value: float) -> None:
            if float(theta[-1]) < LOG_THETA_FLOOR:
                raise _Boundary()
    return maximize(objective, start, what=command, callback=callback)


def fit_streg_frailty(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``streg_frailty``: ``(spec, data) -> ResultBundle`` (module notes)."""
    command = "streg_frailty"
    frame = ModelFrame(spec, data)
    distribution, frailty = frame.option("distribution"), frame.option("frailty")
    metric = resolve_metric(distribution, frame.option("metric"))
    t, _, delta = _survival_columns(frame)
    usable = check_times(t, None, spec.outcome, None)
    if not bool(usable.all()):
        frame.restrict(usable, end_before_entry_note(int((~usable).sum())))
        t, _, delta = _survival_columns(frame)
    if float(delta.sum()) == 0:
        raise AnalysisError("no_failures", f"{command}: the sample has no failures.")
    shared = frame.role("shared")[0]
    groups, q = frame.codes(shared)
    if q < 2:
        raise AnalysisError("invalid_groups", f"{command}: the shared frailty needs at least two "
                            f"groups in '{shared}'.")
    unweighted = Weights(torch.ones(frame.n, dtype=FLOAT), None, frame.n, False)
    design = frame.drop_collinear(frame.design(spec.predictors, intercept=True))
    means = centre_regressors(design, unweighted)
    offset_names = frame.role("offset")
    offset = frame.numeric(offset_names[0]) if offset_names else None
    k = len(design.terms)
    weibull = distribution == "weibull"
    size = k + int(weibull) + 1
    if frame.n <= size:
        raise AnalysisError("insufficient_observations", f"{command} needs more observations "
                            f"({frame.n}) than parameters ({size}).")
    events = torch.zeros(q, dtype=FLOAT).index_add_(0, groups, delta)
    if frailty == "invgaussian" and float(events.max()) > MAX_EVENTS:
        raise AnalysisError("resource_limit", f"{command}: a group has more than {MAX_EVENTS} "
                            "failures (inverse-Gaussian Bessel sums are dense in that count).")
    base = FrailtyLikelihood(distribution, metric, design.x, t, delta, groups, q, offset, None)
    start = torch.zeros(base.size, dtype=FLOAT)
    start[0], anc = _start(distribution, metric, t, delta)
    if weibull:
        start[k] = anc
    null = _newton(base, start, f"{command} (model without frailty)")
    if not null.converged:
        raise AnalysisError("nonconvergence", f"{command}: the model without frailty did not "
                            f"converge ({null.diagnostics.get('message')}); check for covariates "
                            "that separate failures from survivors.")
    log_time = float((delta * torch.log(t)).sum())
    null_ll = float(null.value)
    model = FrailtyLikelihood(distribution, metric, design.x, t, delta, groups, q, offset, frailty)
    best, attempts = None, []
    for theta0 in STARTS:
        trial = torch.cat([null.theta, torch.tensor([math.log(theta0)], dtype=FLOAT)])
        try:
            result = _newton(model, trial, command, watch=True)
        except _Boundary:
            attempts.append({"start_theta": theta0, "outcome": "boundary"})
            continue
        except AnalysisError as error:
            attempts.append({"start_theta": theta0, "outcome": error.code})
            continue
        attempts.append({"start_theta": theta0, "outcome": "converged" if result.converged
                         else "not converged", "log_likelihood": float(result.value) + log_time})
        if result.converged and (best is None or result.value > best.value):
            best = result
    interior = best is not None and float(best.value) - null_ll > BOUNDARY_GAIN * max(1.0, abs(null_ll))
    if best is None and base.boundary_score(null.theta) > 0:
        # The likelihood increases away from theta = 0, so the maximum is interior.
        raise AnalysisError("nonconvergence", f"{command}: no Newton run of the frailty model "
                            "converged although the likelihood increases from theta = 0; "
                            "check the scale of the covariates.")
    chosen = best if interior else null
    jacobian = torch.eye(chosen.theta.numel(), dtype=FLOAT)
    jacobian[:k, :k] = constant_shift(means)
    covariance = kernel_call(information_inverse, -chosen.hessian)
    params, covariance = jacobian @ chosen.theta, jacobian @ covariance @ jacobian.T
    terms = list(design.terms)
    equations: list[str | None] = [spec.outcome] * k
    if weibull:
        terms.append("/ln_p")
        equations.append(None)
    if interior:
        terms.append("/ln_the")
        equations.append(None)
    theta = math.exp(float(params[-1])) if interior else 0.0
    log_likelihood = float(chosen.value) + log_time
    null_log_likelihood = null_ll + log_time
    parameters = len(terms)
    slopes = [i for i, term in enumerate(design.terms) if term != "Intercept"]
    tests = {"model": wald_test(params, covariance, slopes,
                                label="Wald chi2 test of the main-equation covariates"),
             "theta": chibar_test(2.0 * (log_likelihood - null_log_likelihood),
                                  f"LR test of theta = 0 ({frailty} frailty; chibar2(01): chi2(1) "
                                  "tail halved)")}
    z = critical_value(spec.alpha, None)
    se = covariance.diagonal().sqrt()
    ratio = "hazard_ratios" if metric == "ph" else "time_ratios"
    extra: dict[str, Any] = {
        "distribution": distribution, "metric": metric,
        ratio: {terms[i]: {"ratio": safe_exp(float(params[i])),
                           "std_error": safe_exp(float(params[i])) * float(se[i]),
                           "ci_low": safe_exp(float(params[i] - z * se[i])),
                           "ci_high": safe_exp(float(params[i] + z * se[i]))} for i in slopes},
        "frailty": {"distribution": frailty, "shared": shared, "theta": theta,
                    "boundary": not interior, "starts": attempts,
                    "score_at_zero": base.boundary_score(null.theta)},
        "null_log_likelihood": null_log_likelihood,
        "log_likelihood_scale": "log time (Stata): time-scale ll + sum d ln t",
    }
    if interior:
        j = terms.index("/ln_the")
        value, s = float(params[j]), float(se[j])
        extra["theta"] = {"estimate": theta, "std_error": theta * s,
                          "ci_low": safe_exp(value - z * s), "ci_high": safe_exp(value + z * s)}
    else:
        extra["notes"] = ["The frailty variance is estimated at the boundary theta = 0: the fit "
                          "is the model without frailty."]
    if weibull:
        j = terms.index("/ln_p")
        value, s = float(params[j]), float(se[j])
        extra["p"] = {"estimate": safe_exp(value), "std_error": safe_exp(value) * s,
                      "ci_low": safe_exp(value - z * s), "ci_high": safe_exp(value + z * s)}
    t_sum = float(t.sum())
    metrics = {"log_likelihood": log_likelihood, "log_likelihood_no_frailty": null_log_likelihood,
               "theta": theta, "aic": -2 * log_likelihood + 2 * parameters,
               "bic": -2 * log_likelihood + math.log(frame.n) * parameters,
               "df_model": len(slopes), "n_groups": q, "n_failures": float(delta.sum()),
               "time_at_risk": t_sum}
    title = (f"{'Weibull' if weibull else 'Exponential'} {'PH' if metric == 'ph' else 'AFT'} "
             f"regression with shared {frailty} frailty")
    provenance = {"distribution": distribution, "metric": metric, "frailty": frailty,
                  "derivatives": "analytic", "integration": "closed-form marginal likelihood",
                  "log_likelihood_scale": "log survival time (Stata streg)"}
    return build_result(
        frame, terms=terms, params=params, covariance=covariance, use_t=False,
        equations=equations, metrics=metrics, nobs=frame.n, tests=tests, extra=extra,
        inference={"correction": "observed information (inverse negative Hessian)",
                   "covariance": "nonrobust"},
        title=title, categories=design.categories, provenance=provenance,
        solver="newton_observed_hessian",
        solver_diagnostics={"converged": True, "iterations": chosen.iterations},
        optimizer=optimizer_record(chosen))


def streg_frailty(*, data: Any, time: str, shared: str, x: Sequence[str] | None = None,
                  failure: str | None = None, distribution: str = "weibull",
                  metric: str | None = None, frailty: str = "gamma", offset: str | None = None,
                  categorical: Sequence[str] | None = None, missing: str = "raise",
                  alpha: float = 0.05) -> ResultBundle:
    """Exponential or Weibull regression with a shared gamma or inverse-Gaussian frailty.

    Model
        Conditional hazard ``z_i h(t | x)`` with the ``oe.streg`` exponential/Weibull
        ``h`` (PH or AFT metric) and a frailty ``z_i`` shared by the records of group i,
        gamma or inverse Gaussian with mean 1 and variance theta (Stata ``streg,
        shared() frailty()``).

    Estimator
        Maximum likelihood of the closed-form marginal likelihood (the frailty integrated
        out analytically) by Newton-Raphson with analytic derivatives, jointly in
        ``(b, /ln_p, /ln_the)``; theta = 0 (the model without frailty) when the likelihood
        has no interior maximum.

    Parameters
        data, time, failure, x, offset, categorical, missing, alpha: as in ``oe.streg``
            (no delayed entry).
        shared: group column of the shared frailty.
        distribution: ``'weibull'`` (default) or ``'exponential'``.
        metric: ``'ph'`` (default) or ``'aft'``.
        frailty: ``'gamma'`` (default) or ``'invgaussian'``.

    Result
        Coefficients with observed-information z tests, including ``/ln_p`` and
        ``/ln_the``. ``metrics``: log_likelihood (Stata's log-time scale),
        log_likelihood_no_frailty, theta, aic, bic, n_groups, n_failures. ``tests``:
        ``theta`` (LR chibar2(01) of theta = 0), ``model`` (Wald). ``extra``: hazard or
        time ratios, ``theta`` and ``p`` with exp-transformed intervals.

    Stata
        ``streg x1 x2, distribution(weibull) frailty(gamma) shared(g)`` is
        ``oe.streg_frailty(data=df, time='t', failure='d', x=['x1', 'x2'], shared='g')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(2)
        >>> g = np.repeat(np.arange(50), 4); z = rng.gamma(2, 0.5, 50)[g]
        >>> x = rng.normal(size=200); t = rng.weibull(1.3, 200) / (z * np.exp(0.5 * x)) ** (1 / 1.3)
        >>> df = pd.DataFrame({"t": t, "d": 1, "x": x, "g": g})
        >>> print(oe.streg_frailty(data=df, time="t", failure="d", x=["x"], shared="g").summary())
    """
    spec = build_spec(
        "streg_frailty", outcome=time, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), covariance="nonrobust",
        missing=missing, alpha=alpha,
        columns={"failure": failure, "shared": shared, "offset": offset},
        options={"distribution": distribution, "metric": metric, "frailty": frailty})
    return fit(spec, data=data)
