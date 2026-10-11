"""GARCH-MIDAS: a GARCH short-run component times a MIDAS long-run component.

Engle, R. F., Ghysels, E. and Sohn, B. (2013), "Stock market volatility and
macroeconomic fundamentals", Review of Economics and Statistics 95(3), 776-797, with the
conventions of the authors-adjacent R package mfGARCH 0.2.2 (Conrad and Kleen 2020,
"Two are better than one", Journal of Applied Econometrics 35, 19-45; source
github.com/cran/mfGARCH: R/helper_functions.R ``calculate_phi``, ``llh_mf``,
src/calculate_g.cpp, src/sum_tau.cpp). For day t of low-frequency period s(t):

    y_t   = mu + sqrt(tau_t g_t) Z_t,                      Z_t ~ N(0, 1)  (quasi-ML)
    tau_t = exp(m + theta sum_(k=1..K) phi_k(w1, w2) X_(s(t)-k))         long run, per period
    u_t   = (y_t - mu) / sqrt(tau_t)
    g_t   = (1 - alpha - beta - gamma/2) + (alpha + gamma 1(u_(t-1) < 0)) u_(t-1)^2
            + beta g_(t-1)                                               short run
    phi_k = c_k / sum_j c_j,   c_k = (k/(K+1))^(w1-1) (1 - k/(K+1))^(w2-1)

X is one value per period: a covariate column (constant within each period) or, by
default, the realized variance RV_s = sum_(t in s) y_t^2 (Engle, Ghysels and Sohn's
fixed-span RV). The first K periods only supply lags; the likelihood is that of the days
of periods K+1..S (as mfGARCH). The short-run recursion starts at its unconditional mean,
g = 1 on the first likelihood day (mfGARCH starts at the sample variance of y instead).
``weighting="beta_restricted"`` fixes w1 = 1 (estimating w2; K >= 2), ``"beta_unrestricted"``
estimates both (K >= 3); with K = 1 there is no weight parameter.

Evaluation has no loop over time: the short-run recursion and all its derivative paths
are first-order filters (``arima.filters.inverse_filter``) of known inputs at given
parameters, and the long-run component is a K-column product with the lagged covariate.
Scores are analytic; the Hessian is the numerical derivative of the analytic gradient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arch.recursions import lagged
from openecon.econometrics.arima.filters import inverse_filter
from openecon.econometrics.core import (
    ModelFrame, build_result, information_criteria, make_spec, ml_covariance, table,
)
from openecon.engines import optimize
from openecon.engines.contracts import KernelError
from openecon.engines.distributions import normal_isf
from openecon.models import ModelSpec, ResultBundle

MAX_LAGS = 400                 # largest K (low-frequency lags of the long-run component)
_WEIGHT_BOUNDS = (1e-3, 500.0)
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_CORRECTIONS = {
    "robust": "Huber-White quasi-maximum-likelihood sandwich (Bollerslev-Wooldridge), N/(N-1); "
              "mfGARCH's rob.std.err without the N/(N-1) factor",
    "opg": "outer product of the per-observation analytic scores",
    "nonrobust": "observed information: numerical Hessian of the analytic gradient",
}


@dataclass(frozen=True)
class MidasLayout:
    lags: int
    weighting: str          # beta_restricted | beta_unrestricted
    asymmetric: bool
    mean: bool

    @property
    def weight_names(self) -> list[str]:
        if self.lags == 1:
            return []
        return ["w2"] if self.weighting == "beta_restricted" else ["w1", "w2"]

    @property
    def names(self) -> list[str]:
        return [*(["mu"] if self.mean else []), "alpha", *(["gamma"] if self.asymmetric else []),
                "beta", "m", "theta", *self.weight_names]

    def position(self, name: str) -> int:
        return self.names.index(name)

    def terms(self, outcome: str) -> tuple[list[str], list[str]]:
        terms, equations = [], []
        for name in self.names:
            if name == "mu":
                terms.append("Intercept")
                equations.append(outcome)
            elif name in {"alpha", "gamma", "beta"}:
                terms.append(f"GARCH:{name}")
                equations.append("GARCH")
            else:
                terms.append(f"MIDAS:{name}")
                equations.append("MIDAS")
        return terms, equations


def beta_weights(lags: int, w1: float, w2: float, derivatives: bool = False):
    """phi_1..phi_K (mfGARCH ``calculate_phi``) and d phi / d (w1, w2) as ``[2, K]``."""
    grid = torch.arange(1, lags + 1, dtype=torch.float64) / (lags + 1)
    log_left, log_right = grid.log(), torch.log1p(-grid)
    log_c = (w1 - 1.0) * log_left + (w2 - 1.0) * log_right
    phi = torch.softmax(log_c, 0)
    if not derivatives:
        return phi, None
    d_phi = torch.stack([phi * (log_left - phi @ log_left), phi * (log_right - phi @ log_right)])
    return phi, d_phi


@dataclass
class MidasEvaluation:
    value: float
    gradient: Tensor | None
    scores: Tensor | None
    loglik: Tensor
    residual: Tensor           # y - mu on the likelihood days
    tau: Tensor                # per likelihood day
    g: Tensor
    log_tau_periods: Tensor    # ln tau of the periods K+1..S


class MidasLikelihood:
    """Gaussian quasi-likelihood of one GARCH-MIDAS specification."""

    def __init__(self, y: Tensor, period: Tensor, covariate: Tensor, layout: MidasLayout):
        self.layout, self.y, self.period, self.covariate = layout, y, period, covariate
        K = layout.lags
        self.first = int((period >= K).nonzero()[0])          # first likelihood day
        self.days = period[self.first:] - K                     # index into periods K..S-1
        self.n = int(y.shape[0]) - self.first
        count = int(covariate.shape[0])
        rows = torch.arange(K, count)[:, None] - torch.arange(1, K + 1)[None, :]
        self.lagged_covariate = covariate[rows]                 # [S - K, K]: X_(s-k)

    def split(self, theta: Tensor) -> dict[str, float]:
        values = dict(zip(self.layout.names, theta.tolist(), strict=True))
        values.setdefault("mu", 0.0)
        values.setdefault("gamma", 0.0)
        values.setdefault("w1", 1.0)
        values.setdefault("w2", 1.0)
        return values

    @torch.no_grad()
    def evaluate(self, theta: Tensor, *, scores: bool = False, derivatives: bool = True
                 ) -> MidasEvaluation | None:
        layout = self.layout
        if not bool(torch.isfinite(theta).all()):
            return None
        p = self.split(theta)
        alpha, beta, gamma = p["alpha"], p["beta"], p["gamma"]
        omega = 1.0 - alpha - beta - 0.5 * gamma
        low, high = _WEIGHT_BOUNDS
        if not omega > 0.0 or not -1.0 < beta < 1.0 \
                or not all(low < p[name] < high for name in layout.weight_names):
            return None
        phi, d_phi = beta_weights(layout.lags, p["w1"], p["w2"], derivatives)
        combined = self.lagged_covariate @ phi                  # sum_k phi_k X_(s-k)
        log_tau_periods = p["m"] + p["theta"] * combined
        if not bool(torch.isfinite(log_tau_periods).all()) \
                or float(log_tau_periods.abs().max()) > 600.0:
            return None
        log_tau = log_tau_periods[self.days]
        tau = log_tau.exp()
        e = self.y[self.first:] - p["mu"]
        root = (-0.5 * log_tau).exp()
        u = e * root
        u2 = u.square()
        negative = (u < 0).to(torch.float64)
        slope = alpha + gamma * negative                         # coefficient of u^2
        forcing = omega + lagged(slope * u2, 1, 0.0)
        forcing[0] = 1.0                                         # g = 1 on the first day
        g = inverse_filter(forcing, [-beta]) if beta != 0.0 else forcing
        if not bool(torch.isfinite(g).all()) or not bool((g > 0.0).all()):
            return None
        h = tau * g
        ratio = e.square() / h
        loglik = -_HALF_LOG_2PI - 0.5 * (log_tau + g.log()) - 0.5 * ratio
        value = float(loglik.sum())
        if not math.isfinite(value):
            return None
        out = MidasEvaluation(value, None, None, loglik, e, tau, g, log_tau_periods)
        if not derivatives:
            return out
        n = self.n
        later = torch.ones(n, dtype=torch.float64)
        later[0] = 0.0
        rows, log_tau_rows = [], []
        names = layout.names
        for name in names:
            if name == "mu":
                direct = lagged(slope * (-2.0 * u * root), 1, 0.0)
                log_tau_rows.append(torch.zeros(n, dtype=torch.float64))
            elif name == "alpha":
                direct = later * (-1.0) + lagged(u2, 1, 0.0)
                log_tau_rows.append(torch.zeros(n, dtype=torch.float64))
            elif name == "gamma":
                direct = later * (-0.5) + lagged(negative * u2, 1, 0.0)
                log_tau_rows.append(torch.zeros(n, dtype=torch.float64))
            elif name == "beta":
                direct = later * (-1.0) + lagged(g, 1, 0.0)
                log_tau_rows.append(torch.zeros(n, dtype=torch.float64))
            else:
                if name == "m":
                    period_slope = torch.ones_like(combined)
                elif name == "theta":
                    period_slope = combined
                else:
                    period_slope = p["theta"] * (self.lagged_covariate
                                                 @ d_phi[0 if name == "w1" else 1])
                day_slope = period_slope[self.days]
                # u^2 = e^2 / tau: d u^2 = -u^2 d ln tau.
                direct = lagged(slope * (-u2) * day_slope, 1, 0.0)
                log_tau_rows.append(day_slope)
            rows.append(direct)
        d_g = torch.stack(rows)
        if beta != 0.0:
            d_g = inverse_filter(d_g, [-beta])
        d_log_h = torch.stack(log_tau_rows) + d_g / g
        weight = -0.5 + 0.5 * ratio
        score = d_log_h * weight
        if layout.mean:
            score[0] += e / h
        if not bool(torch.isfinite(score).all()):
            return None
        out.gradient = score.sum(dim=1)
        out.scores = score.T.contiguous() if scores else None
        return out


# ---- estimation --------------------------------------------------------------------------------


def _prepare(spec: ModelSpec, data: Any):
    frame = ModelFrame(spec, data)
    period_name = frame.role("period")[0]
    covariate_names = frame.role("covariate")
    names = {spec.outcome, period_name, *covariate_names}
    if spec.time is not None and spec.time in names:
        raise AnalysisError("invalid_spec", "The time column must differ from the outcome, the "
                            "period and the covariate.")
    if len(names) != 2 + len(covariate_names):
        raise AnalysisError("invalid_spec", "The outcome, the period column and the covariate "
                            "must be distinct columns.")
    if spec.time is not None:
        frame.sort_panel()
        order_column = frame.original[spec.time]
        if frame.n < len(order_column):
            # Rows excluded for missing values must not lie inside the series.
            order = torch.as_tensor(order_column.sort_values(kind="stable", na_position="last")
                                    .index.to_numpy(copy=True))
            kept = torch.zeros(len(order_column), dtype=torch.bool)
            kept[torch.as_tensor(frame.positions)] = True
            where = kept[order].nonzero().flatten()
            if int(where[-1] - where[0]) + 1 != len(where):
                raise AnalysisError("time_gaps", "Observations with missing values lie inside "
                                    "the series: GARCH-MIDAS needs an uninterrupted series.")
    elif frame.n > 1:
        positions = torch.as_tensor(frame.positions)
        if bool((positions[1:] - positions[:-1] != 1).any()):
            raise AnalysisError("time_gaps", "Observations with missing values lie inside the "
                                "series: GARCH-MIDAS needs an uninterrupted daily series.")
    column = frame.series(period_name)
    if is_datetime64_any_dtype(column.dtype) or is_bool_dtype(column.dtype):
        raise AnalysisError("invalid_period", "The period column must hold integer period "
                            "numbers (for example 12 * year + month); convert dates first.")
    period_values = frame.numeric(period_name)
    if bool((period_values != period_values.round()).any()):
        raise AnalysisError("invalid_period", "The period column must hold integers.")
    steps = period_values[1:] - period_values[:-1]
    if bool(((steps != 0) & (steps != 1)).any()):
        raise AnalysisError("invalid_period", "Periods must be consecutive integers that never "
                            "decrease in time order (each day belongs to the period of the "
                            "previous day or to the next one); gaps in the low-frequency series "
                            "are not allowed.")
    period = (period_values - period_values[0]).to(torch.int64)
    y = frame.numeric(spec.outcome)
    count = int(period[-1]) + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.int64),
                        (steps != 0).nonzero().flatten() + 1])
    if covariate_names:
        values = frame.numeric(covariate_names[0])
        covariate = values[starts]
        if bool((values != covariate[period]).any()):
            raise AnalysisError("invalid_covariate", f"The covariate '{covariate_names[0]}' must "
                                "be constant within each period (one low-frequency value per "
                                "period).")
        source = covariate_names[0]
    else:
        covariate = torch.zeros(count, dtype=torch.float64).index_add_(0, period, y.square())
        source = "realized variance: sum of squared outcomes within the period"
    return frame, y, period, covariate, source, period_values


def _starts(like: MidasLikelihood) -> list[Tensor]:
    layout = like.layout
    e = like.y[like.first:]
    mu = float(e.mean()) if layout.mean else 0.0
    level = math.log(float((e - mu).square().mean()))
    starts = []
    for alpha, beta in ((0.05, 0.90), (0.10, 0.80), (0.03, 0.95)):
        values = {"mu": mu, "alpha": alpha, "gamma": 0.05, "beta": beta - 0.03 * layout.asymmetric,
                  "m": level, "theta": 0.0, "w1": 1.5, "w2": 3.0}
        starts.append(torch.tensor([values[name] for name in layout.names], dtype=torch.float64))
    return starts


def _maximize(like: MidasLikelihood, tolerance: float, max_iterations: int):
    layout = like.layout
    e = like.y[like.first:]
    scale = torch.ones(len(layout.names), dtype=torch.float64)
    spread = float(like.lagged_covariate.std()) if like.lagged_covariate.numel() > 1 else 1.0
    if layout.mean:
        scale[0] = float(e.std()) or 1.0
    scale[layout.position("theta")] = 1.0 / spread if spread > 0.0 else 1.0
    infeasible = torch.full_like(scale, math.nan)

    def objective(t: Tensor):
        out = like.evaluate(t * scale)
        if out is None:
            return -math.inf, infeasible
        return out.value, out.gradient * scale

    def gradient(t: Tensor) -> Tensor:
        out = like.evaluate(t * scale)
        return infeasible if out is None else out.gradient * scale

    def hessian(t: Tensor) -> Tensor:
        # The GJR indicator makes the gradient kink at a zero residual: small first step.
        return optimize.numerical_hessian(gradient, t, relative_step=1e-6)

    ranked = []
    for number, start in enumerate(_starts(like)):
        out = like.evaluate(start, derivatives=False)
        if out is not None:
            ranked.append((out.value, number, start))
    ranked.sort(key=lambda item: -item[0])
    if not ranked:
        raise AnalysisError("invalid_start", "The GARCH-MIDAS likelihood could not be evaluated "
                            "at any starting value; rescale the outcome or the covariate.")
    failure = "no starting value could be improved."
    for _, number, start in ranked:
        first = like.evaluate(start, scores=True)
        inverse = None
        if first is not None:
            scores = first.scores * scale
            try:
                inverse = optimize.information_inverse(scores.T @ scores)
            except KernelError:
                inverse = None
        try:
            result = optimize.maximize_bfgs(
                objective, start / scale, max_iter=max_iterations, gradient_tol=tolerance,
                hessian_fn=hessian, initial_inverse_hessian=inverse, raise_on_failure=False)
        except KernelError as exc:
            failure = str(exc)
            continue
        if result.converged:
            return result.theta * scale, scale, result, number
        failure = str(result.diagnostics.get("message"))
    raise AnalysisError(
        "nonconvergence", f"The GARCH-MIDAS likelihood maximization did not converge: {failure} "
        "Try fewer lags K, the restricted weighting, a covariate on a comparable scale (for "
        "example returns in percent), or raise max_iterations.")


def fit_garch_midas(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``garch_midas`` (see ``garch_midas`` for the description)."""
    frame, y, period, covariate, source, period_values = _prepare(spec, data)
    lags = int(frame.option("lags"))
    weighting = frame.option("weighting")
    if weighting == "beta_unrestricted" and lags < 3:
        raise AnalysisError("unsupported_combination", "weighting='beta_unrestricted' needs at "
                            "least lags=3: with fewer lags the two beta parameters are not "
                            "separately identified. Use weighting='beta_restricted'.")
    layout = MidasLayout(lags=lags, weighting=weighting,
                         asymmetric=bool(frame.option("asymmetric")), mean=spec.intercept)
    periods = int(covariate.shape[0])
    if periods < lags + 2:
        raise AnalysisError("insufficient_observations", f"GARCH-MIDAS with lags={lags} needs at "
                            f"least {lags + 2} low-frequency periods; the sample has {periods}.")
    tolerance = float(frame.option("tolerance"))
    if not tolerance > 0.0:
        raise AnalysisError("invalid_option", "tolerance must be positive.")
    like = MidasLikelihood(y, period, covariate, layout)
    k = len(layout.names)
    if like.n < k + 20:
        raise AnalysisError("insufficient_observations", f"GARCH-MIDAS with {k} parameters needs "
                            f"at least {k + 20} days after the first {lags} periods; the sample "
                            f"has {like.n}.")
    if not bool((y != y[0]).any()):
        raise AnalysisError("constant_outcome", "The outcome is constant.")
    theta, scale, result, number = _maximize(like, tolerance, int(frame.option("max_iterations")))
    final = like.evaluate(theta, scores=True)
    if final is None:
        raise AnalysisError("numerical_failure", "The likelihood is not defined at the estimates.")
    hessian = result.hessian / scale[:, None] / scale[None, :]
    covariance, info = ml_covariance(frame, hessian=hessian, scores=final.scores, nobs=like.n)
    info["correction"] = _CORRECTIONS[info["covariance"]]
    terms, equations = layout.terms(spec.outcome)
    p = like.split(theta)
    persistence = p["alpha"] + p["beta"] + 0.5 * p["gamma"]
    phi, _ = beta_weights(lags, p["w1"], p["w2"])
    # mfGARCH's variance ratio: share of the variance of ln(period mean of h) due to tau.
    first_period = like.days
    tau_mean = final.log_tau_periods.exp()
    h_sum = torch.zeros_like(tau_mean).index_add_(0, first_period, final.tau * final.g)
    days = torch.zeros_like(tau_mean).index_add_(0, first_period, torch.ones_like(final.g))
    variance_ratio = None
    if tau_mean.shape[0] > 2:
        log_h = (h_sum / days).log()
        variance_ratio = 100.0 * float(tau_mean.log().var() / log_h.var()) \
            if float(log_h.var()) > 0.0 else None
    if p["alpha"] < 0.0 or p["beta"] < 0.0 or p["alpha"] + p["gamma"] < 0.0:
        frame.warn("The short-run GARCH coefficients are negative (alpha, beta or alpha + "
                   "gamma): the fitted short-run component is positive in the sample but not "
                   "guaranteed to stay positive. No constraints other than alpha + beta + "
                   "gamma/2 < 1 are imposed (mfGARCH also requires alpha, beta >= 0).")
    state = {
        "g": float(final.g[-1]), "residual": float(final.residual[-1]),
        "tau": float(final.tau[-1]), "covariate_tail": covariate[-lags:].tolist(),
        "last_period": float(period_values[-1]),
    }
    metrics = {**information_criteria(final.value, k, like.n), "persistence": persistence,
               "variance_ratio": variance_ratio, "iterations": result.iterations}
    extra = {
        "model": {"lags": lags, "weighting": weighting, "asymmetric": layout.asymmetric,
                  "covariate": source, "periods": periods, "burn_in_periods": lags,
                  "burn_in_observations": like.first, "likelihood_observations": like.n,
                  "short_run_start": "g = 1 (its unconditional mean) on the first likelihood day"},
        "weights": phi.tolist(),
        "long_run_tail": final.log_tau_periods[-400:].exp().tolist(),
        "short_run_tail": final.g[-400:].tolist(),
        "state": state,
    }
    fitted = torch.full_like(y, p["mu"])
    return build_result(
        frame, terms=terms, params=theta, covariance=covariance,
        title=f"GARCH-MIDAS(K={lags}) regression, normal quasi-likelihood",
        equations=equations, use_t=False, metrics=metrics, fitted=fitted,
        solver="bfgs_garch_midas_likelihood",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer={"method": result.method, "iterations": result.iterations, "converged": True,
                   "starting_values": f"candidate_{number}",
                   "gradient": "analytic (first-order filters of the short-run recursion)",
                   "hessian": "numerical derivative of the analytic gradient (relative first "
                              "step 1e-6)",
                   "scaled_gradient": result.diagnostics.get("scaled_gradient")},
        inference=info, extra=extra, nobs=like.n,
        provenance={
            "likelihood": "Gaussian (quasi-)likelihood of the days of periods K+1..S",
            "reference": "Engle, Ghysels and Sohn (2013); conventions of mfGARCH 0.2.2 "
                         "(Conrad and Kleen), except the short-run start g = 1",
            "time_order": f"sorted by {spec.time}" if spec.time else "input row order",
        },
    )


def garch_midas(*, data: Any, y: str, period: str, covariate: str | None = None,
                lags: int = 12, weighting: str = "beta_restricted", asymmetric: bool = True,
                time: str | None = None, constant: bool = True, covariance: str | None = None,
                max_iterations: int = 500, tolerance: float = 1e-8, missing: str = "raise",
                alpha: float = 0.05) -> ResultBundle:
    """GARCH-MIDAS: GARCH (or GJR) short-run volatility times a MIDAS long-run component.

    Model (Engle, Ghysels and Sohn 2013; conventions of the R package mfGARCH 0.2.2)
    ----------------------------------------------------------------------------------
    For day t of low-frequency period s:

        y_t   = mu + sqrt(tau_s g_t) Z_t
        tau_s = exp(m + theta sum_(k=1..K) phi_k(w1, w2) X_(s-k))
        g_t   = (1 - alpha - beta - gamma/2) + (alpha + gamma 1(u_(t-1) < 0)) u_(t-1)^2
                + beta g_(t-1),        u_t = (y_t - mu) / sqrt(tau_(s(t)))
        phi_k = c_k / sum c_j,   c_k = (k/(K+1))^(w1-1) (1 - k/(K+1))^(w2-1)

    X is one value per period: ``covariate`` (a column constant within each period, for
    example a monthly macro variable repeated on every day of the month) or, without it,
    the realized variance sum_(t in s) y_t^2 of the period. The first ``lags`` periods
    supply the lags only; the likelihood uses the days of the later periods. ``g`` starts
    at its unconditional mean 1.

    Estimator: Gaussian quasi-maximum likelihood by BFGS with analytic gradients; the
    domain is alpha + beta + gamma/2 < 1 (positive short-run constant) and positive
    weight parameters. Covariance ``"robust"`` (default: QML sandwich with N/(N-1)),
    ``"opg"`` or ``"nonrobust"``; z inference.

    Parameters
    ----------
    data : table with one row per high-frequency observation.
    y : return column.
    period : integer low-frequency period of each row (consecutive integers that never
        decrease in time order, e.g. 12 * year + month).
    covariate : optional covariate column, constant within each period; default realized
        variance.
    lags : K, the number of low-frequency lags (1..400; default 12).
    weighting : ``"beta_restricted"`` (w1 = 1, estimates w2; needs K >= 2) or
        ``"beta_unrestricted"`` (w1 and w2; needs K >= 3). No weight parameter when K = 1.
    asymmetric : GJR short-run component (gamma; default True as mfGARCH) or GARCH(1,1).
    time : optional time column to sort by (otherwise the row order is the time order).
    constant : estimate the mean mu (default) or fix it at zero.
    covariance, max_iterations, tolerance, missing, alpha : as for ``oe.arch``.

    Result
    ------
    Terms ``Intercept``, ``GARCH:alpha``, ``GARCH:gamma``, ``GARCH:beta``, ``MIDAS:m``,
    ``MIDAS:theta``, ``MIDAS:w1``, ``MIDAS:w2``; metrics ``log_likelihood``, ``aic``, ``bic``,
    ``persistence`` (alpha + beta + gamma/2), ``variance_ratio`` (mfGARCH's share of the
    variance of the log period-mean variance explained by tau, in percent); ``extra``:
    the estimated weights, the last 400 long-run and short-run components and the state
    used by ``oe.forecast``.

    Example
    -------
    >>> fit = oe.garch_midas(data=df, y="ret", period="month", covariate="ip_growth", lags=24)
    >>> oe.forecast(fit, 20)
    """
    spec = make_spec(
        "garch_midas", outcome=y, time=time, intercept=constant, covariance=covariance,
        missing=missing, alpha=alpha,
        columns={"period": period, "covariate": covariate},
        options={"lags": lags, "weighting": weighting, "asymmetric": asymmetric,
                 "max_iterations": max_iterations, "tolerance": tolerance},
    )
    from openecon.analysis import fit

    return fit(spec, data=data)


def garch_midas_forecast(result: ResultBundle, steps: int, *, new_period: bool = True,
                         alpha: float = 0.05) -> pd.DataFrame:
    """Variance forecasts of a GARCH-MIDAS fit (``oe.forecast`` dispatches here).

    The forecast days belong to the period after the last sample period when
    ``new_period`` is true (default; its tau uses the last K covariate values, the last
    sample period included, which must then be complete) and to the last sample period
    otherwise. With rho = alpha + beta + gamma/2,
    E_n[g_(n+1)] = (1 - rho) + (alpha + gamma 1(u_n < 0)) u_n^2 + beta g_n and
    E_n[g_(n+k)] = 1 + rho^(k-1) (E_n[g_(n+1)] - 1) (Gaussian innovations: E[u^2 1(u<0)] =
    g/2); the variance forecast is tau times that. Intervals are normal and exact one step
    ahead; parameter uncertainty is ignored.
    """
    if not isinstance(result, ResultBundle) or result.spec.estimator != "garch_midas":
        raise AnalysisError("invalid_result", "garch_midas_forecast needs a result returned by "
                            "oe.garch_midas.")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 10_000:
        raise AnalysisError("invalid_steps", "steps must be an integer between 1 and 10000.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 < alpha < 1.0:
        raise AnalysisError("invalid_option", "alpha must be a number strictly between 0 and 1.")
    if not isinstance(new_period, bool):
        raise AnalysisError("invalid_option", "new_period must be True or False.")
    model, state = result.extra.get("model") or {}, result.extra.get("state")
    try:
        layout = MidasLayout(int(model["lags"]), model["weighting"], bool(model["asymmetric"]),
                             result.spec.intercept)
        values = {c.term: c.estimate for c in result.coefficients}
        terms, _ = layout.terms(result.spec.outcome)
        p = dict(zip(layout.names, [values[term] for term in terms], strict=True))
        g, e, tau = float(state["g"]), float(state["residual"]), float(state["tau"])
        tail = [float(value) for value in state["covariate_tail"]]
        if len(tail) != layout.lags or not all(math.isfinite(v) for v in (g, e, tau, *tail)) \
                or not (g > 0.0 and tau > 0.0):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise AnalysisError("invalid_result", "The result does not carry a valid GARCH-MIDAS "
                            "specification and end-of-sample state.") from None
    mu, a, b = p.get("mu", 0.0), p["alpha"], p["beta"]
    gamma = p.get("gamma", 0.0)
    rho = a + b + 0.5 * gamma
    if new_period:
        phi, _ = beta_weights(layout.lags, p.get("w1", 1.0), p.get("w2", 1.0))
        lags = torch.tensor(tail[::-1], dtype=torch.float64)       # X_S, X_(S-1), ...
        tau_next = math.exp(p["m"] + p["theta"] * float(phi @ lags))
    else:
        tau_next = tau
    u = e / math.sqrt(tau)
    first = (1.0 - rho) + (a + gamma * (u < 0)) * u * u + b * g
    expected = [1.0 + rho ** k * (first - 1.0) for k in range(steps)]
    variance = [tau_next * value for value in expected]
    if not all(math.isfinite(v) and v > 0.0 for v in variance):
        raise AnalysisError("non_finite_result", "The variance forecasts are not positive and "
                            "finite.")
    critical = normal_isf(0.5 * alpha)
    sd = [math.sqrt(v) for v in variance]
    return table(
        {"step": list(range(1, steps + 1)), "mean_forecast": [mu] * steps,
         "variance_forecast": variance, "long_run": [tau_next] * steps,
         "short_run": expected, "std_error": sd,
         "ci_low": [mu - critical * s for s in sd], "ci_high": [mu + critical * s for s in sd]},
        title=f"GARCH-MIDAS forecasts of {result.spec.outcome}", steps=steps, alpha=alpha,
        new_period=new_period, long_run_component=tau_next,
        variance_forecast_definition="tau times the conditional expectation of g",
        std_error_definition="innovation uncertainty only (no parameter uncertainty)")
