"""Random-intercept Tobit (censored normal) by adaptive Gauss-Hermite quadrature.

    y*_ij = x_ij'b + u_j + e_ij,   u_j ~ N(0, s_u^2),   e_ij ~ N(0, s_e^2),

observed as y = y* when ll < y* < ul, as left-censored at ``ll`` when
y <= ll and right-censored at ``ul`` when y >= ul (Stata ``metobit`` /
``xttobit, re`` with constant limits). Conditional on u_j the row densities
are phi((y - eta)/s_e)/s_e, Phi((ll - eta)/s_e) and Phi((eta - ul)/s_e);
the random intercept is integrated by mode-curvature adaptive Gauss-Hermite
quadrature with modes recomputed at every parameter value (``agq``) or by the
non-adaptive rule. With no censored rows the integrand is Gaussian in u and
adaptive quadrature is exact, so the fit equals the Gaussian random-intercept
ML (``mixed``). Coordinates b, ln s_e, ln s_u; exact Torch derivatives; full
observed information; variances by the delta method.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, kernel_call, make_spec,
    wald_test,
)
from openecon.econometrics.mixed import common
from openecon.econometrics.mixed.agq import RandomInterceptQuadrature
from openecon.econometrics.mixed.extended_common import Objective, maximize, optimizer_record
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import least_squares

_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


def _censored_tail(w):
    """ln Phi(w), lambda = phi/Phi and d lambda/dw = -lambda (w + lambda)."""
    log_cdf = torch.special.log_ndtr(w)
    ratio = torch.exp(-0.5 * w.square() - _HALF_LOG_2PI - log_cdf)
    return log_cdf, ratio, -ratio * (w + ratio)


class TobitPieces:
    def __init__(self, y, left, right):
        self.y, self.left, self.right = y, left, right
        self.limit_left = torch.where(left, y, torch.zeros_like(y))
        self.limit_right = torch.where(right, y, torch.zeros_like(y))

    def __call__(self, log_sigma):
        sigma = torch.exp(log_sigma)

        def pieces(eta):
            y, left, right = self.y, self.left, self.right
            if eta.ndim == 2:
                y, left, right = y[:, None], left[:, None], right[:, None]
            z = (y - eta) / sigma
            value_u = -0.5 * z.square() - _HALF_LOG_2PI - log_sigma
            d1_u = z / sigma
            d2_u = -torch.ones_like(z) / sigma.square()
            lc, lr, ls = _censored_tail(z)                     # left: Phi((ll - eta)/s)
            rc, rr, rs = _censored_tail(-z)                    # right: Phi((eta - ul)/s)
            value = torch.where(left, lc, torch.where(right, rc, value_u))
            d1 = torch.where(left, -lr / sigma, torch.where(right, rr / sigma, d1_u))
            d2 = torch.where(left, ls / sigma.square(), torch.where(right, rs / sigma.square(),
                                                                    d2_u))
            return value, d1, d2

        return pieces


def fit_metobit(spec, data):
    """Entry point of ``metobit``: ``(spec, data) -> ResultBundle``."""
    with torch.device("cpu"), torch.no_grad():
        frame = ModelFrame(spec, data)
        group = frame.role("group")[0]
        if group == spec.outcome or group in spec.predictors:
            raise AnalysisError("invalid_spec", "The grouping column must differ from the "
                                "outcome and the regressors.")
        lower, upper = frame.option("ll"), frame.option("ul")
        if lower is None and upper is None:
            raise AnalysisError("invalid_spec", "Give a left (ll) and/or right (ul) censoring "
                                "limit; without censoring use oe.mixed.")
        for limit in (lower, upper):
            if limit is not None and (isinstance(limit, bool) or not math.isfinite(limit)):
                raise AnalysisError("invalid_option", "Censoring limits must be finite numbers.")
        if lower is not None and upper is not None and not lower < upper:
            raise AnalysisError("invalid_option", "ll must be smaller than ul.")
        design = frame.drop_collinear(frame.design(), None)
        if not design.terms:
            raise AnalysisError("invalid_spec", "At least one fixed-effect term is required.")
        x, y = design.x, frame.numeric(spec.outcome)
        left = y <= lower if lower is not None else torch.zeros_like(y, dtype=torch.bool)
        right = y >= upper if upper is not None else torch.zeros_like(y, dtype=torch.bool)
        observed = ~(left | right)
        if int(observed.sum()) <= x.shape[1] + 2:
            raise AnalysisError("insufficient_observations", "Too few uncensored observations "
                                "to identify the regression and both variances.")
        values = torch.where(left, torch.full_like(y, lower if lower is not None else 0.0),
                             torch.where(right, torch.full_like(y, upper if upper is not None
                                                               else 0.0), y))
        codes, count = frame.codes(group)
        if count < 3:
            raise AnalysisError("insufficient_groups", "At least three groups are required.")
        p = x.shape[1]
        k = p + 2
        points = frame.option("intpoints")
        adaptive = frame.option("intmethod") == "mcaghermite"
        frame.workspace_plan("random-intercept Tobit quadrature likelihood", {
            "row node likelihood and derivative tape": 8 * frame.n * (2 * points + 2) * (k + 8),
            "group nodes and modes": 8 * count * (2 * points + 8) * (k + 4),
        })
        quadrature = RandomInterceptQuadrature(codes, count, points, adaptive)
        pieces = TobitPieces(values, left, right)

        def value(theta):
            eta = x @ theta[:p]
            try:
                return quadrature.log_likelihood(pieces(theta[p]), eta, torch.exp(theta[p + 1]))
            except KernelError:
                return theta.sum() * math.nan

        ols = least_squares(x[observed], y[observed], drop_collinear=False)
        sd = float((ols.ssr / max(int(observed.sum()) - p, 1)) ** 0.5)
        if not math.isfinite(sd) or sd <= 0:
            raise AnalysisError("perfect_fit", "The uncensored outcomes are fitted exactly.")
        starts = [torch.cat([ols.beta, torch.tensor([math.log(sd * a), math.log(sd * b)],
                                                    dtype=torch.float64)])
                  for a, b in ((0.8, 0.5), (0.6, 1.0))]
        objective = Objective(value, 40 * frame.n * (2 * points + 6), frame.option("max_work"))
        fit, raw = maximize(objective, starts)
        theta = fit.theta
        if float(torch.exp(theta[p + 1] - theta[p])) < 1e-4:
            raise AnalysisError("boundary_solution", "The random-intercept variance is estimated "
                                "at zero; fit a pooled Tobit model instead.")

        def report(t):
            return torch.cat([t[:p], torch.exp(2 * t[p + 1:p + 2]), torch.exp(2 * t[p:p + 1])])

        with torch.enable_grad():
            jacobian = torch.autograd.functional.jacobian(report, theta)
        covariance = jacobian @ raw @ jacobian.T
        covariance = (covariance + covariance.T) / 2
        params = report(theta)
        terms = [*design.terms, f"/var(_cons[{group}])", "/var(Residual)"]
        eta = x @ theta[:p]
        doubled = min(2 * points, 128)
        check = RandomInterceptQuadrature(codes, count, doubled, adaptive)
        loglik_doubled = float(kernel_call(check.log_likelihood, pieces(theta[p]), eta,
                                           torch.exp(theta[p + 1])))
        log_likelihood = float(fit.value)
        criteria = information_criteria(log_likelihood, k, frame.n)
        sizes = common.group_sizes(codes, count)
        slopes = [i for i, term in enumerate(design.terms) if term != "Intercept"]
        tests = {"model": wald_test(params, covariance, slopes,
                                    label="Wald chi2 test of the slopes")} if slopes else {}
        bundle = build_result(
            frame, terms=terms, params=params, covariance=covariance,
            title="Mixed-effects Tobit regression",
            equations=[spec.outcome] * p + [group, "Residual"], use_t=False,
            metrics={"log_likelihood": log_likelihood, "aic": criteria["aic"],
                     "bic": criteria["bic"], "n_groups": count,
                     "n_left_censored": int(left.sum()), "n_right_censored": int(right.sum()),
                     "n_uncensored": int(observed.sum()),
                     "group_size_min": sizes["size_min"], "group_size_avg": sizes["size_avg"],
                     "group_size_max": sizes["size_max"]},
            fitted=eta, solver="native adaptive Gauss-Hermite random-intercept Tobit likelihood",
            optimizer=optimizer_record(fit, objective), tests=tests,
            inference={"covariance": "nonrobust",
                       "correction": "inverse full observed information of (b, ln s_e, ln s_u), "
                                     "delta method to the variances"},
            extra={
                "group": group, "ll": lower, "ul": upper,
                "censoring_rule": "y <= ll left-censored, y >= ul right-censored",
                "intpoints": points, "intmethod": "mcaghermite" if adaptive else "ghermite",
                "adaptation": "mode and curvature recomputed at every parameter value",
                "quadrature_check": {"intpoints": points, "log_likelihood": log_likelihood,
                                     "doubled_intpoints": doubled,
                                     "doubled_log_likelihood": loglik_doubled,
                                     "difference": loglik_doubled - log_likelihood},
                "likelihood_theta": theta.tolist(), "likelihood_covariance": raw.tolist(),
                "theta_parameterization": "b, ln sd(e), ln sd(u)",
                "fitted_target": "latent linear prediction x'b (u = 0)",
            },
            categories=design.categories,
        )
        common.variance_intervals(bundle, [p, p + 1], spec.alpha)
        return bundle


def metobit(*, data: Any, y: str, x=None, group: str, ll: float | None = None,
            ul: float | None = None, intpoints: int = 12, intmethod: str = "mcaghermite",
            categorical=None, intercept: bool = True, missing: str = "raise", alpha: float = 0.05,
            max_work: int = 2_000_000_000):
    """Random-intercept Tobit by adaptive Gauss-Hermite quadrature (Stata ``metobit``, ``xttobit, re``).

    ``y* = x'b + u_g + e`` with normal u and e, left-censored at ``ll`` (rows with
    ``y <= ll``) and/or right-censored at ``ul`` (``y >= ul``). Terms: fixed effects,
    ``/var(_cons[group])`` and ``/var(Residual)``; covariance from the full observed
    information. ``intmethod='mcaghermite'`` recomputes the adaptive modes at every
    parameter value; ``'ghermite'`` is non-adaptive.

    Stata: ``metobit y x || g:, ll(0)`` is ``oe.metobit(data=df, y='y', x=['x'], group='g',
    ll=0)``.
    """
    from openecon.analysis import fit

    return fit(make_spec(
        "metobit", outcome=y, predictors=column_list(x or [], "x"), intercept=intercept,
        categorical=column_list(categorical, "categorical"), missing=missing, alpha=alpha,
        columns={"group": group},
        options={"ll": ll, "ul": ul, "intpoints": intpoints, "intmethod": intmethod,
                 "max_work": max_work}),
        data=data)
