"""Random-intercept ordered logit/probit by adaptive Gauss-Hermite quadrature.

    Pr(y_ij <= k | u_j) = F(cut_k - x_ij'b - u_j),    u_j ~ N(0, s^2),

with F logistic (``meologit``) or standard normal (``meoprobit``), cutpoints
``cut_1 < ... < cut_{J-1}`` estimated directly (no constant; Stata's
``meologit`` and ordinal::clmm parameterization). The random intercept is
integrated by mode-curvature adaptive Gauss-Hermite quadrature with modes
recomputed at every parameter value (``agq``; clmm with nAGQ > 1) or by the
non-adaptive rule. Likelihood coordinates: b, cut_1, ln(cut_k - cut_{k-1}),
ln s; exact Torch gradient and Hessian; BFGS with a Newton polish. The
covariance of (b, cutpoints, var(u)) is the delta-method transform of the
inverse full observed information.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn.functional import logsigmoid

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, kernel_call, make_spec,
    wald_test,
)
from openecon.econometrics.discrete.common import implicit_constant_design
from openecon.econometrics.discrete.ordered import ordered_outcome
from openecon.econometrics.mixed import common
from openecon.econometrics.mixed.agq import RandomInterceptQuadrature
from openecon.econometrics.mixed.extended_common import Objective, maximize, optimizer_record
from openecon.engines.contracts import KernelError

_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


def _log_cdf(link):
    return logsigmoid if link == "logit" else torch.special.log_ndtr


def _log_pdf(link, x):
    if link == "logit":
        return logsigmoid(x) + logsigmoid(-x)
    return -0.5 * x.square() - _HALF_LOG_2PI


def _slope(link, x):
    """f'(x)/f(x)."""
    return 1 - 2 * torch.sigmoid(x) if link == "logit" else -x


def _tail(link, x):
    """ln F(x), r = f/F and dr/dx for a one-sided category."""
    log_cdf = _log_cdf(link)(x)
    ratio = torch.exp(_log_pdf(link, x) - log_cdf)
    return log_cdf, ratio, ratio * _slope(link, x) - ratio.square()


class OrderedPieces:
    """Elementwise ln Pr(y | eta), its first and second eta derivatives (F symmetric)."""

    def __init__(self, link: str, codes, categories: int):
        self.link, self.categories = link, categories
        self.first = codes == 0
        self.last = codes == categories - 1
        self.upper = codes.clamp(max=categories - 2)
        self.lower = (codes - 1).clamp(min=0)

    def __call__(self, cuts):
        link, log_cdf = self.link, _log_cdf(self.link)
        upper_cut, lower_cut = cuts[self.upper], cuts[self.lower]

        def pieces(eta):
            first, last, cu, cl = self.first, self.last, upper_cut, lower_cut
            if eta.ndim == 2:
                first, last, cu, cl = first[:, None], last[:, None], cu[:, None], cl[:, None]
            b, a = cu - eta, cl - eta
            lp1, r1, s1 = _tail(link, b)                       # Pr = F(b)
            lp2, r2, s2 = _tail(link, -a)                      # Pr = 1 - F(a) = F(-a)
            am = torch.where(first, b - 1, a)
            bm = torch.where(last, a + 1, b)
            low_form = log_cdf(bm) + torch.log1p(-torch.exp(log_cdf(am) - log_cdf(bm)))
            high_form = log_cdf(-am) + torch.log1p(-torch.exp(log_cdf(-bm) - log_cdf(-am)))
            lp3 = torch.where(am + bm > 0, high_form, low_form)
            fb = torch.exp(_log_pdf(link, bm) - lp3)
            fa = torch.exp(_log_pdf(link, am) - lp3)
            d1_3 = fa - fb
            d2_3 = _slope(link, bm) * fb - _slope(link, am) * fa - d1_3.square()
            value = torch.where(first, lp1, torch.where(last, lp2, lp3))
            d1 = torch.where(first, -r1, torch.where(last, r2, d1_3))
            d2 = torch.where(first, s1, torch.where(last, s2, d2_3))
            return value, d1, d2

        return pieces


def _cuts(theta, p, count):
    start = theta[p:p + 1]
    if count == 1:
        return start
    return torch.cat([start, start + torch.cumsum(torch.exp(theta[p + 1:p + count]), 0)])


def _fit(spec, data, link):
    with torch.device("cpu"), torch.no_grad():
        command = spec.estimator
        frame = ModelFrame(spec, data)
        group = frame.role("group")[0]
        if group == spec.outcome or group in spec.predictors:
            raise AnalysisError("invalid_spec", "The grouping column must differ from the "
                                "outcome and the regressors.")
        codes_y, labels = ordered_outcome(frame, command)
        categories = len(labels)
        design = implicit_constant_design(frame, None, None)
        x = design.x
        p, cuts_n = x.shape[1], categories - 1
        codes, count = frame.codes(group)
        if count < 3:
            raise AnalysisError("insufficient_groups", "At least three groups are required for "
                                "a random-intercept variance.")
        k = p + cuts_n + 1
        if frame.n <= k:
            raise AnalysisError("insufficient_observations", "More observations than "
                                "parameters are required.")
        points = frame.option("intpoints")
        adaptive = frame.option("intmethod") == "mcaghermite"
        frame.workspace_plan("random-intercept ordered quadrature likelihood", {
            "row node likelihood and derivative tape": 8 * frame.n * (2 * points + 2) * (k + 8),
            "group nodes and modes": 8 * count * (2 * points + 8) * (k + 4),
        })
        quadrature = RandomInterceptQuadrature(codes, count, points, adaptive)
        pieces = OrderedPieces(link, codes_y, categories)

        def value(theta):
            sigma = torch.exp(theta[-1])
            eta = x @ theta[:p]
            try:
                return quadrature.log_likelihood(pieces(_cuts(theta, p, cuts_n)), eta, sigma)
            except KernelError:
                return theta.sum() * math.nan

        shares = torch.bincount(codes_y, minlength=categories).to(torch.float64) / frame.n
        cumulative = torch.cumsum(shares, 0)[:-1].clamp(1e-6, 1 - 1e-6)
        base = torch.logit(cumulative) if link == "logit" else torch.special.ndtri(cumulative)
        gaps = torch.log(torch.diff(base).clamp_min(1e-3)) if cuts_n > 1 else base[:0]
        starts = [torch.cat([torch.zeros(p, dtype=torch.float64), base[:1], gaps,
                             torch.tensor([math.log(sd)], dtype=torch.float64)])
                  for sd in (0.5, 1.0)]
        work = 40 * frame.n * (2 * points + 6)
        objective = Objective(value, work, frame.option("max_work"))
        fit, raw = maximize(objective, starts)
        theta = fit.theta
        sigma = float(torch.exp(theta[-1]))
        if sigma < 1e-4:
            raise AnalysisError("boundary_solution", "The random-intercept variance is estimated "
                                "at zero; fit the pooled ordered model instead.")

        def report(t):
            return torch.cat([t[:p], _cuts(t, p, cuts_n), torch.exp(2 * t[-1:])])

        with torch.enable_grad():
            jacobian = torch.autograd.functional.jacobian(report, theta)
        covariance = jacobian @ raw @ jacobian.T
        covariance = (covariance + covariance.T) / 2
        params = report(theta)
        terms = [*design.terms, *(f"/cut{j}" for j in range(1, categories)),
                 f"/var(_cons[{group}])"]
        doubled = min(2 * points, 128)
        check = RandomInterceptQuadrature(codes, count, doubled, adaptive)
        eta = x @ theta[:p]
        cuts = _cuts(theta, p, cuts_n)
        sigma_t = torch.exp(theta[-1])
        loglik_doubled = float(kernel_call(check.log_likelihood, pieces(cuts), eta, sigma_t))
        posterior = kernel_call(quadrature.posterior_means, pieces(cuts), eta, sigma_t)
        log_likelihood = float(fit.value)
        criteria = information_criteria(log_likelihood, k, frame.n)
        sizes = common.group_sizes(codes, count)
        tests = {"model": wald_test(params, covariance, list(range(p)),
                                    label="Wald chi2 test of the slopes")} if p else {}
        bundle = build_result(
            frame, terms=terms, params=params, covariance=covariance,
            title=f"Mixed-effects ordered {link} regression",
            equations=[spec.outcome] * p + [None] * cuts_n + [group], use_t=False,
            metrics={"log_likelihood": log_likelihood, "aic": criteria["aic"],
                     "bic": criteria["bic"], "n_groups": count, "n_categories": categories,
                     "group_size_min": sizes["size_min"], "group_size_avg": sizes["size_avg"],
                     "group_size_max": sizes["size_max"]},
            solver="native adaptive Gauss-Hermite random-intercept likelihood",
            optimizer=optimizer_record(fit, objective), tests=tests,
            inference={"covariance": "nonrobust",
                       "correction": "inverse full observed information of (b, cutpoints, "
                                     "ln s), delta method to var(u)"},
            extra={
                "link": link, "categories": labels, "group": group,
                "parameterization": "Pr(y <= k | u) = F(cut_k - x'b - u), u ~ N(0, var)",
                "intpoints": points,
                "intmethod": "mcaghermite" if adaptive else "ghermite",
                "adaptation": "mode and curvature recomputed at every parameter value",
                "quadrature_check": {"intpoints": points, "log_likelihood": log_likelihood,
                                     "doubled_intpoints": doubled,
                                     "doubled_log_likelihood": loglik_doubled,
                                     "difference": loglik_doubled - log_likelihood},
                "posterior_mean_summary": {"min": float(posterior.min()),
                                           "max": float(posterior.max()),
                                           "sd": float(posterior.std())},
                "likelihood_theta": theta.tolist(), "likelihood_covariance": raw.tolist(),
                "theta_parameterization": "b, cut_1, ln(cut_k - cut_k-1), ln sd(u)",
            },
            categories=design.categories,
        )
        common.variance_intervals(bundle, [len(terms) - 1], spec.alpha)
        return bundle


def fit_meologit(spec, data):
    return _fit(spec, data, "logit")


def fit_meoprobit(spec, data):
    return _fit(spec, data, "probit")


def _ordered(command, *, data, y, x, group, intpoints, intmethod, categorical, missing, alpha,
             max_work):
    from openecon.analysis import fit

    return fit(make_spec(
        command, outcome=y, predictors=column_list(x or [], "x"), intercept=False,
        categorical=column_list(categorical, "categorical"), missing=missing, alpha=alpha,
        columns={"group": group},
        options={"intpoints": intpoints, "intmethod": intmethod, "max_work": max_work}),
        data=data)


def meologit(*, data: Any, y: str, x=None, group: str, intpoints: int = 12,
             intmethod: str = "mcaghermite", categorical=None, missing: str = "raise",
             alpha: float = 0.05, max_work: int = 2_000_000_000):
    """Random-intercept ordered logit by adaptive Gauss-Hermite quadrature (Stata ``meologit``).

    ``Pr(y <= k | u_j) = L(cut_k - x'b - u_j)`` with ``u_j ~ N(0, var)``; the cutpoints
    replace the constant. ``intmethod='mcaghermite'`` (default) is mode-curvature adaptive
    quadrature with the modes recomputed at every parameter value (R ``ordinal::clmm`` with
    ``nAGQ = intpoints``); ``'ghermite'`` is the non-adaptive rule. Terms: slopes,
    ``/cut1``..``/cut{J-1}``, ``/var(_cons[group])``; covariance from the full observed
    information. ``extra['quadrature_check']`` re-evaluates the likelihood at twice the
    points at the estimate.

    Stata: ``meologit y x || g:, intpoints(12) intmethod(mcaghermite)`` is
    ``oe.meologit(data=df, y='y', x=['x'], group='g')``.
    """
    return _ordered("meologit", data=data, y=y, x=x, group=group, intpoints=intpoints,
                    intmethod=intmethod, categorical=categorical, missing=missing, alpha=alpha,
                    max_work=max_work)


def meoprobit(*, data: Any, y: str, x=None, group: str, intpoints: int = 12,
              intmethod: str = "mcaghermite", categorical=None, missing: str = "raise",
              alpha: float = 0.05, max_work: int = 2_000_000_000):
    """Random-intercept ordered probit by adaptive Gauss-Hermite quadrature (Stata ``meoprobit``).

    ``Pr(y <= k | u_j) = Phi(cut_k - x'b - u_j)``; otherwise as ``oe.meologit``
    (R ``ordinal::clmm(link = 'probit', nAGQ = intpoints)``).
    """
    return _ordered("meoprobit", data=data, y=y, x=x, group=group, intpoints=intpoints,
                    intmethod=intmethod, categorical=categorical, missing=missing, alpha=alpha,
                    max_work=max_work)
