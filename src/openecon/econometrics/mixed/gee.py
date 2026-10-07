"""Generalized estimating equations: Stata's ``xtgee`` (SPSS GEE / GENLIN REPEATED).

Population-averaged model g(E[y_it]) = x_it'b with Var(y_it) = phi V(mu_it) and a
working correlation R(alpha) within panels (``gee_kernels``). Following Liang
and Zeger (1986) and Stata's Methods and formulas, the estimating equations

    sum_i D_i' V_i^-1 (y_i - mu_i) = 0,    V_i = phi A_i^1/2 R_i(alpha) A_i^1/2,

are solved by alternating one Fisher-scoring (IRLS) step for b with moment
estimates of alpha from the Pearson residuals, starting from the
independence GLM, until the largest relative change of b is below 1e-10. The
correlation estimators are those printed in Stata's [XT] xtgee Methods and
formulas (``gee_kernels``): pooled pairs over sum r^2 / N for exchangeable,
per-panel 1/n_i-weighted (Yule-Walker) moments for ar1 and stationary(m),
and pair means over the panel-averaged mean square for nonstationary(m) and
unstructured; panel i uses the upper-left n_i-by-n_i block of one
max(n_i)-by-max(n_i) matrix (position within the panel), and panels with
n_i <= g are dropped for the lag-dependent structures (ar1, stationary(g),
nonstationary(g)), as Stata does. None depends on ``nmp``.
With the standardized design xt_it = x_it (dmu/deta) / sqrt(V(mu_it)) and
r_it = (y_it - mu_it) / sqrt(V(mu_it)) the step is

    b <- b + (sum_i xt_i' R_i^-1 xt_i)^-1 sum_i xt_i' R_i^-1 r_i.

Covariances (z inference): ``nonrobust`` (Stata's conventional) is
phi (sum_i xt_i' R_i^-1 xt_i)^-1 with phi = 1 for the binomial, Poisson and
negative binomial families and the Pearson estimate sum r^2 / N (N - p with
``nmp``) otherwise; ``robust`` is the semi-robust sandwich clustered on the
panels, G/(G-1) B^-1 [sum_i s_i s_i'] B^-1 with s_i = xt_i' R_i^-1 r_i (Stata's
_robust; reproduces the xtpoisson, pa vce(robust) example of the Stata manual).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, kernel_call, make_spec, ml_covariance, wald_test,
)
from openecon.econometrics.glm.families import FAMILY_LINKS, make_family, make_link
from openecon.econometrics.mixed import common
from openecon.econometrics.mixed.gee_kernels import TIMED, PanelLayout, WorkingCorrelation, gee_workspace_plan
from openecon.econometrics.mixed.glmm import offset_of
from openecon.engines import linalg, optimize
from openecon.engines.contracts import KernelError
from openecon.engines.covariance import group_sums
from openecon.models import ModelSpec, ResultBundle

FAMILIES = {"gaussian": "gaussian", "binomial": "binomial", "poisson": "poisson",
            "gamma": "gamma", "nbinomial": "nbinomial", "igaussian": "inverse_gaussian"}
DEFAULT_LINK = {"gaussian": "identity", "binomial": "logit", "poisson": "log",
                "gamma": "reciprocal", "nbinomial": "log", "igaussian": "inverse_squared"}
_FIXED_SCALE = {"binomial", "poisson", "nbinomial"}
_MAX_ITER = 200
_TOL = 1e-10
_MATRIX_LIMIT = 50


def _check_spacing(frame: ModelFrame, corr: str, force: bool) -> None:
    """Timed structures need equally spaced periods within panels (or ``force``)."""
    if corr not in TIMED:
        return
    if frame.spec.time is None:
        raise AnalysisError("invalid_spec", f"corr='{corr}' needs a time column (Stata's "
                            "xtset panel time).")
    codes, _ = frame.codes(frame.spec.panel)
    time = frame.time_index()
    gaps = torch.zeros(len(codes), dtype=torch.bool)
    gaps[1:] = (codes[1:] == codes[:-1]) & (time[1:] - time[:-1] != 1)
    if bool(gaps.any()) and not force:
        raise AnalysisError("unequal_spacing", f"corr='{corr}' needs equally spaced "
                            "observations, but some panels have gaps in the time variable. "
                            "Pass force=True to treat the observations of each panel as "
                            "consecutive (Stata's force option).")


def _drop_short_panels(frame: ModelFrame, corr: str, order: int, command: str) -> None:
    """Stata: only panels with n_i > g enter a lag-dependent correlation structure."""
    if corr not in {"ar1", "stationary", "nonstationary"}:
        return
    lag = 1 if corr == "ar1" else order
    codes, count = frame.codes(frame.spec.panel)
    sizes = torch.bincount(codes, minlength=count)
    short = sizes <= lag
    if bool(short.all()):
        raise AnalysisError("insufficient_panel_length", f"{command}: corr='{corr}' with lag "
                            f"{lag} needs panels with more than {lag} observation(s); none has.")
    if bool(short.any()):
        frame.restrict(~short[codes], f"note: {int(short.sum())} panel(s) "
                       f"({int(sizes[short].sum())} obs) dropped because of too few "
                       f"observations (corr='{corr}' needs more than {lag} per panel).")


def _outcome(frame: ModelFrame, family: str, command: str) -> Tensor:
    y = frame.numeric(frame.spec.outcome)
    problems = {
        "binomial": bool(((y < 0) | (y > 1)).any()),
        "poisson": bool((y < 0).any()), "nbinomial": bool((y < 0).any()),
        "gamma": bool((y <= 0).any()), "igaussian": bool((y <= 0).any()),
    }
    if problems.get(family):
        need = {"binomial": "values in [0, 1] (0/1 outcomes or proportions)",
                "poisson": "nonnegative values", "nbinomial": "nonnegative values",
                "gamma": "positive values", "igaussian": "positive values"}[family]
        raise AnalysisError("invalid_outcome", f"{command} with family '{family}' needs "
                            f"{need} in '{frame.spec.outcome}'.")
    common.require_variation(y, command, frame.spec.outcome)
    return y


class _Equations:
    """Mean, variance and standardized quantities of the GEE at b."""

    def __init__(self, x: Tensor, y: Tensor, offset: Tensor | None, family: Any, link: Any):
        self.x, self.y, self.family, self.link = x, y, family, link
        self.offset = torch.zeros_like(y) if offset is None else offset

    def at(self, beta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eta = self.x @ beta + self.offset
        if not bool(self.link.valid(eta).all()):
            raise KernelError("invalid_mean", "The linear predictor left the domain of the link.")
        mu = self.link.inverse(eta)
        if not bool(self.family.valid_mu(mu).all()):
            raise KernelError("invalid_mean", "The fitted means left the range of the family.")
        variance = self.family.variance(mu)
        if not bool((variance > 0).all()):
            raise KernelError("invalid_mean", "A fitted variance is not positive.")
        root = variance.sqrt()
        resid = (self.y - mu) / root
        design = self.x * (self.link.derivative(eta, mu) / root)[:, None]
        return mu, resid, design


def _independence_start(eq: _Equations) -> Tensor:
    """GLM start: one weighted least-squares step from the family's starting means."""
    mu = eq.family.start_mu(eq.y)
    eta = eq.link.link(mu)
    derivative = eq.link.derivative(eta, mu)
    working = eta - eq.offset + (eq.y - mu) / derivative
    weights = derivative.square() / eq.family.variance(mu)
    fit = linalg.least_squares(eq.x, working, weights, drop_collinear=False)
    return fit.beta


def solve_gee(eq: _Equations, working: WorkingCorrelation,
              fixed: bool = False) -> tuple[Tensor, int]:
    """Alternate Fisher steps and moment updates; return b and the iteration count.

    The correlation parameters do not depend on the scale divisor (``nmp``): Stata's
    estimators are ratios of residual moments (see ``gee_kernels``).
    """
    p = eq.x.shape[1]
    independent = WorkingCorrelation("independent", working.layout)
    beta = _independence_start(eq)
    scale = float(eq.y.square().sum())
    for stage, structure in (("independence", independent), ("gee", working)):
        for iteration in range(1, _MAX_ITER + 1):
            _, resid, design = eq.at(beta)
            if structure is working and not fixed:
                if not float(resid.square().sum()) > 1e-24 * scale:
                    raise KernelError("perfect_fit", "The regressors reproduce the outcome "
                                      "exactly (all Pearson residuals are zero), so the working "
                                      "correlation and the scale are not identified.")
                structure.estimate(resid)
            structure.check()
            solved = structure.solve(torch.cat([design, resid[:, None]], dim=1))
            bread = design.T @ solved[:, :p]
            step = linalg.cholesky_solve(bread, design.T @ solved[:, p],
                                         code="singular_information", what="GEE information")
            beta = beta + step
            if float((step.abs() / beta.abs().clamp_min(1.0)).max()) <= _TOL:
                break
        else:
            raise KernelError("nonconvergence", f"The {stage} iterations did not converge in "
                              f"{_MAX_ITER} steps.")
    return beta, iteration


def fit_gee(frame: ModelFrame, *, family: str, link: str | None, corr: str, order: int,
            scale: Any, nmp: bool, force: bool, command: str, nbk: float = 1.0,
            title: str | None = None) -> ResultBundle:
    """GEE fit on a frame whose spec has ``panel`` (and ``time`` for timed structures)."""
    spec = frame.spec
    link = link or DEFAULT_LINK[family]
    if link not in FAMILY_LINKS[FAMILIES[family]]:
        raise AnalysisError("invalid_spec", f"{command}: link '{link}' is not available for "
                            f"family '{family}'.")
    if corr not in {"stationary", "nonstationary"} and order != 1:
        raise AnalysisError("invalid_spec", "corr_order applies to corr='stationary' or "
                            "'nonstationary' only.")
    if spec.panel in spec.predictors or spec.panel == spec.outcome:
        raise AnalysisError("invalid_spec", f"{command}: the panel column must differ from the "
                            "outcome and the regressors.")
    frame.sort_panel()
    _check_spacing(frame, corr, force)
    _drop_short_panels(frame, corr, order, command)
    layout = kernel_call(PanelLayout, *frame.codes(spec.panel))
    planned_width = frame.design_width()
    plan = gee_workspace_plan(corr, layout, planned_width,
                              resident_bytes=frame.resource_input_bytes, budget_bytes=frame.resource_budget_bytes)
    frame.resource_plans.append(plan.record())
    y = _outcome(frame, family, command)
    design = frame.drop_collinear(frame.design())
    if not design.terms:
        raise AnalysisError("invalid_spec", f"{command} needs at least one regressor or the "
                            "constant.")
    if layout.n_panels < 2:
        raise AnalysisError("insufficient_groups", f"{command} needs at least two panels.")
    n, p = design.x.shape
    if n <= p:
        raise AnalysisError("insufficient_observations", f"{command} needs more observations "
                            "than parameters.")
    glm_family = kernel_call(make_family, FAMILIES[family], k=nbk)
    glm_link = kernel_call(make_link, link, k=nbk)
    x = design.x.clone()
    means = torch.zeros(p, dtype=torch.float64)
    if design.intercept and p > 1:
        means[1:] = x[:, 1:].mean(dim=0)
        x[:, 1:] -= means[1:]
    eq = _Equations(x, y, offset_of(frame), glm_family, glm_link)
    working = kernel_call(WorkingCorrelation, corr, layout, order, width=p,
                          resident_bytes=frame.resource_input_bytes, budget_bytes=frame.resource_budget_bytes)
    beta, iterations = kernel_call(solve_gee, eq, working)
    mu, resid, xt = kernel_call(eq.at, beta)
    pearson = float(resid.square().sum())
    phi_hat = pearson / (n - p if nmp else n)
    solved = working.solve(torch.cat([xt, resid[:, None]], dim=1))
    bread = xt.T @ solved[:, :p]
    if scale is None:
        phi = 1.0 if family in _FIXED_SCALE else phi_hat
        scale_text = "fixed at 1" if family in _FIXED_SCALE else "Pearson chi2 / " + (
            "(N - p)" if nmp else "N")
    elif scale == "x2":
        phi, scale_text = phi_hat, "Pearson chi2 / " + ("(N - p)" if nmp else "N")
    elif scale == "dev":
        deviance = float(glm_family.unit_deviance(y, mu).sum())
        phi, scale_text = deviance / (n - p if nmp else n), "deviance / " + (
            "(N - p)" if nmp else "N")
    elif isinstance(scale, (int, float)) and not isinstance(scale, bool) and scale > 0:
        phi, scale_text = float(scale), "user-specified"
    else:
        raise AnalysisError("invalid_spec", "scale must be None, 'x2', 'dev' or a positive "
                            "number.")
    shift = torch.eye(p, dtype=torch.float64)
    if design.intercept:
        shift[0, 1:] = -means[1:]
    try:
        if spec.covariance == "nonrobust":
            covariance = optimize.information_inverse(bread) * phi
            info = {"covariance": "nonrobust",
                    "correction": f"conventional: phi (sum D'V^-1 D)^-1, phi {scale_text}"}
        else:
            scores = panel_scores(layout, xt, solved[:, p])
            covariance, info = ml_covariance(
                frame, hessian=-bread, scores=scores, kind="cluster", nobs=layout.n_panels,
                clusters=[(torch.arange(layout.n_panels), layout.n_panels)],
                cluster_names=[spec.panel])
            info["covariance"] = "robust"
            info["correction"] = ("semi-robust sandwich clustered on the panels: G/(G-1) "
                                  "B^-1 [sum_i s_i s_i'] B^-1")
    except KernelError as exc:
        raise AnalysisError("singular_information", f"{command}: the GEE information matrix is "
                            "singular; a regressor is not identified.") from exc
    info["df_inference"] = None
    params = shift @ beta
    covariance = shift @ covariance @ shift.T
    slopes = [i for i, term in enumerate(design.terms) if term != "Intercept"]
    sizes = common.group_sizes(layout.codes, layout.n_panels)
    deviance = float(glm_family.unit_deviance(y, mu).sum())
    metrics = {"n_groups": layout.n_panels, "group_size_min": sizes["size_min"],
               "group_size_avg": sizes["size_avg"], "group_size_max": sizes["size_max"],
               "scale": phi, "pearson_chi2": pearson, "deviance": deviance}
    longest = int(layout.sizes.max())
    extra: dict[str, Any] = {
        "family": family, "link": link, "corr": corr, "scale_method": scale_text,
        "pearson_dispersion": phi_hat, "iterations": iterations, "nmp": nmp,
    }
    if corr in {"stationary", "nonstationary"}:
        extra["corr_order"] = order
    if corr in {"exchangeable", "ar1", "stationary"}:
        extra["alpha"] = working.alpha.tolist()
    size = working.matrix.shape[0] if working.matrix is not None else longest
    if size <= _MATRIX_LIMIT:
        extra["working_correlation"] = working.full_matrix(size).tolist()
    else:
        extra["working_correlation_note"] = (f"the {size}-by-{size} working correlation is "
                                             f"not stored (more than {_MATRIX_LIMIT} periods)")
    tests = {"model": wald_test(params, covariance, slopes, label="Wald chi2 test of the slopes")}
    return build_result(
        frame, terms=design.terms, params=params, covariance=covariance,
        title=title or "GEE population-averaged model", use_t=False, metrics=metrics,
        fitted=mu, solver="gee_fisher_scoring",
        solver_diagnostics={"converged": True, "iterations": iterations}, inference=info,
        tests=tests, extra=extra, categories=design.categories,
        provenance={"estimating_equations": "Liang-Zeger GEE, moment estimators of phi and "
                                            "alpha (Stata xtgee conventions)"})


def panel_scores(layout: PanelLayout, design: Tensor, solved_resid: Tensor) -> Tensor:
    """Per-panel scores s_i = xt_i' R_i^-1 r_i, [G, p]."""
    return group_sums(design * solved_resid[:, None], layout.codes, layout.n_panels)


def fit_xtgee(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``xtgee``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    family = frame.option("family")
    if "nbk" in spec.options and family != "nbinomial":
        raise AnalysisError("invalid_spec", "nbk (the k of Var = mu + k mu^2) applies to "
                            "family='nbinomial' only.")
    return fit_gee(frame, family=family, link=frame.option("link"), corr=frame.option("corr"),
                   order=frame.option("corr_order"), scale=frame.option("scale"),
                   nmp=frame.option("nmp"), force=frame.option("force"), command="xtgee",
                   nbk=frame.option("nbk"))


def xtgee(*, data: Any, y: str, x: Sequence[str] | None = None, panel: str,
          time: str | None = None, family: str = "gaussian", link: str | None = None,
          corr: str = "exchangeable", corr_order: int | None = None,
          covariance: str | None = None, scale: str | float | None = None, nmp: bool = False,
          force: bool = False, nbk: float | None = None, offset: str | None = None,
          exposure: str | None = None, categorical: Sequence[str] | None = None,
          intercept: bool = True, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Population-averaged panel model by GEE (Stata ``xtgee``; SPSS ``GENLIN ... /REPEATED``).

    Model
        ``g(E[y_it]) = x_it'b (+ offset)``, ``Var(y_it) = phi V(mu_it)`` and a working
        correlation ``R(alpha)`` among the observations of a panel. b is the marginal
        (population-averaged) effect; the correlation is a nuisance that only affects
        efficiency, and the robust covariance stays valid when R is misspecified.

    Estimator
        Liang-Zeger estimating equations ``sum_i D_i'V_i^-1 (y_i - mu_i) = 0`` solved by
        alternating Fisher-scoring steps for b with moment estimates of phi and alpha from
        the Pearson residuals r_it = (y_it - mu_it)/sqrt(V(mu_it)), starting from the
        independence GLM, with the correlation estimators of Stata's Methods and
        formulas: exchangeable alpha = [sum_i sum_{t != s} r_it r_is / sum_i n_i (n_i - 1)]
        / (sum r^2 / N); ar1 and stationary(m): a_k = sum_i (1/n_i) sum_t r_it r_i,t+k /
        sum_i (1/n_i) sum_t r_it^2; nonstationary(m) and unstructured: a_ts = mean of
        r_it r_is over the panels observing positions t and s / ((1/G) sum_i (1/n_i)
        sum_t r_it^2). Panel i uses the upper-left n_i-by-n_i block (positions within the
        panel); ar1, stationary(g) and nonstationary(g) drop panels with n_i <= g (with a
        warning), as Stata does. R_i^-1 is applied in closed form (exchangeable, ar1) or
        by one batched solve per distinct panel size.

    Parameters
        data: DataFrame. y: outcome. x: regressors. panel: panel identifier.
        time: period variable, required for corr='ar1', 'stationary', 'nonstationary' and
            'unstructured' (observations must be equally spaced unless ``force=True``).
        family: 'gaussian' (default), 'binomial' (0/1 or proportions), 'poisson',
            'gamma', 'nbinomial' (variance mu + k mu^2 with ``nbk`` = k, default 1) or
            'igaussian'.
        link: None for the canonical default (identity, logit, log, reciprocal, log,
            inverse_squared) or 'identity', 'log', 'logit', 'probit', 'cloglog', 'loglog',
            'reciprocal', 'inverse_squared' where valid for the family (``invalid_spec``
            otherwise; ``nbk`` is accepted with family 'nbinomial' only).
        corr: 'exchangeable' (default, Stata's), 'independent', 'ar1', 'stationary',
            'nonstationary', 'unstructured'; corr_order: m for stationary/nonstationary.
        covariance: 'nonrobust' (default; Stata's conventional) or 'robust' (semi-robust
            sandwich clustered on the panels, G/(G-1)).
        scale: None (phi = 1 for binomial/Poisson/nbinomial, Pearson chi2/N otherwise),
            'x2' (Pearson chi2/N), 'dev' (deviance/N) or a positive number; it scales the
            conventional covariance only. nmp: use N - p instead of N in phi (the working
            correlation is unaffected).
        offset / exposure: offset column / positive exposure (log added).
        categorical, intercept, missing, alpha: as everywhere.

    Result
        z tests. ``metrics``: n_groups, group_size_min/avg/max, scale (the phi used),
        pearson_chi2, deviance. ``tests['model']``: Wald chi2 of the slopes. ``extra``:
        family, link, corr, alpha (exchangeable/ar1/stationary), the working correlation
        matrix (up to 50 periods), the Pearson dispersion and the iteration count.

    Stata
        ``xtgee y x1 x2, family(binomial) link(logit) corr(exchangeable) vce(robust)``
        (after ``xtset id t``) is ``oe.xtgee(data=df, y='y', x=['x1', 'x2'], panel='id',
        time='t', family='binomial', covariance='robust')``; ``corr(ar 1)`` is
        ``corr='ar1'``, ``corr(stationary 2)`` is ``corr='stationary', corr_order=2``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> ids = np.repeat(np.arange(200), 5)
        >>> df = pd.DataFrame({"id": ids, "t": np.tile(np.arange(5), 200),
        ...                    "x": rng.normal(size=1000)})
        >>> df["y"] = 1 + 0.5 * df.x + rng.normal(size=200)[ids] + rng.normal(size=1000)
        >>> print(oe.xtgee(data=df, y="y", x=["x"], panel="id", time="t").summary())
    """
    from openecon.analysis import fit

    spec = make_spec(
        "xtgee", outcome=y, predictors=column_list(x, "x"), panel=panel, time=time,
        intercept=intercept, covariance=covariance,
        categorical=column_list(categorical, "categorical"), missing=missing, alpha=alpha,
        columns={"offset": offset, "exposure": exposure},
        options={"family": family, "link": link, "corr": corr, "corr_order": corr_order,
                 "scale": scale, "nmp": True if nmp else None, "force": True if force else None,
                 "nbk": nbk})
    return fit(spec, data=data)
