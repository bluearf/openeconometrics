"""Stata's ``xtlogit``, ``xtprobit`` and ``xtpoisson`` with ``model='re' | 'fe' | 'pa'``.

re  Random effects. Logit and probit: a normal random intercept integrated by
    adaptive Gauss-Hermite quadrature (the ``glmm`` core; 12 points by default,
    Stata's xt default), reported as Stata does: ``/lnsig2u`` (ln of the
    panel-level variance) with sigma_u and rho = sigma_u^2/(sigma_u^2 + c),
    c = pi^2/3 (logit) or 1 (probit), and the LR test of rho = 0 (chibar2(01)).
    Poisson: the gamma random effect with the closed-form likelihood of
    Hausman, Hall and Griliches (``xt_kernels``), ``/lnalpha`` and alpha, and
    the LR test of alpha = 0 against the pooled Poisson model (chibar2(01));
    ``normal=True`` uses the normal random intercept instead (quadrature).
fe  Conditional fixed effects. Logit: the conditional logit likelihood of the
    discrete family (``ConditionalLogitObjective``), panels without outcome
    variation dropped. Poisson: the conditional (multinomial) Poisson
    likelihood, panels with all-zero outcomes dropped. Regressors constant
    within panels are omitted; there is no constant.
pa  Population-averaged: ``xtgee`` with family binomial (logit/probit link) or
    Poisson (log link) and the working correlation ``corr`` (exchangeable by
    default), as Stata's ``xtlogit, pa`` is ``xtgee, family(binomial)``.

Covariances (Stata's vce() lists): re takes nonrobust, robust (clustered on the
panels, G/(G-1)) and cluster (a column nesting the panels); fe logit takes
nonrobust only (Stata's xtlogit, fe allows oim, bootstrap and jackknife); fe
Poisson nonrobust and robust (panel-clustered); pa nonrobust (conventional) and
robust. Model tests as Stata reports them: Wald chi2 for re and pa, LR chi2
for xtlogit, fe (e(chi2type) = LR, with e(r2_p) as ``pseudo_r_squared``) and
Wald chi2 for xtpoisson, fe (e(chi2type) = Wald). The LR tests of the
panel-level variance (rho, sigma_u, alpha) are omitted under robust/cluster.
The re results reproduce the ships example of Stata's [XT] xtpoisson manual.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, kernel_call, lr_test, make_spec,
    ml_covariance, wald_test,
)
from openecon.econometrics.discrete.common import binary_outcome, constant_shift, maximize
from openecon.econometrics.discrete.conditional import ConditionalLogitObjective
from openecon.econometrics.mixed import common, glmm
from openecon.econometrics.mixed.gee import fit_gee
from openecon.econometrics.mixed.glmm_kernels import PooledObjective
from openecon.econometrics.mixed.xt_kernels import (
    ConditionalPoissonObjective, GammaPoissonObjective,
)
from openecon.engines.absorb import demean
from openecon.engines.covariance import group_counts, group_sums
from openecon.engines.inference import critical_value
from openecon.models import ModelSpec, ResultBundle

_LATENT = {"logit": math.pi ** 2 / 3, "probit": 1.0}
_ALLOWED = {
    ("logit", "fe"): {"nonrobust"}, ("probit", "fe"): set(),
    ("poisson", "fe"): {"nonrobust", "robust"},
}
_TITLES = {
    ("logit", "re"): "Random-effects logistic regression",
    ("probit", "re"): "Random-effects probit regression",
    ("poisson", "re"): "Random-effects Poisson regression",
    ("logit", "fe"): "Conditional fixed-effects logistic regression",
    ("poisson", "fe"): "Conditional fixed-effects Poisson regression",
    ("logit", "pa"): "GEE population-averaged logistic model",
    ("probit", "pa"): "GEE population-averaged probit model",
    ("poisson", "pa"): "GEE population-averaged Poisson model",
}
_LINK = {"logit": ("binomial", "logit"), "probit": ("binomial", "probit"),
         "poisson": ("poisson", "log")}
_BOUNDARY_ALPHA = 1e-6


def _validate(frame: ModelFrame, family: str, model: str, command: str) -> None:
    spec = frame.spec
    allowed = _ALLOWED.get((family, model))
    if allowed is not None and not allowed:
        raise AnalysisError("invalid_spec", f"{command} has no fixed-effects model (Stata offers "
                            "none: there is no sufficient statistic for the panel effect). Use "
                            "model='re' or model='pa'.")
    if allowed is not None and spec.covariance not in allowed:
        raise AnalysisError("unsupported_covariance", f"{command}, fe supports only these "
                            f"covariances: {', '.join(sorted(allowed))}.")
    if model == "pa" and spec.covariance == "cluster":
        raise AnalysisError("unsupported_covariance", f"{command}, pa supports nonrobust and "
                            "robust (panel-clustered) covariances.")
    if model != "pa" and ("corr" in spec.options or "corr_order" in spec.options
                          or "force" in spec.options):
        raise AnalysisError("invalid_spec", "corr, corr_order and force apply to model='pa' "
                            "only.")
    if model != "re" and ("intpoints" in spec.options or "intmethod" in spec.options
                          or "normal" in spec.options):
        raise AnalysisError("invalid_spec", "intpoints, intmethod and normal apply to "
                            "model='re' only.")
    if spec.panel in spec.predictors or spec.panel == spec.outcome:
        raise AnalysisError("invalid_spec", f"{command}: the panel column must differ from the "
                            "outcome and the regressors.")


def _fit_xt(spec: ModelSpec, data: Any, family: str) -> ResultBundle:
    command = {"logit": "xtlogit", "probit": "xtprobit", "poisson": "xtpoisson"}[family]
    frame = ModelFrame(spec, data)
    model = frame.option("model")
    _validate(frame, family, model, command)
    title = _TITLES[(family, model)]
    if model == "pa":
        glm_family, link = _LINK[family]
        return fit_gee(frame, family=glm_family, link=link, corr=frame.option("corr"),
                       order=frame.option("corr_order"), scale=None, nmp=False,
                       force=frame.option("force"), command=f"{command}, pa", title=title)
    if model == "fe":
        if spec.intercept:
            frame.notes["constant"] = "absorbed by the panel fixed effects"
        if family == "logit":
            return _conditional(frame, command, title, logit=True)
        return _conditional(frame, command, title, logit=False)
    if family == "poisson" and not frame.option("normal"):
        return _gamma_poisson(frame, command, title)
    data_ = glmm.sample(frame, family, command, spec.panel, [])
    points, method = frame.option("intpoints"), frame.option("intmethod")
    fit = glmm.fit_glmm(data_, family, points, method)
    return _xt_normal_result(data_, fit, family, points, method, title)


def fit_xtlogit(spec: ModelSpec, data: Any) -> ResultBundle:
    return _fit_xt(spec, data, "logit")


def fit_xtprobit(spec: ModelSpec, data: Any) -> ResultBundle:
    return _fit_xt(spec, data, "probit")


def fit_xtpoisson(spec: ModelSpec, data: Any) -> ResultBundle:
    return _fit_xt(spec, data, "poisson")


# ---- random effects ------------------------------------------------------------------


def _ancillary_summary(estimate: float, se: float, alpha: float,
                       transform, derivative) -> dict[str, float]:
    """A transformed ancillary parameter: delta-method se, interval of transformed limits."""
    critical = critical_value(alpha, None)
    value = transform(estimate)
    return {"estimate": value, "std_error": abs(derivative(estimate)) * se,
            "ci_low": transform(estimate - critical * se),
            "ci_high": transform(estimate + critical * se)}


def _structure(codes: Tensor, count: int) -> dict[str, float]:
    sizes = common.group_sizes(codes, count)
    return {"n_groups": count, "group_size_min": sizes["size_min"],
            "group_size_avg": sizes["size_avg"], "group_size_max": sizes["size_max"]}


def _xt_normal_result(data: glmm.GLMMSample, fit: glmm.GLMMFit, family: str, points: int,
                      method: str, title: str) -> ResultBundle:
    """Stata's xtlogit/xtprobit (and xtpoisson, normal) re output: /lnsig2u, sigma_u, rho."""
    frame, design = data.frame, data.design
    p = len(design.terms)
    jacobian = torch.eye(p + 1, dtype=torch.float64)
    jacobian[p, p] = 2.0
    params = torch.cat([fit.theta[:p], 2 * fit.theta[p:]])
    covariance = jacobian @ fit.covariance @ jacobian.T
    lnsig2u, se = float(params[p]), float(covariance[p, p].sqrt())
    alpha = frame.spec.alpha
    sigma = _ancillary_summary(lnsig2u, se, alpha, lambda v: math.exp(v / 2),
                               lambda v: 0.5 * math.exp(v / 2))
    slopes = [i for i, term in enumerate(design.terms) if term != "Intercept"]
    lr = 2 * (fit.log_likelihood - fit.pooled_log_likelihood)
    pooled_name = {"logit": "pooled logit", "probit": "pooled probit",
                   "poisson": "pooled Poisson"}[family]
    tests = {"model": wald_test(params, covariance, slopes, label="Wald chi2 test of the slopes")}
    metrics = {**information_criteria(fit.log_likelihood, p + 1, frame.n),
               **_structure(data.codes, data.groups), "sigma_u": sigma["estimate"]}
    extra = {"sigma_u": sigma, "model": "re", "integration": {
        "method": method, "points": points, "adaptations": fit.adaptations},
        "pooled_log_likelihood": fit.pooled_log_likelihood}
    if family in _LATENT:
        c = _LATENT[family]

        def rho_of(value: float) -> float:
            return math.exp(value) / (math.exp(value) + c)

        rho = _ancillary_summary(lnsig2u, se, alpha, rho_of,
                                 lambda v: rho_of(v) * (1 - rho_of(v)))
        metrics["rho"] = rho["estimate"]
        extra["rho"] = rho
        if frame.spec.covariance == "nonrobust":
            tests["rho"] = common.boundary_lr(lr, 1, pooled_name) | {
                "label": "LR test of rho = 0 (chibar2(01))"}
    elif frame.spec.covariance == "nonrobust":
        tests["sigma_u"] = common.boundary_lr(lr, 1, pooled_name) | {
            "label": "LR test of sigma_u = 0 (chibar2(01))"}
    return build_result(
        frame, terms=[*design.terms, "/lnsig2u"], params=params, covariance=covariance,
        title=title, equations=[frame.spec.outcome] * p + [None], use_t=False, metrics=metrics,
        solver="newton_adaptive_quadrature",
        solver_diagnostics={"converged": True, "iterations": fit.optimizer["iterations"]},
        optimizer=fit.optimizer, inference=fit.info, tests=tests, extra=extra,
        categories=design.categories,
        provenance={"model": "re", "integration": f"{method} with {points} points"})


def _gamma_poisson(frame: ModelFrame, command: str, title: str) -> ResultBundle:
    spec = frame.spec
    frame.sort_panel()
    y = glmm.outcome(frame, "poisson", command)
    design = frame.drop_collinear(frame.design())
    codes, count = frame.codes(spec.panel)
    if count < 2:
        raise AnalysisError("insufficient_groups", f"{command} needs at least two panels.")
    offset = glmm.offset_of(frame)
    p = len(design.terms)
    x = design.x.clone()
    means = torch.zeros(p, dtype=torch.float64)
    if design.intercept and p > 1:
        means[1:] = x[:, 1:].mean(dim=0)
        x[:, 1:] -= means[1:]
    pooled = PooledObjective(x, y, "poisson", offset)
    start = maximize(pooled, torch.zeros(p, dtype=torch.float64), what=f"{command} (pooled)")
    if not start.converged:
        raise AnalysisError("nonconvergence", f"{command}: the pooled Poisson starting model did "
                            "not converge.")
    objective = kernel_call(GammaPoissonObjective, x, y, codes, count, offset)
    best, best_value = None, -math.inf
    for value in (0.1, 0.5, 1.0, 2.0):
        theta = torch.cat([start.theta, torch.tensor([math.log(value)], dtype=torch.float64)])
        current = float(objective.value(theta))
        if current > best_value:
            best, best_value = theta, current
    result = maximize(objective, best, what=command)
    if float(result.theta[p]) < math.log(_BOUNDARY_ALPHA):
        raise AnalysisError("boundary_solution", f"{command}: alpha (the variance of the gamma "
                            "panel effect) is estimated at zero, the boundary of the parameter "
                            "space. Fit the pooled Poisson model instead.")
    if not result.converged:
        raise AnalysisError("nonconvergence", f"{command} did not converge: "
                            f"{result.diagnostics.get('message')}")
    shift = torch.eye(p + 1, dtype=torch.float64)
    if design.intercept:
        shift[:p, :p] = constant_shift(means)
    theta = shift @ result.theta
    covariance, info = _panel_covariance(frame, result.hessian, objective, result.theta, codes,
                                         count, command)
    covariance = shift @ covariance @ shift.T
    lnalpha, se = float(theta[p]), float(covariance[p, p].sqrt())
    alpha_summary = _ancillary_summary(lnalpha, se, spec.alpha, math.exp, math.exp)
    slopes = [i for i, term in enumerate(design.terms) if term != "Intercept"]
    lr = 2 * (result.value - start.value)
    tests = {"model": wald_test(theta, covariance, slopes, label="Wald chi2 test of the slopes")}
    if spec.covariance == "nonrobust":
        tests["alpha"] = common.boundary_lr(lr, 1, "pooled Poisson") | {
            "label": "LR test of alpha = 0 (chibar2(01))"}
    metrics = {**information_criteria(result.value, p + 1, frame.n),
               **_structure(codes, count), "alpha": alpha_summary["estimate"]}
    extra = {"alpha": alpha_summary, "model": "re", "random_effect": "gamma",
             "pooled_log_likelihood": start.value}
    return build_result(
        frame, terms=[*design.terms, "/lnalpha"], params=theta, covariance=covariance,
        title=title, equations=[spec.outcome] * p + [None], use_t=False, metrics=metrics,
        solver="newton_observed_hessian",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=common.optimizer_summary(result), inference=info, tests=tests, extra=extra,
        categories=design.categories, provenance={"model": "re", "random_effect": "gamma"})


def _panel_covariance(frame: ModelFrame, hessian: Tensor, objective: Any, theta: Tensor,
                      codes: Tensor, count: int, command: str) -> tuple[Tensor, dict[str, Any]]:
    spec = frame.spec
    try:
        if spec.covariance == "nonrobust":
            covariance, info = ml_covariance(frame, hessian=hessian, kind="nonrobust")
            info["correction"] = "observed information (OIM)"
        else:
            scores = kernel_call(objective.group_scores, theta)
            if spec.covariance == "robust":
                clusters, names = [(torch.arange(count), count)], [spec.panel]
            else:
                clusters = common.check_cluster_nesting(frame, codes, count, spec.panel)
                names = [spec.cluster]
            covariance, info = ml_covariance(frame, hessian=hessian, scores=scores,
                                             kind="cluster", nobs=count, clusters=clusters,
                                             cluster_names=names)
            info["covariance"] = spec.covariance
            info["correction"] = (f"sandwich clustered on '{info['cluster_column']}' (G/(G-1)), "
                                  "panel scores")
    except AnalysisError as exc:
        if exc.code != "singular_information":
            raise
        raise AnalysisError("singular_information", f"{command}: the information matrix is not "
                            "positive definite; a parameter is not identified.") from exc
    info["df_inference"] = None
    return covariance, info


# ---- conditional fixed effects ----------------------------------------------------------


def _conditional(frame: ModelFrame, command: str, title: str, *, logit: bool) -> ResultBundle:
    spec = frame.spec
    panel = spec.panel
    # Panels without outcome variation are dropped below (Stata), so a constant outcome is
    # reported as no_outcome_variation rather than refused here.
    y = binary_outcome(frame, spec.outcome, command) if logit else glmm.outcome(
        frame, "poisson", command, constant_ok=True)
    codes, count = frame.codes(panel)
    sizes = group_counts(codes, count)
    totals = group_sums(y, codes, count)
    informative = (totals > 0) & (totals < sizes) if logit else totals > 0
    dropped = int((~informative).sum())
    if bool((~informative).all()):
        raise AnalysisError("no_outcome_variation", f"{command}, fe: no panel of '{panel}' "
                            "carries information (" + ("every panel has all positive or all "
                                                       "negative outcomes" if logit else
                                                       "every panel has all-zero outcomes")
                            + ").")
    if dropped:
        reason = "all positive or all negative outcomes" if logit else "all zero outcomes"
        frame.restrict(informative[codes], f"note: {dropped} group(s) "
                       f"({int(sizes[~informative].sum())} obs) dropped because of {reason}.")
        y = binary_outcome(frame, spec.outcome, command) if logit else glmm.outcome(
            frame, "poisson", command, constant_ok=True)
        codes, count = frame.codes(panel)
    design = frame.design(intercept=False)
    within = kernel_call(demean, design.x, [(codes, count)]).values
    design, within = frame.drop_absorbed(design, within)
    k = len(design.terms)
    if k == 0:
        raise AnalysisError("no_within_group_variation", f"{command}, fe: no regressor varies "
                            f"within the panels of '{panel}'.")
    offset = glmm.offset_of(frame)
    if logit:
        objective = kernel_call(ConditionalLogitObjective, within, y, codes, count,
                                torch.ones(count, dtype=torch.float64), offset)
    else:
        objective = kernel_call(ConditionalPoissonObjective, within, y, codes, count, offset)
    start = torch.zeros(k, dtype=torch.float64)
    result = maximize(objective, start, what=command)
    if not result.converged:
        raise AnalysisError(
            "separation_detected" if logit else "nonconvergence",
            f"{command}, fe did not converge: {result.diagnostics.get('message')} "
            + ("A regressor may predict the outcome perfectly within panels." if logit else
               "Check the scale of the regressors."))
    theta, log_likelihood = result.theta, result.value
    covariance, info = _panel_covariance(frame, result.hessian, objective, theta, codes, count,
                                         command)
    null = float(objective.value(start))
    # Stata: xtlogit, fe (clogit) reports LR chi2 against b = 0; xtpoisson, fe reports a
    # Wald chi2 (e(chi2type) = "Wald"), as does every model under robust covariance.
    if logit and spec.covariance == "nonrobust":
        model_test = lr_test(log_likelihood, null, k, label="LR chi2 test against b = 0")
    else:
        model_test = wald_test(theta, covariance, range(k),
                               label="Wald chi2 test of the coefficients")
    tests = {"model": model_test}
    metrics = {**information_criteria(log_likelihood, k, frame.n), **_structure(codes, count),
               "n_groups_dropped": dropped}
    if logit:
        metrics["pseudo_r_squared"] = 1 - log_likelihood / null if null < 0 else None
    extra = {"model": "fe", "null_log_likelihood": null,
             "likelihood": "conditional on the panel totals",
             "constant": "absorbed by the panel fixed effects"}
    return build_result(
        frame, terms=design.terms, params=theta, covariance=covariance, title=title,
        use_t=False, metrics=metrics, solver="newton_observed_hessian",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=common.optimizer_summary(result), inference=info, tests=tests, extra=extra,
        categories=design.categories, provenance={"model": "fe"})


# ---- convenience ---------------------------------------------------------------------


def _spec(estimator: str, *, data: Any, y: str, x: Sequence[str], panel: str, time: str | None,
          model: str, covariance: str | None, cluster: str | None, intpoints: int | None,
          intmethod: str | None, corr: str | None, corr_order: int | None, force: bool,
          offset: str | None, categorical: Sequence[str] | None, missing: str, alpha: float,
          exposure: str | None = None, normal: bool = False) -> ResultBundle:
    from openecon.analysis import fit

    spec = make_spec(
        estimator, outcome=y, predictors=column_list(x, "x"), panel=panel, time=time,
        intercept=model != "fe", covariance=covariance, cluster=cluster,
        categorical=column_list(categorical, "categorical"), missing=missing, alpha=alpha,
        columns={"offset": offset, "exposure": exposure},
        options={"model": model, "intpoints": intpoints, "intmethod": intmethod, "corr": corr,
                 "corr_order": corr_order, "force": True if force else None,
                 "normal": True if normal else None})
    return fit(spec, data=data)


_XT_DOC = """{title} (Stata ``{command}``).

    Models (``model``)
        ``'re'`` (default): random effects. {re}
        ``'fe'``: {fe}
        ``'pa'``: population-averaged GEE, Stata's ``{command}, pa`` = ``xtgee,
        family({gee_family}) link({gee_link})`` with working correlation ``corr``
        (default 'exchangeable'; 'independent', 'ar1', 'stationary', 'nonstationary',
        'unstructured' with ``corr_order``; timed structures need ``time``, ``force``
        treats gaps as consecutive). Conventional covariance with scale 1, or 'robust'.

    Parameters
        data: DataFrame. y: {outcome}. x: regressors. panel: panel identifier.
        time: period variable (needed only by timed GEE correlations).
        covariance: re: 'nonrobust' (default, OIM), 'robust' (clustered on the panels,
            G/(G-1)), 'cluster' with ``cluster`` nesting the panels; {fe_cov}; pa:
            'nonrobust' or 'robust'.
        intpoints / intmethod: quadrature of the normal random effect (default 12 points,
            'mvaghermite', Stata's xt defaults).{extra_params}
        categorical, missing ('raise' or 'drop'), alpha: as everywhere.

    Result
        re: fixed coefficients and the ancillary {ancillary}; ``tests['model']`` (Wald
        chi2 of the slopes) and {lr} (default covariance only). fe: coefficients without
        a constant, {fe_test}, ``metrics['n_groups_dropped']``. pa: see
        ``oe.xtgee``. ``metrics`` always hold log_likelihood (re, fe), aic, bic,
        n_groups and group_size_min/avg/max.

    Stata
        ``xtset id year`` + ``{command} {stata_example}, re`` is ``oe.{command}(data=df,
        {oe_example}, panel='id')``; add ``model='fe'`` / ``model='pa'`` for ``, fe`` /
        ``, pa``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> ids = np.repeat(np.arange(200), 6)
        >>> df = pd.DataFrame({{"id": ids, "x": rng.normal(size=1200)}})
        >>> eta = 0.3 + 0.7 * df.x + rng.normal(size=200)[ids]
        >>> {example_y}
        >>> print(oe.{command}(data=df, y="y", x=["x"], panel="id").summary())
"""


def xtlogit(*, data: Any, y: str, x: Sequence[str], panel: str, time: str | None = None,
            model: str = "re", covariance: str | None = None, cluster: str | None = None,
            intpoints: int | None = None, intmethod: str | None = None, corr: str | None = None,
            corr_order: int | None = None, force: bool = False, offset: str | None = None,
            categorical: Sequence[str] | None = None, missing: str = "raise",
            alpha: float = 0.05) -> ResultBundle:
    return _spec("xtlogit", data=data, y=y, x=x, panel=panel, time=time, model=model,
                 covariance=covariance, cluster=cluster, intpoints=intpoints,
                 intmethod=intmethod, corr=corr, corr_order=corr_order, force=force,
                 offset=offset, categorical=categorical, missing=missing, alpha=alpha)


def xtprobit(*, data: Any, y: str, x: Sequence[str], panel: str, time: str | None = None,
             model: str = "re", covariance: str | None = None, cluster: str | None = None,
             intpoints: int | None = None, intmethod: str | None = None,
             corr: str | None = None, corr_order: int | None = None, force: bool = False,
             offset: str | None = None, categorical: Sequence[str] | None = None,
             missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    return _spec("xtprobit", data=data, y=y, x=x, panel=panel, time=time, model=model,
                 covariance=covariance, cluster=cluster, intpoints=intpoints,
                 intmethod=intmethod, corr=corr, corr_order=corr_order, force=force,
                 offset=offset, categorical=categorical, missing=missing, alpha=alpha)


def xtpoisson(*, data: Any, y: str, x: Sequence[str], panel: str, time: str | None = None,
              model: str = "re", normal: bool = False, covariance: str | None = None,
              cluster: str | None = None, intpoints: int | None = None,
              intmethod: str | None = None, corr: str | None = None,
              corr_order: int | None = None, force: bool = False, offset: str | None = None,
              exposure: str | None = None, categorical: Sequence[str] | None = None,
              missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    return _spec("xtpoisson", data=data, y=y, x=x, panel=panel, time=time, model=model,
                 covariance=covariance, cluster=cluster, intpoints=intpoints,
                 intmethod=intmethod, corr=corr, corr_order=corr_order, force=force,
                 offset=offset, categorical=categorical, missing=missing, alpha=alpha,
                 exposure=exposure, normal=normal)


_RE_BINARY = ("A normal panel effect u_i ~ N(0, sigma_u^2) integrated by adaptive "
              "Gauss-Hermite quadrature; reported as ``/lnsig2u`` = ln sigma_u^2 with "
              "``metrics['sigma_u']``, ``metrics['rho']`` = sigma_u^2/(sigma_u^2 + {c}) and "
              "their delta-method standard errors and transformed intervals in ``extra``.")
xtlogit.__doc__ = _XT_DOC.format(
    title="Panel-data logistic regression", command="xtlogit",
    re=_RE_BINARY.format(c="pi^2/3"),
    fe="conditional (fixed-effects) logit: conditioning on the number of positive outcomes "
       "per panel removes the panel effect (the discrete family's clogit likelihood); panels "
       "with all-positive or all-negative outcomes are dropped and counted.",
    gee_family="binomial", gee_link="logit", outcome="binary outcome coded 0/1",
    fe_cov="fe: 'nonrobust' only (Stata's oim)", extra_params="\n        offset: offset column.",
    ancillary="``/lnsig2u``", lr="``tests['rho']`` (LR test of rho = 0 against the pooled "
                                  "logit, chibar2(01))",
    fe_test="LR chi2 against b = 0 (Stata's) and ``metrics['pseudo_r_squared']``",
    stata_example="union age grade", oe_example="y='union', x=['age', 'grade']",
    example_y='df["y"] = (rng.random(1200) < 1 / (1 + np.exp(-eta))) * 1.0')
xtprobit.__doc__ = _XT_DOC.format(
    title="Panel-data probit regression", command="xtprobit", re=_RE_BINARY.format(c="1"),
    fe="not available (as in Stata): raises ``invalid_spec``.",
    gee_family="binomial", gee_link="probit", outcome="binary outcome coded 0/1",
    fe_cov="fe: none", extra_params="\n        offset: offset column.",
    ancillary="``/lnsig2u``", lr="``tests['rho']`` (LR test of rho = 0 against the pooled "
                                  "probit, chibar2(01))", fe_test="not available",
    stata_example="union age grade", oe_example="y='union', x=['age', 'grade']",
    example_y='df["y"] = (rng.random(1200) < 1 / (1 + np.exp(-eta))) * 1.0')
xtpoisson.__doc__ = _XT_DOC.format(
    title="Panel-data Poisson regression", command="xtpoisson",
    re="A gamma panel effect nu_i (mean 1, variance alpha) with the closed-form negative "
       "binomial panel likelihood (Hausman, Hall and Griliches 1984), reported as "
       "``/lnalpha`` with ``metrics['alpha']``; ``normal=True`` uses a normal random intercept "
       "integrated by quadrature (reported as ``/lnsig2u``, sigma_u).",
    fe="conditional (fixed-effects) Poisson: the multinomial likelihood of the counts given "
       "the panel totals (equivalent to Poisson with panel dummies); panels with all-zero "
       "outcomes are dropped and counted.",
    gee_family="poisson", gee_link="log", outcome="nonnegative integer counts",
    fe_cov="fe: 'nonrobust' or 'robust' (panel-clustered)",
    extra_params="\n        normal: normal instead of gamma random effect (re).\n"
                 "        offset / exposure: offset column / positive exposure (log added).",
    ancillary="``/lnalpha``", lr="``tests['alpha']`` (LR test of alpha = 0 against the "
                                  "pooled Poisson, chibar2(01))",
    fe_test="Wald chi2 of the coefficients (Stata's e(chi2type) = Wald)",
    stata_example="accidents op_75_79 co_65_69, exposure(service)",
    oe_example="y='accidents', x=['op_75_79', 'co_65_69'], exposure='service'",
    example_y='df["y"] = rng.poisson(np.exp(eta))')
