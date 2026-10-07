"""Random-intercept (and random-slope) GLMMs: Stata's ``melogit``, ``meprobit``, ``mepoisson``.

    g(E[y_ij | u_j]) = x_ij'b + z_ij'u_j,   u_j ~ N(0, diag(sigma^2))

with a random intercept and optionally one random slope (independent), logit,
probit or log link. The likelihood integrates u_j out by mean-variance adaptive
Gauss-Hermite quadrature with 7 points per dimension by default (Stata's
``intmethod(mvaghermite) intpoints(7)``); ``glmm_kernels`` has the formulas.
The same core fits ``xtlogit, re`` and ``xtprobit, re`` (12 points, Stata's
default there) with their own reporting (``/lnsig2u``, sigma_u, rho).

Estimation: b starts from the pooled model, ln sigma from a small grid; Newton-
Raphson with the analytic gradient and Hessian of the quadrature sum runs with
the adaptive parameters fixed, which are then recomputed at the new estimates,
until the estimates move by less than 1e-8 (relative). Regressors are centred
internally (exact back-transformation of the constant). The covariance is the
inverse observed information (``nonrobust``), the sandwich clustered on the
groups with G/(G-1) (``robust``, Stata's vce(robust) for me commands) or on a
column nesting the groups (``cluster``). Variances are reported as
``/var(_cons[group])`` with delta-method standard errors and log-scale
intervals. ``tests['model']`` is the Wald chi2 of the slopes and, with the
default covariance, ``tests['lr_vs_pooled']`` the LR test against the pooled
model (chibar2(01) for one variance component, conservative chi2(2) for two);
it is omitted under robust/cluster, as for Stata's mixed. A constant outcome
and a constant (collinear) random slope are refused before estimation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, column_list, information_criteria, kernel_call, make_spec,
    ml_covariance, wald_test,
)
from openecon.econometrics.discrete.common import (
    EXTREME, binary_outcome, constant_shift, maximize, perfectly_predicted,
)
from openecon.econometrics.mixed import common
from openecon.econometrics.mixed.glmm_kernels import PooledObjective, RandomEffectsGLMM
from openecon.models import ModelSpec, ResultBundle

_COMMANDS = {"logit": "melogit", "probit": "meprobit", "poisson": "mepoisson"}
_TITLES = {"logit": "Mixed-effects logistic regression",
           "probit": "Mixed-effects probit regression",
           "poisson": "Mixed-effects Poisson regression"}
_POOLED = {"logit": "logistic model", "probit": "probit model", "poisson": "Poisson model"}
# Latent residual variance of the binary links (Stata's estat icc).
_LATENT = {"logit": math.pi ** 2 / 3, "probit": 1.0}
# A random-effect standard deviation below this (on the scale of the linear predictor)
# is at the boundary of the parameter space.
_BOUNDARY_SD = 1e-4
_OUTER = 30


@dataclass
class GLMMSample:
    frame: ModelFrame
    design: Design
    y: Tensor
    z: Tensor
    z_names: list[str]
    codes: Tensor
    groups: int
    group: str
    offset: Tensor | None
    command: str


@dataclass
class GLMMFit:
    theta: Tensor            # (b in original units, ln sigma_1..q)
    covariance: Tensor       # of theta, original units
    log_likelihood: float
    pooled_log_likelihood: float
    info: dict[str, Any]
    optimizer: dict[str, Any]
    adaptations: int
    posterior: Tensor        # [G, q] empirical Bayes means (not stored in results)


def outcome(frame: ModelFrame, family: str, command: str, *, constant_ok: bool = False) -> Tensor:
    """The 0/1 or count outcome; a constant outcome is refused unless ``constant_ok``."""
    if family == "poisson":
        y = frame.numeric(frame.spec.outcome)
        if bool((y < 0).any()) or bool((y != y.round()).any()):
            raise AnalysisError("invalid_count_outcome", f"{command} needs a nonnegative integer "
                                "count outcome.")
    else:
        y = binary_outcome(frame, frame.spec.outcome, command)
    if not constant_ok:
        common.require_variation(y, command, frame.spec.outcome)
    return y


def offset_of(frame: ModelFrame) -> Tensor | None:
    """offset + ln(exposure) of the current sample, or None."""
    total = None
    names = frame.role("offset")
    if names:
        total = frame.numeric(names[0])
    names = frame.role("exposure")
    if names:
        exposure = frame.numeric(names[0])
        if bool((exposure <= 0).any()):
            raise AnalysisError("invalid_exposure", "Exposure values must be positive.")
        total = torch.log(exposure) if total is None else total + torch.log(exposure)
    return total


def sample(frame: ModelFrame, family: str, command: str, group: str,
           random: Sequence[str]) -> GLMMSample:
    spec = frame.spec
    if group == spec.outcome or group in spec.predictors or group in random:
        raise AnalysisError("invalid_spec", f"{command}: the group column '{group}' must differ "
                            "from the outcome and the regressors.")
    if len(random) > 1:
        raise AnalysisError("invalid_spec", f"{command} supports one random slope (independent "
                            "of the random intercept); give at most one column in random.")
    y = outcome(frame, family, command)
    design = frame.drop_collinear(frame.design())
    if not design.terms:
        raise AnalysisError("invalid_spec", f"{command} needs at least one fixed-effect term.")
    codes, count = frame.codes(group)
    if count < 2:
        raise AnalysisError("insufficient_groups", f"{command}: '{group}' has fewer than two "
                            "groups.")
    columns = [frame.numeric(name) for name in random]
    columns.append(torch.ones(frame.n, dtype=torch.float64))
    z = torch.stack(columns, dim=1)
    common.require_independent_effects(z, [*random, "_cons"], None, command)
    return GLMMSample(frame, design, y, z, [*random, "_cons"], codes, count, group,
                      offset_of(frame), command)


def _pooled(data: GLMMSample, x: Tensor, family: str) -> tuple[Tensor, float]:
    objective = PooledObjective(x, data.y, family, data.offset)
    result = maximize(objective, torch.zeros(x.shape[1], dtype=torch.float64),
                      what=f"{data.command} (pooled starting model)")
    if not result.converged:
        count = 0
        if family != "poisson":
            mean = objective.mean(result.theta)
            count = perfectly_predicted(torch.where(data.y == 1, mean, 1 - mean), EXTREME)
        if count:
            raise AnalysisError(
                "separation_detected",
                f"{data.command}: {count} observation(s) are predicted perfectly by the "
                "regressors (complete or quasi-complete separation), so finite estimates do not "
                "exist. Remove the regressor that separates the outcome.")
        raise AnalysisError("nonconvergence", f"{data.command}: the pooled starting model did "
                            f"not converge: {result.diagnostics.get('message')}")
    return result.theta, result.value


def fit_glmm(data: GLMMSample, family: str, points: int, method: str) -> GLMMFit:
    """Maximize the quadrature likelihood (see the module docstring)."""
    frame, design = data.frame, data.design
    p, q = len(design.terms), data.z.shape[1]
    x = design.x.clone()
    means = torch.zeros(p, dtype=torch.float64)
    if design.intercept and p > 1:
        means[1:] = x[:, 1:].mean(dim=0)
        x[:, 1:] -= means[1:]
    beta, pooled = _pooled(data, x, family)
    objective = kernel_call(RandomEffectsGLMM, x, data.y, data.z, data.codes, data.groups,
                            family, points, offset=data.offset, method=method)
    best, best_value = None, -math.inf
    for sd in (0.25, 0.5, 1.0, 2.0):
        theta = torch.cat([beta, torch.full((q,), math.log(sd), dtype=torch.float64)])
        kernel_call(objective.adapt, theta)
        value = float(objective.value(theta))
        if value > best_value:
            best, best_value = theta, value
    if best is None:
        raise AnalysisError("invalid_start", f"{data.command}: the likelihood is not finite at "
                            "the starting values.")
    theta, result, outer, iterations = best, None, 0, 0
    for outer in range(1, _OUTER + 1):
        kernel_call(objective.adapt, theta)
        result = maximize(objective, theta, what=data.command)
        iterations += result.iterations
        moved = float(((result.theta - theta).abs() / theta.abs().clamp_min(1.0)).max())
        theta = result.theta
        if not result.converged or moved < 1e-8 or method == "ghermite":
            break
    sigma = torch.exp(theta[p:])
    small = [data.z_names[d] for d in range(q) if float(sigma[d]) < _BOUNDARY_SD]
    if small:
        raise AnalysisError(
            "boundary_solution",
            f"{data.command}: the variance of the random effect(s) "
            f"{', '.join(f'{n}[{data.group}]' for n in small)} is estimated at zero (the "
            "boundary of the parameter space), where the likelihood is flat. Fit the pooled "
            "model, or drop that random effect.")
    if not result.converged:
        raise AnalysisError("nonconvergence", f"{data.command} did not converge: "
                            f"{result.diagnostics.get('message')}")
    shift = torch.eye(p + q, dtype=torch.float64)
    if design.intercept:
        shift[:p, :p] = constant_shift(means)
    covariance, info = _covariance(frame, objective, theta, result.hessian, data)
    return GLMMFit(shift @ theta, shift @ covariance @ shift.T, result.value, pooled, info,
                   common.optimizer_summary(result) | {"outer_iterations": outer,
                                                       "iterations": iterations},
                   objective.adaptations, objective.posterior_means(theta))


def _covariance(frame: ModelFrame, objective: RandomEffectsGLMM, theta: Tensor,
                hessian: Tensor, data: GLMMSample) -> tuple[Tensor, dict[str, Any]]:
    kind = frame.spec.covariance
    try:
        if kind == "nonrobust":
            covariance, info = ml_covariance(frame, hessian=hessian, kind="nonrobust")
            info["correction"] = "observed information (OIM)"
            return covariance, info
        scores = kernel_call(objective.group_scores, theta)
        if kind == "robust":
            clusters, names = [(torch.arange(data.groups), data.groups)], [data.group]
        else:
            clusters = common.check_cluster_nesting(frame, data.codes, data.groups, data.group)
            names = [frame.spec.cluster]
        covariance, info = ml_covariance(frame, hessian=hessian, scores=scores, kind="cluster",
                                         nobs=data.groups, clusters=clusters, cluster_names=names)
    except AnalysisError as exc:
        if exc.code != "singular_information":
            raise
        raise AnalysisError("singular_information", f"{data.command}: the information matrix "
                            "is not positive definite, so standard errors are undefined; a "
                            "parameter is not identified.") from exc
    info["covariance"] = kind
    info["correction"] = (f"sandwich clustered on '{info['cluster_column']}' (G/(G-1)), group "
                          "scores of the integrated likelihood")
    return covariance, info


def _structure(data: GLMMSample) -> dict[str, float]:
    sizes = common.group_sizes(data.codes, data.groups)
    return {"n_groups": data.groups, "group_size_min": sizes["size_min"],
            "group_size_avg": sizes["size_avg"], "group_size_max": sizes["size_max"]}


def me_result(data: GLMMSample, fit: GLMMFit, family: str, points: int,
              method: str) -> ResultBundle:
    """Result in Stata's me style: /var(z[group]) ancillary terms."""
    frame, design = data.frame, data.design
    p, q = len(design.terms), data.z.shape[1]
    variances = torch.exp(2 * fit.theta[p:])
    jacobian = torch.eye(p + q, dtype=torch.float64)
    jacobian[p:, p:] = torch.diag(2 * variances)
    covariance = jacobian @ fit.covariance @ jacobian.T
    params = torch.cat([fit.theta[:p], variances])
    names = [f"/var({name}[{data.group}])" for name in data.z_names]
    slopes = [i for i, term in enumerate(design.terms) if term != "Intercept"]
    tests = {"model": wald_test(params, covariance, slopes,
                                label="Wald chi2 test of the slopes")}
    if frame.spec.covariance == "nonrobust":
        tests["lr_vs_pooled"] = common.boundary_lr(
            2 * (fit.log_likelihood - fit.pooled_log_likelihood), q, _POOLED[family])
    criteria = information_criteria(fit.log_likelihood, p + q, frame.n)
    metrics = {"log_likelihood": fit.log_likelihood, "aic": criteria["aic"],
               "bic": criteria["bic"], **_structure(data)}
    if family in _LATENT and q == 1:
        metrics["icc"] = float(variances[0] / (variances[0] + _LATENT[family]))
    extra = {
        "group": data.group, "random_effects": data.z_names, "family": family,
        "integration": {"method": method, "points": points, "adaptations": fit.adaptations},
        "pooled_log_likelihood": fit.pooled_log_likelihood,
        "ln_sd": {name: {"estimate": float(fit.theta[p + d]),
                         "std_error": float(fit.covariance[p + d, p + d].sqrt())}
                  for d, name in enumerate(data.z_names)},
    }
    bundle = build_result(
        frame, terms=[*design.terms, *names], params=params, covariance=covariance,
        title=_TITLES[family], equations=[frame.spec.outcome] * p + [data.group] * q,
        use_t=False, metrics=metrics, solver="newton_adaptive_quadrature",
        solver_diagnostics={"converged": True, "iterations": fit.optimizer["iterations"]},
        optimizer=fit.optimizer, inference=fit.info, tests=tests, extra=extra,
        categories=design.categories,
        provenance={"integration": f"{method} with {points} points per dimension",
                    "derivatives": "analytic (adaptive parameters held fixed between updates)"})
    common.log_intervals(bundle, range(p, p + q),
                         [2 * float(fit.covariance[p + d, p + d].sqrt()) for d in range(q)],
                         frame.spec.alpha)
    return bundle


def _fit_me(spec: ModelSpec, data: Any, family: str) -> ResultBundle:
    command = _COMMANDS[family]
    frame = ModelFrame(spec, data)
    points = frame.option("intpoints")
    method = frame.option("intmethod")
    sample_ = sample(frame, family, command, frame.role("group")[0], frame.role("random"))
    fit = fit_glmm(sample_, family, points, method)
    return me_result(sample_, fit, family, points, method)


def fit_melogit(spec: ModelSpec, data: Any) -> ResultBundle:
    return _fit_me(spec, data, "logit")


def fit_meprobit(spec: ModelSpec, data: Any) -> ResultBundle:
    return _fit_me(spec, data, "probit")


def fit_mepoisson(spec: ModelSpec, data: Any) -> ResultBundle:
    return _fit_me(spec, data, "poisson")


# ---- convenience ---------------------------------------------------------------------

_DOC = """{title} (Stata ``{command}``{spss}).

    Model
        {model} for row i of group j, ``u_j = (u_slope, u_cons)`` independent normal with
        variances ``var(slope[group])`` (only with ``random``) and ``var(_cons[group])``.

    Estimator
        Maximum likelihood with the random effects integrated out by mean-variance
        adaptive Gauss-Hermite quadrature (``intmethod='mvaghermite'``, Stata's default;
        ``'mcaghermite'`` mode-curvature adaptive; ``'ghermite'`` non-adaptive) with
        ``intpoints`` nodes per dimension (7 by default, Stata's). The quadrature runs on
        the standardized effects v = u / sd. Newton-Raphson with the analytic gradient and
        Hessian of the quadrature sum (adaptive locations and scales are recomputed at
        the estimates until they stop moving). Regressors are centred internally.

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        y: {outcome}. x: regressors (a constant is added unless ``intercept=False``).
        group: grouping column (any scalar labels).
        random: at most one column with a random slope (independent of the intercept).
        intpoints: quadrature points per dimension (1..64). intmethod: see above.
        {offset}covariance: ``'nonrobust'`` (default, observed information),
            ``'robust'`` (clustered on the groups, G/(G-1)) or ``'cluster'`` with
            ``cluster`` (a column nesting the groups).
        categorical, intercept, missing ('raise' or 'drop'), alpha: as everywhere.

    Result
        Fixed effects (z tests) and ``/var(_cons[group])`` (and ``/var(slope[group])``) on
        the variance scale with delta-method standard errors and intervals
        ``exp(ln v -/+ z se(ln v))``. ``metrics``: log_likelihood, aic, bic, n_groups,
        group_size_min/avg/max{icc}. ``tests``: ``model`` (Wald chi2 of the slopes) and,
        with the default covariance, ``lr_vs_pooled`` (LR test vs. the pooled model:
        chibar2(01) for the random intercept, conservative chi2(2) with a random slope).
        A constant outcome raises ``constant_outcome``. ``extra``: integration
        settings, pooled log likelihood, ln sd estimates. A variance estimated at zero
        raises ``boundary_solution``{separation}.

    Stata
        ``{command} {stata_example}`` is ``oe.{command}(data=df, {oe_example})``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> g = np.repeat(np.arange(100), 10)
        >>> df = pd.DataFrame({{"g": g, "x": rng.normal(size=1000)}})
        >>> eta = -0.5 + 0.8 * df.x + rng.normal(size=100)[g]
        >>> {example_y}
        >>> print(oe.{command}(data=df, y="y", x=["x"], group="g").summary())
"""


def _convenience(estimator: str, *, data: Any, y: str, x: Sequence[str] | None, group: str,
                 random: Sequence[str] | None, intpoints: int, intmethod: str,
                 covariance: str | None, cluster: str | None,
                 categorical: Sequence[str] | None, intercept: bool, missing: str, alpha: float,
                 offset: str | None = None, exposure: str | None = None) -> ResultBundle:
    from openecon.analysis import fit

    spec = make_spec(
        estimator, outcome=y, predictors=column_list(x, "x"), intercept=intercept,
        covariance=covariance, cluster=cluster, categorical=column_list(categorical, "categorical"),
        missing=missing, alpha=alpha,
        columns={"group": group, "random": column_list(random, "random"), "offset": offset,
                 "exposure": exposure},
        options={"intpoints": intpoints, "intmethod": intmethod})
    return fit(spec, data=data)


def melogit(*, data: Any, y: str, x: Sequence[str] | None = None, group: str,
            random: Sequence[str] | None = None, intpoints: int = 7,
            intmethod: str = "mvaghermite", covariance: str | None = None,
            cluster: str | None = None, categorical: Sequence[str] | None = None,
            intercept: bool = True, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    return _convenience("melogit", data=data, y=y, x=x, group=group, random=random,
                        intpoints=intpoints, intmethod=intmethod, covariance=covariance,
                        cluster=cluster, categorical=categorical, intercept=intercept,
                        missing=missing, alpha=alpha)


def meprobit(*, data: Any, y: str, x: Sequence[str] | None = None, group: str,
             random: Sequence[str] | None = None, intpoints: int = 7,
             intmethod: str = "mvaghermite", covariance: str | None = None,
             cluster: str | None = None, categorical: Sequence[str] | None = None,
             intercept: bool = True, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    return _convenience("meprobit", data=data, y=y, x=x, group=group, random=random,
                        intpoints=intpoints, intmethod=intmethod, covariance=covariance,
                        cluster=cluster, categorical=categorical, intercept=intercept,
                        missing=missing, alpha=alpha)


def mepoisson(*, data: Any, y: str, x: Sequence[str] | None = None, group: str,
              random: Sequence[str] | None = None, offset: str | None = None,
              exposure: str | None = None, intpoints: int = 7, intmethod: str = "mvaghermite",
              covariance: str | None = None, cluster: str | None = None,
              categorical: Sequence[str] | None = None, intercept: bool = True,
              missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    return _convenience("mepoisson", data=data, y=y, x=x, group=group, random=random,
                        intpoints=intpoints, intmethod=intmethod, covariance=covariance,
                        cluster=cluster, categorical=categorical, intercept=intercept,
                        missing=missing, alpha=alpha, offset=offset, exposure=exposure)


_SEPARATION = ("; a regressor that separates the outcome raises ``separation_detected``")
melogit.__doc__ = _DOC.format(
    title="Mixed-effects (random-intercept) logistic regression", command="melogit",
    spss="; SPSS GENLINMIXED with a binary logit target",
    model="``Pr(y_ij = 1 | u_j) = L(x_ij'b + z_ij'u_j)``, L the logistic cdf,",
    outcome="binary outcome coded 0/1", offset="",
    icc=" and icc = var(_cons)/(var(_cons) + pi^2/3) for random-intercept models",
    separation=_SEPARATION, stata_example="y x || g:",
    oe_example="y='y', x=['x'], group='g'",
    example_y='df["y"] = (rng.random(1000) < 1 / (1 + np.exp(-eta))) * 1.0')
meprobit.__doc__ = _DOC.format(
    title="Mixed-effects (random-intercept) probit regression", command="meprobit",
    spss="; SPSS GENLINMIXED with a binary probit target",
    model="``Pr(y_ij = 1 | u_j) = Phi(x_ij'b + z_ij'u_j)``", outcome="binary outcome coded 0/1",
    offset="", icc=" and icc = var(_cons)/(var(_cons) + 1) for random-intercept models",
    separation=_SEPARATION, stata_example="y x || g:",
    oe_example="y='y', x=['x'], group='g'",
    example_y='df["y"] = (rng.random(1000) < 1 / (1 + np.exp(-eta))) * 1.0')
mepoisson.__doc__ = _DOC.format(
    title="Mixed-effects (random-intercept) Poisson regression", command="mepoisson",
    spss="; SPSS GENLINMIXED with a Poisson log-linear target",
    model="``y_ij | u_j ~ Poisson(exp(x_ij'b + offset_ij + z_ij'u_j))``",
    outcome="nonnegative integer counts",
    offset="offset: column added to the linear predictor with coefficient 1; exposure: "
           "positive column whose log is added (Stata's exposure()).\n        ",
    icc="", separation="", stata_example="y x, exposure(t) || g:",
    oe_example="y='y', x=['x'], group='g', exposure='t'",
    example_y='df["y"] = rng.poisson(np.exp(eta))')
