"""Linear mixed models by ML or REML: Stata's ``mixed`` (SPSS MIXED).

    y = X b + Z u + e,    u ~ N(0, G),  e ~ N(0, s2 I)

with one grouping level (random intercept and/or random slopes with a chosen
covariance structure) or two nested levels (a random intercept at the top
level; the random effects above at the lowest level). The likelihood is
evaluated on group sufficient statistics (``lmm_kernels``); its gradient is
analytic and the Hessian of the variance parameters is the Ridders numerical
derivative of that gradient. Conventions (Stata ``mixed``, Methods and
formulas):

* Variance parameters are estimated on the scale of log standard deviations
  (``lns``), the log-Cholesky factor (unstructured) and ``ln sigma_e``, by BFGS
  with a Newton polish; b is profiled out by GLS. Reported: ``/var(z[group])``,
  ``/cov(z1,z2[group])`` and ``/var(Residual)`` with delta-method standard
  errors from the observed information of the (restricted) log likelihood
  with b profiled out; intervals for variances are exp(ln v -/+ z se(ln v)),
  for covariances symmetric.
* Fixed effects: GLS with the model-based covariance (X'V^-1 X)^-1 and z
  inference (Stata's default without dfmethod). The default covariance matrix
  is block diagonal between the fixed effects and the variance parameters
  (Stata: "b is asymptotically uncorrelated with" the variance parameters).
* ``robust`` (ML only, as in Stata) follows Stata's mixed Methods and formulas:
  the sandwich of that block-diagonal matrix with the per-top-level-group
  scores of ALL parameters (b and the variance parameters), times G/(G-1);
  ``cluster`` uses a column that nests the top-level groups. The robust
  covariance is not block diagonal.
* ``tests['model']`` is the Wald chi2 of the fixed slopes; ``tests['lr_vs_linear']``
  the LR test against the linear regression (same ML/REML criterion):
  chibar2(01) with one variance component, conservative chi2 otherwise. It is
  not reported under robust/cluster covariance (a pseudolikelihood), as Stata.
* A constant outcome, an exact fit and collinear random-effects columns are
  refused before estimation; a variance at zero raises ``boundary_solution``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, kernel_call, make_spec,
    ml_covariance, wald_test,
)
from openecon.econometrics.discrete.common import constant_shift
from openecon.econometrics.mixed import common
from openecon.econometrics.mixed.lmm_kernels import CovStructure, MixedLikelihood
from openecon.engines import linalg, optimize
from openecon.engines.contracts import KernelError
from openecon.models import ModelSpec, ResultBundle

_LOG_2PI = math.log(2 * math.pi)
# Relative variance (variance / residual variance, per unit of z^2) below which a
# component is at the boundary of the parameter space: an sd ratio of 1e-4.
_BOUNDARY = 1e-8
# Smallest eigenvalue of a random-effects correlation matrix treated as singular.
_SINGULAR_CORRELATION = 1e-8


class _Setup:
    """The validated sample of one mixed-model specification."""

    def __init__(self, frame: ModelFrame):
        spec = frame.spec
        self.frame = frame
        self.groups = frame.role("group")
        self.random = frame.role("random")
        if not 1 <= len(self.groups) <= 2:
            raise AnalysisError("invalid_spec", "mixed takes one grouping column or two nested "
                                "ones (top level first), e.g. group=['school', 'class'].")
        used = {spec.outcome, *spec.predictors, *self.random}
        for name in self.groups:
            if name in used:
                raise AnalysisError("invalid_spec", f"The grouping column '{name}' must differ from "
                                    "the outcome, the regressors and the random slopes.")
        self.reml = frame.option("method") == "reml"
        self.intercept_re = bool(frame.option("random_intercept"))
        if not self.random and not self.intercept_re:
            raise AnalysisError("invalid_spec", "The model has no random effect: keep the random "
                                "intercept or give random slopes (random=[...]).")
        if self.reml and spec.covariance in {"robust", "cluster"}:
            raise AnalysisError("unsupported_covariance", "Stata's mixed allows robust and cluster "
                                "covariances with ML only; use method='ml'.")
        self.frequency = frame.weights()
        design = frame.drop_collinear(frame.design(), self.frequency)
        if not design.terms:
            raise AnalysisError("invalid_spec", "mixed needs at least one fixed-effect term (the "
                                "constant or a regressor).")
        self.design = design
        self.y = frame.numeric(spec.outcome)
        common.require_variation(self.y, "mixed", spec.outcome)
        columns = [frame.numeric(name) for name in self.random]
        if self.intercept_re:
            columns.append(torch.ones(frame.n, dtype=torch.float64))
        self.z = torch.stack(columns, dim=1)
        self.z_names = [*self.random, *(["_cons"] if self.intercept_re else [])]
        common.require_independent_effects(self.z, self.z_names, self.frequency, "mixed")
        levels = common.grouping_codes(frame, self.groups)
        self.low_codes, self.n_low = levels[-1]
        self.two_level = len(levels) == 2
        if self.two_level:
            self.top_codes, self.n_top = levels[0]
            self.top_of_low = common.parent_codes(self.low_codes, self.top_codes, self.n_low)
        else:
            self.top_codes, self.n_top, self.top_of_low = self.low_codes, self.n_low, None
        self.levels = [common.group_sizes(codes, count, self.frequency) for codes, count in levels]
        if self.levels[-1]["size_max"] <= 1:
            raise AnalysisError("insufficient_group_size", f"Every group of '{self.groups[-1]}' "
                                "has a single observation, so its variance cannot be separated "
                                "from the residual variance.")
        self.nobs = int(round(float(self.frequency.sum()))) if self.frequency is not None \
            else frame.n
        self.p = len(design.terms)
        self.structure = CovStructure(frame.option("covstructure"), self.z.shape[1])
        self.n_variance = self.structure.size + int(self.two_level)
        if self.nobs - self.p - self.n_variance - 1 <= 0:
            raise AnalysisError("insufficient_observations", "The mixed model needs more "
                                "observations than parameters.")


def _likelihood(setup: _Setup, x: Tensor) -> MixedLikelihood:
    return kernel_call(MixedLikelihood, x, setup.y, setup.z, setup.low_codes, setup.n_low,
                       setup.structure, top_of_low=setup.top_of_low, n_top=setup.n_top,
                       frequency=setup.frequency, reml=setup.reml)


def _start(setup: _Setup, like: MixedLikelihood) -> Tensor:
    """Best of a few relative variance scales, with s2 profiled at each."""
    weights = setup.frequency
    z2 = (setup.z.square() if weights is None else setup.z.square() * weights[:, None]).sum(0)
    z2 = (z2 / like.nobs).clamp_min(1e-300)
    best, best_value = None, -math.inf
    for scale in (0.01, 0.1, 1.0, 10.0):
        relative = setup.structure.start(0.25 * scale / z2)
        parts = [relative]
        if setup.two_level:
            parts.append(torch.tensor([0.5 * math.log(0.25 * scale)], dtype=torch.float64))
        theta = torch.cat([*parts, torch.zeros(1, dtype=torch.float64)])
        try:
            sigma2 = like.profiled_sigma2(theta)
        except KernelError:
            continue
        theta = _rescale(setup, theta, sigma2)
        value = float(like.value(theta))
        if value > best_value:
            best, best_value = theta, value
    if best is None:
        raise AnalysisError("invalid_start", "The mixed-model likelihood could not be evaluated "
                            "at the starting values; check the outcome for an exact fit.")
    return best


def _rescale(setup: _Setup, theta: Tensor, sigma2: Tensor) -> Tensor:
    """theta of (G, g, s2) multiplied by sigma2 (relative -> absolute parameters)."""
    m = setup.structure.size
    out = theta.clone()
    out[:m] = setup.structure.scaled(theta[:m], sigma2)
    out[m:] = theta[m:] + 0.5 * torch.log(sigma2)
    return out


def _maximize(setup: _Setup, like: MixedLikelihood, start: Tensor) -> optimize.OptimResult:
    def hessian(theta: Tensor) -> Tensor:
        return optimize.numerical_hessian(like.gradient, theta)

    try:
        result = optimize.maximize_bfgs(like, start, hessian_fn=hessian, raise_on_failure=False)
    except KernelError as exc:
        raise AnalysisError("numerical_failure", f"The mixed-model likelihood could not be "
                            f"maximized: {exc}") from exc
    _check_boundary(setup, result)
    if not result.converged:
        raise AnalysisError("nonconvergence", "The mixed-model likelihood maximization did not "
                            f"converge: {result.diagnostics.get('message')} Centre or rescale "
                            "random-slope variables with large levels or scales, or simplify "
                            "the random-effects covariance structure.")
    return result


def _check_boundary(setup: _Setup, result: optimize.OptimResult) -> None:
    """Refuse a variance component (or a correlation) at the edge of the parameter space."""
    theta = result.theta
    m = setup.structure.size
    sigma2 = torch.exp(2 * theta[-1])
    matrix = setup.structure.matrix(theta[:m])
    weights = setup.frequency
    z2 = (setup.z.square() if weights is None else setup.z.square() * weights[:, None]).sum(0)
    z2 = z2 / setup.nobs
    relative = matrix.diagonal() * z2 / sigma2
    problems = [f"var({name}[{setup.groups[-1]}])"
                for name, value in zip(setup.z_names, relative.tolist(), strict=True)
                if value < _BOUNDARY]
    if setup.two_level and float(torch.exp(2 * theta[m]) / sigma2) < _BOUNDARY:
        problems.insert(0, f"var(_cons[{setup.groups[0]}])")
    if problems:
        raise AnalysisError(
            "boundary_solution",
            f"The maximum likelihood estimate of {', '.join(problems)} lies at zero (the boundary "
            "of the parameter space), where the likelihood is flat. Remove that random effect "
            "(or fit the model without it).")
    if setup.structure.kind in {"unstructured", "exchangeable"} and setup.structure.q > 1:
        sd = matrix.diagonal().sqrt()
        values = torch.linalg.eigvalsh(matrix / sd[:, None] / sd)
        if float(values.min()) < _SINGULAR_CORRELATION:
            raise AnalysisError(
                "boundary_solution",
                "The random effects are estimated as perfectly correlated (a singular "
                "covariance matrix at the boundary of the parameter space). Use "
                "covstructure='independent' or drop a random slope.")


def _variance_terms(setup: _Setup) -> tuple[list[str], list[str], list[bool]]:
    """Names, equations and is-variance flags of the reported variance parameters."""
    low, structure, names = setup.groups[-1], setup.structure, setup.z_names
    terms, equations, variance = [], [], []
    if setup.two_level:
        terms.append(f"/var(_cons[{setup.groups[0]}])")
        equations.append(setup.groups[0])
        variance.append(True)
    for a, b in structure.entries:
        if structure.kind == "identity" or (structure.kind == "exchangeable" and a == b):
            label = f"/var({' '.join(names)}[{low}])"
        elif structure.kind == "exchangeable":
            label = f"/cov({','.join(names)}[{low}])"
        elif a == b:
            label = f"/var({names[a]}[{low}])"
        else:
            label = f"/cov({names[b]},{names[a]}[{low}])"
        terms.append(label)
        equations.append(low)
        variance.append(a == b)
    terms.append("/var(Residual)")
    equations.append("Residual")
    variance.append(True)
    return terms, equations, variance


def _reported(setup: _Setup, theta: Tensor) -> tuple[Tensor, Tensor]:
    """Reported variances/covariances (top, lowest level, residual) and d phi/d theta.

    The parameter order of theta is (lowest-level structure, [top], ln sigma_e) while
    the reported order is (top, lowest level, residual).
    """
    structure, m = setup.structure, setup.structure.size
    rows, jacobian = [], []
    size = len(theta)
    if setup.two_level:
        g = torch.exp(2 * theta[m])
        row = torch.zeros(size, dtype=torch.float64)
        row[m] = 2 * g
        rows.append(g.reshape(1))
        jacobian.append(row[None, :])
    values = structure.reported(structure.matrix(theta[:m]))
    block = torch.zeros((len(values), size), dtype=torch.float64)
    block[:, :m] = structure.reported_jacobian(theta[:m])
    rows.append(values)
    jacobian.append(block)
    sigma2 = torch.exp(2 * theta[-1])
    row = torch.zeros(size, dtype=torch.float64)
    row[-1] = 2 * sigma2
    rows.append(sigma2.reshape(1))
    jacobian.append(row[None, :])
    return torch.cat(rows), torch.cat(jacobian)


def _linear_log_likelihood(setup: _Setup, x: Tensor) -> float:
    """(Restricted) log likelihood of the linear regression of y on X."""
    fit = kernel_call(linalg.least_squares, x, setup.y, setup.frequency, drop_collinear=False)
    n, p = float(setup.nobs), setup.p
    ssr = float(fit.ssr)
    scale = float(linalg.weighted_crossprod(setup.y[:, None], setup.frequency)[0, 0])
    if ssr <= 1e-24 * scale:
        raise AnalysisError("perfect_fit", "The regressors explain the outcome exactly; the "
                            "mixed model is not identified.")
    if not setup.reml:
        return -n / 2 * (_LOG_2PI + math.log(ssr / n) + 1)
    gram = linalg.weighted_crossprod(x, setup.frequency)
    logdet = float(2 * torch.log(torch.linalg.cholesky(gram).diagonal()).sum())
    return -(n - p) / 2 * (_LOG_2PI + math.log(ssr / (n - p)) + 1) - logdet / 2


def _icc(setup: _Setup, reported: Tensor) -> dict[str, float]:
    """Stata's estat icc for random-intercept-only models (residual-scale ICCs)."""
    if setup.random or not setup.intercept_re:
        return {}
    values = reported.tolist()
    total = sum(values)
    if setup.two_level:
        top, low = values[0], values[1]
        return {setup.groups[0]: top / total,
                f"{setup.groups[1]}|{setup.groups[0]}": (top + low) / total}
    return {setup.groups[0]: values[0] / total}


def _covariance(frame: ModelFrame, setup: _Setup, like: MixedLikelihood, x: Tensor,
                theta: Tensor, final: Any, hessian: Tensor, shift: Tensor, jacobian: Tensor,
                v_theta: Tensor) -> tuple[Tensor, dict[str, Any]]:
    """Joint covariance of (b, reported variance parameters), Stata's mixed conventions.

    nonrobust: block diagonal, (X'V^-1 X)^-1 for b and the inverse observed information of
    the likelihood with b profiled out for theta (carried to the reported variances by the
    delta method). robust / cluster (ML only): Stata's _robust with that block-diagonal
    matrix as bread and the per-top-level-group scores of (b, theta), G/(G-1).
    """
    p = setup.p
    k = p + jacobian.shape[0]
    transform = torch.zeros((k, len(theta) + p), dtype=torch.float64)
    transform[:p, :p] = shift
    transform[p:, p:] = jacobian
    kind = frame.spec.covariance
    if kind == "nonrobust":
        joint = torch.block_diag(kernel_call(optimize.information_inverse,
                                             final.xvx / final.sigma2), v_theta)
        info: dict[str, Any] = {
            "covariance": "nonrobust",
            "correction": "fixed effects: (X'V^-1 X)^-1 (model-based GLS); variance "
                          "parameters: observed information of the likelihood with b profiled "
                          "out, delta method; block diagonal (Stata's mixed)"}
        return transform @ joint @ transform.T, info
    scores = kernel_call(like.group_scores, x, setup.y, setup.z, setup.low_codes, theta,
                         setup.frequency)
    if kind == "robust":
        clusters, names = [(torch.arange(setup.n_top), setup.n_top)], [setup.groups[0]]
    else:
        clusters = common.check_cluster_nesting(frame, setup.top_codes, setup.n_top,
                                                setup.groups[0])
        names = [frame.spec.cluster]
    bread = torch.block_diag(-final.xvx / final.sigma2, hessian)
    joint, info = ml_covariance(frame, hessian=bread, scores=scores, kind="cluster",
                                nobs=setup.n_top, clusters=clusters, cluster_names=names)
    info["covariance"] = kind
    info["correction"] = (f"sandwich clustered on '{info['cluster_column']}' for every "
                          "parameter (G/(G-1)): block-diagonal observed information as bread, "
                          "top-level group scores of (b, theta) (Stata's mixed, vce(robust))")
    return transform @ joint @ transform.T, info


def fit_mixed(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``mixed``: ``(spec, data) -> ResultBundle`` (see the module docstring)."""
    frame = ModelFrame(spec, data)
    setup = _Setup(frame)
    design = setup.design
    # Centre the slopes (an exact reparameterization of the constant; unit Jacobian, so the
    # restricted likelihood is unchanged) for a well-conditioned X'V^-1 X.
    x = design.x.clone()
    means = torch.zeros(setup.p, dtype=torch.float64)
    if design.intercept and setup.p > 1:
        w = setup.frequency
        means[1:] = x[:, 1:].mean(0) if w is None else (w[:, None] * x[:, 1:]).sum(0) / w.sum()
        x[:, 1:] -= means[1:]
    shift = constant_shift(means) if design.intercept else torch.eye(setup.p, dtype=torch.float64)
    linear = _linear_log_likelihood(setup, x)          # refuses an exact fit before optimizing
    like = _likelihood(setup, x)
    result = _maximize(setup, like, _start(setup, like))
    theta = result.theta
    final = kernel_call(like.evaluate, theta)
    log_likelihood = float(final.value)
    sigma2 = final.sigma2
    beta = shift @ final.beta
    try:
        v_theta = optimize.information_inverse(-result.hessian)
    except KernelError as exc:
        raise AnalysisError("singular_information", "The information matrix of the variance "
                            "parameters is not positive definite; a variance component is not "
                            "identified by the data.") from exc
    reported, jacobian = _reported(setup, theta)
    covariance, info = _covariance(frame, setup, like, x, theta, final, result.hessian, shift,
                                   jacobian, v_theta)
    v_beta = covariance[:setup.p, :setup.p]
    var_terms, var_equations, is_variance = _variance_terms(setup)
    terms = [*design.terms, *var_terms]
    params = torch.cat([beta, reported])
    slopes = [i for i, term in enumerate(design.terms) if term != "Intercept"]
    tests = {"model": wald_test(beta, v_beta, slopes,
                                label="Wald chi2 test of the fixed slopes")}
    if spec.covariance == "nonrobust":
        # Stata prints no LR test vs. the linear model with a robust (pseudo) likelihood.
        tests["lr_vs_linear"] = common.boundary_lr(2 * (log_likelihood - linear),
                                                   setup.n_variance, "linear model")
    n_params = setup.p + setup.n_variance + 1
    criteria = information_criteria(log_likelihood, n_params, setup.nobs)
    top_level = setup.levels[0]
    icc = _icc(setup, reported)
    metrics = {"log_likelihood": log_likelihood, "aic": criteria["aic"], "bic": criteria["bic"],
               "n_groups": top_level["n_groups"], "group_size_min": top_level["size_min"],
               "group_size_avg": top_level["size_avg"], "group_size_max": top_level["size_max"]}
    if icc:
        metrics["icc"] = next(iter(icc.values()))
    m = setup.structure.size
    extra = {
        "method": "reml" if setup.reml else "ml",
        "likelihood": "restricted (REML)" if setup.reml else "full (ML)",
        "covstructure": setup.structure.kind,
        "random_effects": {"group": setup.groups[-1], "effects": setup.z_names},
        "levels": [{"group": name, **level} for name, level in zip(setup.groups, setup.levels,
                                                                     strict=True)],
        "G": setup.structure.matrix(theta[:m]).tolist(),
        "residual_variance": float(sigma2),
        "icc": icc,
        "theta": theta.tolist(),
        "theta_std_error": v_theta.diagonal().sqrt().tolist(),
        "theta_parameterization": "lowest-level covariance structure parameters (log sd / "
                                  "log-Cholesky), then ln sd of the top-level intercept, then "
                                  "ln sigma_e",
        "linear_log_likelihood": linear,
    }
    if spec.covariance != "nonrobust":
        extra["notes"] = ["theta_std_error is model-based; the reported covariance is robust.",
                          "No LR test vs. the linear model under robust/cluster covariance "
                          "(the likelihood is treated as a pseudolikelihood, as in Stata)."]
    method = "REML" if setup.reml else "ML"
    bundle = build_result(
        frame, terms=terms, params=params, covariance=covariance,
        title=f"Mixed-effects {method} regression",
        equations=[spec.outcome] * setup.p + var_equations, use_t=False, metrics=metrics,
        fitted=design.x @ beta, solver="bfgs_profiled_mixed",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=common.optimizer_summary(result), inference=info, tests=tests, extra=extra,
        categories=design.categories, nobs=setup.nobs,
        provenance={"gradient": "analytic (fixed effects profiled by GLS)",
                    "hessian": "numerical (Ridders) derivative of the analytic gradient",
                    "likelihood_evaluation": "Woodbury identity on group sufficient statistics"},
    )
    positions = [setup.p + i for i, flag in enumerate(is_variance) if flag]
    common.variance_intervals(bundle, positions, spec.alpha)
    return bundle


# ---- convenience ---------------------------------------------------------------------


def mixed(*, data: Any, y: str, x: Sequence[str] | None = None, group: str | Sequence[str],
          random: Sequence[str] | None = None, random_intercept: bool = True,
          covstructure: str = "independent", method: str = "ml",
          covariance: str | None = None, cluster: str | None = None, weights: str | None = None,
          weight_type: str | None = None, categorical: Sequence[str] | None = None,
          intercept: bool = True, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Linear mixed-effects model by ML or REML (Stata ``mixed``; SPSS ``MIXED``).

    Model
        ``y = X b + Z u + e`` with group-specific random effects ``u ~ N(0, G)``
        and ``e ~ N(0, sigma_e^2 I)``. ``group`` names one grouping column, or two
        nested ones with the top level first (``['school', 'class']``: classes within
        schools; a class label is interpreted within its school). The lowest level
        carries a random intercept (unless ``random_intercept=False``) and random
        slopes on the ``random`` columns; the top level of a two-level model has a
        random intercept.

    Estimator
        For given variance parameters, b is the GLS estimate
        ``(X'V^-1 X)^-1 X'V^-1 y``; the (restricted) log likelihood with b profiled out
        is maximized over the log standard deviations / log-Cholesky factor of G, the
        top-level ln sd and ln sigma_e by BFGS with an analytic gradient, polished by
        Newton steps on the numerical (Ridders) Hessian of that gradient. Per group only
        q-by-q matrices are factorized (Woodbury identity on group sufficient
        statistics accumulated once), so a model on a million rows costs a few
        seconds. ``method='reml'`` maximizes Harville's restricted likelihood
        ``-1/2 [(N-p) ln 2pi + ln|V| + ln|X'V^-1 X| + r'V^-1 r]``.

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        y: continuous outcome. x: fixed-effect regressors (may be empty).
        group: grouping column or list of two nested columns (top level first).
        random: columns with random slopes at the lowest level (numeric).
        random_intercept: keep the random intercept at the lowest level (default True).
        covstructure: covariance of the lowest-level random effects:
            ``'independent'`` (default, Stata's), ``'unstructured'``, ``'exchangeable'``
            (common variance and covariance) or ``'identity'`` (common variance).
        method: ``'ml'`` (default, as Stata) or ``'reml'`` (SPSS MIXED's default).
        covariance: ``'nonrobust'`` (default: model-based ``(X'V^-1 X)^-1`` for b, the
            observed information for the variance parameters, block diagonal),
            ``'robust'`` (Stata's vce(robust): sandwich of every parameter clustered on
            the top-level groups, G/(G-1); ML only) or ``'cluster'`` with ``cluster`` (a
            column nesting the top-level groups).
        weights, weight_type: ``'fweight'`` only (rows replicated within their group).
        categorical: regressors to treatment-code. intercept: include the constant.
        missing: ``'raise'`` (default) or ``'drop'``. alpha: test size.

    Result
        Fixed effects (equation = outcome) and the random-effects parameters as ancillary
        terms ``/var(_cons[school])``, ``/var(x[class])``, ``/cov(x,_cons[class])`` and
        ``/var(Residual)`` (variance scale, delta-method standard errors, intervals
        ``exp(ln v -/+ z se/v)`` for variances). ``metrics``: log_likelihood (restricted
        under REML), aic, bic, n_groups and group_size_min/avg/max (top level), icc for
        random-intercept-only models. ``tests``: ``model`` (Wald chi2 of the fixed
        slopes) and, with the default covariance, ``lr_vs_linear`` (LR test vs. the
        linear regression: chibar2(01) for one variance component, conservative chi2
        otherwise). A constant outcome, an exact fit, collinear random-effects columns
        and variances at zero raise ``AnalysisError``. ``extra``: method,
        G matrix, residual variance, ICCs per level, group sizes per level and the
        estimates in the optimization metric. ``oe.mixed_predict`` returns linear
        predictions, fitted values with BLUPs and the BLUPs themselves.

    Stata / SPSS
        ``mixed weight week || id: week, covariance(unstructured)`` is
        ``oe.mixed(data=df, y='weight', x=['week'], group='id', random=['week'],
        covstructure='unstructured')``; ``mixed y x || school: || class:`` is
        ``oe.mixed(data=df, y='y', x=['x'], group=['school', 'class'])``; add
        ``method='reml'`` for ``, reml`` (SPSS ``MIXED ... /METHOD=REML``).

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> g = np.repeat(np.arange(50), 8)
        >>> df = pd.DataFrame({"g": g, "x": rng.normal(size=400)})
        >>> df["y"] = 1 + 0.5 * df.x + rng.normal(size=50)[g] + rng.normal(size=400)
        >>> print(oe.mixed(data=df, y="y", x=["x"], group="g").summary())
    """
    from openecon.analysis import fit

    groups = [group] if isinstance(group, str) else column_list(group, "group")
    spec = make_spec(
        "mixed", outcome=y, predictors=column_list(x, "x"), intercept=intercept,
        covariance=covariance, cluster=cluster, categorical=column_list(categorical, "categorical"),
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"group": groups, "random": column_list(random, "random")},
        options={"random_intercept": None if random_intercept else False,
                 "covstructure": covstructure, "method": method})
    return fit(spec, data=data)
