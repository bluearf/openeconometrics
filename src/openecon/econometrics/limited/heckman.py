"""Heckman selection model: Stata's ``heckman`` (maximum likelihood and two-step).

Model
-----
    outcome:    y = x'b + u1            (observed only when selected)
    selection:  s = 1[z'g + u2 > 0]
    u1 ~ N(0, sigma^2),  u2 ~ N(0, 1),  corr(u1, u2) = rho.

When ``rho != 0`` least squares on the selected sample is inconsistent because
``E[u1 | selected] = rho sigma lambda(z'g)`` with the inverse Mills ratio
``lambda(c) = phi(c) / Phi(c)``.

Maximum likelihood (``method='ml'``, Stata's default)
----------------------------------------------------
    ln L = sum_selected w { ln Phi[(z'g + rho (y - x'b)/sigma) / sqrt(1 - rho^2)]
                            - ((y - x'b)/sigma)^2 / 2 - ln(sqrt(2 pi) sigma) }
         + sum_nonselected w ln Phi(-z'g),

maximized by Newton-Raphson over ``(b, g, athrho, lnsigma)``, ``rho =
tanh(athrho)``, ``sigma = exp(lnsigma)``, with the analytic score and Hessian of
``selection_kernels.HeckmanObjective`` from the two-step estimates. ``rho``,
``sigma`` and ``lambda = rho sigma`` are reported with delta-method standard
errors; ``tests['rho']`` is the likelihood-ratio test of independent equations
against the probit of the selection equation plus the normal regression of
the selected outcomes (``nonrobust``, ``opg``), or the Wald test of
``athrho = 0`` (``robust``, ``cluster``).

Two-step (``method='twostep'``, Heckman 1979)
-------------------------------------------
1. Probit of ``s`` on ``z`` gives ``g`` and its covariance ``V_p``.
2. ``lambda_i = phi(z_i'g) / Phi(z_i'g)`` and ``delta_i = lambda_i (lambda_i + z_i'g)``.
3. Least squares of ``y`` on ``X* = [x, lambda]`` over the selected sample
   gives ``b`` and ``b_lambda``, with residuals ``e``.
4. ``sigma^2 = (e'e + b_lambda^2 sum delta_i) / N_1`` and ``rho = b_lambda / sigma``.
   If ``|rho| > 1`` it is truncated to +-1 and ``sigma = |b_lambda|`` (Stata's
   default ``rhosigma``), with a warning.
5. Heckman's consistent covariance, with ``D = diag(delta_i)``:

       V = sigma^2 (X*'X*)^-1 [X*'(I - rho^2 D) X* + Q] (X*'X*)^-1,
       Q = rho^2 (X*'D Z) V_p (Z'D X*).

   The selection coefficients keep the probit covariance ``V_p``; the
   covariance between the two blocks is ``b_lambda (X*'X*)^-1 (X*'D Z) V_p``,
   which follows from the same expansion of ``lambda(z'g-hat)``.

Both estimators run on regressors centred at their means (the outcome
regressors at the means of the selected rows) and map the two constants back
exactly; see the ``common`` module notes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, column_list, information_criteria, kernel_call, lr_test,
    wald_test,
)
from openecon.econometrics.discrete.kernels import mills_ratio
from openecon.econometrics.limited.common import (
    LIKELIHOOD_KINDS, Weights, binary_outcome, build_spec, centre, check_pweights,
    constant_shift, count, likelihood_covariance, likelihood_weights, maximize, probit_fit,
    require_error_variance, require_observations, resolve_covariance, rho_record, sigma_record,
    slope_indices, solver_fields, uncentre, weight_scale,
)
from openecon.econometrics.limited.selection_kernels import HeckmanObjective
from openecon.econometrics.registry import role_columns
from openecon.engines.linalg import least_squares, weighted_crossprod
from openecon.engines.optimize import information_inverse
from openecon.models import ModelSpec, ResultBundle

# |athrho| beyond which the correlation is at the boundary: |rho| > 1 - 1.2e-5.
BOUNDARY = 6.0


class SelectionSample:
    """Sample, designs and weights shared by ``heckman`` and ``heckprobit``."""

    def __init__(self, spec: ModelSpec, data: Any, command: str, *, method: str = "ml"):
        self.command = command
        select_name = role_columns(spec, "select")[0]
        regressors = role_columns(spec, "select_x")
        if select_name == spec.outcome:
            raise AnalysisError("invalid_spec", f"{command}: the selection indicator must be a "
                                "column different from the outcome.")
        if select_name in regressors or select_name in spec.predictors:
            raise AnalysisError("invalid_spec", f"{command}: the selection indicator cannot be "
                                "a regressor.")
        if spec.outcome in regressors:
            raise AnalysisError("invalid_spec", f"{command}: the outcome cannot be a regressor "
                                "of the selection equation.")
        frame = ModelFrame(spec, data, allow_missing=[spec.outcome])
        if method == "ml":
            check_pweights(frame)
        selected = binary_outcome(frame, select_name, command) == 1
        unobserved = torch.isnan(self._outcome(frame)) & selected
        if bool(unobserved.any()):
            number = int(unobserved.sum())
            if spec.missing == "raise":
                raise AnalysisError(
                    "missing_values",
                    f"{number} selected observation(s) have a missing outcome. Choose "
                    "missing='drop' explicitly to exclude them, or mark them as not selected.")
            frame.restrict(~unobserved, f"Excluded {number} selected observation(s) with a "
                                        "missing outcome.")
            selected = binary_outcome(frame, select_name, command) == 1
        self.frame, self.selected = frame, selected
        self.weights: Weights = likelihood_weights(frame)
        w = self.weights.user
        self.n_selected = count(selected, self.weights)
        self.n_nonselected = self.weights.nobs - self.n_selected
        if self.n_selected == 0 or self.n_nonselected == 0:
            raise AnalysisError(
                "constant_selection",
                f"{command}: '{select_name}' does not vary in the estimation sample, so the "
                "selection equation is not identified. A selection model needs selected and "
                "nonselected observations.")
        raw = self._outcome(frame)
        self.y = torch.where(selected, raw, torch.zeros_like(raw))
        screen = self.weights.for_screen()
        self.design_z: Design = frame.drop_collinear(
            frame.design(regressors, intercept=True, prefix="select:"), screen)
        # The outcome equation is identified by the selected rows only.
        self.design_x: Design = frame.drop_collinear(frame.design(), w * selected)
        if not self.design_x.terms:
            raise AnalysisError("no_parameters", f"{command}: the outcome equation has no "
                                "regressor left; include a constant or a regressor that varies.")
        # Centred regressors in both equations; ``reported`` undoes it exactly.
        self.centres = [centre(self.design_x, w * selected), centre(self.design_z, screen)]
        self._probit: Any = None
        self._regression: Any = None

    def reported(self, theta: Tensor, covariance: Tensor, extra: int) -> tuple[Tensor, Tensor]:
        """Estimates ``(b, g, extra ancillary)`` and covariance for the regressors as given."""
        return uncentre(theta, covariance, constant_shift(self.centres, extra=extra))

    def require_outcome_variance(self) -> None:
        """Refuse selected outcomes that the outcome regressors fit exactly."""
        on = self.selected
        require_error_variance(float(self.selected_regression().ssr), self.y[on],
                               self.weights.user[on], self.command, "the selected outcomes")

    @staticmethod
    def _outcome(frame: ModelFrame) -> Tensor:
        name = frame.spec.outcome
        series = frame.series(name)
        if not (is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype)):
            raise AnalysisError("non_numeric_column", f"Column '{name}' must be numeric.")
        values = torch.as_tensor(series.to_numpy(dtype="float64", na_value=float("nan")).copy(),
                                 dtype=torch.float64)
        if bool(torch.isinf(values).any()):
            raise AnalysisError("non_finite_values", f"Column '{name}' contains non-finite "
                                "values.")
        return values

    def selection_probit(self):
        """Newton result of the probit of the selection indicator on ``z`` (cached)."""
        if self._probit is None:
            self._probit = probit_fit(
                self.design_z.x, self.selected.to(torch.float64), self.weights.user,
                command=self.command, what="probit of the selection equation",
                scale=weight_scale(self.weights))[1]
        return self._probit

    def selected_regression(self):
        """Least squares of the outcome on ``x`` over the selected rows (cached)."""
        if self._regression is None:
            on = self.selected
            self._regression = kernel_call(least_squares, self.design_x.x[on], self.y[on],
                                           self.weights.user[on], drop_collinear=False)
        return self._regression


def _two_step(sample: SelectionSample) -> dict[str, Any]:
    """Heckman's two-step estimates and consistent covariance (see the module notes)."""
    command, on = sample.command, sample.selected
    w = sample.weights.user
    probit = sample.selection_probit()
    gamma = probit.theta
    v_probit = kernel_call(information_inverse, -probit.hessian)
    x1, z1, y1, w1 = sample.design_x.x[on], sample.design_z.x[on], sample.y[on], w[on]
    _, mills, delta = mills_ratio(z1 @ gamma)
    augmented = torch.cat([x1, mills[:, None]], dim=1)
    try:
        solved = kernel_call(least_squares, augmented, y1, w1, drop_collinear=False)
    except AnalysisError as exc:
        if exc.code != "singular_design":
            raise
        raise AnalysisError(
            "collinear_mills_ratio",
            f"{command}: the inverse Mills ratio is collinear with the outcome regressors in "
            "the selected sample. Add a selection regressor that is excluded from the outcome "
            "equation.") from exc
    require_error_variance(float(solved.ssr), y1, w1, command, "the selected outcomes")
    k = x1.shape[1]
    n_selected = float(w1.sum())
    slope = float(solved.beta[k])
    variance = (float(solved.ssr) + slope ** 2 * float((w1 * delta).sum())) / n_selected
    sigma = math.sqrt(variance)
    rho = slope / sigma if sigma > 0 else 0.0
    truncated = abs(rho) > 1
    if truncated:
        rho, sigma = math.copysign(1.0, rho), abs(slope)
    inner = weighted_crossprod(augmented, w1 * (1 - rho ** 2 * delta))
    link = weighted_crossprod(augmented, w1 * delta, z1)                       # X*'D Z
    inner = inner + rho ** 2 * (link @ v_probit @ link.T)
    outcome = sigma ** 2 * (solved.xtx_inv @ inner @ solved.xtx_inv)
    cross = slope * (solved.xtx_inv @ link @ v_probit)                          # [k + 1, q]
    return {"beta": solved.beta, "gamma": gamma, "outcome_cov": (outcome + outcome.T) / 2,
            "probit_cov": v_probit, "cross_cov": cross, "sigma": sigma, "rho": rho,
            "lambda": slope, "truncated": truncated, "probit": probit, "ssr": float(solved.ssr),
            "n_selected": n_selected}


def _regression_log_likelihood(sample: SelectionSample) -> float:
    """Normal ML log likelihood of the outcome regression on the selected sample."""
    total = float(sample.weights.user[sample.selected].sum())
    ssr = float(sample.selected_regression().ssr)
    return -0.5 * total * (math.log(2 * math.pi * ssr / total) + 1)


def _selection_extra(sample: SelectionSample) -> dict[str, Any]:
    return {"select": sample.frame.spec.columns["select"],
            "selection_terms": sample.design_z.terms,
            "n_selected": sample.n_selected, "n_nonselected": sample.n_nonselected}


def _fit_two_step(sample: SelectionSample) -> ResultBundle:
    frame, spec = sample.frame, sample.frame.spec
    if spec.covariance != "nonrobust":
        raise AnalysisError(
            "unsupported_covariance",
            "heckman, method='twostep' has its own consistent covariance; choose "
            "covariance='nonrobust' (the default) or method='ml' for robust and cluster "
            "covariances.")
    if spec.weight_type not in (None, "fweight"):
        raise AnalysisError("unsupported_weights", "heckman, method='twostep' accepts frequency "
                            "weights only; use method='ml' for other weight types.")
    k, q = len(sample.design_x.terms), len(sample.design_z.terms)
    require_observations(sample.n_selected, k + 1)
    sample.require_outcome_variance()
    step = _two_step(sample)
    if step["truncated"]:
        frame.warn("The two-step estimate of rho was outside [-1, 1]; it is truncated to "
                   f"{step['rho']:+.0f} and sigma is set to |lambda| (Stata's rhosigma rule).")
    size = k + q + 1
    params = torch.cat([step["beta"][:k], step["gamma"], step["beta"][k:]])
    order = [*range(k), size - 1]                       # positions of (b, lambda) in the result
    covariance = torch.zeros((size, size), dtype=torch.float64)
    covariance[torch.tensor(order)[:, None], torch.tensor(order)] = step["outcome_cov"]
    covariance[k:k + q, k:k + q] = step["probit_cov"]
    covariance[torch.tensor(order), k:k + q] = step["cross_cov"]
    covariance[k:k + q, torch.tensor(order)] = step["cross_cov"].T
    params, covariance = sample.reported(params, covariance, 1)
    slopes = slope_indices(sample.design_x)
    tests = {"model": wald_test(params, covariance, slopes,
                                label="Wald chi2 test of the outcome slopes")} if slopes else {}
    metrics = {"rho": step["rho"], "sigma": step["sigma"], "lambda": step["lambda"],
               "n_selected": sample.n_selected, "n_censored": sample.n_nonselected}
    extra = {**_selection_extra(sample), "method": "twostep", "rho_truncated": step["truncated"],
             "probit_log_likelihood": step["probit"].value,
             "covariance": "Heckman (1979) two-step covariance; probit covariance for the "
                           "selection equation"}
    info = {"covariance": "nonrobust", "df_inference": None,
            "correction": "Heckman two-step: sigma^2 (X*'X*)^-1 [X*'(I - rho^2 D)X* + Q] "
                          "(X*'X*)^-1; observed-information probit covariance for the "
                          "selection equation"}
    terms = [*sample.design_x.terms, *sample.design_z.terms, "mills:lambda"]
    return build_result(
        frame, terms=terms, params=params, covariance=covariance,
        equations=[spec.outcome] * k + ["select"] * q + ["mills"], use_t=False,
        title="Heckman selection model (two-step)", metrics=metrics,
        fitted=torch.special.ndtr(sample.design_z.x @ step["gamma"]),
        observed=sample.selected.to(torch.float64), nobs=sample.weights.nobs, inference=info,
        tests=tests, extra=extra,
        categories={**sample.design_x.categories, **sample.design_z.categories},
        provenance={"prediction_definition": "fitted selection probability Phi(z'g) against "
                                             "the selection indicator"},
        solver="probit_newton_then_qr_least_squares",
        solver_diagnostics={"converged": True, "iterations": step["probit"].iterations})


def _raise_failure(command: str, athrho: float, message: Any) -> None:
    if abs(athrho) > BOUNDARY:
        sign = "+1" if athrho > 0 else "-1"
        raise AnalysisError(
            "boundary_solution",
            f"{command}: the correlation between the outcome and selection errors runs to "
            f"{sign} (athrho = {athrho:.3g}) and the likelihood keeps increasing, so no "
            "interior maximum exists. Either the two errors are (almost) perfectly correlated "
            "or the model is weakly identified because the selection equation has no regressor "
            "that is excluded from the outcome equation; add one, or use method='twostep'.")
    raise AnalysisError(
        "nonconvergence",
        f"{command} did not converge: {message} The selection likelihood is not globally "
        "concave; check that the selection equation contains a regressor excluded from the "
        "outcome equation and the scale of the regressors.")


def _fit_ml(sample: SelectionSample) -> ResultBundle:
    frame, spec, command = sample.frame, sample.frame.spec, sample.command
    weights = sample.weights
    k, q = len(sample.design_x.terms), len(sample.design_z.terms)
    size = k + q + 2
    require_observations(sample.n_selected, k + 2)
    require_observations(weights.nobs, size)
    sample.require_outcome_variance()
    probit = sample.selection_probit()
    try:
        step = _two_step(sample)
        beta0, rho0, sigma0 = step["beta"][:k], max(-0.9, min(0.9, step["rho"])), step["sigma"]
        origin = "Heckman two-step estimates"
    except AnalysisError as exc:
        if exc.code != "collinear_mills_ratio":
            raise
        beta0, rho0, sigma0 = sample.selected_regression().beta, 0.0, 0.0
        origin = "least squares on the selected sample, the selection probit and rho = 0"
    if not sigma0 > 0:
        total = float(weights.user[sample.selected].sum())
        sigma0 = math.sqrt(float(sample.selected_regression().ssr) / total)
    if not sigma0 > 0:
        raise AnalysisError("constant_outcome", f"{command}: the selected outcomes are fitted "
                            "exactly by the regressors, so sigma is not identified.")
    start = torch.cat([beta0, probit.theta,
                       torch.tensor([math.atanh(rho0), math.log(sigma0)], dtype=torch.float64)])
    objective = kernel_call(HeckmanObjective, sample.design_x.x, sample.design_z.x, sample.y,
                            sample.selected, weights.user)
    try:
        result = maximize(objective, start, what=command, scale=weight_scale(weights))
    except AnalysisError as exc:
        if exc.code != "numerical_failure":
            raise
        _raise_failure(command, 0.0, "the likelihood has no interior maximum along the "
                                     "iteration path.")
    a, t = k + q, k + q + 1
    centred = result.theta
    if not result.converged or abs(float(centred[a])) > BOUNDARY:
        _raise_failure(command, float(centred[a]), result.diagnostics.get("message"))
    log_likelihood = result.value
    covariance, info = likelihood_covariance(frame, result.hessian,
                                             lambda: objective.score_rows(centred), weights)
    theta, covariance = sample.reported(centred, covariance, 2)
    athrho, lnsigma = float(theta[a]), float(theta[t])
    rho = rho_record(athrho, float(covariance[a, a]), spec.alpha)
    sigma = sigma_record(lnsigma, float(covariance[t, t]), spec.alpha)
    scale = sigma["estimate"]
    lam = rho["estimate"] * scale
    jacobian = torch.tensor([scale / math.cosh(athrho) ** 2, lam], dtype=torch.float64)
    block = covariance[a:, a:]
    lam_se = math.sqrt(max(float(jacobian @ block @ jacobian), 0.0))
    slopes = slope_indices(sample.design_x)
    tests: dict[str, Any] = {}
    if slopes:
        tests["model"] = wald_test(theta, covariance, slopes,
                                   label="Wald chi2 test of the outcome slopes")
    comparison = probit.value + _regression_log_likelihood(sample)
    if spec.covariance in LIKELIHOOD_KINDS:
        tests["rho"] = lr_test(log_likelihood, comparison, 1,
                               label="LR test of independent equations (rho = 0)")
    else:
        tests["rho"] = wald_test(theta, covariance, [a],
                                 label="Wald test of independent equations (rho = 0)")
    criteria = information_criteria(log_likelihood, size, weights.nobs)
    metrics = {"log_likelihood": log_likelihood, "aic": criteria["aic"], "bic": criteria["bic"],
               "rho": rho["estimate"], "sigma": scale, "lambda": lam,
               "n_selected": sample.n_selected, "n_censored": sample.n_nonselected}
    extra = {**_selection_extra(sample), "method": "ml", "rho": rho, "sigma": sigma,
             "lambda": {"estimate": lam, "std_error": lam_se,
                        "method": "lambda = rho sigma; delta method on (athrho, lnsigma)"},
             "comparison_log_likelihood": comparison,
             "starting_values": origin}
    terms = [*sample.design_x.terms, *sample.design_z.terms, "/athrho", "/lnsigma"]
    return build_result(
        frame, terms=terms, params=theta, covariance=covariance,
        equations=[spec.outcome] * k + ["select"] * q + [None, None], use_t=False,
        metrics=metrics, fitted=objective.selection_probability(centred, sample.design_z.x),
        observed=sample.selected.to(torch.float64), nobs=weights.nobs, inference=info,
        tests=tests, extra=extra,
        categories={**sample.design_x.categories, **sample.design_z.categories},
        provenance={"prediction_definition": "fitted selection probability Phi(z'g) against "
                                             "the selection indicator"},
        **solver_fields(result))


def fit_heckman(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``heckman``: ``(spec, data) -> ResultBundle``."""
    method = spec.options.get("method", "ml")
    sample = SelectionSample(spec, data, "heckman", method=method)
    return _fit_two_step(sample) if method == "twostep" else _fit_ml(sample)


def heckman(*, data: Any, y: str, x: Sequence[str], select: str, select_x: Sequence[str],
            method: str = "ml", covariance: str | None = None,
            cluster: str | Sequence[str] | None = None, weights: str | None = None,
            weight_type: str | None = None, categorical: Sequence[str] | None = None,
            intercept: bool = True, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Heckman selection model, Stata's ``heckman`` (ML or two-step).

    Model
        ``y = x'b + u1`` is observed only when ``select = 1[z'g + u2 > 0]`` equals one;
        ``u1 ~ N(0, sigma^2)``, ``u2 ~ N(0, 1)`` and ``corr(u1, u2) = rho``. With
        ``rho != 0`` least squares on the selected sample is biased by the omitted term
        ``rho sigma phi(z'g) / Phi(z'g)``. The model is best identified when ``select_x``
        contains at least one regressor that is not in ``x``.

    Estimator
        ``method='ml'`` (default): full maximum likelihood over
        ``(b, g, athrho, lnsigma)`` by Newton-Raphson with the analytic score and Hessian,

            ``ln L = sum_sel w {ln Phi[(z'g + rho e) / sqrt(1 - rho^2)] - e^2/2
                     - ln(sqrt(2 pi) sigma)} + sum_nonsel w ln Phi(-z'g)``,
            ``e = (y - x'b) / sigma``,

        started at the two-step estimates. ``method='twostep'``: Heckman's (1979)
        estimator. A probit of ``select`` on ``z`` gives the inverse Mills ratio
        ``lambda_i``; least squares of ``y`` on ``[x, lambda]`` over the selected rows
        gives ``b`` and ``b_lambda``; ``sigma^2 = (e'e + b_lambda^2 sum delta_i) / N_1``,
        ``rho = b_lambda / sigma`` and the covariance is
        ``sigma^2 (X*'X*)^-1 [X*'(I - rho^2 D)X* + rho^2 (X*'DZ) V_p (Z'DX*)] (X*'X*)^-1``.
        Both methods centre the regressors of the two equations during the computation
        and map the constants back exactly.

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        y: outcome; it may be missing where ``select`` is 0 (values there are ignored).
        x: regressors of the outcome equation; they must be complete in every row.
        select: selection indicator coded 0/1.
        select_x: regressors of the selection equation (a constant is always included).
        method: ``'ml'`` or ``'twostep'``.
        covariance: ML: ``'nonrobust'`` (default; inverse observed information),
            ``'opg'``, ``'robust'`` (``N/(N-1)`` sandwich), ``'cluster'`` (``G/(G-1)``;
            implied by ``cluster``). Two-step: only ``'nonrobust'`` (its own covariance).
        cluster: cluster column, or a list of two columns (ML only).
        weights, weight_type: ML: ``'fweight'``, ``'aweight'``, ``'pweight'`` (robust by
            default), ``'iweight'``. Two-step: ``'fweight'`` only.
        categorical: columns of ``x`` or ``select_x`` to expand into indicators.
        intercept: include a constant in the outcome equation (default True).
        missing: ``'raise'`` (default) or ``'drop'`` incomplete rows; a selected row with a
            missing outcome is incomplete.
        alpha: significance level of the confidence intervals.

    Result
        z statistics. Terms: the outcome equation (equation = the outcome),
        ``select:<term>`` (equation ``select``), then for ML the ancillary ``/athrho`` and
        ``/lnsigma``, for the two-step ``mills:lambda`` (the coefficient of the inverse
        Mills ratio). ``metrics``: [log_likelihood, aic, bic,] rho, sigma, lambda,
        n_selected, n_censored (nonselected). ``tests['model']``: Wald chi2 of the outcome
        slopes. ``tests['rho']`` (ML): LR chi2(1) of ``rho = 0`` against the probit plus
        the normal regression (``nonrobust``, ``opg``) or the Wald chi2(1). ``extra``
        (ML): ``rho``, ``sigma`` and ``lambda`` with delta-method standard errors and
        intervals; (two-step) ``rho_truncated``. The chart sample holds the fitted
        selection probability. ``result.nobs`` counts selected and nonselected rows. A
        correlation running to +-1 raises ``boundary_solution`` (in small samples the
        likelihood can increase all the way to ``|rho| = 1``); selected outcomes that the
        regressors fit exactly raise ``perfect_fit``; a selection indicator without
        variation raises ``constant_selection``.

    Stata / EViews
        ``heckman wage educ age, select(works = married children educ age)`` is
        ``oe.heckman(data=df, y='wage', x=['educ', 'age'], select='works',
        select_x=['married', 'children', 'educ', 'age'])``; add ``method='twostep'`` for
        ``heckman ..., twostep``. EViews: ``heckit`` (two-step) and ``heckit(ml)``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> n = 2000
        >>> df = pd.DataFrame({"educ": rng.normal(size=n), "kids": rng.normal(size=n)})
        >>> u = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=n)
        >>> df["works"] = (0.4 + 0.5 * df.educ - 0.8 * df.kids + u[:, 1] > 0) * 1.0
        >>> df["wage"] = np.where(df.works == 1, 1.0 + 0.7 * df.educ + u[:, 0], np.nan)
        >>> print(oe.heckman(data=df, y="wage", x=["educ"], select="works",
        ...                  select_x=["educ", "kids"]).summary())
    """
    if method == "twostep":
        resolved = covariance
    else:
        resolved = resolve_covariance(covariance, cluster, weight_type)
    spec = build_spec(
        "heckman", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolved, cluster=cluster, weights=weights, weight_type=weight_type,
        missing=missing, alpha=alpha,
        columns={"select": select, "select_x": column_list(select_x, "select_x")},
        options={"method": method})
    return fit(spec, data=data)
