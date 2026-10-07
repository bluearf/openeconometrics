"""ARIMA, seasonal ARIMA and ARMAX estimation (Stata's ``arima``).

The model is a regression with ARMA disturbances,

    y_t = x_t'b + mu_t,
    phi(L) Phi(L^s) (1 - L)^d (1 - L^s)^D mu_t = theta(L) Theta(L^s) e_t,   e_t ~ N(0, sigma^2),

estimated, as Stata does, on the differenced data: with w_t = (1-L)^d (1-L^s)^D y_t
and the regressors differenced the same way,

    w_t = b_0 + (differenced x_t)'b + u_t,    phi(L) Phi(L^s) u_t = theta(L) Theta(L^s) e_t,

so the constant b_0 is the mean of the differenced series (a drift when d > 0).
``method='ml'`` maximizes the exact Gaussian likelihood of the stationary ARMA
process (the Kalman-filter likelihood with the unconditional initial state,
evaluated in closed form by ``kernels.ArmaLikelihood``); ``method='css'``
maximizes the conditional likelihood with presample disturbances and presample
u at zero (Stata's ``condition`` option).

Optimization: sigma^2 is concentrated out and the remaining parameters are found
by BFGS (``engines.optimize.maximize_bfgs``) with the analytic gradient, started
from Hannan-Rissanen values with the Gauss-Newton information as the initial
curvature (samples up to 20,000 observations are also maximized from the
white-noise start and from the conditional estimate, and the highest maximum is
kept: mixed ARMA likelihoods can have several local maxima). The optimizer works
in standardized units, an exact reparameterization: the outcome and the regressors
are centered (when the model has a constant) and scaled to unit variance, and the
estimates, their covariance and the log likelihood are mapped back to the original
units, so neither the units nor the level of the data affect convergence or the
numerical Hessian. The raw AR and MA
coefficients are optimized, as in Stata; points with a non-stationary AR part
(exact likelihood) or a non-invertible MA part are infeasible, so the reported MA
polynomial is always the invertible representation.

Covariance (in the reported parameters b, phi, theta, Phi, Theta, sigma):
``opg`` (default, Stata's default for arima) inverts the outer product of the
per-observation scores of the prediction-error decomposition; ``nonrobust`` is
the inverse observed information, whose Hessian is the Ridders-extrapolated
numerical derivative of the analytic gradient (the sigma row and column are
analytic); ``robust`` is the Huber-White sandwich of the two with N/(N-1). At an
MA unit root the score-based estimators are degenerate and the observed
information is reported instead (see ``_covariance``).
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, replace
from typing import Any

import torch
from pandas.api.types import is_datetime64_any_dtype
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arima.diagnostics import (
    archlm_test, default_lags, jarque_bera_test, ljung_box_test, require_time_column,
)
from openecon.econometrics.arima.filters import difference, inverse_roots
from openecon.econometrics.arima.kernels import ArmaLikelihood, Decomposition
from openecon.econometrics.arima.maximize import maximize
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, column_list, information_criteria, kernel_call, make_spec,
    ml_covariance, wald_test,
)
from openecon.models import ModelSpec, ResultBundle

MAX_STATE = 200          # largest state dimension max(p + P s, q + Q s + 1)
_NEAR_UNIT = 0.99      # inverse-root modulus from which a boundary warning is issued
_UNIT_ROOT = 1.0 - 1e-3  # MA inverse-root modulus near enough to 1 for the OIM fallback
_PILE_UP = 1.0 - 1e-6    # MA inverse-root modulus that is a unit root (pile-up)


@dataclass(frozen=True)
class Orders:
    p: int
    d: int
    q: int
    seasonal_p: int
    seasonal_d: int
    seasonal_q: int
    period: int

    @property
    def lost(self) -> int:
        """Observations consumed by differencing."""
        return self.d + self.seasonal_d * self.period

    @property
    def p_full(self) -> int:
        return self.p + self.seasonal_p * self.period

    @property
    def q_full(self) -> int:
        return self.q + self.seasonal_q * self.period

    @property
    def state(self) -> int:
        return max(self.p_full, self.q_full + 1)

    @property
    def n_arma(self) -> int:
        return self.p + self.q + self.seasonal_p + self.seasonal_q

    def label(self) -> str:
        text = f"ARIMA({self.p},{self.d},{self.q})"
        if self.period:
            text += f"x({self.seasonal_p},{self.seasonal_d},{self.seasonal_q})[{self.period}]"
        return text


def _triple(value: Any, name: str, example: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 3 \
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in value):
        raise AnalysisError("invalid_order", f"{name} must be three non-negative integers, "
                            f"for example {example}.")
    return value


def read_orders(spec: ModelSpec) -> Orders:
    """Validate the order, seasonal and period options of an arima specification."""
    p, d, q = _triple(spec.options.get("order"), "order", "order=(1, 1, 1)")
    seasonal, period = spec.options.get("seasonal"), spec.options.get("period")
    if seasonal is None:
        if period is not None:
            raise AnalysisError("invalid_order", "period is only used together with "
                                "seasonal=(P, D, Q); pass both or neither.")
        sp = sd = sq = period = 0
    else:
        sp, sd, sq = _triple(seasonal, "seasonal", "seasonal=(0, 1, 1) with period=12")
        if period is None:
            raise AnalysisError("invalid_order", "seasonal=(P, D, Q) needs the seasonal period, "
                                "for example period=12 for monthly data.")
        if sp == sd == sq == 0:
            period = 0
    orders = Orders(p, d, q, sp, sd, sq, int(period))
    if d > 3 or sd > 2:
        raise AnalysisError("invalid_order", "At most three regular and two seasonal differences "
                            "are supported; difference the series yourself for more.")
    if orders.state > MAX_STATE:
        raise AnalysisError("model_too_large", f"The model has state dimension {orders.state} "
                            f"(max(p + P*s, q + Q*s + 1)); the limit is {MAX_STATE}. Reduce the "
                            "orders or the seasonal period.")
    return orders


@dataclass
class Prepared:
    """The time-ordered sample of one arima specification, in levels and differenced."""

    frame: ModelFrame
    orders: Orders
    last_period: int | None    # integer time value of the last observation, when available
    levels: Tensor             # outcome in levels, all ordered rows
    design: Design             # predictors in levels (no constant), all ordered rows
    y: Tensor                  # differenced outcome (estimation rows)
    x: Tensor                  # differenced predictors (estimation rows, no constant)


def _gap_error(detail: str) -> AnalysisError:
    return AnalysisError("time_gaps", f"{detail} ARIMA needs a regularly spaced series without "
                         "gaps: fill or interpolate the missing periods, or restrict the sample to "
                         "one uninterrupted stretch.")


def prepare(spec: ModelSpec, data: Any) -> Prepared:
    """Order the sample in time, refuse gaps, difference it and record the rows used."""
    if spec.time is not None and spec.time == spec.outcome:
        raise AnalysisError("invalid_spec", "The time column must differ from the outcome.")
    frame = ModelFrame(spec, data)
    orders = read_orders(spec)
    last_period = None
    if spec.time is not None:
        require_time_column(frame.original[spec.time], spec.time)
        frame.sort_panel()
        column = frame.original[spec.time]
        if frame.n < len(column):
            # Rows excluded for missing values must not lie inside the series.
            order = torch.as_tensor(column.sort_values(kind="stable", na_position="last")
                                    .index.to_numpy(copy=True))
            kept = torch.zeros(len(column), dtype=torch.bool)
            kept[torch.as_tensor(frame.positions)] = True
            where = kept[order].nonzero().flatten()
            if int(where[-1] - where[0]) + 1 != len(where):
                raise _gap_error("Observations with missing values lie inside the series.")
        index = frame.time_index()
        if not is_datetime64_any_dtype(column.dtype):
            if frame.n > 1 and bool((index[1:] - index[:-1] != 1).any()):
                raise _gap_error(f"Time column '{spec.time}' skips periods.")
            last_period = int(index[-1])
    else:
        positions = torch.as_tensor(frame.positions)
        if frame.n > 1 and bool((positions[1:] - positions[:-1] != 1).any()):
            raise _gap_error("Observations with missing values lie inside the series.")
    levels = frame.numeric(spec.outcome)
    design = frame.design(intercept=False)
    lost = orders.lost
    remaining = frame.n - lost
    if remaining < 2:
        raise AnalysisError("insufficient_observations", f"Differencing consumes {lost} "
                            f"observation(s) and leaves {max(remaining, 0)}; the model needs a "
                            "longer series.")
    y = difference(levels, orders.d, orders.seasonal_d, orders.period)
    x = difference(design.x, orders.d, orders.seasonal_d, orders.period)
    if lost:
        frame.restrict(torch.arange(frame.n) >= lost,
                       f"Differencing uses the first {lost} observation(s) as starting values; "
                       f"the estimation sample has {remaining} observations.")
    return Prepared(frame, orders, last_period, levels, design, y, x)


def _likelihood(prepared: Prepared, x: Tensor, method: str,
                y: Tensor | None = None) -> ArmaLikelihood:
    """The likelihood kernel of the differenced sample (``y`` overrides the outcome)."""
    orders = prepared.orders
    return kernel_call(ArmaLikelihood, prepared.y if y is None else y, x, p=orders.p, q=orders.q,
                       seasonal_p=orders.seasonal_p, seasonal_q=orders.seasonal_q,
                       period=orders.period, exact=method == "ml")


def state_record(prepared: Prepared, terms: list[str], x: Tensor, theta: Tensor,
                 decomposition: Decomposition) -> dict[str, Any]:
    """What ``forecast`` needs to continue the recursions after the last observation.

    ``process`` holds the last p* values of the ARMA disturbance u_t,
    ``disturbances`` the estimates of the last q* innovations e_t given the sample
    and ``disturbance_covariance`` their covariance divided by sigma^2 (None when
    negligible, the usual case); ``levels`` holds the last d + D*s values of the
    outcome net of the regression on the non-constant regressors.
    """
    orders, k, n = prepared.orders, len(terms), prepared.y.shape[0]
    beta = theta[:k]
    process = prepared.y - x @ beta if k else prepared.y
    level = prepared.levels
    slopes = [i for i, term in enumerate(terms) if term != "Intercept"]
    if slopes:
        columns = [prepared.design.terms.index(terms[i]) for i in slopes]
        level = level - prepared.design.x[:, columns] @ beta[slopes]
    covariance = decomposition.disturbance_covariance
    negligible = covariance.numel() == 0 or float(covariance.abs().max()) <= 1e-14
    return {
        "process": process[n - orders.p_full:].tolist() if orders.p_full else [],
        "disturbances": decomposition.disturbances.tolist(),
        "disturbance_covariance": None if negligible else covariance.tolist(),
        "levels": level[level.shape[0] - orders.lost:].tolist() if orders.lost else [],
        "last_period": prepared.last_period,
    }


def _roots(values: list[float], *, moving_average: bool) -> list[dict[str, float]]:
    roots = inverse_roots([-value for value in values] if moving_average else values)
    return [{"real": float(z.real), "imag": float(z.imag), "modulus": float(abs(z))}
            for z in roots.tolist()]


def _differenced_label(name: str, orders: Orders) -> str:
    """Stata's operator notation for the differenced outcome (D.y, DS12.y, D2.y)."""
    prefix = ("D" + (str(orders.d) if orders.d > 1 else "")) if orders.d else ""
    prefix += f"S{orders.period}" * orders.seasonal_d
    return f"{prefix}.{name}" if prefix else name


_CORRECTIONS = {
    "opg": "outer product of the per-observation scores of the prediction-error decomposition "
           "(Stata's default vce(opg) for arima)",
    "nonrobust": "observed information (OIM): numerical Hessian of the analytic gradient",
    "robust": "Huber-White sandwich of the observed information and the outer product of "
              "scores: N/(N-1)",
}


def _covariance(frame: ModelFrame, hessian: Tensor, scores: Tensor | None, n: int,
                largest_ma: float) -> tuple[Tensor, dict[str, Any]]:
    """Covariance of (b, phi, theta, Phi, Theta, sigma) by ``spec.covariance``.

    At an MA unit root (the pile-up of an over-differenced series) the likelihood is
    symmetric in theta -> 1/theta with sigma -> sigma |theta|, which makes the theta
    score of every observation the multiple -sigma/2 of its sigma score. The outer
    product of scores is then singular and the sandwich gives theta a zero variance,
    although the observed information is regular. In that one case ``opg`` and
    ``robust`` are replaced by the observed information, with a recorded warning and
    ``inference["requested_covariance"]``.
    """
    kind = frame.spec.covariance
    singular = AnalysisError(
        "singular_information", "The information matrix of the ARIMA model is singular, so "
        "standard errors are undefined. The model is probably over-parameterized (AR and MA "
        "factors that cancel, or more terms than the series supports): reduce the orders.")
    covariance = None
    if kind == "nonrobust" or largest_ma < _PILE_UP:
        try:
            covariance, info = ml_covariance(frame, hessian=hessian, scores=scores, nobs=n)
        except AnalysisError as exc:
            if exc.code != "singular_information":
                raise
            if kind == "nonrobust" or largest_ma < _UNIT_ROOT:
                raise singular from exc
        else:
            if largest_ma >= _UNIT_ROOT and not bool((covariance.diagonal() > 0.0).all()):
                covariance = None
    if covariance is None:
        try:
            covariance, info = ml_covariance(frame, hessian=hessian, nobs=n, kind="nonrobust")
        except AnalysisError as exc:
            raise singular from exc
        info["requested_covariance"] = kind
        frame.warn("The per-observation scores are linearly dependent at the MA unit root, so "
                   f"covariance='{kind}' is degenerate there; the standard errors are those of "
                   "the observed information (covariance='nonrobust').")
    info["correction"] = _CORRECTIONS[info["covariance"]]
    info["sigma"] = ("/sigma is tested one-sided against zero and its confidence interval is "
                     "truncated at zero, as in Stata")
    return covariance, info


def fit_arima(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of the ``arima`` estimator (see the module docstring and ``arima``)."""
    prepared = prepare(spec, data)
    frame, orders = prepared.frame, prepared.orders
    method = frame.option("method")
    tolerance = float(frame.option("tolerance"))
    max_iterations = int(frame.option("max_iterations"))
    if not tolerance > 0.0:
        raise AnalysisError("invalid_option", "tolerance must be positive.")
    n, y = frame.n, prepared.y
    constant = bool(spec.intercept) and bool(frame.option("constant"))
    x = prepared.x
    if constant:
        x = torch.cat([torch.ones((n, 1), dtype=torch.float64), x], dim=1)
    block = Design(x, (["Intercept"] if constant else []) + prepared.design.terms,
                   prepared.design.categories, constant)
    block = frame.drop_collinear(block)
    k = block.x.shape[1]
    kk = k + orders.n_arma
    if n < max(kk + 3, orders.state + 2):
        raise AnalysisError("insufficient_observations",
                            f"{orders.label()} with {kk + 1} parameters needs more than {n} "
                            "observations after differencing.")
    if not bool((y != y[0]).any()):
        raise AnalysisError("constant_outcome", "The (differenced) outcome is constant; there is "
                            "nothing to model. Check the differencing orders.")
    # Estimation runs in standardized units, an exact reparameterization: with a constant
    # in the model the outcome and the regressors are centered at their means; the outcome
    # is divided by its standard deviation c_y and regressor j by its root mean square c_j.
    # The ARMA parameters are unit free; b_j = b*_j c_y / c_j, sigma = sigma* c_y,
    # b_0 = c_y b*_0 + ybar - sum_j b_j xbar_j and ll = ll* - N ln c_y, so the reported
    # vector is A theta* + a and its covariance A V* A'. The optimizer's tolerance and the
    # numerical Hessian are thereby independent of the units and of the level of the data.
    centered = y - y.mean()
    scale_y = float(centered.square().mean().sqrt())
    x_std = block.x.clone()
    mean_y, mean_x = 0.0, torch.zeros(k, dtype=torch.float64)
    if block.intercept:
        mean_y = float(y.mean())
        mean_x[1:] = block.x[:, 1:].mean(dim=0)
        x_std[:, 1:] -= mean_x[1:]
    scale_x = x_std.square().mean(dim=0).sqrt()
    if not (math.isfinite(scale_y) and scale_y > 0.0 and bool(torch.isfinite(scale_x).all())
            and bool((scale_x > 0.0).all())):
        raise AnalysisError("numerical_failure", "The outcome or a regressor is too large or too "
                            "small to be standardized in float64; rescale the data.")
    y_std, x_std = (y - mean_y) / scale_y, (x_std / scale_x).contiguous()
    like = _likelihood(prepared, x_std, method, y_std)
    psi, hessian_c, optimizer = maximize(like, y_std, x_std, tolerance, max_iterations)
    pieces = like.evaluate(psi, derivatives=False)
    if pieces is None or not pieces.ss > 0.0:
        raise AnalysisError("numerical_failure", "The likelihood is not defined at the estimates.")
    sigma_std = math.sqrt(pieces.ss / n)
    log_likelihood = -0.5 * n * (math.log(2.0 * math.pi) + 1.0 + math.log(pieces.ss / n)) \
        - 0.5 * pieces.logdet - n * math.log(scale_y)
    if optimizer.get("starts"):
        # Log likelihoods reached from each start, in the units of the reported one.
        optimizer["starts"] = {name: None if value is None else value - n * math.log(scale_y)
                               for name, value in optimizer["starts"].items()}
    theta_std = torch.cat([psi, torch.tensor([sigma_std], dtype=torch.float64)])
    hessian = kernel_call(like.full_hessian, psi, hessian_c)
    decomposition = kernel_call(like.decompose, theta_std, scores=spec.covariance != "nonrobust")
    units = torch.diag(torch.cat([scale_y / scale_x, torch.ones(orders.n_arma, dtype=torch.float64),
                                  torch.tensor([scale_y], dtype=torch.float64)]))
    if block.intercept:
        units[0, 1:k] = -scale_y * mean_x[1:] / scale_x[1:]
    theta = units @ theta_std
    if block.intercept:
        theta[0] += mean_y
    sigma = float(theta[-1])
    phi, theta_ma, sphi, stheta = like.split(psi)
    roots = {"ar": _roots(phi, moving_average=False), "ma": _roots(theta_ma, moving_average=True),
             "seasonal_ar": _roots(sphi, moving_average=False),
             "seasonal_ma": _roots(stheta, moving_average=True),
             "convention": "inverse roots (eigenvalues of the companion matrix); stationary / "
                           "invertible when every modulus is below 1; seasonal roots are those "
                           "of the polynomial in L^s"}
    largest_ar = max((root["modulus"] for root in roots["ar"] + roots["seasonal_ar"]), default=0.0)
    largest_ma = max((root["modulus"] for root in roots["ma"] + roots["seasonal_ma"]), default=0.0)
    terms = [*block.terms]
    equations: list[str | None] = [_differenced_label(spec.outcome, orders)] * k
    terms += [f"ARMA:L{i}.ar" for i in range(1, orders.p + 1)]
    terms += [f"ARMA:L{i}.ma" for i in range(1, orders.q + 1)]
    equations += ["ARMA"] * (orders.p + orders.q)
    seasonal_name = f"ARMA{orders.period}"
    terms += [f"{seasonal_name}:L{i}.ar" for i in range(1, orders.seasonal_p + 1)]
    terms += [f"{seasonal_name}:L{i}.ma" for i in range(1, orders.seasonal_q + 1)]
    equations += [seasonal_name] * (orders.seasonal_p + orders.seasonal_q)
    terms.append("/sigma")
    equations.append(None)

    if largest_ar >= 1.0:
        frame.warn("The estimated AR polynomial is not stationary (an inverse root has modulus "
                   f"{largest_ar:.4f}); consider differencing the series.")
    elif largest_ar >= _NEAR_UNIT:
        frame.warn(f"The estimated AR polynomial is close to a unit root (largest inverse root "
                   f"{largest_ar:.4f}); consider differencing the series.")
    if largest_ma >= _NEAR_UNIT:
        frame.warn("The estimated MA polynomial is on or near the boundary of the invertibility "
                   f"region (largest inverse root {largest_ma:.4f}); the series may be "
                   "over-differenced and the standard errors are unreliable.")

    covariance, info = _covariance(frame, hessian, decomposition.scores, n, largest_ma)
    covariance = units @ covariance @ units.T
    decomposition = replace(decomposition, scores=None,
                            innovations=decomposition.innovations * scale_y,
                            disturbances=decomposition.disturbances * scale_y)

    residuals = decomposition.innovations
    lags = frame.option("ljung_lags")
    lags = default_lags(n) if lags is None else int(lags)
    if lags >= n:
        raise AnalysisError("invalid_lags", f"ljung_lags must be smaller than the number of "
                            f"observations after differencing ({n}).")
    tests: dict[str, Any] = {}
    tested = [i for i, term in enumerate(terms[:-1]) if term != "Intercept"]
    if tested:
        tests["model"] = wald_test(theta, covariance, tested,
                                   label="Wald chi2 test that all coefficients except the "
                                         "constant are zero")
    if bool((residuals != residuals[0]).any()):
        tests["ljung_box"] = ljung_box_test(
            residuals, lags, fitted_parameters=orders.n_arma,
            label=f"Ljung-Box Q({lags}) test of the residuals")
        tests["jarque_bera"] = jarque_bera_test(residuals,
                                                label="Jarque-Bera normality test of the residuals")
        if n >= 6:
            tests["arch_lm"] = archlm_test(residuals, 1, label="ARCH-LM(1) test of the residuals")

    metrics = {**information_criteria(log_likelihood, kk + 1, n), "sigma": sigma,
               "n_differenced": n, "iterations": optimizer["iterations"]}
    extra = {
        "model_order": {"p": orders.p, "d": orders.d, "q": orders.q, "P": orders.seasonal_p,
                        "D": orders.seasonal_d, "Q": orders.seasonal_q, "period": orders.period},
        "method": method, "ar": phi, "ma": theta_ma,
        "seasonal": {"ar": sphi, "ma": stheta, "period": orders.period},
        "roots": roots,
        "last_state": state_record(prepared, block.terms, block.x, theta, decomposition),
    }
    exact = method == "ml"
    bundle = build_result(
        frame, terms=terms, params=theta, covariance=covariance,
        title=f"{orders.label()} regression" + ("" if exact else " (conditional)"),
        equations=equations, use_t=False, metrics=metrics,
        fitted=prepared.levels[orders.lost:] - residuals,
        solver="bfgs_exact_arma_likelihood" if exact else "bfgs_conditional_arma_likelihood",
        solver_diagnostics={"converged": True, "iterations": optimizer["iterations"]},
        optimizer=optimizer, inference=info, tests=tests, extra=extra,
        categories=block.categories,
        provenance={
            "method": method,
            "standardization": "estimated on the outcome and regressors centered at their means "
                               "(when the model has a constant) and scaled to unit variance; "
                               "coefficients, covariance and log likelihood are reported in the "
                               "original units",
            "likelihood": "exact Gaussian likelihood, stationary initial state (closed form of "
                          "the Kalman-filter prediction-error decomposition)" if exact else
                          "conditional Gaussian likelihood, presample disturbances at zero",
            "differencing": {"d": orders.d, "D": orders.seasonal_d, "period": orders.period,
                             "observations_lost": orders.lost},
            "fitted_values": "one-step-ahead predictions of the outcome in levels; residuals are "
                             "the one-step prediction errors (innovations)",
            "time_order": f"sorted by {spec.time}" if spec.time else "input row order",
        },
    )
    # Stata's /sigma row: one-sided test, interval truncated at zero.
    sigma_row = bundle.coefficients[-1]
    sigma_row.p_value = 0.5 * sigma_row.p_value
    sigma_row.ci_low = max(0.0, sigma_row.ci_low)
    return bundle


def arima(*, data: Any, y: str, order: Any, x: Any = None, time: str | None = None,
          seasonal: Any = None, period: int | None = None, method: str = "ml",
          constant: bool = True, covariance: str | None = None, categorical: Any = None,
          ljung_lags: int | None = None, max_iterations: int = 200, tolerance: float = 1e-8,
          missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """ARIMA, seasonal ARIMA and ARMAX models (Stata's ``arima``).

    Model
    -----
    A regression whose disturbance follows a multiplicative seasonal ARIMA process:

        y_t = x_t'b + mu_t
        phi(L) Phi(L^s) (1-L)^d (1-L^s)^D mu_t = theta(L) Theta(L^s) e_t,    e_t ~ N(0, sigma^2)

    with ``phi(L) = 1 - phi_1 L - ... - phi_p L^p`` and ``theta(L) = 1 + theta_1 L +
    ... + theta_q L^q`` (Stata's signs; ``Phi``, ``Theta`` likewise in ``L^s``). As in
    Stata the outcome and the regressors are differenced first, and the constant is
    the mean of the differenced outcome (the drift when ``d > 0``):
    ``D.y_t = b_0 + D.x_t'b + u_t`` with ARMA ``u_t``. Without ``x`` this is a plain
    ARIMA(p, d, q) or SARIMA(p, d, q)(P, D, Q)_s model.

    Estimator
    ---------
    ``method="ml"`` (default): exact Gaussian maximum likelihood of the stationary
    ARMA process on the differenced data. The likelihood equals the Kalman-filter
    prediction-error decomposition with the unconditional initial state covariance
    (Stata's default); OpenEconometrics evaluates it in closed form (conditional residuals
    from polynomial filters plus an r-by-r correction for the presample state), so
    the cost is linear in the sample size with no loop over time.
    ``method="css"``: conditional likelihood with presample disturbances and
    presample ARMA errors set to zero, ``ll = -N/2 ln(2 pi sigma^2) - sum e_t^2 /
    (2 sigma^2)`` over all N observations (Stata's ``condition`` option).

    sigma^2 is concentrated out and the other parameters are found by BFGS with
    the analytic gradient from Hannan-Rissanen starting values (up to 20,000
    observations, mixed models are also started from white-noise errors and from
    the conditional estimate and the highest maximum is kept). The exact
    likelihood requires a stationary AR part; the MA part is kept in its
    invertible representation. Estimation runs on internally standardized data
    (centered when the model has a constant, unit variance), so the result does
    not depend on the units or the level of ``y`` and ``x``; everything is
    reported in the original units.

    Parameters
    ----------
    data : DataFrame, mapping of columns, or list of row records.
    y : outcome column.
    order : ``(p, d, q)``, three non-negative integers (tuple, list or integer array);
        ``d <= 3``.
    x : optional list of regressor columns (ARMAX / regression with ARMA errors).
    time : optional time column. Rows are sorted by it; integer periods must be
        consecutive (gaps are an error) and datetimes are taken as consecutive.
        Without it the row order is the time order.
    seasonal, period : ``(P, D, Q)`` and the seasonal period ``s >= 2`` (both or
        neither); ``D <= 2``.
    method : ``"ml"`` or ``"css"``.
    constant : include the constant ``b_0`` (default True).
    covariance : ``"opg"`` (default, as Stata: inverse outer product of the
        per-observation scores), ``"nonrobust"`` (inverse observed information,
        Stata's ``vce(oim)``) or ``"robust"`` (Huber-White sandwich with N/(N-1),
        Stata's ``vce(robust)``). At an MA unit root (an over-differenced series)
        the score-based estimators are degenerate; the observed information is
        reported then, with a warning and ``inference["requested_covariance"]``.
    categorical : regressors to expand into treatment-coded indicators.
    ljung_lags : lags of the Ljung-Box test of the residuals
        (default ``min(floor(N/2) - 2, 40)``, Stata's ``wntestq`` default).
    max_iterations, tolerance : BFGS iteration limit and gradient tolerance (on the
        standardized problem).
    missing : ``"raise"`` or ``"drop"``; dropped rows may only be at the start or
        end of the series (interior gaps raise ``time_gaps``).
    alpha : significance level of the confidence intervals.

    Result
    ------
    Coefficients: the regression terms (equation named after the differenced
    outcome, e.g. ``D.y``), ``ARMA:L1.ar``, ``ARMA:L1.ma``, ..., the seasonal terms
    ``ARMA12:L1.ar``, ``ARMA12:L1.ma`` (for ``period=12``) and ``/sigma``; z tests.
    ``metrics``: ``log_likelihood``, ``aic``, ``bic`` (k counts every parameter
    including sigma, N the observations after differencing), ``sigma``,
    ``n_differenced``, ``iterations``. ``tests``: ``model`` (Wald chi2 that all
    coefficients except the constant are zero), ``ljung_box`` (Q test of the
    residuals with ``df = lags`` and the ARMA-adjusted p-value alongside),
    ``jarque_bera`` and ``arch_lm``. ``predictions``: one-step-ahead predictions of
    the outcome in levels against the observed values. ``extra``: ``model_order``,
    ``ar``, ``ma``, ``seasonal``, ``roots`` (inverse roots and moduli) and
    ``last_state`` (what ``oe.forecast`` needs to continue after the sample).
    ``metrics["iterations"]`` counts the BFGS iterations of the screening run that
    supplied the final starting point plus those of the final run.

    Errors are ``AnalysisError``: ``invalid_order`` / ``invalid_spec`` for a bad
    specification, ``time_gaps``, ``repeated_time_values``, ``invalid_time`` for a
    series that is not regularly spaced, ``insufficient_observations``,
    ``constant_outcome``, ``perfect_fit``, ``nonconvergence`` and
    ``singular_information`` for data the model cannot be fitted to.

    Stata: ``arima y x1 x2, arima(1,1,1)``, ``arima y, arima(0,1,1) sarima(0,1,1,12)``,
    ``arima y, ar(1/2) ma(1) condition``. EViews: ``ls d(y) c ar(1) ma(1)``. SPSS:
    ``ARIMA``.

    Example
    -------
    >>> import openecon as oe
    >>> fit = oe.arima(data=df, y="wpi", order=(1, 1, 1), time="quarter")
    >>> print(fit.summary())
    >>> oe.forecast(fit, steps=8)
    """
    if order is None:
        raise AnalysisError("invalid_order", "order is required: pass order=(p, d, q), for "
                            "example order=(1, 0, 1).")
    if not isinstance(constant, bool):
        raise AnalysisError("invalid_spec", "constant must be True or False.")
    spec = build_spec(
        "arima", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), time=time, intercept=constant,
        covariance=covariance, missing=missing, alpha=alpha,
        options={"order": _integer_triple(order, "order"),
                 "seasonal": _integer_triple(seasonal, "seasonal"),
                 "period": _integer(period), "method": method, "ljung_lags": _integer(ljung_lags),
                 "max_iterations": _integer(max_iterations), "tolerance": tolerance},
    )
    from openecon.analysis import fit

    return fit(spec, data=data)


def _integer(value: Any) -> Any:
    """NumPy and other integral scalars as ``int``; anything else is left for the registry."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    try:
        return operator.index(value)
    except TypeError:
        return value


def _integer_triple(value: Any, name: str) -> list[int] | None:
    """``order`` / ``seasonal`` as a list of ints (tuples, lists and integer arrays pass)."""
    if value is None:
        return None
    problem = AnalysisError("invalid_order", f"{name} must be three non-negative integers, "
                            f"for example {name}=(1, 0, 1).")
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise problem
    items = []
    for item in value:
        if isinstance(item, bool):
            raise problem
        try:
            items.append(operator.index(item))
        except TypeError:
            raise problem from None
    return items


def build_spec(estimator: str, **fields: Any) -> ModelSpec:
    """``core.make_spec`` for the convenience function: every rejection is an AnalysisError.

    The registry reports an invalid option value, covariance name, ``alpha`` or
    ``missing`` policy as a validation ``ValueError`` (pydantic's ValidationError);
    here it becomes ``AnalysisError('invalid_spec', ...)`` with the same message.
    """
    try:
        return make_spec(estimator, **fields)
    except AnalysisError:
        raise
    except ValueError as exc:
        errors = exc.errors() if hasattr(exc, "errors") else []
        message = "; ".join(
            (f"{'.'.join(str(part) for part in error.get('loc', ()))}: " if error.get("loc")
             else "") + str(error.get("msg", "")).removeprefix("Value error, ")
            for error in errors) or str(exc)
        raise AnalysisError("invalid_spec", message) from exc
