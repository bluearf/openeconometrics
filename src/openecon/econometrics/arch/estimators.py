"""ARCH-family regression (Stata's ``arch``; EViews' ARCH estimation).

``fit_arch`` is the registry entry point and ``arch`` the Stata-style convenience
function. The likelihood and its analytic scores live in ``kernels``, the optimizer and
starting values in ``maximize``, the derived quantities (persistence, diagnostics, the
end-of-sample state) in ``postfit`` and multi-step forecasts in ``forecast``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import torch
from pandas.api.types import is_datetime64_any_dtype
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arch import postfit
from openecon.econometrics.arch.kernels import ArchLikelihood
from openecon.econometrics.arch.layout import Layout, lag_list, read_layout
from openecon.econometrics.arch.maximize import maximize
from openecon.econometrics.arch.recursions import dense
from openecon.econometrics.arima.diagnostics import require_time_column
from openecon.econometrics.arima.filters import spectral_radius
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, column_list, information_criteria, make_spec,
    ml_covariance, wald_test,
)
from openecon.engines.distributions import normal_isf
from openecon.models import ModelSpec, ResultBundle

_CORRECTIONS = {
    "opg": "outer product of the per-observation analytic scores (Stata's default vce(opg) "
           "for arch)",
    "nonrobust": "observed information (OIM): numerical Hessian of the analytic gradient "
                 "(Stata's vce(oim))",
    "robust": "Huber-White quasi-maximum-likelihood sandwich of the observed information and "
              "the outer product of scores (Bollerslev-Wooldridge): N/(N-1)",
}
_MAX_SCAN_BYTES = 2 ** 30          # transition maps of the scan engine (EGARCH, ARCH-in-mean)
_PRESAMPLE = ("presample ARMA disturbances are zero (Stata's arma0(zero)); presample variances "
              "and squared innovations equal the mean squared residual of the mean equation at "
              "the current parameters (Stata's arch0(xb)), with presample news at their "
              "expected values")


@dataclass
class Prepared:
    """The time-ordered sample of one arch specification."""

    frame: ModelFrame
    layout: Layout
    y: Tensor
    design: Design            # mean equation (constant first)
    z: Tensor                 # variance regressors, [n, k_z]
    z_terms: list[str]
    z_categories: dict[str, Any]
    last_period: int | None   # integer time value of the last observation, when available


def _gap_error(detail: str) -> AnalysisError:
    return AnalysisError("time_gaps", f"{detail} ARCH models need a regularly spaced series "
                         "without gaps: fill or interpolate the missing periods, or restrict the "
                         "sample to one uninterrupted stretch.")


def prepare(spec: ModelSpec, data: Any, *, screen: bool = True) -> Prepared:
    """Order the sample in time, refuse gaps and build the mean and variance designs.

    ``screen=False`` keeps collinear columns (used when a fitted model is run over new
    data, where the fitted terms are selected by name).
    """
    variance_columns = spec.columns.get("variance_x") or []
    variance_columns = [variance_columns] if isinstance(variance_columns, str) \
        else list(variance_columns)
    if spec.time is not None and spec.time in {spec.outcome, *spec.predictors, *variance_columns}:
        raise AnalysisError("invalid_spec", "The time column must differ from the outcome and "
                            "the regressors.")
    if spec.outcome in variance_columns:
        raise AnalysisError("invalid_spec", "The outcome must not be a variance regressor.")
    frame = ModelFrame(spec, data)
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
    y = frame.numeric(spec.outcome)
    design = frame.design(intercept=spec.intercept)
    if screen:
        design = frame.drop_collinear(design)
    z = torch.empty((frame.n, 0), dtype=torch.float64)
    z_terms: list[str] = []
    z_categories: dict[str, Any] = {}
    if variance_columns:
        block = frame.design(variance_columns, intercept=True, prefix="HET:")
        if screen:
            block = frame.drop_collinear(block)
        z, z_terms, z_categories = block.x[:, 1:].contiguous(), block.terms[1:], block.categories
    layout = read_layout(spec, design.x.shape[1], z.shape[1])
    return Prepared(frame, layout, y, design, z, z_terms, z_categories, last_period)


def fit_arch(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of the ``arch`` estimator (see ``arch`` for the full description)."""
    prepared = prepare(spec, data)
    frame, layout = prepared.frame, prepared.layout
    ix, n, y = layout.index, frame.n, prepared.y
    tolerance = float(frame.option("tolerance"))
    max_iterations = int(frame.option("max_iterations"))
    if not tolerance > 0.0:
        raise AnalysisError("invalid_option", "tolerance must be positive.")
    needed = layout.k_free + layout.max_lag + 5
    if n < needed:
        raise AnalysisError("insufficient_observations",
                            f"{layout.label()} with {layout.k_free} parameters needs at least "
                            f"{needed} observations; the sample has {n}.")
    if not bool((y != y[0]).any()):
        raise AnalysisError("constant_outcome", "The outcome is constant; there is no variance "
                            "to model.")

    # Estimation runs on regressors centred at their means (both equations): with a
    # constant in the equation this is an exact linear reparameterization, mapped back
    # below, that keeps the constant and the slopes of a regressor with a large offset
    # (a calendar year, a price level) from being numerically collinear.
    x, z = prepared.design.x, prepared.z
    back = torch.eye(ix.k, dtype=torch.float64)       # reported = back @ centred parameters
    if prepared.design.intercept and layout.k_x > 1:
        centre = x[:, 1:].mean(dim=0)
        x[:, 1:] -= centre                             # the design is owned by this fit
        back[ix.x[0], ix.x[1:]] = -centre
    if layout.k_z:
        centre = z.mean(dim=0)
        z -= centre
        back[ix.c, ix.z] = -centre
    like = ArchLikelihood(y, x, z, layout)
    if not like.linear:
        # EGARCH and ARCH-in-mean: the derivative recursion carries one transition map
        # per observation, of the square of the longest lags.
        size = sum(like.scan_lags())
        if 8 * n * size * size > _MAX_SCAN_BYTES:
            raise AnalysisError(
                "model_too_large",
                f"{layout.label()} with lags up to {layout.max_lag} on {n} observations needs "
                f"{8 * n * size * size / 2 ** 30:.1f} GiB for the derivative recursions of an "
                "EGARCH or ARCH-in-mean model (the limit is "
                f"{_MAX_SCAN_BYTES / 2 ** 30:.0f} GiB). Use shorter lags or a shorter sample; "
                "GARCH, GJR and power-ARCH models without ARCH-in-mean have no such limit.")
    fit = maximize(like, tolerance, max_iterations)
    final = like.evaluate(fit.theta, scores=True, signs=fit.signs, flat=fit.flat)
    if final is None:
        raise AnalysisError("numerical_failure", "The likelihood is not defined at the "
                            "estimates.")
    try:
        covariance, info = ml_covariance(frame, hessian=fit.hessian,
                                         scores=final.scores @ fit.transform, nobs=n)
    except AnalysisError as exc:
        if exc.code != "singular_information":
            raise
        raise AnalysisError(
            "singular_information", "The information matrix of the ARCH model is singular, so "
            "standard errors are undefined. A parameter is probably not identified (an ARCH "
            "coefficient at zero leaves the GARCH coefficients undetermined, or AR and MA "
            "factors cancel): simplify the model.") from exc
    transform = back @ fit.transform
    theta = back @ fit.theta
    covariance = transform @ covariance @ transform.T
    info["correction"] = _CORRECTIONS[info["covariance"]]
    if layout.restricted:
        info["restriction"] = ("IGARCH: the last GARCH coefficient is one minus the other ARCH "
                               "and GARCH coefficients; its standard error follows from them")

    terms, equations = layout.terms(spec.outcome, prepared.design.terms, prepared.z_terms)
    persist = postfit.persistence(layout, theta)
    unconditional = postfit.unconditional_variance(layout, theta, persist)
    if not layout.restricted and persist is not None and persist >= 1.0:
        frame.warn(f"The estimated variance equation is not covariance stationary (persistence "
                   f"{persist:.4f} >= 1): shocks to the variance do not die out and the "
                   "unconditional variance does not exist. Consider model='igarch'.")
    if layout.kind != "egarch":
        # Sufficient conditions for a positive variance: every coefficient non-negative
        # (for GJR also a_i + g_i, the coefficient of a negative shock).
        negative = [terms[i] for i in [*ix.a, *ix.b] if float(theta[i]) < 0.0]
        negative += [terms[j] for i, j in zip(ix.a, ix.g, strict=False)
                     if float(theta[i] + theta[j]) < 0.0]
        if not layout.k_z and float(theta[ix.c]) <= 0.0:
            negative.append(terms[ix.c])
        if negative:
            frame.warn("The variance equation has negative coefficients "
                       f"({', '.join(negative)}): no positivity constraints are imposed (as in "
                       "Stata). The fitted conditional variance is positive at every observation "
                       "of the sample, but it is not guaranteed to stay positive in forecasts or "
                       "on other data; consider fewer ARCH/GARCH lags or model='egarch'.")
    if fit.flat is not None:
        frame.warn("The maximum of the EGARCH likelihood lies on a kink (a standardized "
                   "residual is exactly zero), where the curvature of the GED density is "
                   "unbounded. The density kernel of the "
                   f"{int(fit.flat.sum())} observation(s) on the kink is left out of the "
                   "scores and of the Hessian, so that the standard errors stay defined.")
    if layout.ar_lags and spectral_radius(dense(theta[ix.ar].tolist(), layout.ar_lags)) >= 1.0:
        frame.warn("The estimated AR polynomial of the disturbance is not stationary; consider "
                   "differencing the outcome.")
    if layout.ma_lags and spectral_radius(dense(theta[ix.ma].tolist(), layout.ma_lags,
                                                -1.0)) >= 1.0:
        frame.warn("The estimated MA polynomial of the disturbance is not invertible; the "
                   "residuals of the conditional recursion are unreliable.")

    tests: dict[str, Any] = {}
    tested = [i for i in ix.x if terms[i] != "Intercept"] + ix.m + ix.ar + ix.ma
    if tested:
        tests["model"] = wald_test(theta, covariance, tested,
                                   label="Wald chi2 test that all mean-equation coefficients "
                                         "except the constant are zero")
    tests.update(postfit.diagnostics(final, layout, frame.option("test_lags")))

    metrics = {**information_criteria(final.value, layout.k_free, n), "persistence": persist,
               "unconditional_variance": unconditional, "iterations": fit.record["iterations"]}
    extra = {
        "model": layout.describe(),
        "persistence": persist,
        "unconditional_variance": unconditional,
        "distribution": postfit.distribution_record(layout, theta, covariance,
                                                    normal_isf(0.5 * spec.alpha)),
        "state": postfit.state_record(layout, final, prepared.last_period),
        "conditional_variance_tail": postfit.variance_tail(final),
    }
    if layout.kind == "egarch" and persist is not None and abs(persist) < 1.0 and not layout.k_z:
        extra["unconditional_log_variance"] = float(theta[ix.c]) / (1.0 - persist)
    return build_result(
        frame, terms=terms, params=theta, covariance=covariance, title=layout.title(),
        equations=equations, use_t=False, metrics=metrics, fitted=y - final.residual,
        solver="bfgs_arch_likelihood",
        solver_diagnostics={"converged": True, "iterations": fit.record["iterations"]},
        optimizer=fit.record, inference=info, tests=tests, extra=extra,
        categories={**prepared.design.categories, **prepared.z_categories},
        provenance={
            "likelihood": f"conditional {layout.dist} likelihood of the innovations "
                          "e_t = sqrt(h_t) z_t",
            "presample": _PRESAMPLE,
            "fitted_values": "one-step-ahead conditional mean (regression, ARCH-in-mean and "
                             "ARMA parts); residuals are the innovations e_t",
            "time_order": f"sorted by {spec.time}" if spec.time else "input row order",
        },
    )


def _integer(value: Any) -> Any:
    """NumPy and other integral scalars as ``int``; anything else is left for the registry."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    try:
        return operator.index(value)
    except TypeError:
        return value


def arch(*, data: Any, y: str, x: Any = None, time: str | None = None, arch: Any = 1,
         garch: Any = None, model: str = "garch", dist: str = "normal",
         archm: str | None = None, ar: Any = None, ma: Any = None, constant: bool = True,
         variance_x: Any = None, covariance: str | None = None, categorical: Any = None,
         test_lags: int | None = None, max_iterations: int = 500, tolerance: float = 1e-8,
         missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """ARCH-family regression: ARCH, GARCH, GJR/TARCH, EGARCH, PARCH, IGARCH (Stata's ``arch``).

    Model
    -----
    A regression whose innovation has a time-varying conditional variance h_t:

        y_t = x_t'b + psi g(h_t) + u_t
        u_t = sum_j rho_j u_(t-j) + sum_k theta_k e_(t-k) + e_t,      e_t = sqrt(h_t) z_t

    with z_t independent, mean 0, variance 1 (``dist``). ``archm`` adds the
    ARCH-in-mean term with g(h) = h (``"variance"``), sqrt(h) (``"sd"``) or ln h
    (``"log"``); ``ar`` and ``ma`` give the disturbance an ARMA structure with Stata's
    signs. The variance equation is chosen by ``model`` (lags i in ``arch``, j in
    ``garch``):

    ``"arch"``, ``"garch"``
        h_t = omega + sum a_i e_(t-i)^2 + sum b_j h_(t-j)             (Engle; Bollerslev)
    ``"gjr"`` / ``"tarch"``
        h_t = omega + sum a_i e_(t-i)^2 + sum g_i e_(t-i)^2 1(e_(t-i) < 0) + sum b_j h_(t-j)
        (Glosten-Jagannathan-Runkle; EViews' TARCH). Stata's ``tarch()`` term multiplies
        1(e > 0) instead: its ``arch`` coefficient is ``a + g`` and its ``tarch``
        coefficient ``-g`` here.
    ``"egarch"``
        ln h_t = omega + sum a_i z_(t-i) + sum g_i (|z_(t-i)| - sqrt(2/pi)) + sum b_j ln h_(t-j)
        with z = e / sqrt(h) (Nelson; Stata's earch, earch_a and egarch terms).
    ``"parch"``
        h_t^(phi/2) = omega + sum a_i |e_(t-i)|^phi + sum b_j h_(t-j)^(phi/2), with the power
        phi estimated (Ding-Granger-Engle without asymmetry; Stata's parch and pgarch).
    ``"igarch"``
        the GARCH equation with sum a_i + sum b_j = 1 imposed (the last GARCH coefficient
        is not free); the constant omega is kept.

    ``variance_x`` adds regressors z_t to the variance equation as Stata's ``het()`` does:
    multiplicative heteroskedasticity ``exp(l_0 + z_t'l)`` replaces omega (``l_0 + z_t'l``
    in the EGARCH equation for ln h).

    Estimator
    ---------
    Full maximum likelihood on the raw parameters (no positivity or stationarity
    constraints are imposed, as in Stata; a parameter point with a non-positive variance
    is infeasible; negative variance coefficients at the estimates are reported with a
    warning). BFGS with analytic gradients from the recursive derivatives of h_t; the
    Hessian for the convergence test and the observed information is the numerical
    derivative of the analytic gradient. Regressors of both equations are centred at
    their means during estimation (an exact reparameterization of the constants, mapped
    back afterwards). Presample values follow Stata's defaults: ARMA disturbances start
    at zero (``arma0(zero)``) and presample variances and squared innovations equal the
    mean squared residual of the mean equation at the current parameters (``arch0(xb)``).
    GARCH, GJR, IGARCH and power-ARCH models without ARCH-in-mean are evaluated without
    any loop over time; EGARCH and ARCH-in-mean models loop over time for the state path
    only.

    The EGARCH likelihood has a kink wherever a standardized residual is zero. Hessians
    are taken on the smooth piece through the estimates, and a maximum that lies on a
    kink is found by an active-set Newton method (``provenance["optimizer"]`` then lists
    ``kink_observations``; with GED errors the density kernel of those observations,
    whose curvature is unbounded at zero, is left out of the scores and the Hessian, and
    a warning says so). Because nothing is constrained, the likelihood can lack a
    regular maximum: a negative ARCH/GARCH coefficient can drive one conditional
    variance to zero (unbounded likelihood), and a power-ARCH power of 1 or less puts a
    cusp at every zero residual. Both are reported as ``nonconvergence`` with that
    explanation.

    Parameters
    ----------
    data : DataFrame, mapping of columns, or list of row records.
    y : outcome column.
    x : optional list of regressors of the mean equation.
    time : optional time column (integer periods or datetimes). Rows are sorted by it and
        integer periods must be consecutive. Without it the row order is the time order.
    arch : ARCH order q (lags 1..q) or a list of lags; default 1. For ``"egarch"`` each lag
        has an ``earch`` (signed) and an ``earch_a`` (absolute) term, for ``"gjr"`` an
        ``arch`` and a ``tarch`` term.
    garch : GARCH order p or a list of lags; default 1 (none for ``model="arch"``).
    model : ``"arch"``, ``"garch"``, ``"gjr"`` (alias ``"tarch"``), ``"egarch"``, ``"parch"``
        or ``"igarch"``.
    dist : ``"normal"``, ``"t"`` (Student t; reports ``/lndfm2 = ln(df - 2)``) or ``"ged"``
        (generalized error distribution; reports ``/lnshape = ln(shape)``).
    archm : ``None``, ``"variance"``, ``"sd"`` or ``"log"``.
    ar, ma : AR and MA orders or lists of lags of the disturbance, e.g. ``ar=1, ma=[1, 4]``.
    constant : include the constant of the mean equation (default True).
    variance_x : optional list of variance regressors.
    covariance : ``"opg"`` (default, as in Stata: inverse outer product of the
        per-observation scores, vce(opg)), ``"nonrobust"`` (inverse observed information,
        vce(oim)) or ``"robust"`` (quasi-ML sandwich with N/(N-1), vce(robust); valid when
        the innovation distribution is misspecified). All coefficient tests are z tests.
    categorical : regressors (of either equation) to expand into treatment-coded indicators.
    test_lags : lags of the diagnostics of the standardized residuals (default 5 for the
        ARCH-LM test and ``min(floor(N/2) - 2, 40)`` for the Ljung-Box tests).
    max_iterations, tolerance : BFGS iteration limit and gradient tolerance.
    missing : ``"raise"`` or ``"drop"``; dropped rows may only be at the start or end of the
        series (interior gaps raise ``time_gaps``).
    alpha : significance level of the confidence intervals.

    Result
    ------
    Coefficients, z tests: the mean equation (equation named after the outcome),
    ``ARCHM:sigma2`` | ``ARCHM:sigma`` | ``ARCHM:lnsigma2``, ``ARMA:L1.ar``, ``ARMA:L1.ma``,
    ``HET:<z>`` and ``HET:Intercept`` (with ``variance_x``), ``ARCH:L1.arch``,
    ``ARCH:L1.tarch``, ``ARCH:L1.garch`` (``earch``, ``earch_a``, ``egarch`` for EGARCH;
    ``parch``, ``pgarch`` for power ARCH), ``ARCH:Intercept``, ``POWER:power``, and
    ``/lndfm2`` or ``/lnshape``.
    ``metrics``: ``log_likelihood``, ``aic``, ``bic``, ``persistence`` (sum a + sum b; plus
    half the threshold terms for GJR; sum b for EGARCH; E|z|^phi sum a + sum b for power
    ARCH; 1 for IGARCH), ``unconditional_variance`` (omega / (1 - persistence) for
    stationary GARCH and GJR models) and ``iterations``.
    ``tests``: ``model`` (Wald chi2 of the mean-equation coefficients except the constant),
    ``ljung_box`` and ``ljung_box_squared`` (standardized residuals and their squares),
    ``arch_lm_residuals`` (Engle's LM test on the standardized residuals) and
    ``jarque_bera``.
    ``predictions``: the one-step conditional mean against the outcome.
    ``extra``: ``model``, ``persistence``, ``unconditional_variance``, ``distribution`` (df
    or shape with its interval), ``state`` (last innovations, disturbances and variances,
    used by ``oe.forecast``) and ``conditional_variance_tail`` (the last 400 h_t).

    Errors are ``AnalysisError``: ``invalid_spec`` / ``invalid_lags`` for a bad
    specification, ``time_gaps``, ``repeated_time_values``, ``invalid_time`` for a series
    that is not regularly spaced, ``insufficient_observations``, ``constant_outcome``,
    ``perfect_fit``, ``numerical_failure`` (an outcome whose variance is outside 1e-120 to
    1e120: rescale it), ``nonconvergence`` and ``singular_information``.

    Stata: ``arch y x, arch(1) garch(1)``; ``arch y, earch(1) egarch(1)``;
    ``arch y, arch(1) tarch(1) garch(1)``; ``arch y, parch(1) pgarch(1)``;
    ``arch y x, arch(1) garch(1) archm ar(1) distribution(t) vce(robust)``.
    EViews: ``arch(1,1) y c x``, ``arch(1,1,egarch) y c``, ``arch(1,1,thrsh=1) y c``.

    Example
    -------
    >>> import openecon as oe
    >>> fit = oe.arch(data=df, y="ret", arch=1, garch=1, dist="t", time="day")
    >>> print(fit.summary())
    >>> oe.forecast(fit, steps=10)          # mean and conditional-variance forecasts
    """
    if not isinstance(constant, bool):
        raise AnalysisError("invalid_spec", "constant must be True or False.")
    arch_lags = lag_list(arch, "arch")
    spec = make_spec(
        "arch", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), time=time, intercept=constant,
        covariance=covariance, missing=missing, alpha=alpha,
        columns={"variance_x": column_list(variance_x, "variance_x")},
        options={"model": model, "dist": dist, "arch": [1] if arch_lags is None else arch_lags,
                 "garch": lag_list(garch, "garch"), "archm": archm,
                 "ar": lag_list(ar, "ar") or None, "ma": lag_list(ma, "ma") or None,
                 "test_lags": _integer(test_lags), "max_iterations": _integer(max_iterations),
                 "tolerance": tolerance},
    )
    from openecon.analysis import fit

    return fit(spec, data=data)
