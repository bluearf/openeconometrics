"""Native panel residual diagnostics with explicit, finite sample contracts.

Methods: Drukker (2003), Stata Journal 3(2), 168–177, and Baum's modified
Wald test (xttest3, variance-of-variance correction in v1.0.8). All numerical
work uses float64 Torch; grouping uses index_add, never observation loops.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Sequence

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, column_list, kernel_call, make_spec
from openecon.econometrics.postest.common import matched_frame, require_result
from openecon.engines.absorb import group_means
from openecon.engines.covariance import group_sums
from openecon.engines.distributions import chi2_sf, f_sf
from openecon.engines.linalg import least_squares
from openecon.models import ResultBundle

_DRUKKER = "https://doi.org/10.1177/1536867X0300300206"
_BAUM = "https://ideas.repec.org/c/boc/bocode/s414801.html"


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise AnalysisError("numerical_failure", f"The {name} is not finite; rescale the model inputs.")
    return value


def _scale(values: Tensor, *, what: str) -> tuple[Tensor, float]:
    if not bool(torch.isfinite(values).all()):
        raise AnalysisError("non_finite_values", f"The {what} contain non-finite values; rescale the inputs.")
    scale = float(values.abs().max())
    if scale <= 0:
        raise AnalysisError("degenerate_residuals", f"The {what} have no variation; the diagnostic is undefined.")
    return values / scale, scale


def _time_index(frame: ModelFrame, delta: Any) -> tuple[Tensor, int | str]:
    """Require an explicit grid for datetimes instead of ranking away gaps."""
    series = frame.sample[frame.spec.time]
    if is_bool_dtype(series.dtype):
        raise AnalysisError("invalid_time", "The time column must contain integer periods or datetimes.")
    if is_datetime64_any_dtype(series.dtype):
        if delta is None or isinstance(delta, (bool, int, float)):
            raise AnalysisError("invalid_time_delta", "Datetime time needs an explicit fixed time_delta, e.g. '1D'. For calendar months, use integer monthly periods.")
        try:
            step = pd.Timedelta(delta)
            if pd.isna(step) or step.value <= 0:
                raise ValueError
            index = torch.from_numpy(series.dt.as_unit("ns").astype("int64").to_numpy(copy=True))
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnalysisError("invalid_time_delta", "time_delta must be a positive fixed-duration interval, e.g. '1D'.") from exc
        duration = step.value
        label: int | str = str(step)
    else:
        if delta is None:
            delta = 1
        if isinstance(delta, bool) or not isinstance(delta, Integral) or not 1 <= delta <= 2**63 - 1:
            raise AnalysisError("invalid_time_delta", "Numeric time_delta must be a positive integer period increment.")
        # Validate finite integer-valued periods before an int64 conversion.
        if not pd.api.types.is_integer_dtype(series.dtype):
            values = frame.numeric(frame.spec.time)
            if bool((values != values.round()).any()) or bool((values.abs() > 2**53 - 1).any()):
                raise AnalysisError("invalid_time", "Numeric time must contain exact integer periods; use an integer dtype for large indices.")
        if pd.api.types.is_integer_dtype(series.dtype) and (int(series.min()) < -(2**63)
                                                            or int(series.max()) > 2**63 - 1):
            raise AnalysisError("invalid_time", "Integer time values must fit int64; recode the time origin.")
        index, duration, label = frame.time_index(), int(delta), int(delta)
    lower, upper = int(index.min()), int(index.max())
    if upper - lower > 2**63 - 1:
        raise AnalysisError("invalid_time", "The time span exceeds exact int64 spacing; recode the time origin.")
    origin = index - lower
    if bool((origin % duration != 0).any()):
        raise AnalysisError("invalid_time", "Time values must lie on the declared time_delta grid.")
    return origin // duration, label


def xtserial(*, data: Any, y: str, x: Sequence[str], panel: str, time: str,
             categorical: Sequence[str] | None = None, missing: str = "drop",
             time_delta: Any = None) -> dict[str, Any]:
    """Wooldridge–Drukker F test of no idiosyncratic first-order correlation.

    The first-difference regression and the regression of its residuals on
    lagged residuals both omit the intercept. The null coefficient is -0.5;
    the second regression uses panel-cluster CR1 covariance and G-1 df.
    Gaps and missing observations never become artificial adjacent periods.
    Numeric time defaults to increment 1; datetimes need e.g. time_delta='1D'.
    x must declare at least one predictor; time-invariant/collinear columns
    are omitted from the differenced regression and recorded in the result.
    """
    spec = make_spec("xtreg", outcome=y, predictors=column_list(x, "x"),
                     panel=panel, time=time, categorical=column_list(categorical, "categorical"),
                     covariance="robust", missing=missing, options={"model": "fd"})
    frame = ModelFrame(spec, data)
    frame.sort_panel()
    codes, groups = frame.codes(panel)
    periods, delta = _time_index(frame, time_delta)
    design = frame.design(intercept=False)
    levels = frame.numeric(y)
    adjacent = (codes[1:] == codes[:-1]) & (periods[1:] - periods[:-1] == 1)
    current = adjacent.nonzero().flatten() + 1
    if len(current) < 2:
        raise AnalysisError("insufficient_observations", "xtserial needs consecutive first differences and at least two panels with three consecutive complete observations.")
    dy, _ = _scale(levels[current] - levels[current - 1], what="first differences of the outcome")
    dx = design.x[current] - design.x[current - 1]
    fit = kernel_call(least_squares, dx, dy)
    rank = len(fit.kept)
    if len(current) <= rank:
        raise AnalysisError("insufficient_observations", "The first-difference regression needs more observations than its design rank.")
    if float(fit.ssr) <= 1e-28 * float(dy.square().sum()):
        raise AnalysisError("degenerate_residuals", "The first-difference regression fits exactly; xtserial residual inference is undefined.")
    residuals = torch.full((frame.n,), math.nan, dtype=torch.float64)
    residuals[current] = fit.resid
    keep = adjacent & torch.isfinite(residuals[1:]) & torch.isfinite(residuals[:-1])
    rows = keep.nonzero().flatten() + 1
    lagged, response = residuals[rows - 1], residuals[rows]
    participating = codes[rows]
    active, dense = torch.unique(participating, return_inverse=True)
    g, n = len(active), len(rows)
    if g < 2 or n < 2:
        raise AnalysisError("insufficient_panels", "xtserial needs at least two panels with three consecutive complete observations.")
    denominator = float(lagged.square().sum())
    if denominator <= 1e-28 * float(response.square().sum()) or denominator <= 0:
        raise AnalysisError("degenerate_residuals", "Lagged first-difference residuals have no usable variation.")
    correlation = float(lagged @ response) / denominator
    errors = response - correlation * lagged
    scores = kernel_call(group_sums, lagged * errors, dense, g)
    factor = g / (g - 1)  # CR1: G/(G-1) * (N-1)/(N-K); K=1, no intercept.
    variance = float(scores.square().sum()) / denominator**2 * factor
    if variance <= 0 or not math.isfinite(variance):
        raise AnalysisError("singular_covariance", "The panel-cluster variance is zero or non-finite; xtserial is undefined for this sample.")
    statistic = _finite((correlation + .5)**2 / variance, "Wooldridge F statistic")
    omitted = [design.terms[i] for i in fit.omitted]
    warnings = [*frame.warnings]
    if omitted:
        warnings.append("Omitted differenced design columns with no independent variation: " + ", ".join(omitted))
    if g < groups:
        warnings.append(f"{groups - g} panel(s) do not supply two consecutive first-difference residuals.")
    return {"method": "wooldridge_drukker", "label": "Wooldridge test for panel serial correlation",
            "null": "No first-order serial correlation in idiosyncratic errors",
            "statistic": statistic, "distribution": "F", "df": 1, "df2": g - 1,
            "p_value": kernel_call(f_sf, statistic, 1, g - 1),
            "correlation": _finite(correlation, "lagged-residual coefficient"),
            "std_error": math.sqrt(variance), "null_correlation": -.5,
            "nobs": n, "nobs_fd": len(current), "nobs_original": len(frame.original),
            "n_groups": g, "n_groups_input": groups, "design_rank": rank,
            "omitted_terms": omitted, "sample_positions": torch.as_tensor(frame.positions, dtype=torch.int64)[rows].tolist(),
            "time_delta": delta, "warnings": warnings,
            "provenance": {"method_source": _DRUKKER, "dtype": "float64", "covariance": "panel_cluster_CR1",
                           "stata_parity_validated": False}}


def xttest3(result: ResultBundle, *, data: Any) -> dict[str, Any]:
    """Modified Wald test of equal idiosyncratic FE variances across panels.

    Requires the original data of an unweighted xtreg(model='fe') result.
    The stored dataset/sample and fitted coefficients are verified. At least
    three observations and nondegenerate squared residuals are required in
    every panel; an undefined variance-of-variance is never silently skipped.
    """
    result = require_result(result, "fixed-effects result")
    spec = result.spec
    if spec.estimator != "xtreg" or spec.options.get("model", "fe") != "fe":
        raise AnalysisError("unsupported_model", "xttest3 requires an xtreg(model='fe') result.")
    if spec.weights is not None:
        raise AnalysisError("unsupported_weights", "xttest3 currently supports unweighted fixed-effects results only.")
    table, positions = matched_frame(result, data, "fixed-effects result")
    frame = ModelFrame(spec, table)
    frame.sort_panel()
    if frame.positions != positions or result.nobs != frame.n:
        raise AnalysisError("data_mismatch", "The saved FE estimation rows do not match the reconstructed sample.")
    codes, groups = frame.codes(spec.panel)
    counts = torch.bincount(codes, minlength=groups).to(torch.float64)
    if groups < 2:
        raise AnalysisError("insufficient_panels", "xttest3 needs at least two panels.")
    if bool((counts < 3).any()):
        raise AnalysisError("insufficient_panel_observations", "xttest3 needs at least three estimation observations in every panel; T_i≤2 makes the variance of squared FE residuals undefined.")
    design = frame.design(intercept=False)
    y = frame.numeric(spec.outcome)
    within_x = design.x - kernel_call(group_means, design.x, codes, groups)[codes]
    within_y = y - kernel_call(group_means, y, codes, groups)[codes]
    design, within_x = frame.drop_absorbed(design, within_x)
    fit = kernel_call(least_squares, within_x, within_y, drop_collinear=False)
    stored = {coefficient.term: coefficient.estimate for coefficient in result.coefficients}
    if len(stored) != len(result.coefficients) or set(stored) != {"Intercept", *design.terms}:
        raise AnalysisError("result_mismatch", "The saved FE coefficient terms do not match this specification and dataset.")
    beta = torch.tensor([stored[term] for term in design.terms], dtype=torch.float64)
    if not torch.allclose(beta, fit.beta, atol=1e-9, rtol=1e-8):
        raise AnalysisError("result_mismatch", "The saved FE coefficients do not match the fitted specification and dataset.")
    intercept = float(y.mean() - design.x.mean(dim=0) @ fit.beta)
    if not math.isclose(stored["Intercept"], intercept, abs_tol=1e-9, rel_tol=1e-8):
        raise AnalysisError("result_mismatch", "The saved FE constant does not match the fitted specification and dataset.")
    residuals, scale = _scale(fit.resid, what="fixed-effects residuals")
    squared = residuals.square()
    variances = kernel_call(group_sums, squared, codes, groups) / counts
    deviations = (squared - variances[codes]).square()
    variance_of_variances = kernel_call(group_sums, deviations, codes, groups) / (counts * (counts - 1))
    if bool((variance_of_variances <= 1e-28 * variances.square()).any()):
        raise AnalysisError("degenerate_group_variance", "Squared FE residuals have no usable variation in at least one panel; the modified Wald test is undefined.")
    pooled = float((residuals - residuals.mean()).square().mean())
    statistic = _finite(float(((variances - pooled).square() / variance_of_variances).sum()), "modified Wald statistic")
    return {"method": "modified_wald_groupwise", "label": "Modified Wald test for panel heteroskedasticity",
            "null": "Equal idiosyncratic residual variance in every panel",
            "statistic": statistic, "distribution": "chi2", "df": groups,
            "p_value": kernel_call(chi2_sf, statistic, groups), "nobs": frame.n,
            "n_groups": groups, "t_min": int(counts.min()), "t_max": int(counts.max()),
            "residual_scale": scale, "warnings": list(frame.warnings),
            "provenance": {"method_source": _BAUM, "variance_denominator": "T_i * (T_i - 1)",
                           "dtype": "float64", "result_id": result.id, "stata_parity_validated": False}}
