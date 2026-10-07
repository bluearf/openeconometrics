"""Conventions shared by areg, reghdfe and cnsreg: weights, sums of squares, tests.

Everything here follows the Methods and formulas of Stata's regress:

* **Weights.** aweights and pweights are rescaled to sum to the number of
  observations N before entering the weighted least-squares problem; the
  estimator counts N = number of rows. fweights replicate observations: they
  enter unscaled and N = sum of weights, so results equal those of the
  duplicated-row data set. pweights are sampling weights and require a
  robust or cluster covariance.
* **Covariance names.** ``robust`` (Stata's ``vce(robust)``) is HC1 for every
  linear estimator. The convenience functions write ``HC1`` into the spec;
  a spec built directly may say ``robust`` and is mapped by
  :func:`covariance_kind` before the covariance is computed.
* **Sums of squares.** With a constant in the model the total sum of squares
  is taken about the weighted mean of y; without one it is uncentered
  (Stata's noconstant R-squared). The residual sum of squares is weighted.
* **Model test.** Classical F = (MSS / df_model) / (RSS / df_resid) for the
  conventional covariance; the Wald F built on the reported covariance
  otherwise (df2 = residual, or G-1 with clusters), which is the statistic
  Stata prints with vce(robust) and vce(cluster). When that covariance has
  lower rank than the number of tested coefficients (fewer clusters than
  slopes), the test is rank-reduced, flagged and warned about; Stata prints
  a missing F in that situation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, kernel_call, wald_test
from openecon.engines.absorb import absorbed_degrees_of_freedom
from openecon.engines.distributions import f_sf
from openecon.engines.linalg import LeastSquares, least_squares


def resolve_covariance(covariance: str | None, cluster: Any, weight_type: str | None) -> str | None:
    """Convenience-function covariance: Stata's vce(robust) alias and the pweight default."""
    if covariance == "robust":
        return "HC1"
    if covariance is None and weight_type == "pweight" and not cluster:
        return "HC1"
    return covariance


def covariance_kind(frame: ModelFrame) -> str:
    """The covariance to compute: ``robust`` in a directly built spec means HC1."""
    return "HC1" if frame.spec.covariance == "robust" else frame.spec.covariance


def check_pweights(frame: ModelFrame) -> None:
    if frame.spec.weight_type == "pweight" and frame.spec.covariance == "nonrobust":
        raise AnalysisError(
            "unsupported_covariance",
            "pweights are sampling weights and need a robust covariance: choose covariance='HC1' "
            "(Stata's vce(robust)) or give a cluster column.")


def check_observations(frame: ModelFrame, columns: int, *, what: str = "design columns") -> None:
    """Reject a sample with no more distinct observations than parameters.

    The rank of a design is bounded by the number of distinct rows, so with
    ``n <= columns`` the collinearity screen would omit columns and blame
    collinearity for what is really a shortage of observations (Stata's
    "insufficient observations"). Frequency weights do not add rank, so the
    row count is compared, not the weighted N.
    """
    if frame.n <= columns:
        raise AnalysisError(
            "insufficient_observations",
            f"The model has {columns} {what} but only {frame.n} observation(s); it needs more "
            "observations than parameters.")


def estimation_weights(frame: ModelFrame) -> tuple[Tensor | None, int]:
    """Weights as the least-squares kernel receives them and N as Stata counts it.

    aweights/pweights are rescaled to sum to the number of rows (N = rows);
    fweights enter unscaled with N = sum of weights.
    """
    weights = frame.weights()
    if weights is None:
        return None, frame.n
    if frame.spec.weight_type == "fweight":
        return weights, int(round(float(weights.sum())))
    return weights * (frame.n / float(weights.sum())), frame.n


def weighted_mean(values: Tensor, weights: Tensor | None) -> Tensor:
    """Column means of ``values`` ([n] or [n, m]) in the weighted inner product."""
    if weights is None:
        return values.mean(dim=0)
    if values.ndim == 1:
        return (weights * values).sum() / weights.sum()
    return (weights[:, None] * values).sum(dim=0) / weights.sum()


def sum_of_squares(values: Tensor, weights: Tensor | None, *, centered: bool) -> float:
    """``sum_i w_i (v_i - vbar)^2`` (centered) or ``sum_i w_i v_i^2``."""
    if centered:
        values = values - weighted_mean(values, weights)
    squares = values.square()
    return float((squares * weights).sum() if weights is not None else squares.sum())


def solve_least_squares(x: Tensor, y: Tensor, weights: Tensor | None) -> LeastSquares:
    """QR least squares on a design that already passed the collinearity screen.

    ``tol=0`` disables the kernel's own (uncentered) screen so that the
    centered decision of ``ModelFrame.drop_collinear`` stands; an exactly
    singular design is still rejected by the kernel's condition-number test.
    """
    fit = kernel_call(least_squares, x, y, weights, drop_collinear=True, tol=0.0)
    if fit.omitted:
        raise AnalysisError("singular_design", "The design is exactly rank deficient after the "
                            "collinearity screen; rescale or drop the affected regressors.")
    return fit


# The residual sum of squares of an exact fit is pure rounding: residuals carry an absolute
# error of about eps times the magnitude of y, so RSS floors at roughly eps^2 (~5e-32) times
# the UNCENTERED sum w y^2 whatever the level of y. The guard sits ~2e3 eps^2 above that
# floor: an outcome with a large offset no longer triggers it unless its residual variation
# is below the float64 resolution of its own level (sigma/level < 1e-14), which float64 data
# cannot represent anyway. (A purely centered scale cannot serve as the criterion: a true
# exact fit of y = 1e10 + 2x leaves RSS/TSS_centered ~ 1e-11, far above any usable cutoff.)
_EXACT_FIT = 1e-28


def check_fit(rss: float, tss: float | None, df_resid: float, scale: float, *,
              absorbed: bool = False, resolution: float = 0.0) -> None:
    """Reject the degenerate fits whose standard errors would be undefined.

    ``scale`` is the uncentered ``sum w y^2`` against which an exact fit is judged.
    ``resolution`` is an additional residual sum of squares below which the fit
    cannot be told from an exact one: iterative demeaning leaves an error of
    its convergence tolerance in every column, so the residuals of an exact
    fit stop at that level instead of at rounding (see ``reghdfe``).
    """
    if df_resid <= 0:
        raise AnalysisError("insufficient_observations", "The model has no residual degrees of "
                            "freedom: it needs more observations than parameters (including "
                            "absorbed fixed effects).")
    if tss is not None and tss <= 0:
        raise AnalysisError("constant_outcome", "The outcome has no variation in the estimation "
                            "sample; nothing can be explained.")
    if rss <= max(_EXACT_FIT * scale, resolution):
        what = "regressors and absorbed fixed effects" if absorbed else "regressors"
        level = ("rounding noise" if rss <= _EXACT_FIT * scale
                 else "below the resolution of the demeaning tolerance")
        raise AnalysisError("perfect_fit", f"The {what} fit the outcome exactly (the residual "
                            f"sum of squares is {level}), so the error variance and every "
                            "standard error are zero. Remove the regressor that reproduces the "
                            "outcome.")


def classical_f_test(mss: float, df_model: int, rss: float, df_resid: float, *,
                     label: str) -> dict[str, Any]:
    """``F(df_model, df_resid) = (MSS/df_model) / (RSS/df_resid)``; None without restrictions."""
    if df_model <= 0:
        return {"statistic": None, "df": 0, "df2": df_resid, "p_value": None,
                "distribution": "F", "label": label}
    statistic = (max(mss, 0.0) / df_model) / (rss / df_resid)
    return {"statistic": statistic, "df": df_model, "df2": df_resid,
            "p_value": kernel_call(f_sf, statistic, df_model, df_resid),
            "distribution": "F", "label": label}


def robust_wald_test(frame: ModelFrame, params: Tensor, covariance: Tensor, indices: list[int], *,
                     df_inference: float, label: str, expected_rank: int | None = None,
                     ) -> dict[str, Any]:
    """Wald F test of the selected coefficients, flagged when its covariance block is rank
    deficient below ``expected_rank`` (default: the number of tested coefficients).

    A cluster covariance has rank at most G - 1, so with fewer clusters than
    tested coefficients the generalized Wald statistic uses fewer restrictions.
    Stata prints a missing model F in that case; here the reduced test is kept,
    marked ``rank_deficient`` and recorded as a warning.
    """
    test = wald_test(params, covariance, indices, df_resid=df_inference, label=label)
    expected = len(indices) if expected_rank is None else expected_rank
    if test["statistic"] is not None and test["df"] < expected:
        test.update({"rank_deficient": True, "restrictions": expected,
                     "label": f"{label} (rank-reduced to {test['df']} of {expected} restrictions)"})
        frame.warn(f"The covariance of the {expected} coefficients in the model test has rank "
                   f"{test['df']} (typically fewer clusters than coefficients); the F test uses "
                   f"{test['df']} restrictions, where Stata would report a missing F.")
    return test


def model_test(frame: ModelFrame, params: Tensor, covariance: Tensor, indices: list[int], *,
               df_inference: float, classical: tuple[float, int, float, float] | None,
               label: str) -> dict[str, Any]:
    """The overall model test Stata prints: classical F under the conventional covariance
    (``classical = (mss, df_model, rss, df_resid)``), the Wald F on the reported
    covariance otherwise."""
    if classical is not None and frame.spec.covariance == "nonrobust":
        mss, df_model, rss, df_resid = classical
        return classical_f_test(mss, df_model, rss, df_resid, label=label)
    return robust_wald_test(frame, params, covariance, indices, df_inference=df_inference,
                            label=label)


def nested_in_clusters(dimensions: Sequence[tuple[Tensor, int]],
                       clusters: Sequence[tuple[Tensor, int]]) -> list[bool]:
    """Whether each absorbed dimension nests within ANY of the cluster dimensions.

    reghdfe marks a fixed effect as nested when every one of its levels lies
    inside a single cluster of some cluster variable; the check runs against
    each cluster column so that the result cannot depend on their order.
    """
    flags = [False] * len(dimensions)
    for cluster in clusters:
        nested = kernel_call(absorbed_degrees_of_freedom, list(dimensions), cluster).nested
        flags = [known or found for known, found in zip(flags, nested, strict=True)]
    return flags


def solver_diagnostics(fit: LeastSquares, **extra: Any) -> dict[str, Any]:
    return {"condition_number": fit.condition_number, "rank": fit.rank,
            "condition_number_basis": "unit-norm column-scaled weighted design", **extra}


def ones(n: int) -> Tensor:
    return torch.ones((n, 1), dtype=torch.float64)
