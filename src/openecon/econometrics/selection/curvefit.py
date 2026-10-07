"""Curve estimation: ``oe.curvefit`` (SPSS CURVEFIT).

Eleven one-predictor models, each fitted by ordinary least squares after the
transformation that makes it linear in its parameters:

    linear        y = b0 + b1 x
    logarithmic   y = b0 + b1 ln(x)
    inverse       y = b0 + b1 / x
    quadratic     y = b0 + b1 x + b2 x^2
    cubic         y = b0 + b1 x + b2 x^2 + b3 x^3
    compound      y = b0 * b1^x              ln(y) = ln(b0) + ln(b1) x
    power         y = b0 * x^b1              ln(y) = ln(b0) + b1 ln(x)
    s             y = exp(b0 + b1 / x)       ln(y) = b0 + b1 / x
    growth        y = exp(b0 + b1 x)         ln(y) = b0 + b1 x
    exponential   y = b0 * exp(b1 x)         ln(y) = ln(b0) + b1 x
    logistic      y = 1 / (1/u + b0 * b1^x)  ln(1/y - 1/u) = ln(b0) + ln(b1) x

R-squared, F and its p-value are those of the transformed regression, as in
SPSS; the reported parameters are back-transformed to the equations above.

Numerics. The regressor is centred and scaled before powers are taken (z = (g(x)
- m) / h), the model is solved by Householder QR (``engines.linalg``), and the
coefficients of the raw powers follow from the binomial expansion, so a
predictor such as a calendar year does not make the cubic model collinear.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call
from openecon.econometrics.selection import common as c
from openecon.engines.linalg import least_squares

MODELS = ("linear", "logarithmic", "inverse", "quadratic", "cubic", "compound", "power", "s",
          "growth", "exponential", "logistic")
GRID = 400

# model: (outcome transform, regressor transform, degree, b0 is exp(a0), b1 is exp(a1))
_FORMS: dict[str, tuple[str, str, int, bool, bool]] = {
    "linear": ("identity", "x", 1, False, False),
    "logarithmic": ("identity", "log", 1, False, False),
    "inverse": ("identity", "inverse", 1, False, False),
    "quadratic": ("identity", "x", 2, False, False),
    "cubic": ("identity", "x", 3, False, False),
    "compound": ("log", "x", 1, True, True),
    "power": ("log", "log", 1, True, False),
    "s": ("log", "inverse", 1, False, False),
    "growth": ("log", "x", 1, False, False),
    "exponential": ("log", "x", 1, True, False),
    "logistic": ("logistic", "x", 1, True, True),
}
_EQUATIONS = {
    "linear": ("y = b0 + b1*x", "y = b1*x"),
    "logarithmic": ("y = b0 + b1*ln(x)", "y = b1*ln(x)"),
    "inverse": ("y = b0 + b1/x", "y = b1/x"),
    "quadratic": ("y = b0 + b1*x + b2*x^2", "y = b1*x + b2*x^2"),
    "cubic": ("y = b0 + b1*x + b2*x^2 + b3*x^3", "y = b1*x + b2*x^2 + b3*x^3"),
    "compound": ("y = b0 * b1^x", "y = b1^x"),
    "power": ("y = b0 * x^b1", "y = x^b1"),
    "s": ("y = exp(b0 + b1/x)", "y = exp(b1/x)"),
    "growth": ("y = exp(b0 + b1*x)", "y = exp(b1*x)"),
    "exponential": ("y = b0 * exp(b1*x)", "y = exp(b1*x)"),
    "logistic": ("y = 1 / (1/u + b0 * b1^x)", "y = 1 / (1/u + b1^x)"),
}
# A residual sum of squares below this share of the total is an exact fit to rounding.
_PERFECT = 1e-22


@torch.no_grad()
def polynomial_fit(target: Tensor, base: Tensor, degree: int, intercept: bool
                   ) -> dict[str, Any] | str:
    """OLS of ``target`` on the powers 1..degree of ``base`` (and a constant).

    Returns the raw-power coefficients ``a_0 .. a_degree`` (``a_0`` is None
    without a constant), the standardized representation used for prediction
    and the fit statistics, or a string saying why the model cannot be fitted.
    """
    n = target.numel()
    q = degree
    df2 = n - q - int(intercept)
    if df2 <= 0:
        return f"it needs at least {q + int(intercept) + 1} observations"
    centre = float(base.mean()) if intercept else 0.0
    spread = float((base - centre).square().mean().sqrt())
    if not spread > 1e-12 * max(abs(centre), 1e-300):
        return "the transformed predictor does not vary"
    z = (base - centre) / spread
    columns = [torch.ones_like(z)] if intercept else []
    columns += [z ** power for power in range(1, degree + 1)]
    fit = least_squares(torch.stack(columns, dim=1), target)
    if fit.omitted:
        return "the predictor has too few distinct values"
    mean = float(target.mean()) if intercept else 0.0
    total = float((target - mean).square().sum())
    if not total > 0 or (intercept and total <= 1e-24 * n * mean * mean):
        return "the transformed outcome does not vary"
    sse = float(fit.ssr)
    perfect = sse <= _PERFECT * total
    r_squared = 1.0 if perfect else 1.0 - sse / total
    statistic = None if perfect else ((total - sse) / q) / (sse / df2)
    standardized = fit.beta.tolist()
    if not intercept:
        standardized = [0.0, *standardized]
    # Raw-power coefficients from  sum_j c_j ((g - m) / h)^j  by the binomial theorem.
    raw = []
    for i in range(degree + 1):
        raw.append(sum(standardized[j] / spread ** j * math.comb(j, i) * (-centre) ** (j - i)
                       for j in range(i, degree + 1)))
    return {
        "raw": [raw[0] if intercept else None, *raw[1:]], "standardized": standardized,
        "centre": centre, "spread": spread, "r_squared": r_squared, "statistic": statistic,
        "df1": q, "df2": df2, "p_value": None if perfect else c.f_upper(statistic, q, df2),
        "perfect": perfect,
    }


class CurveTables(TableSet):
    """The curve-estimation tables; the text form abbreviates the plotting grid."""

    def __str__(self) -> str:
        shown = TableSet(dict(self), title=self.title)
        shown.attrs = self.attrs
        grid = self.get("fitted")
        if grid is not None and len(grid) > 8:
            shown["fitted"] = grid.iloc[[0, 1, 2, len(grid) - 3, len(grid) - 2, len(grid) - 1]]
            return TableSet.__str__(shown).replace(
                "[fitted]", f"[fitted] (rows 0-2 and {len(grid) - 3}-{len(grid) - 1} of "
                f"{len(grid)})", 1)
        return TableSet.__str__(shown)

    __repr__ = __str__


def _transform(kind: str, x: Tensor) -> Tensor:
    return x if kind == "x" else x.log() if kind == "log" else 1.0 / x


def curvefit(data: Any, y: str, x: str, *, models: list[str] | None = None,
             upper_bound: float | None = None, intercept: bool = True,
             missing: str = "drop") -> TableSet:
    """Curve estimation: fit and compare one-predictor curve models (SPSS CURVEFIT).

    Models (``b0`` is SPSS's "Constant"):

    =============  ==============================  ================================
    model          equation                        fitted as
    =============  ==============================  ================================
    linear         y = b0 + b1 x                   y on x
    logarithmic    y = b0 + b1 ln(x)               y on ln(x)
    inverse        y = b0 + b1 / x                 y on 1/x
    quadratic      y = b0 + b1 x + b2 x^2          y on x, x^2
    cubic          y = b0 + b1 x + b2 x^2 + b3 x^3 y on x, x^2, x^3
    compound       y = b0 * b1^x                   ln(y) on x
    power          y = b0 * x^b1                   ln(y) on ln(x)
    s              y = exp(b0 + b1 / x)            ln(y) on 1/x
    growth         y = exp(b0 + b1 x)              ln(y) on x
    exponential    y = b0 * exp(b1 x)              ln(y) on x
    logistic       y = 1 / (1/u + b0 * b1^x)       ln(1/y - 1/u) on x
    =============  ==============================  ================================

    Each model is an ordinary least-squares regression of the (transformed)
    outcome on the (transformed) predictor. ``r_squared``, the F statistic
    ``((SST - SSE) / df1) / (SSE / df2)`` and its p-value refer to that
    transformed regression, exactly as SPSS reports them, so R-squared values
    of models with different outcome transformations are not directly
    comparable. Multiplicative constants are back-transformed: ``b0 =
    exp(intercept)`` for compound, power, exponential and logistic, and ``b1 =
    exp(slope)`` for compound and logistic.

    A model whose transformation is undefined for the data is skipped with a
    note instead of failing the call: models in ln(y) need a positive outcome,
    logarithmic and power need a positive predictor, inverse and s a nonzero
    predictor, and logistic needs 0 < y < ``upper_bound``.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y, x : numeric outcome and predictor columns.
    models : names of the models to fit, in the order to report (default: all
        eleven).
    upper_bound : the upper asymptote ``u`` of the logistic model; it must
        exceed the largest outcome value. Without it ``1/u = 0`` (SPSS's
        default), i.e. ln(1/y) is regressed on x.
    intercept : False fits every model without its constant (SPSS
        ``/NOCONSTANT``): R-squared is then measured about the origin, df2 =
        n - df1, and ``b0`` is not reported (the multiplicative models have
        b0 = 1).
    missing : ``"drop"`` (default) excludes rows with a missing ``y`` or ``x``
        (listwise); ``"raise"`` rejects them.

    Returns
    -------
    TableSet with tables

    - ``summary``: one row per requested model: ``equation``, ``r_squared``,
      ``statistic`` (F), ``df1``, ``df2``, ``p_value``, ``b0``, ``b1``, ``b2``,
      ``b3`` (missing where a model has no such parameter or was skipped).
    - ``fitted``: 400 evenly spaced values of ``x`` between its minimum and
      maximum with the fitted curve of each estimated model on the scale of y,
      for plotting.

    ``attrs``: ``n``, ``n_missing``, ``fitted_models``, ``skipped``,
    ``intercept``, ``upper_bound`` and ``notes``.

    Equivalent command: SPSS ``CURVEFIT /VARIABLES=y WITH x /CONSTANT
    /MODEL=LINEAR LOGARITHMIC INVERSE QUADRATIC CUBIC COMPOUND POWER S GROWTH
    EXPONENTIAL LGSTIC /UPPERBOUND=u``. In Stata each row is a ``regress`` of
    the transformed variables, e.g. ``regress lny x`` for the growth model.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ...         "y": [2.7, 7.4, 20.1, 54.6, 148.4, 403.4]}
    >>> result = oe.curvefit(data, "y", "x", models=["linear", "growth"])
    >>> round(float(result["summary"].loc["growth", "b1"]), 2)
    1.0
    """
    c.check_name(y, "y")
    c.check_name(x, "x")
    requested = c.name_list(list(MODELS) if models is None else models, "models", minimum=1,
                            noun="model name", example="linear")
    unknown = [name for name in requested if name not in MODELS]
    if unknown:
        raise AnalysisError("invalid_option", f"Unknown model(s): {', '.join(unknown)}. "
                            f"Available: {', '.join(MODELS)}.")
    c.check_flag(intercept, "intercept")
    if upper_bound is not None:
        if isinstance(upper_bound, bool) or not isinstance(upper_bound, (int, float)) \
                or not math.isfinite(upper_bound) or upper_bound <= 0:
            raise AnalysisError("invalid_option", "upper_bound must be a positive number larger "
                                "than every outcome value.")
        upper_bound = float(upper_bound)
    frame = c.source(data, [y, x], numeric=[y, x])
    keep, dropped = c.listwise(frame, [y, x], missing, numeric=[y, x])
    outcome, predictor = c.column(frame, y, keep), c.column(frame, x, keep)
    n = outcome.numel()
    if n < 3:
        raise AnalysisError("insufficient_observations", "Curve estimation needs at least "
                            "three complete observations.")
    low, high = float(predictor.min()), float(predictor.max())
    if not high > low:
        raise AnalysisError("zero_variance", f"The predictor '{x}' does not vary.")
    if not float(outcome.max()) > float(outcome.min()):
        raise AnalysisError("zero_variance", f"The outcome '{y}' does not vary.")

    # Why a transformation is unavailable for these data (None when it is fine).
    positive_y = bool((outcome > 0).all())
    blocked_x = {
        "x": None,
        "log": None if low > 0 else "the predictor has non-positive values (ln(x) is undefined)",
        "inverse": None if bool((predictor != 0).all())
        else "the predictor has zero values (1/x is undefined)",
    }
    blocked_y = {
        "identity": None,
        "log": None if positive_y
        else "the outcome has non-positive values (ln(y) is undefined)",
        "logistic": None if positive_y
        else "the outcome has non-positive values (ln(1/y - 1/u) is undefined)",
    }
    if blocked_y["logistic"] is None and upper_bound is not None \
            and not upper_bound > float(outcome.max()):
        blocked_y["logistic"] = (f"upper_bound={upper_bound:g} does not exceed the largest "
                                 f"outcome value ({float(outcome.max()):g})")
    inverse_bound = 0.0 if upper_bound is None else 1.0 / upper_bound
    targets: dict[str, Tensor] = {"identity": outcome}
    if positive_y:
        targets["log"] = outcome.log()
        if blocked_y["logistic"] is None:
            targets["logistic"] = (1.0 / outcome - inverse_bound).log()
    grid = torch.linspace(low, high, GRID, dtype=torch.float64)
    cache: dict[tuple[str, str, int], dict[str, Any] | str] = {}
    rows, notes, fitted_names, skipped = [], [], [], []
    curves: dict[str, list[float | None]] = {}
    for name in requested:
        y_kind, x_kind, degree, exp_b0, exp_b1 = _FORMS[name]
        equation = _EQUATIONS[name][0 if intercept else 1]
        reason = blocked_y[y_kind] or blocked_x[x_kind]
        fit: dict[str, Any] | str | None = None
        if reason is None:
            key = (y_kind, x_kind, degree)
            if key not in cache:
                cache[key] = kernel_call(polynomial_fit, targets[y_kind],
                                         _transform(x_kind, predictor), degree, intercept)
            fit = cache[key]
            if isinstance(fit, str):
                reason, fit = fit, None
        if fit is None:
            notes.append(f"{name}: not fitted because {reason}.")
            skipped.append(name)
            rows.append([equation, *[None] * 9])
            continue
        raw = list(fit["raw"]) + [None] * (3 - degree)
        if exp_b0 and raw[0] is not None:
            raw[0] = _exp(raw[0])
        if exp_b1:
            raw[1] = _exp(raw[1])
        lost = [f"b{j}" for j, value in enumerate(raw) if value is not None
                and c.finite(value) is None]
        raw = [c.finite(value) for value in raw]
        if lost:
            verb = "is" if len(lost) == 1 else "are"
            notes.append(f"{name}: {', '.join(lost)} {verb} not representable in double "
                         "precision (overflow or underflow of the back-transformation) and "
                         f"{verb} left missing; rescale the predictor, e.g. measure it in "
                         "larger units.")
        if fit["perfect"]:
            notes.append(f"{name}: the fit is exact; the F statistic is not defined.")
        rows.append([equation, fit["r_squared"], fit["statistic"], fit["df1"], fit["df2"],
                     fit["p_value"], *raw])
        fitted_names.append(name)
        # Fitted curve on the grid, from the well-conditioned standardized form.
        base = _transform(x_kind, grid)
        z = (base - fit["centre"]) / fit["spread"]
        eta = sum(coefficient * z ** power for power, coefficient
                  in enumerate(fit["standardized"]))
        if y_kind == "log":
            eta = eta.exp()
        elif y_kind == "logistic":
            eta = 1.0 / (inverse_bound + eta.exp())
        curves[name] = [c.finite(value) for value in eta.tolist()]
    summary = c.result_table(rows, index=requested, columns=[
        "equation", "r_squared", "statistic", "df1", "df2", "p_value", "b0", "b1", "b2", "b3"])
    summary.index.name = "model"
    fitted = c.result_table({x: grid.tolist(), **curves}, columns=[x, *fitted_names])
    return CurveTables(
        {"summary": summary, "fitted": fitted}, title=f"Curve estimation of {y} on {x}",
        n=n, n_missing=dropped, fitted_models=fitted_names, skipped=skipped,
        intercept=intercept, upper_bound=upper_bound, missing="listwise", notes=notes)


def _exp(value: float) -> float:
    """exp(value); inf on overflow and NaN on underflow below the normal range, where the
    result (0 or a subnormal number) would no longer carry the coefficient."""
    try:
        result = math.exp(value)
    except OverflowError:
        return math.inf
    return result if result >= sys.float_info.min else math.nan
