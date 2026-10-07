"""Toda--Yamamoto modified Wald tests in a lag-augmented levels VAR."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_integer_dtype
import torch

from openecon.analysis import _coerce_frame, _frame_hasher, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table, wald_test
from openecon.econometrics.unitroot.common import check_count, check_flag, check_magnitude
from openecon.econometrics.var import kernels
from openecon.econometrics.var.common import check_alpha, check_design_size, variable_list
from openecon.econometrics.var.estimators import _screen_system, check_model_size

SOURCE = "https://doi.org/10.1016/0304-4076(94)01616-8"
_MAX_WORK_BYTES = 512 * 1024**2
_MAX_QR_WORK = 2_000_000_000


def _budget(n: int, k: int, p: int, deterministic: int) -> tuple[int, int, int]:
    t, m = n - p, k * p + deterministic
    if t <= m:
        raise AnalysisError("insufficient_observations", "The augmented VAR needs more usable "
                            "observations than regressors per equation.")
    check_design_size(t, m)
    check_model_size(k * m)
    # Covers lag/design/QR copies, outcome/residuals and small matrix operations.
    # It excludes the caller's already resident table and Python result objects.
    work_bytes = 8 * (8 * t * m + 4 * n * k + 6 * m * m + 8 * k * k + 4 * m * k
                      + 4 * (k * m) ** 2)
    if work_bytes > _MAX_WORK_BYTES or t * m * m > _MAX_QR_WORK:
        raise AnalysisError("work_budget_exceeded", "The augmented VAR exceeds its bounded "
                            "512 MiB numerical workspace or QR work budget. Reduce lags, "
                            "variables, or observations; this procedure uses an in-memory sample.")
    return t, m, work_bytes


def _ordered_sample(data: Any, names: list[str], time: str | None) -> pd.DataFrame:
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    selected = names + ([time] if time is not None else [])
    absent = [name for name in selected if name not in frame.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
    chosen = frame.loc[:, selected].reset_index(drop=True)
    if bool(chosen.isna().any().any()):
        raise AnalysisError("missing_values", "Toda--Yamamoto needs complete observations; "
                            "no missing rows are silently removed.")
    if time is None:
        return chosen
    dates = chosen[time]
    if bool(dates.duplicated().any()):
        raise AnalysisError("repeated_time_values", "Every time period must occur exactly once.")
    if is_bool_dtype(dates.dtype):
        raise AnalysisError("invalid_time", "time must contain exact integer periods or datetimes.")
    if not is_integer_dtype(dates.dtype) and not is_datetime64_any_dtype(dates.dtype):
        periods = _numeric(dates, time)
        if bool(((periods != periods.round()) | (periods.abs() > 2**53 - 1)).any()):
            raise AnalysisError("invalid_time", "Floating periods must be exact integers within "
                                "2**53-1; use integer storage for larger period keys.")
    chosen = chosen.sort_values(time, kind="stable").reset_index(drop=True)
    dates = chosen[time]
    if is_datetime64_any_dtype(dates.dtype):
        if len(dates) < 3 or pd.infer_freq(pd.DatetimeIndex(dates)) is None:
            raise AnalysisError("time_gaps", "Datetime observations must follow one regular "
                                "calendar frequency without missing dates.")
    elif int(dates.iloc[-1]) - int(dates.iloc[0]) != len(dates) - 1:
        # Python integers preserve signed/unsigned keys beyond float64 or int64.
        raise AnalysisError("time_gaps", "Integer time periods must be consecutive.")
    return chosen


def tycausality(*, data: Any, y: Any, lags: int, dmax: int = 1,
                time: str | None = None, constant: bool = True, trend: bool = False,
                causes: Any = None, effects: Any = None, joint: bool = True,
                alpha: float = 0.05) -> TableSet:
    """Toda--Yamamoto modified Wald tests of predictive noncausality.

    Fits one levels VAR with ``lags + dmax`` lags and tests ONLY the first
    ``lags`` coefficients of the named causes in each effect equation. The
    extra ``dmax`` lags remain unrestricted. ``dmax`` is a caller-supplied
    upper integration order (0, 1 or 2); no unit-root or lag-selection pretest
    is performed. A valid upper bound and correctly specified base lag order
    and deterministic terms are required for the asymptotic chi-square law.

    By default, tests every directed pair and, for three or more variables,
    every equation's joint exclusion of all other variables. ``causes``
    selects a joint group to test separately in each of ``effects``; the two
    explicitly selected groups must be disjoint. ``joint=False`` selects
    pairwise tests only. This is predictive Granger noncausality, not a
    structural causal effect.

    Complete regularly spaced data are sorted by ``time`` when supplied;
    otherwise input row order supplies time. In-memory CPU float64 QR uses
    the original paper's ML residual covariance U'U/T, with no finite-sample
    df correction. Assumptions include nonsingular iid innovations; this
    function does not add HAC, bootstrap, breaks, or asymmetric decompositions.

    Parameters
    ----------
    data : table
        Complete observations in levels.
    y : ordered list[str]
        At least two endogenous variables in system order.
    lags : int
        Base VAR lag order k; only lags 1..k enter each restriction.
    dmax : int
        Declared upper integration order 0, 1 or 2; adds unrestricted lags.
    time : str or None
        Distinct regular calendar or exact consecutive integer period column.
    constant, trend : bool
        Include an intercept and optional linear trend in every equation.
    causes, effects : ordered list[str] or None
        Cause group and effect equations; explicit groups must be disjoint.
    joint : bool
        Test explicit causes jointly; otherwise include all-other joint rows.
        False requests pairwise tests only.
    alpha : float
        Significance level for asymptotic modified-Wald decisions.

    Returns
    -------
    TableSet
    Returns a ``TableSet`` with ``tests`` and the full augmented-model
    ``coefficients`` (including nuisance lags). Attributes record lag roles,
    residual covariance, sample hash, workspace estimate, and test assumptions.
    Both tables support publication LaTeX output.
    """
    names = variable_list(y)
    if not 2 <= len(names) <= 50:
        raise AnalysisError("invalid_spec", "y must contain between 2 and 50 endogenous variables.")
    base = check_count(lags, "lags", minimum=1)
    augmentation = check_count(dmax, "dmax")
    if augmentation > 2:
        raise AnalysisError("unsupported_integration_order", "This implementation supports "
                            "a declared maximum integration order dmax of 0, 1, or 2.")
    for value, name in ((constant, "constant"), (trend, "trend"), (joint, "joint")):
        check_flag(value, name)
    alpha = check_alpha(alpha)
    if time is not None and (not isinstance(time, str) or not time or time in names):
        raise AnalysisError("invalid_spec", "time must name a distinct nonempty column.")
    causes_given = causes is not None
    selected_causes = variable_list(causes, "causes") if causes_given else names
    selected_effects = variable_list(effects, "effects") if effects is not None else names
    if set(selected_causes + selected_effects) - set(names):
        raise AnalysisError("invalid_spec", "causes and effects must belong to y.")
    if causes_given and effects is not None and set(selected_causes) & set(selected_effects):
        raise AnalysisError("invalid_spec", "Explicit causes and effects must be disjoint.")
    p, k = base + augmentation, len(names)
    raw = _coerce_frame(data)
    t, m, work_bytes = _budget(len(raw), k, p, int(constant) + int(trend))
    chosen = _ordered_sample(raw, names, time)
    groups: list[tuple[str, list[str]]] = []
    for effect in selected_effects:
        available = [name for name in selected_causes if name != effect]
        if not available:
            continue
        if causes_given and joint:
            groups.append((effect, available))
        else:
            groups.extend((effect, [cause]) for cause in available)
            if joint and len(available) > 1:
                groups.append((effect, available))
    if not groups:
        raise AnalysisError("invalid_spec", "The selected groups define no directed noncausality test.")
    with torch.device("cpu"), torch.no_grad():
        levels = torch.stack([_numeric(chosen[name], name) for name in names], dim=1)
        check_magnitude(levels, names)
        regressors = [f"L{j}.{name}" for name in names for j in range(1, p + 1)]
        pieces = [kernels.lag_block(levels, p)]
        if trend:
            pieces.append(torch.arange(p + 1, len(levels) + 1, dtype=torch.float64)[:, None])
            regressors.append("trend")
        if constant:
            pieces.append(torch.ones((t, 1), dtype=torch.float64))
            regressors.append("Intercept")
        z = torch.cat(pieces, dim=1)
        outcome = levels[p:]
        change = torch.eye(m, dtype=torch.float64)
        work = outcome
        if constant:
            mean_z, mean_y = z[:, :-1].mean(dim=0), outcome.mean(dim=0)
            z[:, :-1] -= mean_z
            work = outcome - mean_y
            change[-1, :-1] = -mean_z
        _screen_system(z, regressors, constant)
        fit = kernel_call(kernels.fit_system, z, work, names)
        coef = fit.coef @ change.T
        if constant:
            coef[:, -1] += mean_y
        xtx_inv = change @ fit.xtx_inv @ change.T
        covariance = torch.kron(fit.sigma_ml, xtx_inv.contiguous())
        params = coef.reshape(-1)
        if not bool(torch.isfinite(params).all()) or not bool(torch.isfinite(covariance).all()):
            raise AnalysisError("numerical_failure", "The reported coefficients/covariance "
                                "exceed float64 precision; rescale the series.")
        rows, tested = [], []
        for effect, cause_group in groups:
            equation = names.index(effect)
            positions = [equation * m + names.index(cause) * p + lag
                         for cause in cause_group for lag in range(base)]
            test = wald_test(params, covariance, positions)
            q = len(positions)
            if test["df"] != q or test["statistic"] is None \
                    or not math.isfinite(test["statistic"]):
                raise AnalysisError("singular_test_covariance", "The tested restriction "
                                    "covariance is numerically singular; no reduced-rank "
                                    "Toda--Yamamoto decision is reported.")
            rows.append({"effect": effect, "causes": ", ".join(cause_group),
                         "statistic": test["statistic"], "df": q, "p_value": test["p_value"],
                         "reject": test["p_value"] < alpha})
            tested.append([f"{effect}:{regressors[index % m]}" for index in positions])
        coef_rows = [{"equation": name, "term": term, "estimate": float(coef[i, j]),
                      "std_error": math.sqrt(float(covariance[i * m + j, i * m + j])),
                      "role": "augmentation" if j < k * p and j % p >= base
                              else "base_lag" if j < k * p else "deterministic"}
                     for i, name in enumerate(names) for j, term in enumerate(regressors)]
        tests = table(rows, title="Toda--Yamamoto modified Wald tests", distribution="chi2",
                      alpha=alpha, null="First base-lag coefficients of causes are jointly zero")
        coefficients = table(coef_rows, title="Augmented levels VAR coefficients",
                             covariance="ML Sigma = U'U/T")
        return TableSet({"tests": tests, "coefficients": coefficients},
                        title="Toda--Yamamoto predictive noncausality", method="modified Wald",
                        source=SOURCE, variables=names, base_lags=base, dmax=augmentation,
                        fitted_lags=p, constant=constant, trend=trend, observations=t,
                        levels=len(levels), regressors_per_equation=m, residual_df=t - m,
                        sigma_ml=fit.sigma_ml, covariance_divisor=t, distribution="chi2",
                        asymptotic=True, tested_terms=tested, alpha=alpha,
                        sample_sha256=_frame_hasher(chosen).hexdigest(),
                        time_order=f"sorted by {time}" if time else "input row order",
                        estimated_workspace_bytes=work_bytes, condition_number=fit.condition_number,
                        device="cpu", dtype="float64", inference_scope="iid innovations, "
                        "correctly specified base VAR and deterministic terms, valid dmax upper bound",
                        notes=["The added dmax lags are unrestricted nuisance coefficients.",
                               "Chi-square p-values are asymptotic, not exact finite-sample probabilities.",
                               "No lag selection, integration-order test or residual pretest is performed."])
