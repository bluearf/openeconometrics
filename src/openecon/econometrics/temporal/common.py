"""Bounded explicit-calendar inputs and complete temporal-disaggregation results."""
from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Real
import re
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.resources import workspace_budget_bytes

MAX_HIGH = 240
MAX_INDICATORS = 8
CONDITION_LIMIT = 1e12


@dataclass
class Prepared:
    y: torch.Tensor
    x: torch.Tensor
    C: torch.Tensor
    settings: dict[str, Any]

    @property
    def n(self):
        return self.x.shape[0]

    @property
    def m(self):
        return len(self.y)


def _list(value, name):
    if isinstance(value, torch.Tensor):
        if value.ndim not in (1, 2):
            raise AnalysisError("unsupported_input", f"{name} needs a vector or two-dimensional matrix.")
        if value.device.type != "cpu" or value.requires_grad or value.is_complex():
            raise AnalysisError("unsupported_input", f"{name} needs a real CPU tensor without gradients.")
        if value.numel() > MAX_HIGH * MAX_INDICATORS:
            raise AnalysisError("resource_limit", f"{name} exceeds the bounded input budget.")
        return value.tolist()
    if isinstance(value, pd.Series) or type(value).__module__.startswith("numpy"):
        if getattr(value, "ndim", 0) not in (1, 2):
            raise AnalysisError("unsupported_input", f"{name} needs a vector or two-dimensional matrix.")
        if value.size > MAX_HIGH * MAX_INDICATORS:
            raise AnalysisError("resource_limit", f"{name} exceeds the bounded input budget.")
        return value.tolist()
    if not isinstance(value, (list, tuple)):
        raise AnalysisError("unsupported_input", f"{name} needs a finite positional vector or matrix; Dataset/table collection is unsupported.")
    if len(value) > MAX_HIGH:
        raise AnalysisError("resource_limit", f"{name} exceeds {MAX_HIGH} observations.")
    return list(value)


def _tensor(value, name, *, matrix=False):
    rows = _list(value, name)
    if not rows:
        raise AnalysisError("empty_data", f"{name} cannot be empty.")
    if matrix:
        if not isinstance(rows[0], (list, tuple)):
            rows = [[item] for item in rows]
        width = len(rows[0])
        if not 1 <= width <= MAX_INDICATORS:
            raise AnalysisError("resource_limit", f"indicator requires 1..{MAX_INDICATORS} columns.")
        if any(not isinstance(row, (list, tuple)) or len(row) != width for row in rows):
            raise AnalysisError("invalid_input", "indicator matrix must be rectangular.")
        values = [v for row in rows for v in row]
    else:
        values = rows
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in values):
        raise AnalysisError("invalid_input", f"{name} requires real numeric values, without booleans or missing cells.")
    result = torch.tensor(rows, dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(result).all()):
        raise AnalysisError("missing_values", f"{name} contains missing or nonfinite values; complete calendars cannot drop rows.")
    magnitudes = result.abs()
    if bool((magnitudes > 1e50).any()) or bool(((magnitudes > 0) & (magnitudes < 1e-50)).any()):
        raise AnalysisError("invalid_input", f"{name} nonzero magnitudes must lie in [1e-50,1e50]; rescale units.")
    return result


def _periods(value, frequency, name):
    if frequency not in ("Y", "Q", "M"):
        raise AnalysisError("invalid_calendar", "frequency must be Y, Q or M (December year/quarter ends).")
    labels = _list(value, name)
    pattern = {"Y": r"\d{4}", "Q": r"\d{4}Q[1-4]", "M": r"\d{4}-(0[1-9]|1[0-2])"}[frequency]
    if any(not isinstance(v, str) or re.fullmatch(pattern, v) is None for v in labels):
        raise AnalysisError("invalid_calendar", f"{name} needs canonical {frequency} period strings.")
    try:
        periods = pd.PeriodIndex(labels, freq={"Y":"Y-DEC", "Q":"Q-DEC", "M":"M"}[frequency])
    except (ValueError, TypeError) as error:
        raise AnalysisError("invalid_calendar", f"Invalid {name}.") from error
    if len(periods) < 2 or any(b != a + 1 for a, b in zip(periods.asi8[:-1], periods.asi8[1:])):
        raise AnalysisError("invalid_calendar", f"{name} must be strictly ordered, unique, complete and regular.")
    return periods


def _releases(as_of, low_releases, high_releases, m, n):
    given = (as_of is not None, low_releases is not None, high_releases is not None)
    if not any(given):
        return dict(as_of=None, low_releases=None, high_releases=None,
                    release_policy="no release metadata supplied; retrospective complete-data benchmarking only")
    if not all(given):
        raise AnalysisError("invalid_release", "Provide as_of, low_releases and high_releases together.")
    low, high = _list(low_releases, "low_releases"), _list(high_releases, "high_releases")
    if len(low) != m or len(high) != n:
        raise AnalysisError("invalid_release", "Release vectors must match the complete low/high calendars.")
    try:
        timestamps = [pd.Timestamp(v) for v in [as_of, *low, *high]]
    except (ValueError, TypeError, OverflowError) as error:
        raise AnalysisError("invalid_release", "Releases and as_of must be valid timestamps.") from error
    if any(pd.isna(v) for v in timestamps) or len({v.tzinfo is None for v in timestamps}) != 1:
        raise AnalysisError("invalid_release", "Release timestamps must be complete with consistent timezone awareness.")
    cutoff, *releases = timestamps
    if any(v > cutoff for v in releases):
        raise AnalysisError("future_release", "A supplied value was released after as_of; no future releases or silent truncation are allowed.")
    return dict(as_of=cutoff.isoformat(), low_releases=[v.isoformat() for v in releases[:m]],
                high_releases=[v.isoformat() for v in releases[m:]],
                release_policy="all supplied low and high values released on or before declared as_of; no vintage selection or forecast scoring")


def prepare(low, indicator, *, low_periods, high_periods, low_frequency="Y",
            high_frequency="Q", aggregation="sum", as_of=None,
            low_releases=None, high_releases=None, device="cpu", weights=None):
    if device != "cpu":
        raise AnalysisError("unsupported_device", "Temporal disaggregation supports native CPU float64 only.")
    if weights is not None:
        raise AnalysisError("unsupported_weights", "Sampling/regression weights are outside this benchmarking contract.")
    if aggregation not in ("sum", "mean", "first", "last"):
        raise AnalysisError("invalid_option", "aggregation must be sum, mean, first or last.")
    if (low_frequency, high_frequency) not in (("Y", "Q"), ("Y", "M"), ("Q", "M")):
        raise AnalysisError("invalid_calendar", "Supported calendars are Y→Q, Y→M and Q→M with December fiscal end.")
    low_index = _periods(low_periods, low_frequency, "low_periods")
    high_index = _periods(high_periods, high_frequency, "high_periods")
    m, n = len(low_index), len(high_index)
    if n > MAX_HIGH:
        raise AnalysisError("resource_limit", f"At most {MAX_HIGH} high-frequency periods are supported.")
    ratio = {("Y","Q"):4, ("Y","M"):12, ("Q","M"):3}[(low_frequency, high_frequency)]
    if n != m * ratio or high_index[0] != low_index[0].asfreq(high_index.freq, how="start") or high_index[-1] != low_index[-1].asfreq(high_index.freq, how="end"):
        raise AnalysisError("invalid_calendar", "High periods must exactly cover all low periods; partial ends and ragged calendars are unsupported.")
    estimate = 8 * (24 * (n + m) ** 2 + 12 * n * MAX_INDICATORS)
    budget = workspace_budget_bytes()
    if estimate > budget:
        raise AnalysisError("resource_limit", "Dense temporal-disaggregation workspace exceeds the configured budget.")
    y, x = _tensor(low, "low"), _tensor(indicator, "indicator", matrix=True)
    if len(y) != m or len(x) != n:
        raise AnalysisError("invalid_input", "Values must match their explicit calendars exactly.")
    C = torch.zeros((m,n), dtype=torch.float64, device="cpu")
    for i in range(m):
        if aggregation in ("sum", "mean"):
            C[i, i*ratio:(i+1)*ratio] = 1. if aggregation == "sum" else 1. / ratio
        else:
            C[i,i*ratio + (ratio-1 if aggregation == "last" else 0)] = 1.
    releases = _releases(as_of, low_releases, high_releases, m, n)
    if releases["as_of"] is not None and high_index[-1].end_time.date() > pd.Timestamp(releases["as_of"]).date():
        raise AnalysisError("future_period", "The complete high-frequency calendar ends after as_of; future forecast spans are unsupported.")
    settings = dict(low=y.tolist(), indicator=x.tolist(), low_periods=list(map(str,low_index)),
                    high_periods=list(map(str,high_index)), low_frequency=low_frequency,
                    high_frequency=high_frequency, aggregation=aggregation, ratio=ratio,
                    n_low=m, n_high=n, n_indicators=x.shape[1], device="cpu", weights=None,
                    precision="float64", missing="raise; no row deletion", **releases,
                    input_domain="bounded positional finite vectors/matrices; no Dataset collection",
                    estimated_workspace_bytes=estimate, workspace_budget_bytes=budget,
                    work_limit="n_high<=240; indicators<=8; dense O((n_high+n_low)^3)")
    return Prepared(y,x,C,settings)


def checked_solve(A, B, name="linear system"):
    if not bool(torch.isfinite(A).all()) or not bool(torch.isfinite(B).all()):
        raise AnalysisError("numerical_failure", f"Nonfinite {name}.")
    condition = float(torch.linalg.cond(A))
    if not condition < CONDITION_LIMIT:
        raise AnalysisError("numerical_failure", f"{name} condition number exceeds {CONDITION_LIMIT:g}; no ridge fallback.")
    try:
        solution = torch.linalg.solve(A,B)
    except torch.linalg.LinAlgError as error:
        raise AnalysisError("numerical_failure", f"Cannot solve {name}.") from error
    scale = float(A.abs().max()) * float(solution.abs().max()) * A.shape[1] + float(B.abs().max())
    if not bool(torch.isfinite(solution).all()) or float((A@solution-B).abs().max()) > 2e-10 * max(scale,1e-300):
        raise AnalysisError("numerical_failure", f"{name} backward error gate failed.")
    return solution


def check_constraints(p, values):
    if values.shape != (p.n,) or not bool(torch.isfinite(values).all()):
        raise AnalysisError("numerical_failure", "Reconstructed values must be finite and aligned.")
    errors = (p.C @ values-p.y).abs()
    error = float(errors.max())
    scales = torch.maximum(p.y.abs(), p.C.abs() @ values.abs()).clamp_min(1e-300)
    # Cancellation can satisfy backward error while destroying a tiny nonzero
    # benchmark. Require accuracy relative to every nonzero observed benchmark
    # too. A zero benchmark retains the local backward-error convention.
    scales = torch.where(p.y != 0, p.y.abs(), scales)
    if bool((errors > 2e-9 * scales).any()):
        raise AnalysisError("numerical_failure", "Low-frequency aggregation error gate failed.")
    return error


def output(method, p, values, *, tables=None, settings=None):
    error = check_constraints(p,values)
    metadata = dict(p.settings, method=method, complete_inputs_saved=True,
                    maximum_aggregation_error=error, stochastic_draws=0,
                    aggregation_error_gate="each nonzero low benchmark: 2e-9*abs(observed); zero benchmarks: local backward error 2e-9*max(sum(abs(C*reconstructed)),1e-300)",
                    negative_output_count=int((values<0).sum()),
                    nonnegativity="not constrained; no silent clipping",
                    inference="deterministic benchmarking; sampling covariance/SE/df/p/CI/likelihood not applicable")
    metadata.update(settings or {})
    high, low = metadata["high_periods"], metadata["low_periods"]
    result = {"series":table([dict(high_period=period, low_period=low[i//metadata["ratio"]], value=float(values[i])) for i,period in enumerate(high)]),
              "aggregation":table([dict(low_period=period, observed=float(p.y[i]), reconstructed=float((p.C@values)[i]), error=float((p.C@values-p.y)[i])) for i,period in enumerate(low)])}
    if tables:
        if {"series", "aggregation", "settings"}.intersection(tables):
            raise AnalysisError("invalid_result", "Additional tables cannot replace the common complete contract.")
        result.update(tables)
    result["settings"] = table([[key,json.dumps(value,allow_nan=False)] for key,value in metadata.items()],columns=["setting","json"])
    return TableSet(result,title=method+" — temporal disaggregation",**metadata)
