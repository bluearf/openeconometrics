"""Complete positional input and persistence contract for bounded survival stages."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.resources import workspace_budget_bytes

MAX_N = 4096
MAX_OUTPUT = 256
MAX_SUPPORT = 256
MIN_TIME = 1e-12
MAX_TIME = 1e12


@dataclass
class IntervalData:
    lo: torch.Tensor
    hi: torch.Tensor
    kind: list[str]
    settings: dict[str, Any]

    @property
    def n(self):
        return len(self.lo)


@dataclass
class EventData:
    t: torch.Tensor
    event: torch.Tensor
    settings: dict[str, Any]

    @property
    def n(self):
        return len(self.t)


def positional_list(value, name, *, limit=MAX_N):
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.device.type != "cpu" or value.requires_grad or value.is_complex():
            raise AnalysisError("unsupported_input", f"{name} requires a real one-dimensional CPU tensor without gradients.")
        if value.numel() > limit:
            raise AnalysisError("resource_limit", f"{name} exceeds {limit} observations.")
        result = value.tolist()
    elif isinstance(value, pd.Series) or type(value).__module__.startswith("numpy"):
        if getattr(value, "ndim", 0) != 1:
            raise AnalysisError("unsupported_input", f"{name} requires a one-dimensional positional vector.")
        if value.size > limit:
            raise AnalysisError("resource_limit", f"{name} exceeds {limit} observations.")
        result = value.tolist()
    elif isinstance(value, (list, tuple)):
        if len(value) > limit:
            raise AnalysisError("resource_limit", f"{name} exceeds {limit} observations.")
        result = list(value)
    else:
        raise AnalysisError("unsupported_input", f"{name} requires a positional vector; Dataset/table collection is unsupported.")
    if not result:
        raise AnalysisError("empty_data", f"{name} cannot be empty.")
    return result


def numeric_vector(value, name, *, limit=MAX_N, allow_zero=False):
    rows = positional_list(value, name, limit=limit)
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in rows):
        raise AnalysisError("invalid_input", f"{name} requires real numeric values without booleans or missing cells.")
    if any(v > MAX_TIME or (v != 0 and v < MIN_TIME) or (v == 0 and not allow_zero) for v in rows):
        raise AnalysisError("invalid_input", f"{name} requires positive times in [1e-12,1e12]" + (" or zero." if allow_zero else "."))
    if any(not math.isfinite(float(v)) for v in rows):
        raise AnalysisError("missing_values", f"{name} contains missing or nonfinite values; rows cannot be silently dropped.")
    return torch.tensor(rows, dtype=torch.float64, device="cpu")


def controls(device="cpu", weights=None):
    if device != "cpu":
        raise AnalysisError("unsupported_device", "These survival stages support native CPU float64 only.")
    if weights is not None:
        raise AnalysisError("unsupported_weights", "Weights, clusters and survey inference are outside this iid survival contract.")


def _settings(n):
    return dict(n=n, device="cpu", precision="float64", weights=None,
                missing="raise; no row deletion or endpoint recoding", complete_inputs_saved=True,
                sample="independent iid single-event observations; independent noninformative censoring",
                input_domain="bounded positional vectors; no Dataset collection",
                max_observations=MAX_N,
                excluded_scope=["covariates", "delayed entry", "recurrent events", "clusters", "sampling weights"])


def interval_data(lower, upper, *, device="cpu", weights=None):
    controls(device, weights)
    lo = numeric_vector(lower, "lower", allow_zero=True)
    upper_rows = positional_list(upper, "upper")
    if len(upper_rows) != len(lo):
        raise AnalysisError("invalid_input", "lower and upper must have equal lengths.")
    hi_values, canonical, kinds = [], [], []
    for left, right in zip(lo.tolist(), upper_rows):
        if right is None or (isinstance(right, Real) and not isinstance(right, bool) and right == math.inf):
            if left == 0:
                raise AnalysisError("invalid_interval", "(0,infinity) contributes no information and is unsupported.")
            hi_values.append(math.inf)
            canonical.append(None)
            kinds.append("right")
            continue
        if isinstance(right, bool) or not isinstance(right, Real):
            raise AnalysisError("invalid_interval", "upper requires finite positive times or explicit None/+infinity for right censoring.")
        if right < MIN_TIME or right > MAX_TIME or right < left:
            raise AnalysisError("invalid_interval", "Finite upper must be in [1e-12,1e12] and at least lower.")
        if not math.isfinite(float(right)):
            raise AnalysisError("invalid_interval", "upper requires finite positive times or explicit None/+infinity for right censoring.")
        hi_values.append(float(right))
        canonical.append(float(right))
        kinds.append("exact" if left == right else "left" if left == 0 else "interval")
    settings = _settings(len(lo))
    settings.update(lower=lo.tolist(), upper=canonical, censoring_kind=kinds,
                    endpoints="finite (lower,upper]; exact lower=upper>0; left lower=0; right upper=None means T>lower",
                    exact_observation="continuous density for parametric ML; atom membership for Turnbull NPMLE")
    return IntervalData(lo, torch.tensor(hi_values, dtype=torch.float64, device="cpu"), kinds, settings)


def event_data(time, event, *, device="cpu", weights=None):
    controls(device, weights)
    t = numeric_vector(time, "time")
    rows = positional_list(event, "event")
    if len(rows) != len(t):
        raise AnalysisError("invalid_input", "time and event must have equal lengths.")
    if any(isinstance(v, bool) or not isinstance(v, Real) or v < 0 or v > 2**31-1 or not math.isfinite(float(v)) or int(v) != v for v in rows):
        raise AnalysisError("invalid_event", "event must be integer cause codes: 0=censor, positive=cause (<=2^31-1).")
    e = torch.tensor(rows, dtype=torch.int64, device="cpu")
    settings = _settings(len(t))
    settings.update(time=t.tolist(), event=e.tolist(), event_coding="0=censor; positive integer codes=mutually exclusive causes",
                    tie_policy="failures at a time use the common risk set including censoring at that same time")
    return EventData(t, e, settings)


def prediction_times(times, *, default=None):
    if times is None:
        if default is None:
            return None
        times = default
    t = numeric_vector(times, "times", limit=MAX_OUTPUT, allow_zero=True)
    if len(t) > 1 and not bool((t[1:] > t[:-1]).all()):
        raise AnalysisError("invalid_input", "times must be strictly increasing, unique and prespecified.")
    return t


def confidence(level):
    if isinstance(level, bool) or not isinstance(level, Real) or not 0 < float(level) < 1:
        raise AnalysisError("invalid_option", "level must be a finite probability strictly between zero and one.")
    z = float(torch.special.ndtri(torch.tensor((1 + float(level)) / 2, dtype=torch.float64, device="cpu")))
    if not math.isfinite(z):
        raise AnalysisError("invalid_option", "level is too close to one for float64 normal inference.")
    return float(level), z


def workspace(n, outputs, *, support=0):
    if outputs > MAX_OUTPUT or support > MAX_SUPPORT:
        raise AnalysisError("resource_limit", "At most 256 complete output components and 256 support cells are supported; no thinning.")
    estimate = 8 * (12 * n * max(outputs, support, 1) + 8 * max(outputs, support, 1)**2)
    budget = workspace_budget_bytes()
    if estimate > budget:
        raise AnalysisError("resource_limit", "Complete survival covariance/support workspace exceeds the configured budget.")
    return dict(estimated_workspace_bytes=estimate, workspace_budget_bytes=budget,
                output_components=outputs, support_cells=support)


def output(method, frames, settings, notes=()):
    metadata = dict(settings, method=method, contract="survival_ext_v1", notes=list(notes))
    try:
        serialized = [(key, json.dumps(value, allow_nan=False)) for key, value in metadata.items()]
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("invalid_result", "Saved survival state must be finite and fully JSON serializable.") from error
    if "settings" in frames:
        raise AnalysisError("invalid_result", "Procedure frames cannot replace the complete settings contract.")
    tables = {name: frame if isinstance(frame, pd.DataFrame) else table(frame) for name, frame in frames.items()}
    tables["settings"] = table(serialized, columns=["setting", "json"])
    return TableSet(tables, title=method + " — survival analysis", **metadata)
