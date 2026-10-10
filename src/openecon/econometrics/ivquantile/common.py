"""Bounded iid input and canonical saved state for scalar IVQR."""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
from collections.abc import Mapping, Sequence

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi.common import _decode_index, _encode_index, _index_envelope
from openecon.resources import plan_workspace

SCHEMA = "openecon.ivquantile.v1"
MAX_N = 4096
MAX_WIDTH = 16
MAX_GRID = 257
MAX_EVALUATIONS = 1024
MAX_JSON = 32 * 1024**2


def fail(message, code="invalid_spec"):
    raise AnalysisError(code, message)


def count(value, name, lo, hi):
    if isinstance(value, bool) or not isinstance(value, Integral) or not lo <= value <= hi:
        fail(f"{name} must be an integer in {lo}..{hi}.", "invalid_option")
    return int(value)


def real(value, name, lo, hi, *, strict=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        fail(f"{name} must be a finite real number.", "invalid_option")
    if not (lo < value < hi if strict else lo <= value <= hi):
        fail(f"{name} is outside its declared numerical domain.", "invalid_option")
    return float(value)


def names(value, name):
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        fail(f"{name} requires an explicit list of column names.")
    result = list(value)
    if any(not isinstance(v, str) or not v or len(v.encode()) > 128 for v in result):
        fail(f"{name} requires nonempty string column names of at most 128 bytes.")
    if len(set(result)) != len(result):
        fail(f"{name} cannot repeat a column.")
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, allow_nan=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def seal(value):
    return dict(value, sha256=digest(value))


def controls(*, grid, quantile, confidence, bandwidth, inference, intercept,
             missing, device, weights, cluster, max_work, max_evaluations,
             refinement_tolerance):
    if device != "cpu":
        fail("IVQR supports CPU float64 only.", "unsupported_device")
    if weights is not None or cluster is not None:
        fail("This IVQR contract is iid and unweighted; clusters and weights are unsupported.",
             "unsupported_inference")
    if type(intercept) is not bool or missing not in ("raise", "drop"):
        fail("Declare boolean intercept and missing='raise' or 'drop'.")
    if inference not in ("weak", "identified"):
        fail("inference must be 'weak' or caller-declared 'identified'.", "invalid_option")
    quantile = real(quantile, "quantile", 0.05, 0.95)
    confidence = real(confidence, "confidence", 0, 1, strict=True)
    if isinstance(grid, (str, bytes)) or not isinstance(grid, Sequence):
        fail("Supply a fixed ordered finite alpha grid.", "invalid_option")
    if not 3 <= len(grid) <= MAX_GRID:
        fail(f"The alpha grid requires 3..{MAX_GRID} points.", "invalid_option")
    grid = [real(v, "alpha grid", -1e8, 1e8) for v in grid]
    if any(a >= b for a, b in zip(grid, grid[1:])):
        fail("The alpha grid must be strictly increasing, without duplicates.", "invalid_option")
    max_evaluations = count(max_evaluations, "max_evaluations", len(grid), MAX_EVALUATIONS)
    max_work = count(max_work, "max_work", 1, 10**12)
    refinement_tolerance = real(refinement_tolerance, "refinement_tolerance", 1e-10, 1e-3)
    if bandwidth is not None:
        bandwidth = real(bandwidth, "bandwidth", 1e-12, 1e12)
    return dict(grid=grid, quantile=quantile, confidence=confidence, bandwidth=bandwidth,
                inference=inference, intercept=intercept, missing=missing,
                device="cpu", weights=None, cluster=None, max_work=max_work,
                max_evaluations=max_evaluations, refinement_tolerance=refinement_tolerance)


def _index_size(value):
    encoded = _encode_index(value)
    raw = json.dumps(encoded, allow_nan=False)
    if len(raw.encode()) > 2 * 1024**2:
        fail("IVQR row index exceeds its 2 MiB envelope.", "unsupported_index")
    return encoded


def data_state(data, y, x, endogenous, instruments, config):
    if not isinstance(data, pd.DataFrame):
        fail("IVQR needs a resident pandas DataFrame; Dataset collection is unsupported.",
             "unsupported_input")
    if data.columns.has_duplicates:
        fail("Input column names must be unique.", "duplicate_columns")
    if not 1 <= len(data) <= MAX_N:
        fail(f"IVQR requires at most {MAX_N} original physical rows.", "resource_limit")
    xs, zs = names(x, "x"), names(instruments, "instruments")
    ys, ds = names([y], "y"), names([endogenous], "endogenous")
    roles = ys + ds + xs + zs
    if len(set(roles)) != len(roles) or not zs:
        fail("Outcome, endogenous, exogenous and excluded instrument roles must be distinct; instruments cannot be empty.")
    if "_cons" in roles:
        fail("'_cons' is reserved for the explicit intercept.")
    width = len(xs) + len(zs) + int(config["intercept"])
    if width > MAX_WIDTH or len(xs) > 12 or len(zs) > 8:
        fail("IVQR requires at most 12 exogenous, 8 excluded and 16 total QR columns.", "resource_limit")
    absent = [v for v in roles if v not in data]
    if absent:
        fail(f"Required IVQR columns are absent: {absent}.", "missing_columns")
    original_bytes = sum(int(data[v].memory_usage(index=False, deep=True)) for v in roles)
    plan_workspace("IVQR input and full profile admission", {
        "source, sample, typed index and serialization":
            8 * (original_bytes + int(data.index.memory_usage(deep=True))) + len(data) * 2048,
        "QR and density solver live buffers": len(data) * (width * 12 + 48) * 8,
        "all complete profile coefficients, bases and covariance":
            config["max_evaluations"] * (width * width * 8 + width * 8 + 64) * 48,
    })
    work = config["max_evaluations"] * (len(data) * width**2 * 240
                                             + len(data) * width * (200 + 20 * width) * 16)
    if work > config["max_work"]:
        fail(f"IVQR worst declared QR/profile work {work} exceeds max_work.", "work_budget_exceeded")
    columns = []
    for name in roles:
        series = data[name]
        if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype) or is_complex_dtype(series.dtype):
            fail(f"'{name}' requires real numeric dtype, without categorical/boolean coercion.", "invalid_data")
        values = []
        for value in series:
            if pd.isna(value):
                values.append(None)
            else:
                value = value.item() if hasattr(value, "item") else value
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                    fail(f"'{name}' has a nonfinite value.", "non_finite_values")
                if abs(value) > 1e12:
                    fail("IVQR inputs require absolute values <=1e12; rescale explicitly.", "numerical_domain")
                values.append(int(value) if isinstance(value, Integral) else float(value))
        columns.append(dict(name=name, dtype=str(series.dtype), values=values))
    source = dict(columns=columns, index=_index_size(data.index), n_original=len(data))
    spec = dict(y=y, x=xs, endogenous=endogenous, instruments=zs)
    return source, spec, work


def restore_source(source):
    if not isinstance(source, Mapping) or set(source) != {"columns", "index", "n_original"}:
        fail("Invalid IVQR source schema.", "invalid_state")
    n = count(source["n_original"], "saved n_original", 1, MAX_N)
    columns = source["columns"]
    if not isinstance(columns, list) or not 3 <= len(columns) <= MAX_WIDTH + 2:
        fail("Invalid IVQR saved source column envelope.", "invalid_state")
    try:
        _index_envelope(source["index"], n)
        index = _decode_index(source["index"])
        if len(index) != n or _encode_index(index) != source["index"]:
            raise ValueError("Index identity mismatch.")
        values = {}
        for col in columns:
            if not isinstance(col, Mapping) or set(col) != {"name", "dtype", "values"}:
                raise ValueError("Invalid source descriptor.")
            if not isinstance(col["name"], str) or col["name"] in values or not isinstance(col["dtype"], str):
                raise ValueError("Invalid source name/dtype.")
            if not isinstance(col["values"], list) or len(col["values"]) != n:
                raise ValueError("Invalid saved column length.")
            if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                                     or not math.isfinite(v)) for v in col["values"]):
                raise ValueError("Invalid saved numeric cell.")
            values[col["name"]] = pd.Series(col["values"], dtype=col["dtype"]).array
        return pd.DataFrame(values, index=index)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        fail(f"Saved source/index cannot be restored: {exc}", "invalid_state")


def geometry(data, spec, config):
    roles = [spec["y"], spec["endogenous"], *spec["x"], *spec["instruments"]]
    mask = data[roles].notna().all(axis=1)
    if not bool(mask.all()) and config["missing"] == "raise":
        fail("Missing model cells require explicit missing='drop'.", "missing_values")
    positions = mask.to_numpy().nonzero()[0].tolist()
    selected = data.iloc[positions]
    n = len(selected)
    kx = len(spec["x"]) + int(config["intercept"])
    width = kx + len(spec["instruments"])
    if n < max(32, width + 4):
        fail("IVQR requires at least 32 complete iid rows and QR width+4.", "insufficient_observations")
    block = torch.tensor(selected[roles].to_numpy(dtype="float64", na_value=float("nan")),
                         dtype=torch.float64, device="cpu")
    y, d = block[:, 0], block[:, 1]
    x = block[:, 2:2 + len(spec["x"])]
    z = block[:, 2 + len(spec["x"]):]
    if config["intercept"]:
        x = torch.cat([torch.ones((n, 1), dtype=torch.float64, device="cpu"), x], dim=1)
    w = torch.cat([x, z], dim=1)
    # Full rank in reporting units is assessed after column normalization.
    scale = torch.linalg.vector_norm(w, dim=0)
    if bool((scale == 0).any()):
        fail("The QR design has an all-zero column.", "singular_design")
    _, r = torch.linalg.qr(w / scale, mode="reduced")
    if float(r.diagonal().abs().min()) <= 1e-10:
        fail("The QR design is collinear; terms cannot be silently omitted.", "singular_design")
    if kx:
        qx = torch.linalg.qr(x, mode="reduced")[0]
        zr = z - qx @ (qx.T @ z)
    else:
        zr = z
    a = zr.T @ zr / n
    return y, d, x, z, w, a, positions
