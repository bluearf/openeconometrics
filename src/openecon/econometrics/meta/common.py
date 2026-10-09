"""Strict complete-study input, finite inference and portable provenance."""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real

import torch

from openecon.analysis import _coerce_frame, _frame_hasher, _json_scalar, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import normal_isf, normal_sf, t_isf, t_sf

SOURCES = {
    "effects": "https://wviechtb.github.io/metafor/reference/escalc.html",
    "model": "https://wviechtb.github.io/metafor/reference/rma.uni.html",
    "prediction": "https://wviechtb.github.io/metafor/reference/predict.rma.html",
    "asymmetry": "https://wviechtb.github.io/metafor/reference/regtest.html",
}


def level_check(level):
    if isinstance(level, bool) or not isinstance(level, Real) or not math.isfinite(level) or not .5 < level < 1:
        raise AnalysisError("invalid_level", "level must be finite and strictly between .5 and 1.")
    return float(level)


def sample(data, names, study=None, *, dependence="independent", minimum=1, maximum=2000):
    if dependence != "independent":
        raise AnalysisError("unsupported_dependence", "Only independent study effects are supported; dependent effects require a covariance model.")
    if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise AnalysisError("invalid_spec", "Numeric column roles must be distinct nonempty names.")
    if study is not None and (not isinstance(study, str) or not study or study in names):
        raise AnalysisError("invalid_spec", "study must be a distinct column name.")
    raw = _coerce_frame(data)
    if not minimum <= len(raw) <= maximum or len(names) > 32 or len(raw)*len(names)*8*64 > 128*1024**2:
        raise AnalysisError("resource_limit", f"This in-memory route requires {minimum}..{maximum} studies, <=32 numeric roles and <=128 MiB estimated workspace.")
    roles = [*names, *([study] if study else [])]
    if any(list(raw.columns).count(n) != 1 for n in roles):
        raise AnalysisError("invalid_columns", "Each selected column must exist exactly once.")
    frame = raw.loc[:, roles].copy()
    if bool(frame.isna().any().any()):
        raise AnalysisError("missing_values", "All selected study roles must be complete; no study is silently dropped.")
    with torch.device("cpu"):
        tensors = {n: _numeric(frame[n], n).clone().cpu() for n in names}
    if any(frame[n].dtype.kind == "b" for n in names):
        raise AnalysisError("invalid_numeric", "Boolean values are not quantitative study summaries.")
    ids = [_json_scalar(v) for v in frame[study]] if study else list(range(1, len(frame)+1))
    if len({str(v) for v in ids}) != len(ids):
        raise AnalysisError("duplicate_study", "Study identifiers must be distinct, including their displayed labels.")
    return frame, tensors, ids, _frame_hasher(frame).hexdigest()


def critical(level, inference, df):
    return normal_isf((1-level)/2) if inference == "z" else t_isf((1-level)/2, df)


def probability(value, inference, df):
    return 2*(normal_sf(abs(value)) if inference == "z" else t_sf(abs(value), df))


def checksum(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(values, what):
    if not bool(torch.isfinite(values).all()):
        raise AnalysisError("numerical_failure", f"Non-finite {what}; rescale study summaries.")
