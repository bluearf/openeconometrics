"""Saved separable multivariate hypotheses and bounded one-way sufficient fits.

The hypothesis is L B M = C, with H = D' (L V L')^-1 D and
E_M = M' E M. Marginal intervals use the complete matrix-normal covariance
and residual Student t law; they are pointwise, not familywise intervals.

Primary references:
https://www.statsmodels.org/stable/_modules/statsmodels/multivariate/multivariate_ols.html
https://support.sas.com/documentation/cdl/en/statug/65328/HTML/default/statug_glm_syntax11.htm
https://svn.r-project.org/R/trunk/src/library/stats/R/mlm.R
The existing native/R Hotelling-Lawley F convention is retained, including
its difference from the refined Statsmodels approximation for higher ranks.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import itertools
import json
import math
from numbers import Integral, Real
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.multivariate import common as mc
from openecon.econometrics.postest.index_codec import decode, encode
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.engines.inference import critical_value
from openecon.resources import plan_workspace
from . import common as c

MAX_DESIGN = MAX_OUTCOMES = 128
MAX_TARGETS = 256
MAX_GROUPS = 64
MAX_ROWS = 100_000
MAX_WORK = 250_000_000
MAX_COUNT = 2**53
_SCHEMA = "openecon.manova.geometry.v1"
_EPS = torch.finfo(torch.float64).eps
_TEST_COLUMNS = ["test", "value", "statistic", "df1", "df2", "p_value",
                 "partial_eta_squared", "f_type"]
_ESTIMATE_COLUMNS = ["contrast", "transform", "estimate", "std_error", "null",
                     "statistic", "df", "p_value", "ci_low", "ci_high"]


def _digest(state):
    try:
        return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False,
                                        separators=(",", ":")).encode()).hexdigest()
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise AnalysisError("invalid_state", "Multivariate state must be finite portable JSON.") from None


def _finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise AnalysisError("numerical_failure", "Multivariate arithmetic exceeds finite float64 precision; rescale.")


def _plan(k, p, *, rows=0, groups=0, r=0, d=0):
    if not 1 <= k <= MAX_DESIGN or not 1 <= p <= MAX_OUTCOMES \
            or rows > MAX_ROWS or groups > MAX_GROUPS or r*d > MAX_TARGETS:
        raise AnalysisError("workspace_limit", "Multivariate options admit 128 design columns/outcomes, "
                            "64 groups, 100000 resident rows and 256 joint targets.")
    work = rows*(p*p + 4*p) + (groups+2)*p**3 + k**3 + r*k*p + d*p*p + r*r*k
    if work > MAX_WORK:
        raise AnalysisError("work_limit", "Multivariate moment/projection work exceeds 250 million operations.")
    return plan_workspace("multivariate saved geometry and hypotheses", {
        "resident_selected_rows": rows*(p+3)*96,
        "coefficient_bread_error_and_JSON": (k*p+k*k+p*p)*96,
        "group_moments": groups*p*p*64,
        "projection_and_joint_covariance": (r*d+r*r+d*d+(r*d)**2)*96,
    }).record()


def _shape(value, name):
    if isinstance(value, torch.Tensor) and value.device.type != "cpu":
        raise AnalysisError("unsupported_device", f"{name} requires resident CPU values.")
    try:
        shape = tuple(value.shape) if hasattr(value, "shape") else (len(value), len(value[0]))
    except (TypeError, IndexError, AttributeError):
        raise AnalysisError("invalid_matrix", f"{name} must be a two-dimensional real matrix.") from None
    if len(shape) != 2:
        raise AnalysisError("invalid_matrix", f"{name} must be a two-dimensional real matrix.")
    return shape


def _matrix(value, shape, name):
    if _shape(value, name) != shape:
        raise AnalysisError("invalid_matrix", f"{name} must have shape {shape}.")
    if isinstance(value, torch.Tensor):
        if value.is_complex() or value.dtype == torch.bool:
            raise AnalysisError("invalid_matrix", f"{name} must exclude booleans and complex values.")
        raw = value.tolist()
    elif isinstance(value, pd.DataFrame):
        raw = value.to_numpy().tolist()
    else:
        raw = value.tolist() if hasattr(value, "tolist") else value
    try:
        if any(len(row) != shape[1] for row in raw) or any(
            isinstance(item, (bool, complex)) or not isinstance(item, Real)
            or not math.isfinite(float(item)) for row in raw for item in row
        ):
            raise ValueError("nonreal matrix")
        result = torch.as_tensor(raw, dtype=torch.float64).clone()
    except (TypeError, ValueError, RuntimeError, OverflowError):
        raise AnalysisError("invalid_matrix", f"{name} must contain finite real numbers, excluding booleans.") from None
    _finite(result)
    return result


def _vector(value, length, name):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise AnalysisError("unsupported_device", f"{name} requires CPU values.")
        if value.ndim != 1:
            raise AnalysisError("invalid_state", f"{name} must have {length} real entries.")
        value = value.tolist()
    try:
        if len(value) != length:
            raise ValueError("dimension")
        return _matrix([list(value)], (1, length), name)[0]
    except (TypeError, ValueError):
        raise AnalysisError("invalid_state", f"{name} must have {length} real entries.") from None


def _covariance(matrix, name, *, positive=False, rank_limit=None):
    diagonal = matrix.diagonal()
    if bool((diagonal < 0).any()):
        raise AnalysisError("invalid_covariance", f"{name} has negative variance.")
    present = diagonal > 0
    if bool((matrix[~present] != 0).any()) or bool((matrix[:, ~present] != 0).any()):
        raise AnalysisError("invalid_covariance", f"{name} has nonzero covariance for a zero variance.")
    if not bool(present.any()):
        if positive:
            raise AnalysisError("singular_error_matrix", f"{name} must be positive definite.")
        return 0
    scales = diagonal[present].sqrt()
    correlation = matrix[present][:, present]/torch.outer(scales, scales)
    _finite(correlation)
    if float((correlation-correlation.T).abs().max()) > 1e-12:
        raise AnalysisError("invalid_covariance", f"{name} must be symmetric; it is not repaired.")
    spectrum = torch.linalg.eigvalsh((correlation+correlation.T)/2)
    p = len(matrix)
    if float(spectrum.min()) < -1e-12*p:
        raise AnalysisError("invalid_covariance", f"{name} must be positive semidefinite.")
    rank = int((spectrum > 8*p*_EPS).sum())
    if rank_limit is not None and rank > rank_limit:
        raise AnalysisError("invalid_covariance", f"{name} rank is impossible for its sample count.")
    if positive and (not bool(present.all()) or float(spectrum.min()) <= 1e-11*float(spectrum.max())):
        raise AnalysisError("singular_error_matrix", f"{name} is singular or unresolved at float64 precision.")
    return rank


def _count(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= MAX_COUNT:
        raise AnalysisError("invalid_state", f"{name} must be an exact integer between {minimum} and 2**53.")
    return int(value)


def _names(value, length, prefix):
    if value is None:
        return [f"{prefix}[{i+1}]" for i in range(length)]
    names = c.name_list(value, prefix)
    if len(names) != length:
        raise AnalysisError("invalid_spec", f"{prefix} must supply {length} unique ordered names.")
    return names


def _typed_names(names):
    return [encode(value) for value in names]


def _state(coefficients, offsets, origin, location, bread, error, n, names, columns,
           *, dropped=0, input_kind, design_metadata, oneway=None):
    state = {
        "schema": _SCHEMA, "n": int(n), "df_resid": int(n)-len(columns),
        "n_missing": int(dropped), "input_kind": input_kind,
        "design_columns": list(columns), "typed_design_columns": _typed_names(columns),
        "outcome_columns": list(names), "typed_outcome_columns": _typed_names(names),
        "coefficients": coefficients.tolist(), "coefficient_offsets": offsets.tolist(),
        "outcome_origin": origin.tolist(), "location_design": location.tolist(),
        "bread": bread.tolist(), "residual_sscp": error.tolist(),
        "design_metadata": design_metadata, "oneway": oneway,
    }
    state["sha256"] = _digest(state)
    return state


@resident_cpu
def save_model_geometry(model, outcomes, *, dropped=0, input_kind="resident_glm"):
    """Bounded raw-coordinate geometry, without changing any legacy test table."""
    k, p = model.k, model.m
    if k > MAX_DESIGN or p > MAX_OUTCOMES or model.n > MAX_COUNT:
        return None
    _plan(k, p)
    offsets = model.uncentre @ model.beta
    origin = model.shift.clone()
    location = torch.zeros(k, dtype=torch.float64)
    location[model.columns["Intercept"][0]] = 1.0
    coefficients = offsets + torch.outer(location, origin)
    bread = model.uncentre @ model.full.xtx_inv @ model.uncentre.T
    bread = (bread+bread.T)/2
    error = model.error.clone()
    _finite(offsets, coefficients, bread, error)
    columns = []
    blocks = []
    for term in model.terms:
        indices = model.columns[term.name]
        columns.extend([term.name] if len(indices) == 1
                       else [f"{term.name}[{j+1}]" for j in range(len(indices))])
        blocks.append({"term": term.name, "columns": list(indices)})
    metadata = {
        "basis": "raw sum-to-zero coded design; covariates in original coordinates",
        "blocks": blocks,
        "typed_factor_levels": {name: _typed_names(levels) for name, levels in model.levels.items()},
        "factor_codings": {name: value.tolist() for name, value in model.codings.items()},
        "covariate_centres": {name: float(value) for name, value in model.centres.items()},
    }
    return _state(coefficients, offsets, origin, location, bread, error, model.n,
                  outcomes, columns, dropped=dropped, input_kind=input_kind,
                  design_metadata=metadata)


def _validate_metadata(metadata, k):
    if not isinstance(metadata, dict) or set(metadata) != {
        "basis", "blocks", "typed_factor_levels", "factor_codings", "covariate_centres"
    } or not isinstance(metadata["basis"], str) or metadata["basis"] not in {
        "raw sum-to-zero coded design; covariates in original coordinates", "one-way group cell means"
    }:
        raise AnalysisError("invalid_state", "Saved design metadata is invalid.")
    blocks = metadata["blocks"]
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= k:
        raise AnalysisError("invalid_state", "Saved design blocks are invalid.")
    try:
        terms = [block["term"] for block in blocks]
        indices = [i for block in blocks for i in block["columns"]]
        if any(set(block) != {"term", "columns"} for block in blocks) \
                or any(not isinstance(term, str) or not term for term in terms) or len(set(terms)) != len(terms) \
                or indices != list(range(k)) or any(type(i) is not int for i in indices):
            raise ValueError("design blocks")
        levels, codings, centres = (metadata[name] for name in
                                   ("typed_factor_levels", "factor_codings", "covariate_centres"))
        if not all(isinstance(value, dict) for value in (levels, codings, centres)) \
                or set(levels) != set(codings):
            raise ValueError("factor metadata")
        for name, labels in levels.items():
            if not isinstance(name, str) or not isinstance(labels, list) or not 2 <= len(labels) <= MAX_DESIGN+1 \
                    or len(set(labels)) != len(labels) or any(not isinstance(label, str) for label in labels):
                raise ValueError("typed factor levels")
            if any(encode(decode(label)) != label for label in labels):
                raise ValueError("noncanonical typed factor levels")
            coding = _matrix(codings[name], (len(labels), len(labels)-1), "saved factor coding")
            expected = torch.cat((torch.eye(len(labels)-1, dtype=torch.float64),
                                  -torch.ones(1, len(labels)-1, dtype=torch.float64)))
            if not torch.equal(coding, expected):
                raise ValueError("sum coding")
        for name, value in centres.items():
            if not isinstance(name, str) or not name:
                raise ValueError("centre name")
            c.check_number(value, "saved covariate centre")
    except (KeyError, TypeError, ValueError):
        raise AnalysisError("invalid_state", "Saved ordered design, coding or centering metadata is invalid.") from None
    return [block["term"] if len(block["columns"]) == 1 else f"{block['term']}[{j+1}]"
            for block in blocks for j in range(len(block["columns"]))]


def _validate(result):
    if not isinstance(result, TableSet):
        raise AnalysisError("invalid_result", "Supply a saved MANOVA TableSet with full fit geometry.")
    state = result.attrs.get("manova_contrast_state")
    if not isinstance(state, dict):
        raise AnalysisError("unsupported_saved_geometry", "This MANOVA result has no bounded saved fit geometry; refit a supported design.")
    expected = {"schema", "n", "df_resid", "n_missing", "input_kind", "design_columns",
                "typed_design_columns", "outcome_columns", "typed_outcome_columns",
                "coefficients", "coefficient_offsets", "outcome_origin", "location_design",
                "bread", "residual_sscp", "design_metadata", "oneway", "sha256"}
    if set(state) != expected or state["schema"] != _SCHEMA:
        raise AnalysisError("invalid_state", "Saved MANOVA schema or fields are invalid.")
    columns, names = state["design_columns"], state["outcome_columns"]
    if not isinstance(columns, list) or not isinstance(names, list):
        raise AnalysisError("invalid_state", "Saved MANOVA ordering is invalid.")
    k, p = len(columns), len(names)
    _plan(k, p)
    for labels, typed in ((columns, state["typed_design_columns"]), (names, state["typed_outcome_columns"])):
        if any(not isinstance(name, str) or not name for name in labels) or len(set(labels)) != len(labels) \
                or typed != _typed_names(labels):
            raise AnalysisError("invalid_state", "Saved MANOVA labels/order contain aliases or invalid identities.")
    n, df = _count(state["n"], "saved sample count"), _count(state["df_resid"], "saved residual df")
    dropped = _count(state["n_missing"], "saved missing count", minimum=0)
    if n-k != df or state["input_kind"] not in {
        "resident_glm", "dataset_glm", "resident_oneway", "resident_frequency_oneway", "group_summary",
        "resident_factorial_frequency", "factorial_cell_summary"
    } or result.attrs.get("n") != n or result.attrs.get("df_resid") != df \
            or result.attrs.get("outcomes") != names or result.attrs.get("n_missing") != dropped:
        raise AnalysisError("invalid_state", "Saved MANOVA sample, df or outer metadata disagree.")
    if _validate_metadata(state["design_metadata"], k) != columns:
        raise AnalysisError("invalid_state", "Saved design labels disagree with ordered design blocks.")
    for key, rows, width in (("coefficients", k, p), ("coefficient_offsets", k, p),
                             ("bread", k, k), ("residual_sscp", p, p)):
        _shape(state[key], f"saved {key}")
        if len(state[key]) != rows or any(len(row) != width for row in state[key]):
            raise AnalysisError("invalid_state", "Saved MANOVA matrix dimensions disagree.")
    if _digest({key: value for key, value in state.items() if key != "sha256"}) != state["sha256"]:
        raise AnalysisError("invalid_state", "Saved MANOVA integrity digest disagrees.")
    beta = _matrix(state["coefficients"], (k, p), "saved coefficients")
    offsets = _matrix(state["coefficient_offsets"], (k, p), "saved coefficient offsets")
    origin = _vector(state["outcome_origin"], p, "saved outcome origin")
    location = _vector(state["location_design"], k, "saved location design")
    bread = _matrix(state["bread"], (k, k), "saved bread")
    error = _matrix(state["residual_sscp"], (p, p), "saved residual SSCP")
    if not torch.equal(beta, offsets+torch.outer(location, origin)):
        raise AnalysisError("invalid_state", "Saved raw coefficients disagree with their stable representation.")
    _covariance(bread, "saved between-design bread", positive=True)
    _covariance(error, "saved residual SSCP", rank_limit=df)
    if state["oneway"] is not None:
        _validate_oneway(state, beta, offsets, origin, location, bread, error)
    elif state["input_kind"] in {"resident_oneway", "resident_frequency_oneway", "group_summary"}:
        raise AnalysisError("invalid_state", "Saved one-way fit has no group sufficient state.")
    else:
        if "Intercept" not in columns:
            raise AnalysisError("invalid_state", "Saved raw design has no intercept.")
        expected_location = torch.zeros(k, dtype=torch.float64)
        expected_location[columns.index("Intercept")] = 1.0
        if not torch.equal(location, expected_location):
            raise AnalysisError("invalid_state", "Saved raw design location is invalid.")
    if state["input_kind"] in {"resident_factorial_frequency", "factorial_cell_summary"}:
        from .manova_factorial import validate_factorial_result
        validate_factorial_result(result, state)
    return state, offsets, origin, location, bread, error


def _validate_oneway(state, beta, offsets, origin, location, bread, error):
    moments = state["oneway"]
    if not isinstance(moments, dict) or set(moments) != {"groups", "typed_groups", "counts", "group_sscp", "sample"}:
        raise AnalysisError("invalid_state", "Saved one-way sufficient state is invalid.")
    groups = _groups(moments["groups"])
    k, p = beta.shape
    if len(groups) != k or moments["typed_groups"] != _typed_names(groups):
        raise AnalysisError("invalid_state", "Saved one-way group order disagrees.")
    counts = _counts(moments["counts"], groups, minimum=1)
    if sum(counts) != state["n"] or len(moments["group_sscp"]) != k \
            or not torch.equal(location, torch.ones(k, dtype=torch.float64)):
        raise AnalysisError("invalid_state", "Saved one-way sample/location disagrees.")
    count_tensor = torch.tensor(counts, dtype=torch.float64)
    if not torch.equal(bread, torch.diag(1/count_tensor)):
        raise AnalysisError("invalid_state", "Saved one-way mean covariance bread disagrees with counts.")
    blocks = []
    for value, count in zip(moments["group_sscp"], counts, strict=True):
        block = _matrix(value, (p, p), "saved group SSCP")
        _covariance(block, "saved group SSCP", rank_limit=count-1)
        blocks.append(block)
    if not torch.allclose(torch.stack(blocks).sum(0), error, rtol=16*_EPS, atol=0):
        raise AnalysisError("invalid_state", "Saved pooled residual SSCP disagrees with group sufficient state.")
    sample = moments["sample"]
    if state["input_kind"] == "group_summary":
        if sample is not None or state["n_missing"] != 0:
            raise AnalysisError("invalid_state", "Declared summaries cannot claim raw sample positions.")
        return
    if not isinstance(sample, dict) or set(sample) != {
        "n_input_rows", "source_positions", "frequencies", "positive_complete_positions", "group_codes"
    }:
        raise AnalysisError("invalid_state", "Saved resident sample provenance is invalid.")
    rows = _count(sample["n_input_rows"], "saved physical input rows")
    _plan(k, p, rows=rows, groups=k, r=k, d=p)
    positions = sample["source_positions"]
    if not isinstance(positions, list) or any(type(i) is not int or not 0 <= i < rows for i in positions) \
            or positions != sorted(set(positions)) or len(positions)+state["n_missing"] != rows:
        raise AnalysisError("invalid_state", "Saved missing/complete row positions disagree.")
    frequencies = sample["frequencies"]
    if not isinstance(frequencies, list) or len(frequencies) != len(positions):
        raise AnalysisError("invalid_state", "Saved complete frequency alignment is invalid.")
    frequencies = [_count(value, "saved frequency", minimum=0) for value in frequencies]
    positive = [i for i, value in enumerate(frequencies) if value > 0]
    codes = sample["group_codes"]
    if sample["positive_complete_positions"] != positive or not isinstance(codes, list) \
            or len(codes) != len(positive) or any(type(code) is not int or not 0 <= code < k for code in codes) \
            or [sum(frequencies[i] for i, code in zip(positive, codes, strict=True) if code == j)
                for j in range(k)] != counts \
            or (state["input_kind"] == "resident_oneway" and any(value != 1 for value in frequencies)):
        raise AnalysisError("invalid_state", "Saved physical rows, group frequencies or replication counts disagree.")
    for j, block in enumerate(blocks):
        _covariance(block, "saved physical group SSCP", rank_limit=sum(code == j for code in codes)-1)


def _full_rank_rows(matrix, name):
    scale = matrix.abs().amax(1)
    if bool((scale == 0).any()):
        raise AnalysisError("rank_deficient_contrast", f"{name} must have full represented rank.")
    spectrum = torch.linalg.svdvals(matrix/scale[:, None])
    if len(spectrum) != len(matrix) or float(spectrum.min()) <= 1e-11*float(spectrum.max()):
        raise AnalysisError("rank_deficient_contrast", f"{name} is rank deficient or numerically unresolved.")


def _right(matrix, transform):
    # Separate the common row level from differences; retain the exact represented
    # coefficient sums, including small nonzero sums, rather than canonicalizing.
    try:
        sums = torch.tensor([math.fsum(column) for column in transform.T.tolist()], dtype=torch.float64)
    except (ValueError, OverflowError):
        raise AnalysisError("numerical_failure", "The represented response transform exceeds finite float64 precision.") from None
    result = (matrix-matrix[:, :1]) @ transform + torch.outer(matrix[:, 0], sums)
    _finite(result)
    return result


def _left(matrix, values):
    try:
        result = torch.tensor([[math.fsum(a*b for a, b in zip(row, column, strict=True))
                                for column in values.T.tolist()] for row in matrix.tolist()],
                              dtype=torch.float64)
    except (ValueError, OverflowError):
        raise AnalysisError("numerical_failure", "The represented contrast exceeds finite float64 arithmetic.") from None
    _finite(result)
    return result


def _projection(error, transform):
    projected = transform.T @ error @ transform
    _finite(projected)
    # Existing saved RM geometry has no original residual subject rows. A
    # positive but cancellation-dominated projection cannot be certified by an
    # eigenvalue test alone; explicitly refuse that geometry.
    absolute = transform.abs().T @ error.abs() @ transform.abs()
    _finite(absolute)
    roundoff = 8*len(error)*_EPS*absolute.diagonal()
    if bool((projected.diagonal() <= 128*roundoff).any()):
        raise AnalysisError("unresolved_precision", "Projected residual covariance is unresolved against its formation roundoff; rescale/refit a stable response basis.")
    projected = (projected+projected.T)/2
    _covariance(projected, "projected residual SSCP", positive=True)
    return projected


def _joint(state, offsets, origin, location, bread, error, L, M, null, alpha,
           contrast_names, transform_names, *, procedure):
    k, p = offsets.shape
    r, width = _shape(L, "L")
    if width != k or not 1 <= r <= k:
        raise AnalysisError("invalid_contrast", "L needs one or more independent rows in saved design-column order.")
    if M is None:
        d = p
    else:
        height, d = _shape(M, "M")
        if height != p or not 1 <= d <= p:
            raise AnalysisError("invalid_contrast", "M needs independent columns in saved outcome/cell order.")
    plan = _plan(k, p, r=r, d=d)
    if isinstance(L, pd.DataFrame):
        if L.columns.has_duplicates or list(L.columns) != state["design_columns"]:
            raise AnalysisError("invalid_contrast", "L column labels must equal saved design order.")
        if contrast_names is None:
            contrast_names = list(L.index)
    if isinstance(M, pd.DataFrame):
        if M.index.has_duplicates or list(M.index) != state["outcome_columns"]:
            raise AnalysisError("invalid_contrast", "M row labels must equal saved outcome/cell order.")
        if transform_names is None:
            transform_names = list(M.columns)
    left = _matrix(L, (r, k), "L")
    m = torch.eye(p, dtype=torch.float64) if M is None else _matrix(M, (p, d), "M")
    _full_rank_rows(left, "L")
    _full_rank_rows(m.T, "M")
    labels = _names(contrast_names, r, "L")
    transforms = (list(state["outcome_columns"]) if M is None and transform_names is None
                  else _names(transform_names, d, "M"))
    if isinstance(null, pd.DataFrame) and (list(null.index) != labels or list(null.columns) != transforms):
        raise AnalysisError("invalid_contrast", "Null labels must match contrast and transform order.")
    constant = torch.zeros((r, d), dtype=torch.float64) if null is None else _matrix(null, (r, d), "null")
    df = state["df_resid"]
    if df < d or (min(r, d) > 1 and df < d+1):
        raise AnalysisError("insufficient_observations", "Projected MANOVA requires residual df >= outcome dimension, and strictly greater for higher-rank hypotheses so every classical F law is defined.")
    a = left @ bread @ left.T
    _finite(a)
    a = (a+a.T)/2
    _covariance(a, "contrast covariance bread", positive=True)
    projected = _projection(error, m)
    estimate = _left(left, _right(offsets, m))
    origin_transformed = _right(origin[None, :], m)[0]
    estimate += torch.outer(_left(left, location[:, None])[:, 0], origin_transformed)
    delta = estimate-constant
    _finite(estimate, delta)
    hypothesis = delta.T @ torch.linalg.solve(a, delta)
    hypothesis = (hypothesis+hypothesis.T)/2
    sigma = projected/df
    covariance = torch.kron(a.contiguous(), sigma.contiguous())
    _finite(hypothesis, sigma, covariance)
    _covariance(hypothesis, "hypothesis SSCP", rank_limit=min(r, d))
    _covariance(covariance, "complete target covariance", positive=True)
    se = covariance.diagonal().sqrt().reshape(r, d)
    if procedure == "rm_mtest":
        # Legacy original-cell coefficients have already been rounded after
        # restoring their common response level. Stable subsequent differences
        # cannot recover those digits. Refuse a requested target if that saved
        # representation cannot resolve it against its estimate/inference scale.
        raw_beta = offsets+torch.outer(location, origin)
        absolute_target = left.abs() @ raw_beta.abs() @ m.abs()
        rounding_bound = 8*(k+p)*_EPS*absolute_target
        _finite(rounding_bound)
        if bool((rounding_bound > 1e-8*torch.maximum(estimate.abs(), se)).any()):
            raise AnalysisError("unresolved_precision", "Saved original-cell RM coefficients cannot resolve this target against their float64 rounding; refit in a stable response basis.")
    statistic = delta/se
    critical = critical_value(alpha, df)
    lower, upper = estimate-critical*se, estimate+critical*se
    _finite(se, statistic, lower, upper)
    from .manova import multivariate_tests
    tests = multivariate_tests(hypothesis, projected, r, df)
    chol = torch.linalg.cholesky(projected)
    half = torch.linalg.solve_triangular(chol, hypothesis, upper=False)
    whitened = torch.linalg.solve_triangular(chol, half.T, upper=False)
    roots = torch.linalg.eigvalsh((whitened+whitened.T)/2).clamp_min(0).flip(0)
    targets = [f"target[{i+1}]" for i in range(r*d)]
    target_order = [[labels[i], transforms[j]] for i in range(r) for j in range(d)]
    rows = [[labels[i], transforms[j], float(estimate[i, j]), float(se[i, j]),
             float(constant[i, j]), float(statistic[i, j]), df,
             c.t_two_sided(float(statistic[i, j]), df), float(lower[i, j]), float(upper[i, j])]
            for i in range(r) for j in range(d)]
    specification = {"L": left.tolist(), "M": m.tolist(), "null": constant.tolist(),
                     "contrast_names": labels, "transform_names": transforms,
                     "target_order": target_order, "covariance_order": "row-major: L row then M column"}
    specification["sha256"] = _digest(specification)
    output = TableSet({
        "estimates": table(rows, columns=_ESTIMATE_COLUMNS),
        "target_covariance": table(covariance.tolist(), index=targets, columns=targets),
        "target_order": table([[target, *labels_] for target, labels_ in zip(targets, target_order, strict=True)],
                              columns=["target", "contrast", "transform"]),
        "hypothesis_sscp": table(hypothesis.tolist(), index=transforms, columns=transforms),
        "error_sscp": table(projected.tolist(), index=transforms, columns=transforms),
        "roots": table([[i+1, float(root)] for i, root in enumerate(roots)], columns=["root", "eigenvalue"]),
        "multivariate": table(tests, columns=_TEST_COLUMNS),
        "L": table(left.tolist(), index=labels, columns=state["design_columns"]),
        "M": table(m.tolist(), index=state["outcome_columns"], columns=transforms),
        "null": table(constant.tolist(), index=labels, columns=transforms),
    }, title="Saved general multivariate hypothesis", procedure=procedure,
        n=state["n"], n_missing=state["n_missing"], df_resid=df, outcomes=state["outcome_columns"],
        alpha=alpha, precision="float64", device="cpu", resource_plan=plan,
        contrast_spec=specification, source_state_sha256=state["sha256"],
        target_order=target_order, covariance_order=specification["covariance_order"],
        pointwise_intervals=True, familywise_intervals=False, sphericity_required=False,
        inference="fixed predeclared separable hypothesis under independent Gaussian rows/subjects and common unrestricted outcome covariance",
        hotelling_f_convention="existing native/R approximation; differs from refined Statsmodels for higher ranks")
    return saved_summary(output)


@resident_cpu
def manova_contrast(result: TableSet, L, *, M=None, null=None, alpha=0.05,
                    contrast_names=None, transform_names=None):
    """Test saved full-rank L B M = null, with complete covariance and pointwise t CI.

    L follows saved raw sum-coded design order. M follows outcome order; its
    default is the identity. All original fit outcomes/order/df/covariance and
    the finite state digest are revalidated. Multivariate F laws retain their
    exact/approximate/upper-bound labels. Intervals are not familywise and do
    not include data-driven contrast/model selection uncertainty.
    """
    alpha = c.check_alpha(alpha)
    state, offsets, origin, location, bread, error = _validate(result)
    output = _joint(state, offsets, origin, location, bread, error, L, M, null, alpha,
                    contrast_names, transform_names, procedure="manova_contrast")
    output.attrs["manova_contrast_state"] = state
    output.attrs["manova_contrast_available"] = True
    if "factorial_moment_state" in result.attrs:
        output.attrs["factorial_moment_state"] = result.attrs["factorial_moment_state"]
        from .manova_factorial import stabilize_saved_query
        stabilize_saved_query(output)
    return saved_summary(output)


def _rm_geometry(result):
    if not isinstance(result, TableSet) or not result.attrs.get("within"):
        raise AnalysisError("invalid_result", "Supply a native saved rm_anova TableSet.")
    state = result.attrs.get("rm_contrast_state")
    if not isinstance(state, dict):
        raise AnalysisError("unsupported_saved_geometry", "The RM result has no supported saved original-cell geometry.")
    try:
        expected_fields = {"schema", "df_resid", "n_subjects", "within", "cell_order", "typed_cell_order",
                           "between_design_columns", "between_levels", "between_codings", "coefficients",
                           "bread", "residual_cell_covariance", "target", "assumptions", "resource_plan", "sha256"}
        if set(state) != expected_fields or state["schema"] != "openecon.rm.contrast.v1":
            raise ValueError("schema")
        columns, cells = state["between_design_columns"], state["cell_order"]
        k, p = len(columns), len(cells)
        _plan(k, p)
        n, df = _count(state["n_subjects"], "saved RM subject count"), _count(state["df_resid"], "saved RM df")
        dropped = _count(result.attrs.get("n_missing", 0), "saved RM missing count", minimum=0)
        if n-k != df or state["within"] != result.attrs["within"] \
                or result.attrs.get("n_subjects") != n or result.attrs.get("df_error_between") != df \
                or result.attrs.get("n") not in {n, n*p} \
                or result.attrs.get("within_cells", p) != p \
                or any(not isinstance(name, str) or not name for name in columns) \
                or len(set(columns)) != k or len(set(state["typed_cell_order"])) != p \
                or state["typed_cell_order"] != [encode(tuple(cell)) for cell in cells] \
                or any(len(cell) != len(state["within"]) for cell in cells):
            raise ValueError("ordering or sample")
        within = state["within"]
        if len(set(within)) != len(within) or any(not isinstance(name, str) or not name for name in within):
            raise ValueError("within labels")
        levels = [list(dict.fromkeys(cell[j] for cell in cells)) for j in range(len(within))]
        if cells != [list(cell) for cell in itertools.product(*levels)]:
            raise ValueError("complete ordered within cells")
        between = result.attrs.get("between", [])
        if set(state["between_levels"]) != set(between) or set(state["between_codings"]) != set(between) \
                or math.prod(len(state["between_levels"][name]) for name in between) != k:
            raise ValueError("between geometry")
        for name in between:
            levels_ = state["between_levels"][name]
            if len(set(_typed_names(levels_))) != len(levels_) or len(levels_) < 2:
                raise ValueError("between levels")
            coding = _matrix(state["between_codings"][name], (len(levels_), len(levels_)-1), "saved RM factor coding")
            expected = torch.cat((torch.eye(len(levels_)-1, dtype=torch.float64),
                                  -torch.ones(1, len(levels_)-1, dtype=torch.float64)))
            if not torch.equal(coding, expected):
                raise ValueError("between sum coding")
        from .glm import build_terms
        expected_columns = []
        for term in build_terms(between, [], "full"):
            width = math.prod(len(state["between_levels"][name])-1 for name in term.factors)
            expected_columns.extend([term.name] if width == 1 else
                                    [f"{term.name}[{j+1}]" for j in range(width)])
        if expected_columns != columns:
            raise ValueError("between design aliases/order")
        if _digest({key: value for key, value in state.items() if key != "sha256"}) != state["sha256"]:
            raise ValueError("digest")
        beta = _matrix(state["coefficients"], (k, p), "saved RM coefficients")
        bread = _matrix(state["bread"], (k, k), "saved RM bread")
        covariance = _matrix(state["residual_cell_covariance"], (p, p), "saved RM covariance")
        _covariance(bread, "saved RM bread", positive=True)
        _covariance(covariance, "saved RM covariance", rank_limit=df)
        names = [f"cell[{i+1}]" for i in range(p)]
        # Retain the entire original state digest; cell labels/ordering are
        # reported separately, not converted into potentially aliased text.
        geometry = {"design_columns": columns, "outcome_columns": names, "n": n,
                    "n_missing": dropped, "df_resid": df, "sha256": state["sha256"]}
        origin = beta[0].clone()
        location = torch.zeros(k, dtype=torch.float64)
        location[0] = 1
        offsets = beta-torch.outer(location, origin)
    except (KeyError, TypeError, ValueError, AttributeError):
        raise AnalysisError("invalid_state", "Saved original-cell RM state, sample or ordering is invalid.") from None
    if result.attrs.get("rm_moment_required") or result.attrs.get("procedure") in {"rm_anova_fweight", "rm_anova_summary"}:
        if "rm_moment_state" not in result.attrs:
            raise AnalysisError("invalid_state", "This RM moment result requires its complete saved sample moments.")
    if "rm_moment_state" in result.attrs:
        from .rm_moments import validate_rm_result
        validate_rm_result(result, state)
    return state, geometry, offsets, origin, location, bread, covariance*df


@resident_cpu
def rm_mtest(result: TableSet, L, *, M, null=None, alpha=0.05,
             contrast_names=None, transform_names=None):
    """Joint saved between-design × original-cell RM hypothesis, without refitting.

    L follows between_design_columns; M rows follow the recorded cell_order.
    Independent Gaussian subjects with common unrestricted cell covariance are
    required. Sphericity is unnecessary. Weighted/incomplete subjects, selected
    contrasts and heteroscedastic group covariance laws are outside this scope.
    Numerically unresolved covariance projections are explicitly refused.
    The legacy original-cell coefficient representation also requires its
    rounding bound 8*(k+p)*eps*(abs(L) @ abs(B) @ abs(M)) to be no greater
    than 1e-8*max(abs(estimate), SE) for every requested marginal target.
    """
    alpha = c.check_alpha(alpha)
    original, state, offsets, origin, location, bread, error = _rm_geometry(result)
    output = _joint(state, offsets, origin, location, bread, error, L, M, null, alpha,
                    contrast_names, transform_names, procedure="rm_mtest")
    output.attrs["cell_order"] = original["cell_order"]
    output.attrs["typed_cell_order"] = original["typed_cell_order"]
    output.attrs["within"] = original["within"]
    output.attrs["between"] = result.attrs.get("between", [])
    output.attrs["n_subjects"] = state["n"]
    output.attrs["df_error_between"] = state["df_resid"]
    output.attrs["rm_contrast_state"] = original
    if "rm_moment_state" in result.attrs:
        output.attrs["rm_moment_state"] = result.attrs["rm_moment_state"]
        output.attrs["rm_moment_required"] = True
        from .manova_factorial import stabilize_saved_query
        stabilize_saved_query(output)
    return saved_summary(output)


def _groups(values):
    if isinstance(values, (str, bytes)):
        raise AnalysisError("invalid_groups", "Supply ordered finite scalar group labels.")
    try:
        values = [mc.label(value) for value in values]
    except (TypeError, ValueError, OverflowError):
        raise AnalysisError("invalid_groups", "Groups must be finite JSON scalar labels.") from None
    if not 2 <= len(values) <= MAX_GROUPS or any(
        not isinstance(value, (str, bool, int, float)) or
        (isinstance(value, float) and not math.isfinite(value)) for value in values
    ) or len(set(_typed_names(values))) != len(values) or len(set(map(str, values))) != len(values) \
            or len(set(values)) != len(values):
        raise AnalysisError("invalid_groups", "Groups must be 2..64 unique finite scalar labels without aliases.")
    return values


def _counts(values, groups, *, minimum):
    if isinstance(values, pd.Series):
        if values.index.has_duplicates or list(values.index) != groups:
            raise AnalysisError("invalid_spec", "Count order must equal group mean order.")
        values = values.tolist()
    elif isinstance(values, Mapping):
        if set(values) != set(groups):
            raise AnalysisError("invalid_spec", "Counts must label every group exactly once.")
        values = [values[group] for group in groups]
    elif isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise AnalysisError("unsupported_device", "Counts require CPU values.")
        values = values.tolist()
    try:
        if len(values) != len(groups):
            raise ValueError("counts length")
        counts = [_count(value, "group count", minimum=minimum) for value in values]
    except (TypeError, ValueError):
        raise AnalysisError("invalid_spec", "Supply one exact integer count per group.") from None
    if sum(counts) > MAX_COUNT:
        raise AnalysisError("invalid_spec", "Total sample count exceeds exact float64 integer precision.")
    return counts


def _oneway(origin, offsets, scatters, counts, groups, names, alpha, *, attrs):
    k, p = offsets.shape
    plan = _plan(k, p, groups=k, r=k, d=p)
    n, df = sum(counts), sum(counts)-k
    if df < p or (min(k-1, p) > 1 and df < p+1):
        raise AnalysisError("insufficient_observations", "One-way MANOVA needs df >= outcome dimension, and strictly greater when both hypothesis ranks exceed one.")
    counts_tensor = torch.tensor(counts, dtype=torch.float64)
    mean_offset = (counts_tensor[:, None]*offsets).sum(0)/n
    deviations = offsets-mean_offset
    error = torch.stack(scatters).sum(0)
    hypothesis = deviations.T @ (counts_tensor[:, None]*deviations)
    error, hypothesis = (error+error.T)/2, (hypothesis+hypothesis.T)/2
    sigma = error/df
    bread = torch.diag(1/counts_tensor)
    means = offsets+origin
    _finite(error, hypothesis, sigma, means)
    _covariance(error, "pooled residual SSCP", positive=True)
    covariance = torch.kron(bread, sigma)
    _finite(covariance)
    se = covariance.diagonal().sqrt().reshape(k, p)
    critical = critical_value(alpha, df)
    lower, upper = means-critical*se, means+critical*se
    _finite(se, lower, upper)
    from .manova import multivariate_tests
    tests = multivariate_tests(hypothesis, error, k-1, df)
    chol = torch.linalg.cholesky(error)
    half = torch.linalg.solve_triangular(chol, hypothesis, upper=False)
    whitened = torch.linalg.solve_triangular(chol, half.T, upper=False)
    roots = torch.linalg.eigvalsh((whitened+whitened.T)/2).clamp_min(0).flip(0)
    columns = [f"group[{i+1}]" for i in range(k)]
    state = _state(means, offsets, origin, torch.ones(k, dtype=torch.float64), bread, error, n, names, columns,
                   dropped=attrs["n_missing"], input_kind=attrs["input_kind"],
                   design_metadata={"basis": "one-way group cell means", "blocks": [
                       {"term": columns[i], "columns": [i]} for i in range(k)],
                       "typed_factor_levels": {}, "factor_codings": {}, "covariate_centres": {}},
                   oneway={"groups": groups, "typed_groups": _typed_names(groups), "counts": counts,
                           "group_sscp": [value.tolist() for value in scatters],
                           "sample": attrs.pop("sample", None)})
    target_order = [[groups[i], names[j]] for i in range(k) for j in range(p)]
    targets = [f"target[{i+1}]" for i in range(k*p)]
    output = TableSet({
        "multivariate": table(tests, columns=_TEST_COLUMNS),
        "univariate": table([[names[j], float(hypothesis[j, j]), k-1, float(error[j, j]), df,
                              float(hypothesis[j, j]/(k-1)/(error[j, j]/df)),
                              c.f_upper(float(hypothesis[j, j]/(k-1)/(error[j, j]/df)), k-1, df)]
                             for j in range(p)],
                            columns=["outcome", "hypothesis_ss", "df1", "error_ss", "df2", "statistic", "p_value"]),
        "means": table([[groups[i], names[j], counts[i], float(means[i, j]), float(se[i, j]),
                         df, float(lower[i, j]), float(upper[i, j])]
                        for i in range(k) for j in range(p)],
                       columns=["group", "outcome", "n", "estimate", "std_error", "df", "ci_low", "ci_high"]),
        "counts": table([[group, count] for group, count in zip(groups, counts, strict=True)], columns=["group", "n"]),
        "group_sscp": table([[groups[i], names[j], names[h], float(scatters[i][j, h])]
                             for i in range(k) for j in range(p) for h in range(p)],
                            columns=["group", "outcome", "other_outcome", "sscp"]),
        "coefficients": table(means.tolist(), index=columns, columns=names),
        "bread": table(bread.tolist(), index=columns, columns=columns),
        "residual_covariance": table(sigma.tolist(), index=names, columns=names),
        "coefficient_covariance": table(covariance.tolist(), index=targets, columns=targets),
        "target_order": table([[target, *labels] for target, labels in zip(targets, target_order, strict=True)],
                              columns=["target", "group", "outcome"]),
        "hypothesis_sscp": table(hypothesis.tolist(), index=names, columns=names),
        "error_sscp": table(error.tolist(), index=names, columns=names),
        "roots": table([[i+1, float(root)] for i, root in enumerate(roots)], columns=["root", "eigenvalue"]),
    }, title="One-way multivariate analysis of variance", procedure="manova_oneway",
        n=n, df_resid=df, outcomes=names, groups=groups, alpha=alpha,
        precision="float64", device="cpu", pointwise_intervals=True, familywise_intervals=False,
        covariance_order="row-major: group then outcome", target_order=target_order,
        manova_contrast_state=state, manova_contrast_available=True, resource_plan=plan,
        inference="independent Gaussian observations with common unrestricted outcome covariance; declared frequencies denote independent replications",
        hotelling_f_convention="existing native/R approximation", **attrs)
    return saved_summary(output)


@resident_cpu
def manova_summary(group_means: pd.DataFrame, group_covariances, counts, *, alpha=0.05):
    """One-way MANOVA from declared group means, counts and n-1 sample covariance.

    Group means are a labelled DataFrame. Covariances/count mappings must label
    every group exactly; covariance outcome labels follow the mean columns.
    Individual covariances may be PSD but their rank cannot exceed n_g-1; the
    pooled residual matrix must be PD. No raw rows or missing-data pattern are
    inferred from summaries. Common-covariance Gaussian inference is conditional
    on the declared independent samples, with complete persisted sufficient state.
    """
    alpha = c.check_alpha(alpha)
    if not isinstance(group_means, pd.DataFrame) or group_means.index.has_duplicates \
            or group_means.columns.has_duplicates:
        raise AnalysisError("invalid_spec", "Group means must be a DataFrame with unique ordered group/outcome labels.")
    names = c.name_list(list(group_means.columns), "outcomes")
    groups = _groups(group_means.index)
    k, p = len(groups), len(names)
    _plan(k, p, groups=k, r=k, d=p)
    counts = _counts(counts, groups, minimum=2)
    if not isinstance(group_covariances, Mapping) or set(group_covariances) != set(groups):
        raise AnalysisError("invalid_spec", "Group covariances must map every group exactly once.")
    means = _matrix(group_means, (k, p), "group means")
    scatters = []
    for group, count in zip(groups, counts, strict=True):
        value = group_covariances[group]
        if isinstance(value, pd.DataFrame) and (value.index.has_duplicates or value.columns.has_duplicates
                or list(value.index) != names or list(value.columns) != names):
            raise AnalysisError("invalid_spec", "Group covariance labels must equal outcome order.")
        covariance = _matrix(value, (p, p), "group sample covariance")
        _covariance(covariance, "group sample covariance", rank_limit=count-1)
        scatter = covariance*(count-1)
        _finite(scatter)
        scatters.append(scatter)
    origin = means[0].clone()
    output = _oneway(origin, means-origin, scatters, counts, groups, names, alpha,
                     attrs={"n_missing": 0, "input_kind": "group_summary", "missing": "declared complete summaries",
                            "weight_type": None, "weights": None, "physical_rows": None,
                            "covariance_divisor": "group n-1"})
    output.attrs["procedure"] = "manova_summary"
    return saved_summary(output)


@resident_cpu
def manova_oneway(data: Any, y, group: str, *, weights=None, weight_type="fweight",
                  missing="drop", alpha=0.05):
    """Resident one-way MANOVA, optionally using exact integer frequency counts.

    Frequency rows are accumulated as anchored group moments without expanding
    the observations. Counts are nonnegative exact integers with total <=2**53;
    zero-frequency rows are excluded after validating complete numeric inputs.
    Missingness is listwise across the group, outcomes and optional weight.
    Other weight types, Dataset inputs, covariates and repeated subjects require
    their separately scoped procedures. Full sufficient geometry is saved.
    """
    alpha = c.check_alpha(alpha)
    c.check_choice(weight_type, "weight_type", ("fweight",))
    c.check_choice(missing, "missing", ("drop", "raise"))
    names, group = c.name_list(y, "y"), c.check_name(group, "group")
    weights = None if weights is None else c.check_name(weights, "weights")
    used = [*names, group, *([] if weights is None else [weights])]
    if len(set(used)) != len(used):
        raise AnalysisError("invalid_spec", "Outcome, group and weight columns must have distinct roles.")
    if isinstance(data, Dataset):
        raise AnalysisError("unsupported_data", "Frequency one-way MANOVA requires resident data.")
    if isinstance(data, Mapping):
        data = {name: data[name] for name in used if name in data}
        sizes = []
        for value in data.values():
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                raise AnalysisError("unsupported_device", "One-way MANOVA inputs require CPU values.")
            try:
                sizes.append(len(value))
            except TypeError:
                raise AnalysisError("invalid_data", "Resident columns must be sized; iterators are not admitted.") from None
        rows = max(sizes, default=0)
    elif isinstance(data, (pd.DataFrame, list, tuple)):
        rows = len(data)
    else:
        raise AnalysisError("invalid_data", "Supply resident rows or sized numeric columns.")
    _plan(1, len(names), rows=rows)
    if isinstance(data, (list, tuple)) and all(isinstance(row, Mapping) for row in data):
        data = [{name: row[name] for name in used if name in row} for row in data]
    numeric = [*names, *([] if weights is None else [weights])]
    frame, keep, dropped = mc.select(data, used, numeric=numeric, missing=missing)
    for name in numeric:
        if pd.api.types.is_bool_dtype(frame[name].dtype):
            raise AnalysisError("invalid_weights" if name == weights else "invalid_matrix", "Boolean outcomes/frequencies are not admitted.")
    if weights is None:
        frequencies = [1]*len(frame)
    else:
        frequencies = []
        for value in frame[weights].array:
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) \
                    or value < 0 or value > MAX_COUNT or value != int(value):
                raise AnalysisError("invalid_weights", "Frequency weights must be nonnegative exact integers no greater than 2**53.")
            frequencies.append(int(value))
    if sum(frequencies) > MAX_COUNT:
        raise AnalysisError("invalid_weights", "Total frequency exceeds exact float64 integer precision.")
    # Validate all complete numeric rows, including zero-frequency rows.
    raw = mc.matrix(frame, names)
    positive = [i for i, value in enumerate(frequencies) if value > 0]
    if not positive:
        raise AnalysisError("empty_sample", "No positive-frequency observations remain.")
    labels = []
    lookup = {}
    codes = []
    for i in positive:
        label = mc.label(frame[group].iloc[i])
        key = encode(label)
        if key not in lookup:
            lookup[key] = len(labels)
            labels.append(label)
            if len(labels) > MAX_GROUPS:
                raise AnalysisError("workspace_limit", "One-way MANOVA admits at most 64 groups.")
        codes.append(lookup[key])
    groups = _groups(labels)
    ordered = sorted(range(len(groups)), key=lambda i: (type(groups[i]).__name__, groups[i]))
    inverse = {old: new for new, old in enumerate(ordered)}
    groups = [groups[i] for i in ordered]
    codes = [inverse[code] for code in codes]
    k, p = len(groups), len(names)
    _plan(k, p, rows=rows, groups=k, r=k, d=p)
    raw = raw[positive]
    w = torch.tensor([frequencies[i] for i in positive], dtype=torch.float64)
    code = torch.tensor(codes, dtype=torch.int64)
    counts = [sum(frequencies[i] for i, current in zip(positive, codes, strict=True) if current == j)
              for j in range(k)]
    count_tensor = torch.tensor(counts, dtype=torch.float64)
    origin = raw[0].clone()
    centred = raw-origin
    offsets = torch.zeros((k, p), dtype=torch.float64).index_add_(0, code, w[:, None]*centred)/count_tensor[:, None]
    offsets += torch.zeros_like(offsets).index_add_(0, code, w[:, None]*(centred-offsets[code]))/count_tensor[:, None]
    deviations = centred-offsets[code]
    scatters = []
    for i in range(k):
        rows_ = code == i
        block = deviations[rows_]
        scatter = block.T @ (w[rows_, None]*block)
        scatter = (scatter+scatter.T)/2
        _finite(scatter)
        _covariance(scatter, "frequency group SSCP", rank_limit=counts[i]-1)
        scatters.append(scatter)
    attrs = {"n_missing": dropped, "input_kind": "resident_oneway" if weights is None else "resident_frequency_oneway",
             "missing": "listwise including weight" if weights is not None else "listwise",
             "weight_type": None if weights is None else "fweight", "weights": weights,
             "weight_sum": sum(counts), "physical_rows": len(raw), "n_input_rows": rows,
             "n_zero_weight": len(frame)-len(positive),
             "sample": {"n_input_rows": rows,
                        "source_positions": [i for i, flag in enumerate(keep.tolist()) if flag],
                        "frequencies": frequencies, "positive_complete_positions": positive,
                        "group_codes": codes}}
    return _oneway(origin, offsets, scatters, counts, groups, names, alpha, attrs=attrs)
