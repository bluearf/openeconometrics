"""Complete-subject wide ANOVA and predeclared within-subject contrasts.

The ANOVA adapter preserves the existing ``stats.rm_anova`` numerical core,
including its first-order Mauchly approximation and SPSS Huynh-Feldt convention.
Saved contrast inference uses the exact one-sample Gaussian Hotelling law:
https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/1samphot.htm
R's second-order Mauchly approximation is deliberately not claimed here:
https://www.stat.ethz.ch/R-manual/R-devel/library/stats/html/mauchly.test.html
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Integral
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.postest import index_codec
from openecon.econometrics.stats import common as sc
from openecon.econometrics.stats.rm_anova import _contrasts, rm_anova
from openecon.resources import plan_workspace
from . import common as c

_SCHEMA = "openecon.rm_wide.v1"
_MODEL = "iid_multivariate_normal"
_MAX_SUBJECTS = 10_000
_MAX_CELLS = 32
_MAX_WORK = 100_000_000
_MAX_IDENTITIES = 1_048_576
_CORE = ("within", "sphericity", "between", "descriptives")
_STATE = ("sample", "means", "covariance", "mean_covariance", "within_basis",
          "contrast_covariance")
_CONTRAST = ("contrasts", "null", "subject_contrasts", "estimates", "joint_covariance", "joint")


def _error(message: str, code: str = "invalid_rm_state") -> None:
    raise AnalysisError(code, message)


def _model(value: Any) -> str:
    if value != _MODEL or not isinstance(value, str):
        _error("Declare iid_multivariate_normal sampling; dependent subjects, weights and "
               "other sampling models are outside this contract.", "unsupported_sampling_model")
    return value


def _plan(n: int, k: int, q: int = 0) -> tuple[dict, int]:
    work = n * k * k * 4 + k**3 * 4 + n * q * k + n * q * q + q**3
    if not 2 <= n <= _MAX_SUBJECTS or not 2 <= k <= _MAX_CELLS or work > _MAX_WORK:
        _error("Resident repeated measures require 2..10000 subjects, 2..32 measurements "
               "and estimated work at most 100 million.", "work_budget")
    plan = plan_workspace("resident wide repeated measures", {
        "selected wide sample, centering, transformed and saved float64 copies": 8 * n * k * 12,
        "long numeric values, subject/cell codes and pandas copies": 8 * n * k * 24,
        "subject masks, positions, portable identity strings and hash serialization": 64 * n + 6 * _MAX_IDENTITIES,
        "covariance, contrast bases and numerical decompositions": 8 * k * k * 24,
        "saved subject contrast vectors, joint moments and inference": 8 * (n * q * 8 + q * q * 16),
    }).record()
    return plan, work


def _cpu(value: Any) -> None:
    if isinstance(value, Tensor) and (value.device.type != "cpu" or value.is_complex()):
        _error("Only resident real CPU measurements and contrasts are supported.", "unsupported_domain")


def _input(data: Any, names: list[str], subject: str | None) -> tuple[pd.DataFrame, dict, int]:
    used = names + ([] if subject is None else [subject])
    if isinstance(data, pd.DataFrame):
        n = len(data)
        plan, work = _plan(n, len(names))
        if data.columns.has_duplicates or any(name not in data for name in used):
            _error("Required columns must each exist exactly once.", "invalid_columns")
        return data.loc[:, used].copy(), plan, work
    if isinstance(data, Mapping):
        if any(name not in data for name in used):
            _error("A required measurement or subject column is absent.", "missing_columns")
        try:
            lengths = [len(data[name]) for name in used]
        except TypeError as exc:
            raise AnalysisError("invalid_data", "Columns must be resident sequences.") from exc
        if len(set(lengths)) != 1:
            _error("Every selected column must have the same number of rows.", "invalid_data")
        plan, work = _plan(lengths[0], len(names))
        for name in used:
            _cpu(data[name])
        try:
            return pd.DataFrame({name: data[name] for name in used}), plan, work
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_data", "Columns must be one-dimensional resident sequences.") from exc
    if isinstance(data, (list, tuple)):
        plan, work = _plan(len(data), len(names))
        if any(not isinstance(row, Mapping) or any(name not in row for name in used) for row in data):
            _error("Supply resident records with every required column.", "invalid_data")
        return pd.DataFrame([{name: row[name] for name in used} for row in data]), plan, work
    _error("Wide repeated measures accept resident DataFrames, columns or records; Dataset "
           "and summary-statistic inputs are unsupported.", "unsupported_domain")


def _identities(values, *, subject: bool) -> list[str]:
    encoded, size = [], 0
    for value in values:
        if subject:
            try:
                missing = pd.isna(value)
                if isinstance(missing, bool) and missing:
                    _error("Subject IDs must be nonmissing and unique.", "invalid_subjects")
                if hasattr(missing, "item") and not hasattr(missing, "__len__") and bool(missing):
                    _error("Subject IDs must be nonmissing and unique.", "invalid_subjects")
                if isinstance(value, tuple) and any(pd.isna(item) for item in value):
                    _error("Subject IDs must be nonmissing and unique.", "invalid_subjects")
            except (TypeError, ValueError):
                _error("Subject IDs must be supported nonmissing scalar or tuple identities.", "invalid_subjects")
        try:
            code = index_codec.encode(value)
        except (TypeError, ValueError, OverflowError, RecursionError, AnalysisError) as exc:
            raise AnalysisError("invalid_subjects", "Row identities need supported portable scalar types.") from exc
        size += len(code)
        if size > _MAX_IDENTITIES:
            _error("Portable row identities exceed the one MiB budget.", "work_budget")
        encoded.append(code)
    if subject and len(set(encoded)) != len(encoded):
        _error("Subject IDs must be nonmissing and unique.", "invalid_subjects")
    return encoded


def _tensor(frame: pd.DataFrame, names: list[str]) -> Tensor:
    c.require_numeric(frame, names)
    if any(pd.api.types.is_bool_dtype(frame[name].dtype) for name in names):
        _error("Measurements must be real numeric values, rather than booleans.", "invalid_values")
    try:
        x = torch.as_tensor(frame.to_numpy(dtype="float64", na_value=math.nan), dtype=torch.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise AnalysisError("invalid_values", "Measurements must be representable as float64.") from exc
    if not bool(torch.isfinite(x).all()) or bool((x.abs() > c.LARGEST).any()):
        _error("Measurements must be finite and at most 1e150 in magnitude; rescale inputs.", "nonfinite_values")
    return x


def _moments(x: Tensor) -> tuple[Tensor, Tensor]:
    shifted = x - x[0]
    location = shifted.mean(0)
    location += (shifted - location).mean(0)
    dev = shifted - location
    mean, covariance = x[0] + location, dev.T @ dev / (len(x) - 1)
    covariance = (covariance + covariance.T) / 2
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(covariance).all()):
        _error("Sample moments exceed float64 support; rescale measurements.", "numerical_failure")
    return mean, covariance


def _hash(x: Tensor, attrs: Mapping) -> str:
    identity = {key: attrs[key] for key in (
        "measurement_columns", "sample_positions", "subject_labels_encoded", "row_labels_encoded",
        "physical_subjects", "excluded_subjects", "subject", "missing", "sampling_model")}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode())
    digest.update(x.detach().contiguous().numpy().astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _core(x: Tensor, alpha: float) -> TableSet:
    n, k = x.shape
    long = pd.DataFrame({"subject": torch.arange(n).repeat_interleave(k).tolist(),
                         "measurement": pd.Categorical(list(range(k)) * n,
                                                       categories=list(range(k)), ordered=True),
                         "outcome": x.reshape(-1).tolist()})
    try:
        return rm_anova(long, "outcome", "subject", ["measurement"], alpha=alpha, missing="raise")
    except (OverflowError, ZeroDivisionError) as exc:
        raise AnalysisError("numerical_failure", "The existing repeated-measures core exceeds "
                            "float64 support; rescale the measurements.") from exc


def _wide(x: Tensor, attrs: dict, plan: dict, work: int) -> TableSet:
    names, positions = attrs["measurement_columns"], attrs["sample_positions"]
    n, k = x.shape
    mean, covariance = _moments(x)
    basis = _contrasts([k])[0][:, 1:]
    labels = [f"Contrast{i + 1}" for i in range(k - 1)]
    # Form within differences before multiplication: B' S B suffers catastrophic
    # cancellation when between-subject levels dwarf within-subject variation.
    within_vectors = (x - x[:, :1]) @ basis
    within_mean, within_covariance = _moments(within_vectors)
    base = _core(x, attrs["anova_alpha"])
    stable_ss = n * float(within_mean.square().sum())
    stable_error = (n - 1) * float(within_covariance.trace())
    precision = torch.finfo(torch.float64).eps
    for reported, stable in ((float(base["within"].iloc[0]["ss"]), stable_ss),
                             (float(base["within"].iloc[4]["ss"]), stable_error)):
        floor = 1e-24 * max(stable_ss + stable_error, 1e-300)
        rounding = 8 * precision * float(x.abs().max()) * math.sqrt(n * k * max(stable, 0))
        if not math.isfinite(reported) or not math.isfinite(stable) \
                or (reported == 0 and stable > floor) \
                or abs(reported - stable) > max(2e-10 * abs(stable), rounding, floor):
            _error("The existing ANOVA core cannot resolve the within-subject sums of squares "
                   "at this subject-level scale; rescale or remove the common subject level "
                   "before analysis.", "unresolved_within_scale")
    tables = dict(base)
    tables.update({
        "sample": c.frame(x, columns=names, index=positions),
        "means": c.frame(mean[:, None], columns=["mean"], index=names),
        "covariance": c.frame(covariance, columns=names, index=names),
        "mean_covariance": c.frame(covariance / n, columns=names, index=names),
        "within_basis": c.frame(basis, columns=labels, index=names),
        "contrast_covariance": c.frame(within_covariance, columns=labels, index=labels),
    })
    meta = dict(base.attrs)
    meta.update(attrs)
    meta.update(procedure="rm_anova_wide", state_schema=_SCHEMA, alpha=attrs["anova_alpha"],
                precision="float64", device="cpu", covariance_divisor=n - 1,
                sample_position_base=0, resource_plan=plan, estimated_work=work,
                source_content_sha256=_hash(x, attrs),
                assumptions="Independent subjects with iid multivariate Gaussian measurement vectors; "
                            "the sampling model is declared, not tested. Classical uncorrected F requires "
                            "sphericity; GG/HF corrected F and Mauchly chi-square are approximations.",
                sphericity_convention="Existing first-order Mauchly chi-square; SPSS Huynh-Feldt epsilon",
                within_resolution_guard="Within SS/error must agree with stable subject-difference "
                                        "references; nonzero terms truncated by the unchanged core are refused",
                supported_domain="One ordered within-subject factor, complete retained subjects, no weights or between factors",
                stata_parity_validated=False)
    return TableSet(tables, title="Complete-subject wide repeated-measures ANOVA", **meta)


@c.procedure
@torch.no_grad()
def rm_anova_wide(data, columns, *, subject=None, missing="raise", alpha=0.05,
                  sampling_model="iid_multivariate_normal") -> TableSet:
    """Adapt ordered wide measurements to the existing one-factor ANOVA core.

    Each row is one independent subject. ``subject`` optionally names unique,
    nonmissing portable IDs; otherwise physical row positions identify subjects.
    ``missing='drop_subject'`` excludes an entire subject if any measurement is
    missing. Raw measurements, row/subject identities, covariance and the ANOVA
    tables form a portable state accepted by ``rm_restore`` and ``rm_contrasts``.
    The model declaration is an assumption, never a normality/independence test.
    """
    names = c.name_list(columns, "columns", minimum=2)
    if len(names) > _MAX_CELLS:
        _error("At most 32 ordered measurements are supported.", "work_budget")
    if subject is not None:
        c.check_name(subject, "subject")
        if subject in names:
            _error("Subject ID and measurement roles must be distinct.", "invalid_spec")
    c.check_choice(missing, "missing", ("raise", "drop_subject"))
    alpha, sampling_model = sc.check_alpha(alpha), _model(sampling_model)
    frame, plan, work = _input(data, names, subject)
    physical = len(frame)
    ids = _identities(frame[subject].tolist() if subject else range(physical), subject=True)
    rows = _identities(frame.index, subject=False)
    selected = frame.loc[:, names]
    c.require_numeric(selected, names)
    complete = ~selected.isna().any(axis=1)
    if not bool(complete.all()) and missing == "raise":
        _error("Every retained subject needs all measurements; choose missing='drop_subject' "
               "to exclude incomplete subjects explicitly.", "missing_values")
    positions = complete.to_numpy().nonzero()[0].tolist()
    if len(positions) < 2:
        _error("At least two complete subjects are required.", "insufficient_observations")
    x = _tensor(selected.loc[complete], names)
    attrs = {"measurement_columns": names, "sample_positions": positions,
             "subject_labels_encoded": [ids[i] for i in positions],
             "row_labels_encoded": [rows[i] for i in positions], "subject": subject,
             "physical_subjects": physical, "excluded_subjects": physical - len(positions),
             "n_missing_subjects": physical - len(positions), "missing": missing,
             "sampling_model": sampling_model, "anova_alpha": alpha}
    return _wide(x, attrs, plan, work)


def _shape(spec: Any) -> tuple[int, int]:
    if isinstance(spec, pd.DataFrame):
        return spec.shape
    if not isinstance(spec, Mapping) or set(spec) != {"columns", "index", "data"}:
        _error("Saved tables require split columns/index/data records.")
    columns, rows, index = spec["columns"], spec["data"], spec["index"]
    if not isinstance(columns, list) or not isinstance(rows, list) or not isinstance(index, list):
        _error("Saved table geometry is invalid.")
    n, k = len(rows), len(columns)
    if len(index) != n or n > _MAX_SUBJECTS or k > _MAX_CELLS:
        _error("Saved tables exceed their bounded geometry.")
    if any(not isinstance(row, list) or len(row) != k for row in rows):
        _error("Saved table rows do not match their columns.")
    return n, k


def _same(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    if list(actual.columns) != list(expected.columns) or list(actual.index) != list(expected.index) \
            or actual.shape != expected.shape:
        _error(f"Saved {label} geometry or order does not match the sample.")
    for name in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[name].dtype):
            try:
                a = torch.as_tensor(pd.to_numeric(actual[name], errors="raise").to_numpy(dtype="float64", na_value=math.nan), dtype=torch.float64)
                b = torch.as_tensor(expected[name].to_numpy(dtype="float64", na_value=math.nan), dtype=torch.float64)
            except (ValueError, TypeError, OverflowError) as exc:
                raise AnalysisError("invalid_rm_state", f"Saved {label} values are invalid.") from exc
            # JSON preserves binary64 values; only matrix-operation rounding is admitted.
            scale = torch.maximum(b.abs(), torch.full_like(b, 1e-300))
            matches = (a - b).abs() <= 2e-12 * scale
            matches |= torch.isnan(a) & torch.isnan(b)
            if not bool(matches.all()):
                _error(f"Saved {label} values do not match the complete sample.")
        elif actual[name].tolist() != expected[name].tolist():
            _error(f"Saved {label} labels do not match the complete sample.")


def _portable(result: Any) -> tuple[dict, dict[str, pd.DataFrame]]:
    if isinstance(result, str):
        if len(result) > 64_000_000:
            _error("Saved RM payload exceeds its resident storage budget.")
        try:
            result = json.loads(result)
        except (ValueError, RecursionError) as exc:
            raise AnalysisError("invalid_rm_state", "Saved RM JSON is invalid.") from exc
    if isinstance(result, TableSet):
        attrs, specs = result.attrs, dict(result)
    elif isinstance(result, Mapping) and set(result) == {"attrs", "tables"}:
        attrs, specs = result["attrs"], result["tables"]
    else:
        _error("Supply an rm_anova_wide result or portable attrs/tables payload.")
    if not isinstance(attrs, dict) or not isinstance(specs, Mapping):
        _error("Saved RM metadata or tables are invalid.")
    procedure = attrs.get("procedure")
    keys = set(_CORE + _STATE + (_CONTRAST if procedure == "rm_contrasts" else ()))
    if attrs.get("state_schema") != _SCHEMA or procedure not in ("rm_anova_wide", "rm_contrasts") \
            or set(specs) != keys:
        _error("Saved state must use the complete supported RM schema.")
    names = attrs.get("measurement_columns")
    try:
        names = c.name_list(names, "measurement_columns", minimum=2)
    except AnalysisError as exc:
        raise AnalysisError("invalid_rm_state", "Saved measurement order is invalid.") from exc
    n, k = _shape(specs["sample"])
    if k != len(names):
        _error("Saved sample does not match measurement names.")
    _plan(n, k)
    shapes = {"within": (8, 8), "sphericity": (1, 7), "between": (2, 6),
              "descriptives": (k, 4), "sample": (n, k), "means": (k, 1),
              "covariance": (k, k), "mean_covariance": (k, k),
              "within_basis": (k, k - 1), "contrast_covariance": (k - 1, k - 1)}
    if procedure == "rm_contrasts":
        q, width = _shape(specs["contrasts"])
        if width != k or not 1 <= q < k or n <= q:
            _error("Saved contrast dimensions are invalid.")
        _plan(n, k, q)
        shapes.update(contrasts=(q, k), null=(q, 1), subject_contrasts=(n, q),
                      estimates=(q, 8), joint_covariance=(q, q), joint=(1, 5))
    for key, spec in specs.items():
        if _shape(spec) != shapes[key]:
            _error(f"Saved {key} exceeds its declared shape.")
    tables = {key: value if isinstance(value, pd.DataFrame) else pd.DataFrame(**value)
              for key, value in specs.items()}
    return dict(attrs), tables


def _state(result: Any) -> TableSet:
    attrs, tables = _portable(result)
    names, sample = attrs["measurement_columns"], tables["sample"]
    n, k = sample.shape
    physical, excluded = attrs.get("physical_subjects"), attrs.get("excluded_subjects")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (physical, excluded)) \
            or not n <= physical <= _MAX_SUBJECTS or excluded != physical - n:
        _error("Saved physical/excluded subject counts are invalid.")
    plan, work = _plan(physical, k)
    positions = attrs.get("sample_positions")
    if not isinstance(positions, list) or len(positions) != n \
            or any(isinstance(i, bool) or not isinstance(i, Integral) for i in positions) \
            or positions != sorted(set(positions)) or any(i < 0 or i >= physical for i in positions):
        _error("Saved subject positions must preserve unique physical row order.")
    if list(sample.columns) != names or list(sample.index) != positions:
        _error("Saved raw measurements do not preserve column and subject order.")
    for field in ("subject_labels_encoded", "row_labels_encoded"):
        values = attrs.get(field)
        if not isinstance(values, list) or len(values) != n or any(not isinstance(v, str) for v in values) \
                or sum(map(len, values)) > _MAX_IDENTITIES:
            _error("Saved portable identities are invalid or exceed their budget.")
        try:
            decoded = [index_codec.decode(v) for v in values]
            if _identities(decoded, subject=field == "subject_labels_encoded") != values:
                _error("Saved identities are not canonical.")
        except (ValueError, TypeError, IndexError, KeyError, OverflowError, RecursionError, AnalysisError) as exc:
            raise AnalysisError("invalid_rm_state", "Saved identities cannot be restored safely.") from exc
    subject = attrs.get("subject")
    if subject is not None and (not isinstance(subject, str) or not subject or subject in names):
        _error("Saved subject and measurement roles overlap or are invalid.")
    if subject is None and attrs["subject_labels_encoded"] != _identities(positions, subject=True):
        _error("Positional subject IDs do not agree with retained positions.")
    missing = attrs.get("missing")
    if missing not in ("raise", "drop_subject") or (missing == "raise" and excluded != 0):
        _error("Saved missing policy does not agree with excluded subjects.")
    _model(attrs.get("sampling_model"))
    try:
        alpha = sc.check_alpha(attrs.get("anova_alpha"))
        x = _tensor(sample, names)
    except AnalysisError as exc:
        raise AnalysisError("invalid_rm_state", "Saved alpha or raw measurements are invalid.") from exc
    if attrs.get("source_content_sha256") != _hash(x, attrs):
        _error("Saved complete-sample content hash does not match measurements/order/identities.")
    expected_metadata = {"n_subjects": n, "n": n * k, "n_missing": 0,
                         "n_missing_subjects": excluded, "within_cells": k,
                         "within": ["measurement"], "between": [], "df_error_between": n - 1,
                         "ss_type": 3, "sample_position_base": 0, "precision": "float64",
                         "device": "cpu", "covariance_divisor": n - 1}
    if any(attrs.get(key) != value or (isinstance(value, int) and
           (isinstance(attrs.get(key), bool) or not isinstance(attrs.get(key), Integral)))
           for key, value in expected_metadata.items()):
        _error("Saved counts, design, precision or covariance convention are inconsistent.")
    metadata = {key: attrs[key] for key in ("measurement_columns", "sample_positions",
                "subject_labels_encoded", "row_labels_encoded", "subject", "physical_subjects",
                "excluded_subjects", "n_missing_subjects", "missing", "sampling_model")}
    metadata["anova_alpha"] = alpha
    fresh = _wide(x, metadata, plan, work)
    for key in _CORE + _STATE:
        _same(tables[key], fresh[key], key)
    if attrs["procedure"] == "rm_contrasts":
        restored = _contrast(fresh, tables["contrasts"], tables["null"].iloc[:, 0].tolist(),
                             attrs.get("alpha"), list(tables["contrasts"].index), attrs["sampling_model"])
        for key in _CONTRAST:
            _same(tables[key], restored[key], key)
        for key in ("contrast_content_sha256", "q", "df_marginal", "df_numerator", "df_denominator"):
            if attrs.get(key) != restored.attrs[key]:
                _error(f"Saved {key} does not agree with declared contrasts.")
        return restored
    if attrs.get("alpha") != alpha:
        _error("Saved alpha is inconsistent with ANOVA settings.")
    return fresh


@c.procedure
@torch.no_grad()
def rm_restore(payload) -> TableSet:
    """Validate a portable RM TableSet or JSON attrs/tables split payload.

    Admission precedes table construction. Raw measurements, typed identities,
    ordering, hashes, moments and all reported ANOVA/contrast tables are checked.
    Restoration reconstructs the bounded existing ANOVA core; it never trusts
    saved covariance, degrees of freedom or inferential tables as input.
    """
    return _state(payload)


def _array(value: Any, shape: tuple[int, ...], name: str) -> Tensor:
    _cpu(value)
    if isinstance(value, pd.DataFrame):
        if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
               or pd.api.types.is_complex_dtype(dtype) for dtype in value.dtypes):
            _error(f"{name} must contain real numeric coefficients.", "invalid_contrasts")
        value = value.to_numpy()
    try:
        if bool(torch.as_tensor(value).is_complex()):
            _error(f"{name} must contain real coefficients.", "invalid_contrasts")
        output = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise AnalysisError("invalid_contrasts", f"{name} must be real numeric values of the declared shape.") from exc
    if tuple(output.shape) != shape or not bool(torch.isfinite(output).all()):
        _error(f"{name} must have shape {shape} with finite real values.", "invalid_contrasts")
    return output


def _contrast(base: TableSet, contrasts, null, alpha, contrast_names, sampling_model) -> TableSet:
    alpha, sampling_model = sc.check_alpha(alpha), _model(sampling_model)
    names = base.attrs["measurement_columns"]
    n, k = base["sample"].shape
    _cpu(contrasts)
    try:
        shape = contrasts.shape if hasattr(contrasts, "shape") else (len(contrasts), len(contrasts[0]))
    except (IndexError, TypeError) as exc:
        raise AnalysisError("invalid_contrasts", "Supply a q-by-k contrast matrix.") from exc
    if len(shape) != 2 or shape[1] != k or not 1 <= shape[0] < k or n <= shape[0]:
        _error("Contrasts need 1..k-1 rows, k columns and more complete subjects than rows.", "invalid_contrasts")
    q = int(shape[0])
    plan, work = _plan(base.attrs["physical_subjects"], k, q)
    if isinstance(contrasts, pd.DataFrame):
        if list(contrasts.columns) != names:
            _error("Labeled contrast columns must exactly match the saved measurement order.", "invalid_contrasts")
        if contrast_names is None:
            contrast_names = list(contrasts.index)
    labels = [f"Contrast{i + 1}" for i in range(q)] if contrast_names is None else c.name_list(contrast_names, "contrast_names")
    if len(labels) != q:
        _error("Supply exactly one unique contrast name per row.", "invalid_contrasts")
    matrix = _array(contrasts, (q, k), "contrasts")
    scale = matrix.abs().amax(1)
    if bool((scale <= 0).any()):
        _error("Every contrast must be nonzero.", "invalid_contrasts")
    normalized = matrix / scale[:, None]
    try:
        nonzero_sums = any(math.fsum(row) != 0 for row in matrix.tolist())
    except OverflowError as exc:
        raise AnalysisError("invalid_contrasts", "Contrast coefficient sums exceed float64 support; "
                            "rescale coefficients.") from exc
    if nonzero_sums:
        _error("Every stored binary64 contrast row must sum exactly to zero; near-zero "
               "sums are refused rather than silently changing the coefficients.", "invalid_contrasts")
    singular = torch.linalg.svdvals(normalized)
    if float(singular.min()) <= 1e-12 * float(singular.max()):
        _error("Predeclared contrast rows must have full row rank; no pseudoinverse is used.", "rank_deficient_contrasts")
    null_vector = torch.zeros(q, dtype=torch.float64) if null is None else _array(null, (q,), "null")
    x = _tensor(base["sample"], names)
    # Zero-sum contrasts cancel the per-subject common level before multiplication.
    subject_vectors = (x - x[:, :1]) @ normalized.T
    estimate_scaled, covariance_scaled = _moments(subject_vectors)
    mean_covariance_scaled = covariance_scaled / n
    standard = mean_covariance_scaled.diag().sqrt()
    rounding_floor = 64 * torch.finfo(torch.float64).eps * subject_vectors.abs().amax(0)
    if not bool(torch.isfinite(standard).all()) or bool((standard <= 0).any()) \
            or bool((covariance_scaled.diag().sqrt() <= rounding_floor).any()):
        _error("Every selected contrast needs resolved subject variation.", "singular_covariance")
    correlation = mean_covariance_scaled / standard[:, None] / standard[None, :]
    correlation = (correlation + correlation.T) / 2
    roots = torch.linalg.eigvalsh(correlation)
    if float(roots.min()) <= 1e-12 * float(roots.max()):
        _error("The selected joint contrast covariance is singular or unresolved; no pseudoinverse is used.", "singular_covariance")
    vectors = subject_vectors * scale
    estimate, covariance = estimate_scaled * scale, mean_covariance_scaled * scale[:, None] * scale[None, :]
    se = standard * scale
    delta = estimate_scaled / standard - null_vector / scale / standard
    t_squared = float(delta @ torch.linalg.solve(correlation, delta))
    f = (n - q) / (q * (n - 1)) * t_squared
    t = (estimate - null_vector) / se
    critical = sc.t_critical(alpha, n - 1)
    low, high = estimate - critical * se, estimate + critical * se
    if not math.isfinite(f) or any(not bool(torch.isfinite(value).all()) for value in
                                  (vectors, estimate, covariance, se, t, low, high)) \
            or bool((covariance.diag() <= 0).any()) or bool((se <= 0).any()):
        _error("Contrast estimates, moments or inference exceed float64 support; rescale inputs.", "nonfinite_result")
    tables = dict(base)
    tables.update({
        "contrasts": c.frame(matrix, columns=names, index=labels),
        "null": c.frame(null_vector[:, None], columns=["null"], index=labels),
        "subject_contrasts": c.frame(vectors, columns=labels, index=base.attrs["sample_positions"]),
        "joint_covariance": c.frame(covariance, columns=labels, index=labels),
        "estimates": c.frame([[float(estimate[i]), float(se[i]), float(null_vector[i]), float(t[i]),
                                n - 1, sc.t_two_sided(float(t[i]), n - 1), float(low[i]), float(high[i])]
                               for i in range(q)], columns=["estimate", "std_error", "null", "t", "df",
                                                            "p_value", "ci_lower", "ci_upper"], index=labels),
        "joint": c.frame([[t_squared, f, q, n - q, sc.f_upper(f, q, n - q)]],
                         columns=["t_squared", "statistic", "df_numerator", "df_denominator", "p_value"],
                         index=["predeclared_contrasts"]),
    })
    digest = hashlib.sha256(base.attrs["source_content_sha256"].encode())
    digest.update(json.dumps(labels, separators=(",", ":")).encode())
    digest.update(matrix.numpy().astype("<f8", copy=False).tobytes())
    digest.update(null_vector.numpy().astype("<f8", copy=False).tobytes())
    attrs = dict(base.attrs)
    attrs.update(procedure="rm_contrasts", alpha=alpha, q=q, df_marginal=n - 1,
                 df_numerator=q, df_denominator=n - q, contrast_content_sha256=digest.hexdigest(),
                 resource_plan=plan, estimated_work=work,
                 contrast_assumptions="Predeclared zero-sum full-row-rank contrasts of iid Gaussian subject "
                                      "measurement vectors; unknown positive-definite selected contrast covariance, n>q. "
                                      "Predeclaration and Gaussian independence are assumptions, not verified from outcomes.",
                 joint_distribution="T_squared * (n-q)/(q*(n-1)) ~ F(q,n-q)",
                 intervals="Marginal two-sided Student t intervals with n-1 df; no familywise coverage claim",
                 covariance_scope="Full covariance of contrast sample means, C S C' / n")
    return TableSet(tables, title="Predeclared within-subject contrasts", **attrs)


@c.procedure
@torch.no_grad()
def rm_contrasts(result, contrasts, *, null=None, alpha=0.05, contrast_names=None,
                 sampling_model="iid_multivariate_normal") -> TableSet:
    """Marginal t and exact joint Hotelling F tests from validated saved wide state.

    ``contrasts`` is q-by-k in the original measurement order. Every stored
    binary64 row must sum exactly to zero and the rows must be independent;
    near-zero sums are refused. ``null`` defaults to zero.
    Reported confidence intervals are marginal; this method neither adjusts a
    family of tests nor supports contrasts selected using the observed outcomes.
    All raw subject vectors, coefficients, nulls and joint covariance are saved.
    """
    base = _state(result)
    if base.attrs["procedure"] != "rm_anova_wide":
        _error("rm_contrasts needs the saved rm_anova_wide state.", "invalid_result")
    return _contrast(base, contrasts, null, alpha, contrast_names, sampling_model)
