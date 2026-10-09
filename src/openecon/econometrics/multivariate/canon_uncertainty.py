"""IID uncertainty for fixed simple CCA roots and identified coefficient charts.

References: Anderson (1999), doi:10.1006/jmva.1999.1810 (simple nonzero
canonical roots); Bickel and Freedman (1981), doi:10.1214/aos/1176345637
(bootstrap of regular functionals). QR/SVD follows Bjorck and Golub (1973),
doi:10.1090/S0025-5718-1973-0348991-3. Population regularity is assumed;
the numerical checks below do not establish it from a finite sample.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.postest.index_codec import encode
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.resources import plan_workspace

from . import common as c
from .summary import Summary

MAX_ROWS = 10_000
MAX_VARIABLES = 16
MAX_COMPONENTS = 4
MAX_REPLICATIONS = 1_999
MAX_WORK = 250_000_000
MAX_EXPORT_BYTES = 32 * 1024**2
MAX_LABEL_BYTES = 4096
MAX_IDENTITY_BYTES = 2 * 1024**2
MAX_IDENTITY_DEPTH = 32
MAX_IDENTITY_NODES = 100_000
GEOMETRY_TOLERANCE = 1e-8
MIN_DIRECTION_COSINE = math.sqrt(.5)
STATE_SCHEMA = "openecon.canon_bootstrap.v1"
_DIAGNOSTICS = ["minimum_retained_correlation", "maximum_correlation",
                "minimum_required_correlation_gap", "x_correlation_condition",
                "y_correlation_condition", "minimum_relative_anchor",
                "minimum_x_signed_direction_cosine", "minimum_y_signed_direction_cosine"]


@dataclass
class _Fit:
    vector: Tensor
    roots: Tensor
    mean: Tensor
    sd: Tensor
    covariance: Tensor
    correlation: Tensor
    diagnostics: Tensor
    anchors: list[str] | None
    matrices: dict[str, Tensor]


def _rows(data: Any, names: list[str]) -> int:
    if isinstance(data, (Dataset, Summary)):
        raise AnalysisError("unsupported_input", "CCA bootstrap requires resident raw rows; Dataset and summary matrices are unsupported.")
    if isinstance(data, pd.DataFrame):
        return len(data)
    if isinstance(data, Mapping):
        if any(name not in data for name in names):
            raise AnalysisError("missing_columns", "A declared CCA variable is absent.")
        lengths = []
        for name in names:
            column = data[name]
            if isinstance(column, Tensor) and column.device.type != "cpu":
                raise AnalysisError("unsupported_device", "CCA bootstrap requires CPU inputs.")
            try:
                lengths.append(len(column))
            except TypeError:
                raise AnalysisError("invalid_data", "Supply sized resident numeric columns.") from None
        if len(set(lengths)) != 1:
            raise AnalysisError("invalid_data", "Resident columns have different lengths.")
        return lengths[0]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return len(data)
    raise AnalysisError("invalid_data", "Supply a resident DataFrame, sized column mapping or row records.")


def _selected(data: Any, names: list[str]) -> Any:
    if isinstance(data, pd.DataFrame):
        if data.columns.nlevels != 1:
            raise AnalysisError("invalid_data", "CCA bootstrap requires one source column level with declared string variables.")
        if data.columns.has_duplicates:
            raise AnalysisError("duplicate_columns", "Data must have unique column names.")
        if any(name not in data.columns for name in names):
            raise AnalysisError("missing_columns", "A declared CCA variable is absent.")
        return data.loc[:, names]
    if isinstance(data, Mapping):
        for name in names:
            column = data[name]
            if isinstance(column, Tensor):
                if column.ndim != 1 or column.layout != torch.strided or column.requires_grad:
                    raise AnalysisError("invalid_data", "Tensor measurement columns must be plain one-dimensional strided CPU values without an autograd graph.")
                invalid = column.dtype == torch.bool or column.is_complex()
            else:
                invalid = any(_nonreal_measurement(value) for value in column)
            if invalid:
                raise AnalysisError("non_numeric_column", "CCA measurements exclude Boolean and complex values, including mixed resident columns.")
        return {name: data[name] for name in names}
    if not all(isinstance(row, Mapping) for row in data):
        raise AnalysisError("invalid_data", "Each resident record must map column names to numeric values.")
    if any(_nonreal_measurement(row.get(name)) for row in data for name in names):
        raise AnalysisError("non_numeric_column", "CCA measurements exclude Boolean and complex values, including mixed resident records.")
    return [{name: row.get(name) for name in names} for row in data]


def _nonreal_measurement(value: Any) -> bool:
    return isinstance(value, (bool, complex)) or getattr(getattr(value, "dtype", None), "kind", None) in ("b", "c")


def _identity(value: Any, counter: list[int]) -> str:
    # Bound traversal and scalar allocation before invoking the recursive codec.
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        counter[0] += 1
        if depth > MAX_IDENTITY_DEPTH or counter[0] > MAX_IDENTITY_NODES:
            raise AnalysisError("resource_limit", "Source identities exceed bounded tuple depth or traversal work.")
        if isinstance(item, (str, bytes)) and len(item) > MAX_LABEL_BYTES:
            raise AnalysisError("resource_limit", "A source identity exceeds 4096 text/byte units.")
        if isinstance(item, int) and item.bit_length() > 4 * MAX_LABEL_BYTES \
                or isinstance(item, Decimal) and item.__sizeof__() > MAX_LABEL_BYTES:
            raise AnalysisError("resource_limit", "A numeric source identity exceeds bounded scalar storage.")
        if isinstance(item, tuple):
            if len(item) > MAX_IDENTITY_NODES - counter[0]:
                raise AnalysisError("resource_limit", "Source identities exceed bounded tuple traversal work.")
            pending.extend((child, depth + 1) for child in item)
    try:
        code = encode(value)
    except (TypeError, ValueError, OverflowError, RecursionError, AnalysisError):
        raise AnalysisError("invalid_index", "Source labels need supported lossless scalar/tuple identities.") from None
    if len(code.encode()) > MAX_LABEL_BYTES:
        raise AnalysisError("resource_limit", "An encoded source identity exceeds 4096 bytes.")
    return code


def _identities(index: pd.Index, flags: list[bool], column_names: list[Any]):
    counter, used = [0], 0
    outputs = [[], [], []]
    values = ([value for value, keep in zip(index, flags, strict=True) if keep],
              list(index.names), column_names)
    for source, output in zip(values, outputs, strict=True):
        for value in source:
            code = _identity(value, counter)
            used += len(code.encode())
            if used > MAX_IDENTITY_BYTES:
                raise AnalysisError("resource_limit", "Complete source identities exceed the 2 MiB portable budget.")
            output.append(code)
    return *outputs, used


def _finite(*values: Tensor) -> None:
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise AnalysisError("numerical_failure", "CCA arithmetic exceeds finite float64 precision; rescale the measurements.")


def _parameters(values: Tensor, x_names: list[str], y_names: list[str], components: int,
                target: str, anchors: list[str] | None, point: _Fit | None = None) -> _Fit:
    # A real observed origin removes irrelevant large common levels before
    # centering; correlation arithmetic uses standardized rows, not products
    # of raw scales that can themselves underflow or overflow.
    shifted = values - values[0]
    centered, offset = c.centre(shifted)
    covariance = centered.T @ centered / (len(values) - 1)
    covariance = (covariance + covariance.T) / 2
    mean = values[0] + offset
    _finite(mean, covariance)
    variance = covariance.diagonal()
    if bool((variance <= 0).any()):
        raise AnalysisError("zero_variance", "Every CCA variable must vary in every admitted sample.")
    if bool((variance < torch.finfo(c.FLOAT).tiny).any()):
        raise AnalysisError("numerical_failure", "A raw sample variance is subnormal in float64 before CCA restandardization; rescale the measurements.")
    sd = variance.sqrt()
    standardized = centered / sd
    correlation = standardized.T @ standardized / (len(values) - 1)
    correlation = (correlation + correlation.T) / 2
    correlation.diagonal().fill_(1)
    _finite(sd, correlation)
    p = len(x_names)
    rxx, ryy, rxy = correlation[:p, :p], correlation[p:, p:], correlation[:p, p:]
    spectra = [torch.linalg.eigvalsh(block) for block in (rxx, ryy)]
    conditions = []
    for spectrum in spectra:
        _finite(spectrum)
        if float(spectrum[0]) <= 1e-10 * float(spectrum[-1]):
            raise AnalysisError("singular_matrix", "Both CCA correlation blocks must have full, conditioned rank in every fit; no repair or variable deletion is performed.")
        conditions.append(float(spectrum[-1] / spectrum[0]))
    qx, rx = torch.linalg.qr(standardized[:, :p], mode="reduced")
    qy, ry = torch.linalg.qr(standardized[:, p:], mode="reduced")
    u, roots, vh = torch.linalg.svd(qx.T @ qy, full_matrices=False)
    _finite(roots)
    if float(roots[components - 1]) <= GEOMETRY_TOLERANCE:
        raise AnalysisError("unidentified_correlation", "Every retained fixed canonical correlation must be positive away from numerical zero.")
    if float(roots[0]) >= 1 - GEOMETRY_TOLERANCE:
        raise AnalysisError("unidentified_correlation", "Canonical-correlation uncertainty requires interior roots away from perfect correlation.")
    following = torch.cat((roots[1:], torch.zeros(1, dtype=c.FLOAT)))
    gap = float((roots[:components] - following[:components]).min())
    if gap <= GEOMETRY_TOLERANCE:
        raise AnalysisError("unidentified_correlation", "All retained roots must be simple and separated from adjacent roots, including the retained/omitted boundary.")
    matrices: dict[str, Tensor] = {}
    fixed, anchor_min, x_cos, y_cos = None, math.nan, math.nan, math.nan
    vector = roots[:components].clone()
    if target == "coefficients":
        scale = math.sqrt(len(values) - 1)
        a = torch.linalg.solve_triangular(rx, u[:, :components], upper=True) * scale
        b = torch.linalg.solve_triangular(ry, vh.T[:, :components], upper=True) * scale
        _finite(a, b)
        norms = torch.linalg.vector_norm(a, dim=0)
        if anchors is None:
            ordered = a.abs().sort(dim=0, descending=True).values
            if p > 1 and bool(((ordered[0] - ordered[1]) <= GEOMETRY_TOLERANCE * norms).any()):
                raise AnalysisError("unidentified_anchor", "A default point sign anchor needs a unique strongest standardized X coefficient; declare anchors explicitly.")
            fixed = [x_names[int(row)] for row in a.abs().argmax(dim=0)]
        else:
            fixed = anchors
        anchor_values = a[[x_names.index(name) for name in fixed], torch.arange(components)]
        relative = anchor_values.abs() / norms
        anchor_min = float(relative.min())
        if anchor_min <= GEOMETRY_TOLERANCE:
            raise AnalysisError("unidentified_anchor", "Every fixed X coefficient sign anchor must remain away from numerical zero relative to its standardized coefficient norm.")
        signs = anchor_values.sign()
        a, b = a * signs, b * signs
        raw_a, raw_b = a / sd[:p, None], b / sd[p:, None]
        x_cos = y_cos = 1.0
        if point is not None:
            for key, section in (("x_coefficients", slice(0, p)),
                                 ("y_coefficients", slice(p, None))):
                metric = point.correlation[section, section]
                chol = torch.linalg.cholesky(metric)
                raw = raw_a if key.startswith("x") else raw_b
                # Evaluate both coefficient directions in the same point
                # covariance metric, independently of each draw's scales.
                current = chol.T @ (point.sd[section, None] * raw)
                reference = chol.T @ (point.sd[section, None] * point.matrices[key])
                cosine = (current * reference).sum(0) / (torch.linalg.vector_norm(current, dim=0) * torch.linalg.vector_norm(reference, dim=0))
                _finite(cosine)
                if bool((cosine <= MIN_DIRECTION_COSINE).any()):
                    raise AnalysisError("component_crossing", "A signed canonical coefficient direction left its local 45-degree point-covariance chart; no relabelling, permutation or alignment is applied.")
                if key.startswith("x"):
                    x_cos = float(cosine.min())
                else:
                    y_cos = float(cosine.min())
        matrices = {"x_coefficients": raw_a, "y_coefficients": raw_b,
                    "x_standardized_coefficients": a, "y_standardized_coefficients": b,
                    "x_loadings": rxx @ a, "y_loadings": ryy @ b,
                    "x_cross_loadings": rxy @ b, "y_cross_loadings": rxy.T @ a}
        vector = torch.cat([torch.cat([roots[j:j + 1],
                                       *(matrix[:, j] for matrix in matrices.values())])
                            for j in range(components)])
    _finite(vector, *matrices.values())
    diagnostics = torch.tensor([float(roots[components - 1]), float(roots[0]), gap,
                                *conditions, anchor_min, x_cos, y_cos], dtype=c.FLOAT)
    return _Fit(vector, roots, mean, sd, covariance, correlation, diagnostics, fixed, matrices)


def _parameter_order(x_names: list[str], y_names: list[str], components: int, target: str):
    order = []
    for j in range(components):
        axis = f"Can{j + 1}"
        order.append(["correlation", axis, None, None])
        if target == "coefficients":
            for kind in ("raw_coefficient", "standardized_coefficient", "within_loading", "cross_loading"):
                for block, names in (("X", x_names), ("Y", y_names)):
                    order.extend([kind, axis, block, name] for name in names)
    labels = [":".join(item for item in entry if item is not None) for entry in order]
    return order, labels


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _state_digest(result: TableSet) -> str:
    payload = {"title": result.title,
               "attrs": {key: value for key, value in result.attrs.items() if key != "state_content_sha256"},
               "tables": {name: {"index": list(frame.index), "columns": list(frame.columns),
                                 "index_names": list(frame.index.names), "column_names": list(frame.columns.names),
                                 "data": frame.to_numpy().tolist()}
                          for name, frame in result.items()}}
    encoded = json.dumps(_json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_EXPORT_BYTES:
        raise AnalysisError("resource_limit", "The complete portable CCA result exceeds 32 MiB.")
    return hashlib.sha256(encoded).hexdigest()


def _validate_saved(result: TableSet) -> None:
    """Validate a restored complete result's integrity; never refit or repair it."""
    try:
        valid = isinstance(result, TableSet) and result.attrs.get("state_schema") == STATE_SCHEMA \
            and result.attrs.get("procedure") == "canon_bootstrap" \
            and result.attrs.get("state_content_sha256") == _state_digest(result)
    except (TypeError, ValueError, AttributeError, AnalysisError):
        valid = False
    if not valid:
        raise AnalysisError("invalid_state", "Saved CCA bootstrap tables, identities or settings fail their complete-state integrity receipt.")


@c.procedure
@resident_cpu
def canon_bootstrap(data: Any, x: list[str], y: list[str], *, components: int,
                    target: str = "correlations", anchors: list[str] | None = None,
                    replications: int = 199, confidence: float = .95,
                    seed: int = 0, missing: str = "drop") -> TableSet:
    """Full IID marginal percentile uncertainty for a fixed CCA functional.

    Complete joint rows are resampled with replacement. Means, unbiased
    covariance, marginal scales and both block QR decompositions are refitted
    in every draw. The caller fixes the component count. Correlations are in
    descending order and every retained root must be positive, interior and
    simple, including separation from the first omitted root.

    ``target='correlations'`` estimates only those roots and rejects anchors.
    ``target='coefficients'`` also estimates all raw and standardized X/Y
    coefficients and within/cross loadings. Raw canonical variates have sample
    variance one and positive within-pair correlation. One ordered X-variable
    sign anchor per axis is fixed from the caller or the unique strongest
    standardized point coefficient. Every draw's signed X and Y directions
    must remain inside a 45-degree chart in the fixed point covariance metric.
    Axes are never permuted, relabelled or Procrustes aligned.

    IID complete-case rows, fixed dimension/count, finite fourth moments,
    nonsingular interior population moments and simple nonzero interior roots
    are assumptions. Coefficient inference additionally assumes fixed sign
    anchors and a valid local axis chart. Sample checks do not certify these
    population conditions. First-order marginal percentile coverage of each
    coordinate additionally requires nonzero first-order variance; a smooth
    loading or coefficient can violate that condition despite resolved axes.
    no exact coverage, familywise regions, selected-count inference, zero-root
    tests, p-values or inference df are provided. All requested draws are
    attempted and any failure refuses the entire result without replacement.
    Full sample, typed identities, draw indices/vectors/moments and diagnostics
    persist for independent replay. Raw resident CPU float64 input only;
    weights, clusters, Dataset, summary input and non-CPU fitting are excluded.
    """
    if any(isinstance(block, (list, tuple)) and len(block) > MAX_VARIABLES for block in (x, y)):
        raise AnalysisError("workspace_limit", "CCA bootstrap admits at most 16 total variables.")
    x_names, y_names = c.name_list(x, "x"), c.name_list(y, "y")
    if set(x_names) & set(y_names):
        raise AnalysisError("invalid_spec", "CCA variable sets must be disjoint.")
    names = x_names + y_names
    if len(names) > MAX_VARIABLES:
        raise AnalysisError("workspace_limit", "CCA bootstrap admits at most 16 total variables.")
    target = c.check_choice(target, "target", ("correlations", "coefficients"))
    components = c.check_count(components, "components", maximum=min(MAX_COMPONENTS, len(x_names), len(y_names)))
    replications = c.check_count(replications, "replications", minimum=19, maximum=MAX_REPLICATIONS)
    confidence = c.check_number(confidence, "confidence", minimum=0, maximum=1, exclusive=True)
    if confidence == 1 or (replications + 1) * (1 - confidence) / 2 < 1 - 1e-12:
        raise AnalysisError("insufficient_replications", "Each requested percentile tail needs at least one expected order statistic; increase replications or lower confidence.")
    seed = c.check_count(seed, "seed", minimum=0, maximum=2**63 - 1)
    c.check_choice(missing, "missing", ("drop", "raise"))
    if anchors is not None:
        if target != "coefficients":
            raise AnalysisError("invalid_option", "anchors apply only to coefficient uncertainty.")
        if not isinstance(anchors, (list, tuple)) or len(anchors) != components \
                or any(not isinstance(name, str) or name not in x_names for name in anchors):
            raise AnalysisError("invalid_spec", "anchors must contain one ordered X-variable name per component; repeated anchors are allowed.")
        anchors = list(anchors)
    w, rows = len(names), _rows(data, names)
    if w > MAX_VARIABLES or rows > MAX_ROWS:
        raise AnalysisError("workspace_limit", "CCA bootstrap admits at most 10000 resident physical rows and 16 total variables.")
    if any(len(name) > MAX_LABEL_BYTES or len(json.dumps(name).encode()) > MAX_LABEL_BYTES for name in names):
        raise AnalysisError("resource_limit", "CCA variable names exceed their 4096-byte portable label domain.")
    order, labels = _parameter_order(x_names, y_names, components, target)
    d, roots_count = len(labels), min(len(x_names), len(y_names))
    work = (replications + 1) * (rows * w * w + 64 * w**3) + 2 * replications * d * d + MAX_IDENTITY_NODES
    if work > MAX_WORK:
        raise AnalysisError("work_limit", "Cumulative CCA moment/refit and complete joint-covariance work exceeds 250 million operations.")
    # Admission includes all indices and tables, full metadata and repeated
    # escaped labels. This bound is checked before input coercion/tensor/draws.
    cells = rows * w + replications * (d + 2 * w + roots_count + len(_DIAGNOSTICS)) \
        + d * d + 8 * d + 4 * w * w + 6 * w * components
    export_bytes = 96 * cells + 16 * replications * rows + 96 * (rows + replications) \
        + 8 * MAX_IDENTITY_BYTES + 65536
    export_bytes += (4 * d + 24) * sum(len(json.dumps(label).encode()) for label in labels)
    if export_bytes > MAX_EXPORT_BYTES:
        raise AnalysisError("resource_limit", "Complete CCA bootstrap state exceeds its conservative 32 MiB export admission.")
    plan = plan_workspace("fixed CCA iid bootstrap", {
        "selected_and_refit_blocks": rows * w * 96,
        "row_indices_and_masks": rows * 64,
        "QR_and_spectral_workspace": 64 * w * w * 8,
        "full_draw_vectors_roots_and_moments": replications * (d + 2 * w + roots_count + len(_DIAGNOSTICS)) * 32,
        "all_draw_indices": replications * rows * 16,
        "joint_covariance_and_quantiles": 64 * d * d + 32 * replications * d,
        "portable_result_admission": export_bytes,
    }).record()
    sample, keep, dropped = c.select(_selected(data, names), names, missing=missing)
    if any(pd.api.types.is_bool_dtype(sample[name].dtype) for name in names):
        raise AnalysisError("non_numeric_column", "CCA bootstrap variables must be real numeric measurements, excluding booleans.")
    n = len(sample)
    if n < max(20, 2 * w + 1):
        raise AnalysisError("insufficient_observations", "CCA bootstrap needs at least max(20,2*(p+q)+1) complete rows.")
    column_names = list(data.columns.names) if isinstance(data, pd.DataFrame) else [None]
    index_codes, index_names, column_codes, identity_bytes = _identities(keep.index, keep.tolist(), column_names)
    values = c.matrix(sample, names)
    point = _parameters(values, x_names, y_names, components, target, anchors)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.empty((replications, d), dtype=c.FLOAT)
    indices = torch.empty((replications, n), dtype=torch.int64)
    diagnostics = torch.empty((replications, len(_DIAGNOSTICS)), dtype=c.FLOAT)
    means = torch.empty((replications, w), dtype=c.FLOAT)
    deviations = torch.empty_like(means)
    roots = torch.empty((replications, roots_count), dtype=c.FLOAT)
    failures = []
    for replication in range(replications):
        draw = torch.randint(n, (n,), generator=generator, device="cpu")
        indices[replication] = draw
        try:
            fitted = _parameters(values[draw], x_names, y_names, components, target,
                                 point.anchors, point if target == "coefficients" else None)
            draws[replication], diagnostics[replication] = fitted.vector, fitted.diagnostics
            means[replication], deviations[replication], roots[replication] = fitted.mean, fitted.sd, fitted.roots
        except (AnalysisError, torch.linalg.LinAlgError) as error:
            failures.append({"replication": replication + 1, "code": getattr(error, "code", "numerical_failure"), "message": str(error)})
    if failures:
        error = AnalysisError("bootstrap_failure", f"{len(failures)} of {replications} CCA refits failed. No uncertainty result is returned and no replicate is dropped or replaced.")
        error.failures = failures
        error.replications_attempted = replications
        error.successful_replications = replications - len(failures)
        raise error
    shifts = draws - draws[0]
    mean_shift = shifts.mean(0)
    centered = shifts - mean_shift
    correction = centered.mean(0)
    centered -= correction
    mean_shift += correction
    joint = centered.T @ centered / (replications - 1)
    joint = (joint + joint.T) / 2
    if bool(((centered.abs().amax(0) > 0) & (joint.diagonal() < torch.finfo(c.FLOAT).tiny)).any()):
        raise AnalysisError("numerical_failure", "A nonconstant joint CCA bootstrap variance underflows normal float64 precision; rescale the measurements.")
    se = joint.diagonal().sqrt()
    bounds = torch.quantile(draws, torch.tensor([(1 - confidence) / 2, (1 + confidence) / 2], dtype=c.FLOAT), dim=0, interpolation="linear")
    _finite(joint, se, bounds)
    positions = [i for i, flag in enumerate(keep.tolist()) if flag]
    identity = {"x": x_names, "y": y_names, "positions": positions, "index_codes": index_codes,
                "index_names": index_names, "index_nlevels": keep.index.nlevels,
                "column_names": column_codes, "target": target, "components": components,
                "anchors": point.anchors, "physical_rows": rows}
    source_digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    source_digest.update(values.numpy().tobytes())
    undefined = torch.full_like(point.vector, math.nan)
    axes, rep_index = c.numbered("Can", components), range(1, replications + 1)
    tables = {
        "estimates": c.frame(torch.stack((point.vector, se, bounds[0], bounds[1], (draws[0] - point.vector) + mean_shift, undefined, undefined), dim=1), columns=["estimate", "std_error", "ci_lower", "ci_upper", "bootstrap_bias", "p_value", "df"], index=labels),
        "covariance": c.frame(joint, columns=labels, index=labels),
        "replicates": c.frame(draws, columns=labels, index=rep_index),
        "resample_indices": table(indices.numpy(), columns=range(n), index=rep_index),
        "replicate_diagnostics": c.frame(diagnostics, columns=_DIAGNOSTICS, index=rep_index),
        "replicate_means": c.frame(means, columns=names, index=rep_index),
        "replicate_standard_deviations": c.frame(deviations, columns=names, index=rep_index),
        "replicate_correlations": c.frame(roots, columns=c.numbered("Can", roots_count), index=rep_index),
        "point_correlations": c.frame(point.roots[:, None], columns=["correlation"], index=c.numbered("Can", roots_count)),
        "point_covariance": c.frame(point.covariance, columns=names, index=names),
        "point_correlation": c.frame(point.correlation, columns=names, index=names),
        "point_diagnostics": c.frame(point.diagnostics[None], columns=_DIAGNOSTICS, index=["point"]),
        "descriptives": c.frame(torch.stack((point.mean, point.sd), dim=1), columns=["mean", "std_dev"], index=names),
        "sample": c.frame(values, columns=names, index=positions),
        "parameter_order": table(order, columns=["kind", "component", "block", "variable"], index=labels),
    }
    for key, matrix in point.matrices.items():
        tables["point_" + key] = c.frame(matrix, columns=axes, index=x_names if key.startswith("x") else y_names)
    output = TableSet(tables, title=f"Fixed CCA {target} iid bootstrap uncertainty",
        procedure="canon_bootstrap", state_schema=STATE_SCHEMA, target=target,
        components=components, fixed_count=True, x=x_names, y=y_names, variables=names,
        n=n, physical_rows=rows, n_missing=dropped, missing="listwise", missing_policy=missing,
        sample_positions=positions, sample_index_codes=index_codes, source_index_names=index_names,
        source_index_nlevels=keep.index.nlevels, source_column_names=column_codes, source_column_nlevels=1,
        source_content_sha256=source_digest.hexdigest(), source_identity_bytes=identity_bytes,
        source_identity_byte_limit=MAX_IDENTITY_BYTES, source_identity_label_byte_limit=MAX_LABEL_BYTES,
        source_identity_depth_limit=MAX_IDENTITY_DEPTH, source_identity_node_limit=MAX_IDENTITY_NODES,
        replications=replications, successful_replications=replications, failed_replications=[],
        confidence=confidence, seed=seed, sign_anchors=point.anchors,
        anchor_selection=None if target == "correlations" else "unique strongest standardized point X coefficient" if anchors is None else "caller declared",
        parameter_order=order, parameter_labels=labels, parameter_dimension=d,
        geometry_tolerance=GEOMETRY_TOLERANCE, maximum_block_correlation_condition=1e10,
        minimum_direction_cosine=MIN_DIRECTION_COSINE if target == "coefficients" else None,
        coefficient_normalization="unit sample variance for each raw canonical variate; positive within-pair correlation",
        axis_identity="fixed decreasing simple correlations; coefficient signs fixed by X anchors; every signed draw coefficient remains in a 45-degree point-covariance chart; no permutation or alignment" if target == "coefficients" else "fixed decreasing simple correlations; no coefficient orientation is an inferential target",
        covariance_divisor=replications - 1, quantile_interpolation="linear",
        minimum_raw_sample_variance=torch.finfo(c.FLOAT).tiny,
        minimum_nonconstant_bootstrap_variance=torch.finfo(c.FLOAT).tiny,
        rng="torch.Generator CPU randint(n, (n,)); draws in increasing replication order",
        rng_version=torch.__version__, resample_index_base=0,
        resample_indices_reference="complete sample row ordinal", precision="float64", device="cpu",
        inference_target=f"fixed simple canonical {target} functional",
        inferential_assumptions="IID complete-case rows; fixed dimension and component count; finite fourth moments; interior nonsingular population block moments; simple nonzero canonical roots away from one with retained/omitted separation; coefficient target additionally fixed nonzero sign anchors and valid local direction charts; first-order coverage of each coordinate requires nonzero first-order variance, including smooth coefficients/loadings; population conditions assumed, not certified by sample checks",
        uncertainty="first-order IID marginal percentile bootstrap", familywise_intervals=False,
        p_values_available=False, inference_df_available=False, reestimate_moments_every_draw=True,
        restandardize_every_draw=True, resource_plan=plan, estimated_work=work,
        estimated_complete_export_bytes=export_bytes, portable_export_limit_bytes=MAX_EXPORT_BYTES,
        notes=["No exact coverage, simultaneous region, selected-count inference or zero-root test is claimed.",
               "All requested draws must pass; all failed refits are collected and the complete uncertainty is refused.",
               "Complete JSON preserves sample, row identities, all draw indices/parameters/moments and fixed identification settings; console previews are incomplete."])
    saved_summary(output)
    output.attrs["state_content_sha256"] = _state_digest(output)
    return output
