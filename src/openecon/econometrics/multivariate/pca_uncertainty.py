"""IID uncertainty for fixed, simple covariance/correlation PCA eigenpairs.

Every draw recomputes centred sample moments; correlation PCA also recomputes
all sample standard deviations. This is uncertainty for a fixed eigenpair
functional, not a test or confidence region for a selected component count.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype

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
MAX_REPLICATIONS = 1_999
MAX_PARAMETERS = 128
MAX_WORK = 250_000_000
MAX_EXPORT_BYTES = 32 * 1024**2
MAX_INDEX_BYTES = 1024**2
MAX_INDEX_NODES = 100_000
MAX_INDEX_DEPTH = 32
MAX_INDEX_TEXT = 64_000
EIGENVALUE_TOLERANCE = 1e-8
ANCHOR_TOLERANCE = 1e-8
MIN_DIRECTION_COSINE = math.sqrt(.5)


def _rows(data, names):
    if isinstance(data, (Dataset, Summary)):
        raise AnalysisError("unsupported_input", "PCA bootstrap requires resident raw rows; Dataset and summary matrices are unsupported.")
    if isinstance(data, pd.DataFrame):
        return len(data)
    if isinstance(data, Mapping):
        if any(name not in data for name in names):
            raise AnalysisError("missing_columns", "A declared PCA variable is absent.")
        lengths = []
        for name in names:
            value = data[name]
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                raise AnalysisError("unsupported_device", "PCA bootstrap requires CPU inputs.")
            try:
                lengths.append(len(value))
            except TypeError:
                raise AnalysisError("invalid_data", "Supply sized resident numeric columns.") from None
        if len(set(lengths)) != 1:
            raise AnalysisError("invalid_data", "Resident columns have different lengths.")
        return lengths[0]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return len(data)
    raise AnalysisError("invalid_data", "Supply a resident DataFrame, sized column mapping or row records.")


def _selected(data, names):
    if isinstance(data, pd.DataFrame):
        if data.columns.nlevels != 1:
            raise AnalysisError("invalid_data", "PCA bootstrap requires one source column level with declared string variables.")
        if data.columns.has_duplicates:
            raise AnalysisError("duplicate_columns", "Data must have unique column names.")
        if any(name not in data.columns for name in names):
            raise AnalysisError("missing_columns", "A declared PCA variable is absent.")
        return data.loc[:, names]
    if isinstance(data, Mapping):
        return {name: data[name] for name in names}
    if not all(isinstance(row, Mapping) for row in data):
        raise AnalysisError("invalid_data", "Each resident record must map column names to numeric values.")
    return [{name: row.get(name) for name in names} for row in data]


def _finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise AnalysisError("numerical_failure", "PCA arithmetic exceeds finite float64 precision; rescale the measurements.")


def _index_codes(index, flags, column_names):
    """Bound label traversal/encoding while preserving the shared typed codec."""
    codes, names, columns = [], [], []
    used_bytes = nodes = 0
    for values, output in ((zip(index, flags, strict=True), codes),
                           (((name, True) for name in index.names), names),
                           (((name, True) for name in column_names), columns)):
        for value, keep in values:
            if not keep:
                continue
            pending = [(value, 0)]
            while pending:
                item, depth = pending.pop()
                nodes += 1
                if depth > MAX_INDEX_DEPTH or nodes > MAX_INDEX_NODES:
                    raise AnalysisError("resource_limit", "Source identities exceed bounded tuple depth or traversal work.")
                if isinstance(item, (str, bytes)) and len(item) > MAX_INDEX_TEXT:
                    raise AnalysisError("resource_limit", "One source identity exceeds 64000 text/byte units.")
                if isinstance(item, tuple):
                    if len(item) > MAX_INDEX_NODES-nodes:
                        raise AnalysisError("resource_limit", "Source identities exceed bounded tuple traversal work.")
                    pending.extend((child, depth+1) for child in item)
            try:
                code = encode(value)
            except (TypeError, ValueError, OverflowError, RecursionError, AnalysisError):
                raise AnalysisError("invalid_index", "Source row labels need supported lossless scalar/tuple identities.") from None
            used_bytes += len(code.encode())
            if used_bytes > MAX_INDEX_BYTES:
                raise AnalysisError("resource_limit", "Encoded source row identities exceed one MiB.")
            output.append(code)
    return codes, names, columns


def _parameters(x, names, matrix, components, anchors, point_vectors=None):
    # An actual row origin prevents irrelevant large levels from corrupting the
    # sample covariance. Each bootstrap draw receives its own origin and mean.
    shifted = x-x[0]
    centred, offset = c.centre(shifted)
    covariance = centred.T @ centred/(len(x)-1)
    covariance = (covariance+covariance.T)/2
    mean = x[0]+offset
    variance = covariance.diagonal()
    _finite(mean, covariance)
    if bool((variance <= 0).any()):
        raise AnalysisError("zero_variance", "Every PCA variable must vary in every admitted sample.")
    if bool((variance < torch.finfo(c.FLOAT).tiny).any()):
        raise AnalysisError("numerical_failure", "A raw sample variance is subnormal in float64 before PCA restandardization; rescale the measurements.")
    sd = variance.sqrt()
    correlation = covariance/torch.outer(sd, sd)
    correlation = (correlation+correlation.T)/2
    correlation.diagonal().fill_(1)
    _finite(sd, correlation)
    # The bootstrap domain excludes singular or unresolved complete moments.
    spectrum = torch.linalg.eigvalsh(correlation)
    if float(spectrum[0]) <= 1e-11*float(spectrum[-1]):
        raise AnalysisError("singular_matrix", "Complete PCA moments are singular or numerically unresolved.")
    target = covariance if matrix == "covariance" else correlation
    roots, vectors = torch.linalg.eigh(target)
    roots, vectors = roots.flip(0), vectors.flip(1)
    _finite(roots, vectors)
    spectral_scale = float(roots[0])
    if spectral_scale <= 0 or float(roots[components-1]) <= EIGENVALUE_TOLERANCE*spectral_scale:
        raise AnalysisError("unidentified_component", "Retained eigenvalues must be positive away from numerical zero.")
    gaps = []
    for j in range(components):
        adjacent = []
        if j:
            adjacent.append(roots[j-1]-roots[j])
        if j+1 < len(names):
            adjacent.append(roots[j]-roots[j+1])
        gap = float(torch.stack(adjacent).min())/spectral_scale
        if gap <= EIGENVALUE_TOLERANCE:
            raise AnalysisError("unidentified_component", "Every retained root must be simple and separated from all adjacent roots, including the retained/discarded boundary.")
        gaps.append(gap)
    vectors = vectors[:, :components]
    if anchors is None:
        anchors = [names[int(vectors[:, j].abs().argmax())] for j in range(components)]
    anchor_values = torch.tensor([float(vectors[names.index(name), j])
                                  for j, name in enumerate(anchors)], dtype=c.FLOAT)
    if bool((anchor_values.abs() <= ANCHOR_TOLERANCE).any()):
        raise AnalysisError("unidentified_anchor", "Every recorded sign anchor must remain away from zero.")
    vectors = vectors*anchor_values.sign()
    cosines = torch.ones(components, dtype=c.FLOAT)
    if point_vectors is not None:
        cosines = (point_vectors*vectors).sum(0)
        if bool((cosines <= MIN_DIRECTION_COSINE).any()):
            raise AnalysisError("component_crossing", "A replicated signed direction left its local point-axis chart; no axis permutation, sign relabelling or Procrustes alignment is applied.")
    loadings = vectors*roots[:components].sqrt()
    parameters = torch.cat([torch.cat((roots[j:j+1], vectors[:, j], loadings[:, j]))
                            for j in range(components)])
    _finite(parameters)
    diagnostics = torch.tensor([min(gaps), float(anchor_values.abs().min()),
                                float(cosines.min()), float(roots[:components].min())], dtype=c.FLOAT)
    return parameters, anchors, vectors, loadings, roots, mean, sd, covariance, target, diagnostics


@c.procedure
@resident_cpu
def pca_bootstrap(data: Any, columns: list[str], *, components: int,
                  matrix: str = "correlation", anchors: list[str] | None = None,
                  replications: int = 199, confidence: float = .95, seed: int = 0,
                  missing: str = "drop") -> TableSet:
    """Full IID percentile uncertainty for a fixed covariance/correlation PCA.

    Complete rows are resampled independently and their means, unbiased
    covariance and standard deviations are re-estimated. The correlation
    target is restandardized in every draw. The caller fixes the component
    count; components remain in decreasing-root order with recorded signs.
    Every retained root, anchor and local signed direction must be resolved in
    the point fit and every draw. All requested draws must succeed; failures
    refuse the entire result. No draw is aligned, permuted or silently removed.

    The target is the fixed simple-eigenpair functional under IID complete-case
    sampling, fixed dimension, finite fourth moments, interior nonsingular
    population moments, separated retained roots and anchors away from zero.
    These population conditions are assumptions, not certified by sample
    guards. Intervals are first-order marginal percentiles, not simultaneous
    regions or exact coverage. P-values/df, selected counts, weights, clusters,
    summary input, Dataset and non-CPU calculation are unsupported.
    """
    names = c.name_list(columns, "columns", minimum=2)
    matrix = c.check_choice(matrix, "matrix", ("covariance", "correlation"))
    components = c.check_count(components, "components", maximum=len(names))
    replications = c.check_count(replications, "replications", minimum=19, maximum=MAX_REPLICATIONS)
    confidence = c.check_number(confidence, "confidence", minimum=0, maximum=1, exclusive=True)
    if confidence == 1 or (replications+1)*(1-confidence)/2 < 1-1e-12:
        raise AnalysisError("insufficient_replications", "Each requested percentile tail needs at least one expected bootstrap order statistic; increase replications or lower confidence.")
    seed = c.check_count(seed, "seed", minimum=0, maximum=2**63-1)
    c.check_choice(missing, "missing", ("drop", "raise"))
    if anchors is not None:
        if not isinstance(anchors, (list, tuple)) or len(anchors) != components \
                or any(not isinstance(name, str) or name not in names for name in anchors):
            raise AnalysisError("invalid_spec", "anchors must contain one ordered analysed-variable name per component.")
        anchors = list(anchors)
    p, rows = len(names), _rows(data, names)
    q = components*(1+2*p)
    if p > MAX_VARIABLES or rows > MAX_ROWS or q > MAX_PARAMETERS:
        raise AnalysisError("workspace_limit", "PCA bootstrap admits 10000 physical rows, 16 variables and 128 joint parameters.")
    if any(len(json.dumps(name).encode()) > 4096 for name in names):
        raise AnalysisError("resource_limit", "PCA variable labels exceed the bounded portable result domain.")
    work = (replications+1)*(rows*p*p+64*p**3)+2*replications*q*q+MAX_INDEX_NODES
    if work > MAX_WORK:
        raise AnalysisError("work_limit", "Cumulative PCA moment/refit and joint-covariance work exceeds 250 million operations.")
    # Include complete draw indices, numeric tables, JSON and repeated labels;
    # restoration must never depend on a preview or an unbounded output object.
    float_cells = rows*p+replications*(q+2*p+4)+q*q+8*q+6*p*p
    export_bytes = 32*float_cells+8*replications*rows+64*(rows+replications)+4*MAX_INDEX_BYTES
    export_bytes += 16*q*sum(len(json.dumps(name).encode()) for name in names)
    if export_bytes > MAX_EXPORT_BYTES:
        raise AnalysisError("resource_limit", "Complete PCA bootstrap state exceeds its conservative 32 MiB export admission.")
    plan = plan_workspace("fixed PCA iid bootstrap", {
        "selected_and_refit_blocks": rows*p*64,
        "sample_indices_and_masks": rows*64,
        "spectral_workspace": 64*p*p*8,
        "all_draw_parameters_and_moments": replications*(q+2*p+4)*32,
        "all_draw_indices": replications*rows*16,
        "joint_covariance_and_quantiles": q*q*64+q*replications*32,
        "portable_result_admission": export_bytes,
    }).record()
    sample, keep, dropped = c.select(_selected(data, names), names, missing=missing)
    if any(is_bool_dtype(sample[name].dtype) for name in names):
        raise AnalysisError("non_numeric_column", "PCA bootstrap variables must be real numeric measurements, excluding booleans.")
    n = len(sample)
    if n < max(20, 2*p+1):
        raise AnalysisError("insufficient_observations", "PCA bootstrap needs at least max(20,2*p+1) complete rows.")
    source_column_names = list(data.columns.names) if isinstance(data, pd.DataFrame) else [None]
    index_codes, index_names, column_names = _index_codes(keep.index, keep.tolist(), source_column_names)
    x = c.matrix(sample, names)
    point = _parameters(x, names, matrix, components, anchors)
    parameters, fixed_anchors, vectors, loadings, roots, mean, sd, covariance, target, diagnostic = point
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.empty((replications, q), dtype=c.FLOAT)
    indices = torch.empty((replications, n), dtype=torch.int64)
    diagnostics = torch.empty((replications, 4), dtype=c.FLOAT)
    means = torch.empty((replications, p), dtype=c.FLOAT)
    deviations = torch.empty_like(means)
    failures = []
    for b in range(replications):
        draw = torch.randint(n, (n,), generator=generator, device="cpu")
        indices[b] = draw
        try:
            fitted = _parameters(x[draw], names, matrix, components, fixed_anchors, vectors)
            draws[b], diagnostics[b], means[b], deviations[b] = fitted[0], fitted[-1], fitted[5], fitted[6]
        except (AnalysisError, torch.linalg.LinAlgError) as error:
            failures.append({"replication": b+1, "code": getattr(error, "code", "numerical_failure"),
                             "message": str(error)})
    if failures:
        error = AnalysisError("bootstrap_failure", f"{len(failures)} of {replications} PCA refits failed. No uncertainty result is returned and no replicate is dropped.")
        error.failures = failures
        error.replications_attempted = replications
        error.successful_replications = replications-len(failures)
        raise error
    centred = draws-draws.mean(0)
    joint = centred.T @ centred/(replications-1)
    joint = (joint+joint.T)/2
    if bool(((centred.abs().amax(0) > 0) & (joint.diagonal() < torch.finfo(c.FLOAT).tiny)).any()):
        raise AnalysisError("numerical_failure", "A nonconstant joint bootstrap variance underflows normal float64 precision; rescale covariance-PCA measurements.")
    se = joint.diagonal().sqrt()
    bounds = torch.quantile(draws, torch.tensor([(1-confidence)/2, (1+confidence)/2], dtype=c.FLOAT),
                            dim=0, interpolation="linear")
    _finite(joint, se, bounds)
    order = []
    for j in range(components):
        label = f"Comp{j+1}"
        order.append(["eigenvalue", label, None])
        order.extend(["eigenvector", label, name] for name in names)
        order.extend(["loading", label, name] for name in names)
    labels = [":".join(value for value in entry if value is not None) for entry in order]
    positions = [i for i, flag in enumerate(keep.tolist()) if flag]
    identity = {"variables": names, "positions": positions, "index_codes": index_codes,
                "index_names": index_names, "index_nlevels": keep.index.nlevels,
                "column_names": column_names, "column_nlevels": 1,
                "matrix": matrix, "components": components, "anchors": fixed_anchors}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    digest.update(x.numpy().tobytes())
    undefined = torch.full_like(parameters, float("nan"))
    tables = {
        "estimates": c.frame(torch.stack((parameters, se, bounds[0], bounds[1], draws.mean(0)-parameters,
                                           undefined, undefined), dim=1),
                             columns=["estimate", "std_error", "ci_lower", "ci_upper", "bootstrap_bias", "p_value", "df"], index=labels),
        "covariance": c.frame(joint, columns=labels, index=labels),
        "replicates": c.frame(draws, columns=labels, index=range(1, replications+1)),
        "resample_indices": table(indices.numpy(), columns=range(n), index=range(1, replications+1)),
        "replicate_diagnostics": c.frame(diagnostics, columns=["minimum_relative_eigenvalue_gap", "minimum_absolute_anchor", "minimum_signed_direction_cosine", "minimum_retained_eigenvalue"], index=range(1, replications+1)),
        "replicate_means": c.frame(means, columns=names, index=range(1, replications+1)),
        "replicate_standard_deviations": c.frame(deviations, columns=names, index=range(1, replications+1)),
        "point_eigenvalues": c.frame(roots[:, None], columns=["eigenvalue"], index=c.numbered("Comp", p)),
        "point_eigenvectors": c.frame(vectors, columns=c.numbered("Comp", components), index=names),
        "point_loadings": c.frame(loadings, columns=c.numbered("Comp", components), index=names),
        "point_covariance": c.frame(covariance, columns=names, index=names),
        "point_matrix": c.frame(target, columns=names, index=names),
        "descriptives": c.frame(torch.stack((mean, sd), dim=1), columns=["mean", "std_dev"], index=names),
        "sample": c.frame(x, columns=names, index=positions),
        "parameter_order": table(order, columns=["kind", "component", "variable"], index=labels),
    }
    output = TableSet(tables, title=f"Fixed {matrix} PCA iid bootstrap uncertainty",
        procedure="pca_bootstrap", matrix=matrix, components=components, variables=names,
        n=n, physical_rows=rows, n_missing=dropped, missing="listwise", missing_policy=missing,
        sample_positions=positions, sample_index_codes=index_codes, source_index_names=index_names,
        source_index_nlevels=keep.index.nlevels, source_content_sha256=digest.hexdigest(),
        source_column_names=column_names, source_column_nlevels=1,
        source_index_byte_limit=MAX_INDEX_BYTES, source_index_node_limit=MAX_INDEX_NODES,
        source_index_depth_limit=MAX_INDEX_DEPTH, source_index_text_limit=MAX_INDEX_TEXT,
        replications=replications, successful_replications=replications, failed_replications=[],
        confidence=confidence, seed=seed, sign_anchors=fixed_anchors,
        anchor_selection="strongest point entry" if anchors is None else "caller declared",
        parameter_order=order, parameter_labels=labels, parameter_dimension=q,
        point_diagnostics=diagnostic.tolist(), eigenvalue_gap_tolerance=EIGENVALUE_TOLERANCE,
        sign_anchor_tolerance=ANCHOR_TOLERANCE, minimum_direction_cosine=MIN_DIRECTION_COSINE,
        axis_identity="fixed descending-root order and signs; every draw stays within a signed 45-degree point-axis chart; no permutation or alignment",
        covariance_divisor=replications-1, quantile_interpolation="linear", covariance_order="component: eigenvalue, ordered eigenvector, ordered loading",
        minimum_nonconstant_bootstrap_variance=torch.finfo(c.FLOAT).tiny,
        minimum_raw_sample_variance=torch.finfo(c.FLOAT).tiny,
        rng="torch.Generator CPU randint(n, (n,)); draws in increasing replication order", rng_version=torch.__version__,
        resample_index_base=0, resample_indices_reference="complete sample row ordinal",
        precision="float64", device="cpu", state_schema="openecon.pca_bootstrap.v1",
        inference_target=f"fixed simple eigenpairs of the population {matrix} functional",
        inferential_assumptions="IID complete-case rows; fixed dimension and component count; finite fourth moments; interior nonsingular population moments; separated retained roots and fixed sign anchors away from zero; population conditions assumed, not certified by sample guards",
        uncertainty="first-order IID marginal percentile bootstrap", familywise_intervals=False,
        p_values_available=False, inference_df_available=False, reestimate_moments_every_draw=True,
        restandardize_every_draw=matrix == "correlation", resource_plan=plan,
        estimated_work=work, estimated_complete_export_bytes=export_bytes,
        notes=["No exact coverage, simultaneous region, selected-count inference or licensed vendor parity is claimed.",
               "All requested replicas must satisfy the same finite/rank/gap/anchor/chart checks; no failed draw is dropped."])
    return saved_summary(output)
