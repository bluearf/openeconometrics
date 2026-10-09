"""IID uncertainty for invariant, fixed-rank PCA subspaces.

The target is the spectral projector, not a basis inside the subspace. Only
the retained/discarded boundary needs to be separated; roots inside either
block may repeat. Projector bilinear functionals and their distinction from
individual eigenvectors are developed by Koltchinskii and Lounici:
https://arxiv.org/abs/1408.4643 . The specialized confidence-set bootstrap in
https://arxiv.org/abs/1703.00871 is not implemented here. Ordinary IID row
resampling below supplies first-order marginal percentile uncertainty, with
no simultaneous confidence ball or finite-sample coverage guarantee.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from decimal import Decimal
from itertools import chain
from typing import Any

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.resources import plan_workspace

from . import common as c
from . import pca_uncertainty as u

MAX_VARIABLES = 15
PROJECTOR_TOLERANCE = 1e-9
MAX_NUMERIC_INDEX_STORAGE = 4096


def _measurement_admission(data, names):
    """Refuse unsupported tensor representations before table coercion."""
    if isinstance(data, Mapping):
        for name in names:
            column = data[name]
            if isinstance(column, torch.Tensor) and (column.ndim != 1
                    or column.layout != torch.strided or column.requires_grad):
                raise AnalysisError("invalid_data", "Tensor measurement columns must be plain one-dimensional strided CPU values without an autograd graph.")


def _numeric_identity_admission(index, flags, column_names):
    """Bound numeric scalar conversion before the shared recursive codec."""
    nodes = 0
    values = chain((value for value, keep in zip(index, flags, strict=True) if keep),
                   index.names, column_names)
    for value in values:
        pending = [(value, 0)]
        while pending:
            item, depth = pending.pop()
            nodes += 1
            if depth > u.MAX_INDEX_DEPTH or nodes > u.MAX_INDEX_NODES:
                raise AnalysisError("resource_limit", "Source identities exceed bounded tuple depth or traversal work.")
            if isinstance(item, int) and item.bit_length() > 4*MAX_NUMERIC_INDEX_STORAGE \
                    or isinstance(item, Decimal) and item.__sizeof__() > MAX_NUMERIC_INDEX_STORAGE:
                raise AnalysisError("resource_limit", "A numeric source identity exceeds bounded scalar storage before encoding.")
            if isinstance(item, tuple):
                if len(item) > u.MAX_INDEX_NODES-nodes:
                    raise AnalysisError("resource_limit", "Source identities exceed bounded tuple traversal work.")
                pending.extend((child, depth+1) for child in item)


def _parameters(x, matrix, components, point_projector=None):
    """Stable moments and a projector; no axis identification or alignment."""
    shifted = x-x[0]
    centred, offset = c.centre(shifted)
    covariance = centred.T @ centred/(len(x)-1)
    covariance = (covariance+covariance.T)/2
    mean = x[0]+offset
    variance = covariance.diagonal()
    u._finite(mean, covariance)
    if bool((variance <= 0).any()):
        raise AnalysisError("zero_variance", "Every PCA variable must vary in every admitted sample.")
    if bool((variance < torch.finfo(c.FLOAT).tiny).any()):
        raise AnalysisError("numerical_failure", "A raw sample variance is subnormal in float64 before PCA restandardization; rescale the measurements.")
    sd = variance.sqrt()
    correlation = covariance/torch.outer(sd, sd)
    correlation = (correlation+correlation.T)/2
    correlation.diagonal().fill_(1)
    u._finite(sd, correlation)
    spectrum = torch.linalg.eigvalsh(correlation)
    if float(spectrum[0]) <= 1e-11*float(spectrum[-1]):
        raise AnalysisError("singular_matrix", "Complete PCA moments are singular or numerically unresolved.")
    target = covariance if matrix == "covariance" else correlation
    roots, vectors = torch.linalg.eigh(target)
    roots, vectors = roots.flip(0), vectors.flip(1)
    u._finite(roots, vectors)
    spectral_scale = float(roots[0])
    if spectral_scale <= 0 or float(roots[components-1]) <= u.EIGENVALUE_TOLERANCE*spectral_scale:
        raise AnalysisError("unidentified_subspace", "The retained PCA subspace must have positive eigenvalues away from numerical zero.")
    gap = float(roots[components-1]-roots[components])/spectral_scale
    if gap <= u.EIGENVALUE_TOLERANCE:
        raise AnalysisError("unidentified_subspace", "The fixed PCA subspace needs a separated retained/discarded boundary; internal repeated roots are permitted.")
    retained = vectors[:, :components]
    projector = retained @ retained.T
    projector = (projector+projector.T)/2
    idempotence = float((projector @ projector-projector).abs().max())
    trace_error = abs(float(torch.trace(projector))-components)
    u._finite(projector)
    if idempotence > PROJECTOR_TOLERANCE or trace_error > PROJECTOR_TOLERANCE:
        raise AnalysisError("numerical_failure", "The PCA projector lost its fixed-rank orthogonal-projector identities.")
    # Sums of nonnegative roots avoid subtracting two nearly equal traces.
    # They equal tr(P A) and tr((I-P) A), independently checked in tests.
    captured = roots[:components].sum()
    residual = roots[components:].sum()
    if float(residual) <= 0:
        raise AnalysisError("numerical_failure", "Discarded PCA variance is not positive at resolved float64 precision; rescale the measurements.")
    total = captured+residual
    triangular = torch.triu_indices(x.shape[1], x.shape[1], device="cpu")
    parameters = torch.cat((projector[triangular[0], triangular[1]],
                            torch.stack((captured, residual, captured/total))))
    distance = 0. if point_projector is None else float(torch.linalg.vector_norm(projector-point_projector))
    diagnostics = torch.tensor([gap, float(roots[components-1]), float(roots[components]),
                                idempotence, trace_error, distance], dtype=c.FLOAT)
    u._finite(parameters, diagnostics)
    return parameters, projector, roots, mean, sd, covariance, target, diagnostics


@c.procedure
@resident_cpu
def pca_subspace_bootstrap(data: Any, columns: list[str], *, components: int,
                           matrix: str = "correlation", replications: int = 199,
                           confidence: float = .95, seed: int = 0,
                           missing: str = "drop") -> TableSet:
    """IID marginal percentile uncertainty for a fixed PCA spectral projector.

    The caller fixes 1 <= components < the number of variables. For covariance
    PCA, each draw re-estimates unbiased complete-case sample covariance. For
    correlation PCA it additionally re-estimates each standard deviation. Only
    the retained/discarded eigenvalue gap is required; repeated roots inside
    either block are admitted. No basis, sign anchors, alignment, permutations
    or individual repeated-root intervals are used.

    ``estimates`` and the full joint ``covariance`` refer to the projector's
    upper triangle in declared variable order, followed by captured variance,
    residual variance and captured trace share. Complete ``replicates``, draw
    indices, sample rows/typed identities, point moments/projector and per-draw
    moments/diagnostics are saved. The covariance can be structurally singular:
    no inverse, Wald statistic, df, p-value or simultaneous region is supplied.

    Assumptions are IID complete-case rows, fixed dimension/rank, finite fourth
    moments, interior nonsingular population moments and a positive population
    boundary gap. First-order percentile coverage of an individual coordinate
    additionally needs a nonzero first-order variance; special axis-aligned
    populations can violate that condition. Sample guards do not certify these
    population assumptions. The method does not support weights, clusters,
    summary input, Dataset, selected ranks or non-CPU calculation. Every draw
    must pass; a failed draw refuses the entire result.
    """
    names = c.name_list(columns, "columns", minimum=2)
    matrix = c.check_choice(matrix, "matrix", ("covariance", "correlation"))
    components = c.check_count(components, "components", maximum=len(names)-1)
    replications = c.check_count(replications, "replications", minimum=19, maximum=u.MAX_REPLICATIONS)
    confidence = c.check_number(confidence, "confidence", minimum=0, maximum=1, exclusive=True)
    if confidence == 1 or (replications+1)*(1-confidence)/2 < 1-1e-12:
        raise AnalysisError("insufficient_replications", "Each requested percentile tail needs at least one expected bootstrap order statistic; increase replications or lower confidence.")
    seed = c.check_count(seed, "seed", minimum=0, maximum=2**63-1)
    c.check_choice(missing, "missing", ("drop", "raise"))
    p, rows = len(names), u._rows(data, names)
    q = p*(p+1)//2+3
    if p > MAX_VARIABLES or rows > u.MAX_ROWS or q > u.MAX_PARAMETERS:
        raise AnalysisError("workspace_limit", "PCA subspace bootstrap admits 10000 physical rows, 15 variables and 128 joint parameters.")
    label_bytes = [len(json.dumps(name).encode()) for name in names]
    if any(size > 4096 for size in label_bytes):
        raise AnalysisError("resource_limit", "PCA variable labels exceed the bounded portable result domain.")
    work = (replications+1)*(rows*p*p+64*p**3)+2*replications*q*q+2*u.MAX_INDEX_NODES
    if work > u.MAX_WORK:
        raise AnalysisError("work_limit", "Cumulative PCA moment/refit and joint-covariance work exceeds 250 million operations.")
    float_cells = rows*p+replications*(q+2*p+6)+q*q+8*q+6*p*p
    export_bytes = 32*float_cells+8*replications*rows+64*(rows+replications)+4*u.MAX_INDEX_BYTES
    export_bytes += 16*q*sum(label_bytes)
    if export_bytes > u.MAX_EXPORT_BYTES:
        raise AnalysisError("resource_limit", "Complete PCA subspace bootstrap state exceeds its conservative 32 MiB export admission.")
    plan = plan_workspace("fixed PCA subspace iid bootstrap", {
        "selected_and_refit_blocks": rows*p*64,
        "sample_indices_and_masks": rows*64,
        "spectral_and_projector_workspace": 64*p*p*8,
        "all_draw_parameters_and_moments": replications*(q+2*p+6)*32,
        "all_draw_indices": replications*rows*16,
        "joint_covariance_and_quantiles": q*q*64+q*replications*32,
        "portable_result_admission": export_bytes,
    }).record()
    _measurement_admission(data, names)
    sample, keep, dropped = c.select(u._selected(data, names), names, missing=missing)
    if any(is_bool_dtype(sample[name].dtype) for name in names):
        raise AnalysisError("non_numeric_column", "PCA bootstrap variables must be real numeric measurements, excluding booleans.")
    n = len(sample)
    if n < max(20, 2*p+1):
        raise AnalysisError("insufficient_observations", "PCA subspace bootstrap needs at least max(20,2*p+1) complete rows.")
    source_column_names = list(data.columns.names) if isinstance(data, pd.DataFrame) else [None]
    _numeric_identity_admission(keep.index, keep.tolist(), source_column_names)
    index_codes, index_names, column_names = u._index_codes(keep.index, keep.tolist(), source_column_names)
    x = c.matrix(sample, names)
    parameters, projector, roots, mean, sd, covariance, target, diagnostic = _parameters(x, matrix, components)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.empty((replications, q), dtype=c.FLOAT)
    indices = torch.empty((replications, n), dtype=torch.int64)
    diagnostics = torch.empty((replications, 6), dtype=c.FLOAT)
    means = torch.empty((replications, p), dtype=c.FLOAT)
    deviations = torch.empty_like(means)
    failures = []
    for b in range(replications):
        draw = torch.randint(n, (n,), generator=generator, device="cpu")
        indices[b] = draw
        try:
            fitted = _parameters(x[draw], matrix, components, projector)
            draws[b], diagnostics[b], means[b], deviations[b] = fitted[0], fitted[-1], fitted[3], fitted[4]
        except (AnalysisError, torch.linalg.LinAlgError) as error:
            failures.append({"replication": b+1, "code": getattr(error, "code", "numerical_failure"),
                             "message": str(error)})
    if failures:
        error = AnalysisError("bootstrap_failure", f"{len(failures)} of {replications} PCA subspace refits failed. No uncertainty result is returned and no replicate is dropped.")
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
    u._finite(joint, se, bounds)
    order = [["projector", names[i], names[j]] for i in range(p) for j in range(i, p)]
    order.extend([["captured_variance", None, None], ["residual_variance", None, None],
                  ["captured_trace_share", None, None]])
    # Ordinals keep covariance labels collision-free even when variable names
    # themselves contain colons or coincide with scalar functional names.
    labels = [f"P[{i+1},{j+1}]" for i in range(p) for j in range(i, p)]
    labels.extend(["captured_variance", "residual_variance", "captured_trace_share"])
    positions = [i for i, flag in enumerate(keep.tolist()) if flag]
    identity = {"variables": names, "positions": positions, "index_codes": index_codes,
                "index_names": index_names, "index_nlevels": keep.index.nlevels,
                "column_names": column_names, "column_nlevels": 1,
                "matrix": matrix, "components": components}
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
        "replicate_diagnostics": c.frame(diagnostics, columns=["relative_boundary_eigenvalue_gap", "minimum_retained_eigenvalue", "maximum_discarded_eigenvalue", "maximum_projector_idempotence_error", "projector_trace_error", "frobenius_distance_to_point_projector"], index=range(1, replications+1)),
        "replicate_means": c.frame(means, columns=names, index=range(1, replications+1)),
        "replicate_standard_deviations": c.frame(deviations, columns=names, index=range(1, replications+1)),
        "point_eigenvalues": c.frame(roots[:, None], columns=["eigenvalue"], index=c.numbered("Comp", p)),
        "point_projector": c.frame(projector, columns=names, index=names),
        "point_covariance": c.frame(covariance, columns=names, index=names),
        "point_matrix": c.frame(target, columns=names, index=names),
        "descriptives": c.frame(torch.stack((mean, sd), dim=1), columns=["mean", "std_dev"], index=names),
        "sample": c.frame(x, columns=names, index=positions),
        "parameter_order": table(order, columns=["kind", "row_variable", "column_variable"], index=labels),
    }
    output = TableSet(tables, title=f"Fixed {matrix} PCA subspace iid bootstrap uncertainty",
        procedure="pca_subspace_bootstrap", matrix=matrix, components=components, variables=names,
        n=n, physical_rows=rows, n_missing=dropped, missing="listwise", missing_policy=missing,
        sample_positions=positions, sample_index_codes=index_codes, source_index_names=index_names,
        source_index_nlevels=keep.index.nlevels, source_content_sha256=digest.hexdigest(),
        source_column_names=column_names, source_column_nlevels=1,
        source_index_byte_limit=u.MAX_INDEX_BYTES, source_index_node_limit=u.MAX_INDEX_NODES,
        source_index_depth_limit=u.MAX_INDEX_DEPTH, source_index_text_limit=u.MAX_INDEX_TEXT,
        source_numeric_index_storage_limit=MAX_NUMERIC_INDEX_STORAGE,
        replications=replications, successful_replications=replications, failed_replications=[],
        confidence=confidence, seed=seed, parameter_order=order, parameter_labels=labels,
        parameter_dimension=q, point_diagnostics=diagnostic.tolist(),
        eigenvalue_gap_tolerance=u.EIGENVALUE_TOLERANCE, projector_identity_tolerance=PROJECTOR_TOLERANCE,
        subspace_identity="fixed rank and ordered variable coordinates; only the retained/discarded boundary gap; invariant under signs and rotations within either spectral block",
        covariance_divisor=replications-1, quantile_interpolation="linear",
        covariance_order="projector upper triangle in declared row/column variable order, captured variance, residual variance, captured trace share",
        covariance_may_be_singular=True, covariance_inverse_available=False,
        minimum_nonconstant_bootstrap_variance=torch.finfo(c.FLOAT).tiny,
        minimum_raw_sample_variance=torch.finfo(c.FLOAT).tiny,
        rng="torch.Generator CPU randint(n, (n,)); draws in increasing replication order", rng_version=torch.__version__,
        resample_index_base=0, resample_indices_reference="complete sample row ordinal",
        precision="float64", device="cpu", state_schema="openecon.pca_subspace_bootstrap.v1",
        inference_target=f"fixed rank-{components} spectral projector and captured/residual trace functionals of the population {matrix}",
        inferential_assumptions="IID complete-case rows; fixed dimension and component count; finite fourth moments; interior nonsingular population moments; positive retained/discarded population boundary gap; marginal first-order coverage requires nonzero functional derivative variance; population conditions assumed, not certified by sample guards",
        uncertainty="first-order IID marginal percentile bootstrap", familywise_intervals=False,
        p_values_available=False, inference_df_available=False, simultaneous_region_available=False,
        individual_eigenvalue_inference_available=False, eigenvector_inference_available=False,
        reestimate_moments_every_draw=True, restandardize_every_draw=matrix == "correlation",
        resource_plan=plan, estimated_work=work, estimated_complete_export_bytes=export_bytes,
        notes=["Internal repeated eigenvalues are admitted; a tie at the retained/discarded boundary is unidentified.",
               "Joint covariance is reported without inversion; projector identities and correlation trace identities can make it singular.",
               "Frobenius distances are diagnostics, not confidence-ball radii; trace(P)=components is an identity, not an inferred target.",
               "No exact coverage, simultaneous region, selected-count inference or licensed vendor parity is claimed.",
               "All requested replicas must satisfy the same finite/rank/boundary-gap checks; no failed draw is dropped."])
    return saved_summary(output)
