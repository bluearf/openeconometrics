"""Fixed PCA functionals under the expanded-IID integer-frequency bootstrap.

The empirical law is ordinary IID resampling of N=sum(frequencies) units,
represented by complete multinomial row counts rather than expanded rows.
Efron's original bootstrap: https://doi.org/10.1214/aos/1176344552 . Spectral
projector identification, including internal repeated roots, is discussed in
Koltchinskii and Lounici: https://arxiv.org/abs/1408.4643 . These routines do
not implement their specialized confidence balls or selected-rank inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu

from . import common as c
from . import frequency_bootstrap as f
from . import pca_subspace as s
from . import pca_uncertainty as u


@dataclass
class _Fit:
    parameters: torch.Tensor
    roots: torch.Tensor
    mean: torch.Tensor
    sd: torch.Tensor
    covariance: torch.Tensor
    target: torch.Tensor
    diagnostics: torch.Tensor
    anchors: list[str] | None = None
    vectors: torch.Tensor | None = None
    loadings: torch.Tensor | None = None
    projector: torch.Tensor | None = None


def _matrix(mean, covariance, matrix):
    """Use actual count-weighted moments, with the raw-variance gate first."""
    covariance = (covariance+covariance.T)/2
    u._finite(mean, covariance)
    variance = covariance.diagonal()
    if bool((variance <= 0).any()):
        raise AnalysisError("zero_variance", "Every PCA variable must vary in every admitted frequency sample.")
    if bool((variance < torch.finfo(c.FLOAT).tiny).any()):
        raise AnalysisError("numerical_failure", "A raw frequency sample variance is subnormal in float64 before PCA restandardization; rescale the measurements.")
    sd = variance.sqrt()
    correlation = covariance/torch.outer(sd, sd)
    correlation = (correlation+correlation.T)/2
    correlation.diagonal().fill_(1)
    u._finite(sd, correlation)
    spectrum = torch.linalg.eigvalsh(correlation)
    if float(spectrum[0]) <= 1e-11*float(spectrum[-1]):
        raise AnalysisError("singular_matrix", "Complete frequency PCA moments are singular or numerically unresolved.")
    target = covariance if matrix == "covariance" else correlation
    roots, vectors = torch.linalg.eigh(target)
    roots, vectors = roots.flip(0), vectors.flip(1)
    u._finite(roots, vectors)
    return covariance, sd, target, roots, vectors


def _axis_parameters(mean, covariance, names, matrix, components, anchors=None,
                     point_vectors=None):
    covariance, sd, target, roots, vectors = _matrix(mean, covariance, matrix)
    scale = float(roots[0])
    if scale <= 0 or float(roots[components-1]) <= u.EIGENVALUE_TOLERANCE*scale:
        raise AnalysisError("unidentified_component", "Retained eigenvalues must be positive away from numerical zero.")
    gaps = []
    for j in range(components):
        adjacent = []
        if j:
            adjacent.append(roots[j-1]-roots[j])
        if j+1 < len(names):
            adjacent.append(roots[j]-roots[j+1])
        gap = float(torch.stack(adjacent).min())/scale
        if gap <= u.EIGENVALUE_TOLERANCE:
            raise AnalysisError("unidentified_component", "Every retained root must be simple and separated from adjacent roots, including the retained/discarded boundary.")
        gaps.append(gap)
    vectors = vectors[:, :components]
    if anchors is None:
        anchors = [names[int(vectors[:, j].abs().argmax())] for j in range(components)]
    anchor_values = torch.tensor([float(vectors[names.index(name), j]) for j, name in enumerate(anchors)], dtype=c.FLOAT)
    if bool((anchor_values.abs() <= u.ANCHOR_TOLERANCE).any()):
        raise AnalysisError("unidentified_anchor", "Every recorded sign anchor must remain away from zero.")
    vectors = vectors*anchor_values.sign()
    cosines = torch.ones(components, dtype=c.FLOAT)
    if point_vectors is not None:
        cosines = (point_vectors*vectors).sum(0)
        if bool((cosines <= u.MIN_DIRECTION_COSINE).any()):
            raise AnalysisError("component_crossing", "A replicated signed direction left its local point-axis chart; no permutation, sign relabelling or alignment is applied.")
    loadings = vectors*roots[:components].sqrt()
    parameters = torch.cat([torch.cat((roots[j:j+1], vectors[:, j], loadings[:, j])) for j in range(components)])
    diagnostics = torch.tensor([min(gaps), float(anchor_values.abs().min()),
                                float(cosines.min()), float(roots[:components].min())], dtype=c.FLOAT)
    u._finite(parameters, diagnostics)
    return _Fit(parameters, roots, mean, sd, covariance, target, diagnostics,
                anchors=list(anchors), vectors=vectors, loadings=loadings)


def _subspace_parameters(mean, covariance, matrix, components, point_projector=None):
    covariance, sd, target, roots, vectors = _matrix(mean, covariance, matrix)
    scale = float(roots[0])
    if scale <= 0 or float(roots[components-1]) <= u.EIGENVALUE_TOLERANCE*scale:
        raise AnalysisError("unidentified_subspace", "The retained subspace must have positive eigenvalues away from numerical zero.")
    gap = float(roots[components-1]-roots[components])/scale
    if gap <= u.EIGENVALUE_TOLERANCE:
        raise AnalysisError("unidentified_subspace", "The fixed subspace needs a separated retained/discarded boundary; internal repeated roots are permitted.")
    retained = vectors[:, :components]
    projector = retained @ retained.T
    projector = (projector+projector.T)/2
    idempotence = float((projector @ projector-projector).abs().max())
    trace_error = abs(float(torch.trace(projector))-components)
    if idempotence > s.PROJECTOR_TOLERANCE or trace_error > s.PROJECTOR_TOLERANCE:
        raise AnalysisError("numerical_failure", "The PCA projector lost its fixed-rank orthogonal-projector identities.")
    captured, residual = roots[:components].sum(), roots[components:].sum()
    if float(residual) <= 0:
        raise AnalysisError("numerical_failure", "Discarded variance is not positive at resolved float64 precision; rescale the measurements.")
    tri = torch.triu_indices(len(mean), len(mean), device="cpu")
    parameters = torch.cat((projector[tri[0], tri[1]], torch.stack((captured, residual, captured/(captured+residual)))))
    distance = 0. if point_projector is None else float(torch.linalg.vector_norm(projector-point_projector))
    diagnostics = torch.tensor([gap, float(roots[components-1]), float(roots[components]),
                                idempotence, trace_error, distance], dtype=c.FLOAT)
    u._finite(parameters, projector, diagnostics)
    return _Fit(parameters, roots, mean, sd, covariance, target, diagnostics, projector=projector)


def _bootstrap(data, columns, *, weights, weight_type, components, matrix, anchors,
               replications, confidence, seed, missing, subspace):
    names = c.name_list(columns, "columns", minimum=2)
    matrix = c.check_choice(matrix, "matrix", ("covariance", "correlation"))
    p = len(names)
    components = c.check_count(components, "components", maximum=p-1 if subspace else p)
    q = p*(p+1)//2+3 if subspace else components*(1+2*p)
    if p > (s.MAX_VARIABLES if subspace else u.MAX_VARIABLES) or q > u.MAX_PARAMETERS:
        raise AnalysisError("workspace_limit", "Frequency PCA admits at most 16 axis/15 subspace variables and 128 joint parameters.")
    if anchors is not None:
        if not isinstance(anchors, (list, tuple)) or len(anchors) != components \
                or any(not isinstance(name, str) or name not in names for name in anchors):
            raise AnalysisError("invalid_spec", "anchors must contain one ordered analysed-variable name per component.")
        anchors = list(anchors)
    label = "fixed PCA subspace frequency bootstrap" if subspace else "fixed PCA eigenpair frequency bootstrap"
    sample = f.prepare(data, names, weights=weights, weight_type=weight_type,
                       replications=replications, confidence=confidence, seed=seed,
                       missing=missing, parameters=q, fit_work=64*p**3,
                       minimum_units=max(20, 2*p+1), label=label)
    moments = f.moments(sample.values, sample.counts)
    point = (_subspace_parameters(moments.mean, moments.covariance, matrix, components)
             if subspace else _axis_parameters(moments.mean, moments.covariance,
                                              names, matrix, components, anchors))
    bcount = sample.replications
    draws = torch.empty((bcount, q), dtype=c.FLOAT)
    counts = torch.empty((bcount, len(sample.values)), dtype=torch.int64)
    diagnostics = torch.empty((bcount, 6 if subspace else 4), dtype=c.FLOAT)
    means = torch.empty((bcount, p), dtype=c.FLOAT)
    deviations = torch.empty_like(means)
    origins = torch.empty_like(means)
    offsets = torch.empty_like(means)
    generator = torch.Generator(device="cpu").manual_seed(sample.seed)
    failures = []
    for b in range(bcount):
        counts[b] = f.draw_counts(sample, generator)
        try:
            current = f.moments(sample.values, counts[b])
            fitted = (_subspace_parameters(current.mean, current.covariance, matrix,
                                          components, point.projector) if subspace else
                      _axis_parameters(current.mean, current.covariance, names, matrix,
                                       components, point.anchors, point.vectors))
            draws[b], diagnostics[b] = fitted.parameters, fitted.diagnostics
            means[b], deviations[b] = fitted.mean, fitted.sd
            origins[b], offsets[b] = current.origin, current.offset
        except (AnalysisError, torch.linalg.LinAlgError) as error:
            failures.append({"replication": b+1,
                             "code": getattr(error, "code", "numerical_failure"),
                             "message": str(error)})
    if failures:
        error = AnalysisError("bootstrap_failure", f"{len(failures)} of {bcount} frequency PCA refits failed. No uncertainty result is returned and no replicate is dropped.")
        error.failures = failures
        error.replications_attempted = bcount
        error.successful_replications = bcount-len(failures)
        raise error
    joint, se, bias, lower, upper = f.joint(point.parameters, draws, sample.confidence)
    if subspace:
        order = [["projector", names[i], names[j]] for i in range(p) for j in range(i, p)]
        order.extend([["captured_variance", None, None], ["residual_variance", None, None],
                      ["captured_trace_share", None, None]])
        labels = [f"P[{i+1},{j+1}]" for i in range(p) for j in range(i, p)]
        labels.extend(["captured_variance", "residual_variance", "captured_trace_share"])
        order_columns = ["kind", "row_variable", "column_variable"]
        diagnostic_columns = ["relative_boundary_eigenvalue_gap", "minimum_retained_eigenvalue",
                              "maximum_discarded_eigenvalue", "maximum_projector_idempotence_error",
                              "projector_trace_error", "frobenius_distance_to_point_projector"]
    else:
        order = []
        for j in range(components):
            component = f"Comp{j+1}"
            order.append(["eigenvalue", component, None])
            order.extend(["eigenvector", component, name] for name in names)
            order.extend(["loading", component, name] for name in names)
        labels = [":".join(value for value in entry if value is not None) for entry in order]
        order_columns = ["kind", "component", "variable"]
        diagnostic_columns = ["minimum_relative_eigenvalue_gap", "minimum_absolute_anchor",
                              "minimum_signed_direction_cosine", "minimum_retained_eigenvalue"]
    undefined = torch.full_like(point.parameters, float("nan"))
    tables = f.source_tables(sample)
    tables.update({
        "estimates": c.frame(torch.stack((point.parameters, se, lower, upper, bias, undefined, undefined), dim=1),
                             columns=["estimate", "std_error", "ci_lower", "ci_upper", "bootstrap_bias", "p_value", "df"], index=labels),
        "covariance": c.frame(joint, columns=labels, index=labels),
        "replicates": c.frame(draws, columns=labels, index=range(1, bcount+1)),
        "replicate_counts": table(counts.numpy(), columns=sample.positions, index=range(1, bcount+1)),
        "replicate_diagnostics": c.frame(diagnostics, columns=diagnostic_columns, index=range(1, bcount+1)),
        "replicate_means": c.frame(means, columns=names, index=range(1, bcount+1)),
        "replicate_standard_deviations": c.frame(deviations, columns=names, index=range(1, bcount+1)),
        "replicate_origins": c.frame(origins, columns=names, index=range(1, bcount+1)),
        "replicate_mean_offsets": c.frame(offsets, columns=names, index=range(1, bcount+1)),
        "point_eigenvalues": c.frame(point.roots[:, None], columns=["eigenvalue"], index=c.numbered("Comp", p)),
        "point_covariance": c.frame(point.covariance, columns=names, index=names),
        "point_matrix": c.frame(point.target, columns=names, index=names),
        "point_origin": c.frame(moments.origin[:, None], columns=["origin"], index=names),
        "point_mean_offset": c.frame(moments.offset[:, None], columns=["offset"], index=names),
        "descriptives": c.frame(torch.stack((point.mean, point.sd), dim=1), columns=["mean", "std_dev"], index=names),
        "parameter_order": table(order, columns=order_columns, index=labels),
    })
    attrs = {**sample.attrs, "matrix": matrix, "components": components,
             "variables": names, "replications": bcount, "successful_replications": bcount,
             "failed_replications": [], "parameter_order": order, "parameter_labels": labels,
             "parameter_dimension": q, "point_diagnostics": point.diagnostics.tolist(),
             "eigenvalue_gap_tolerance": u.EIGENVALUE_TOLERANCE,
             "covariance_divisor": bcount-1, "quantile_interpolation": "linear",
             "covariance_may_be_singular": True, "covariance_inverse_available": False,
             "minimum_nonconstant_bootstrap_variance": torch.finfo(c.FLOAT).tiny,
             "minimum_raw_sample_variance": torch.finfo(c.FLOAT).tiny,
             "uncertainty": "first-order IID-unit marginal percentile bootstrap",
             "familywise_intervals": False, "simultaneous_region_available": False,
             "p_values_available": False, "inference_df_available": False,
             "reestimate_moments_every_draw": True, "restandardize_every_draw": matrix == "correlation",
             "frequency_interpretation": "exact multiplicities of independent identically distributed units; not clustered copies, survey weights or precision weights",
             "replicate_counts_reference": "complete physical source positions, including zero-frequency columns",
             "point_covariance_divisor": sample.total-1,
             "moment_covariance_divisor": sample.total-1,
             "precision": "float64", "device": "cpu"}
    assumptions = "IID expanded complete-case units; fixed dimension and component count; finite fourth moments; interior nonsingular population moments; population conditions assumed, not certified by sample guards"
    notes = ["Every replicate contains N=sum(source frequencies) IID units, represented by exact complete row counts; physical rows are not resampled uniformly.",
             "No expanded N-by-p sample is materialized; native integer-unit sampling work and all count-state output are bounded separately.",
             "No exact coverage, simultaneous region, selected-count inference or licensed vendor parity is claimed.",
             "All requested replicas must satisfy the same finite/rank/identification checks; no failed draw is dropped."]
    if subspace:
        tables["point_projector"] = c.frame(point.projector, columns=names, index=names)
        attrs.update(procedure="pca_subspace_fweight_bootstrap", state_schema="openecon.pca_subspace_fweight_bootstrap.v1",
                     projector_identity_tolerance=s.PROJECTOR_TOLERANCE,
                     subspace_identity="fixed rank and variable coordinates; only the retained/discarded boundary gap; invariant under signs and rotations within spectral blocks",
                     covariance_order="projector upper triangle in declared row/column variable order, captured variance, residual variance, captured trace share",
                     individual_eigenvalue_inference_available=False, eigenvector_inference_available=False,
                     inferential_assumptions=assumptions+"; positive retained/discarded boundary gap; first-order marginal coverage requires nonzero functional derivative variance",
                     inference_target=f"fixed rank-{components} spectral projector and captured/residual trace functionals of the population {matrix}")
        notes.extend(["Internal repeated roots are admitted; a tie at the retained/discarded boundary is unidentified.",
                      "Joint covariance is reported without inversion; projector and correlation trace identities can make it singular.",
                      "Frobenius distances are diagnostics, not confidence-ball radii; trace(P)=components is an identity, not an inferred target."])
    else:
        tables["point_eigenvectors"] = c.frame(point.vectors, columns=c.numbered("Comp", components), index=names)
        tables["point_loadings"] = c.frame(point.loadings, columns=c.numbered("Comp", components), index=names)
        attrs.update(procedure="pca_fweight_bootstrap", state_schema="openecon.pca_fweight_bootstrap.v1",
                     sign_anchors=point.anchors, anchor_selection="strongest point entry" if anchors is None else "caller declared",
                     sign_anchor_tolerance=u.ANCHOR_TOLERANCE, minimum_direction_cosine=u.MIN_DIRECTION_COSINE,
                     axis_identity="fixed descending-root order and signs; each draw stays within the signed 45-degree point-axis chart; no permutation or alignment",
                     covariance_order="component: eigenvalue, ordered eigenvector, ordered loading",
                     inferential_assumptions=assumptions+"; separated retained roots and fixed sign anchors away from zero",
                     inference_target=f"fixed simple eigenpairs of the population {matrix} functional")
    attrs["notes"] = notes
    return f.seal(TableSet(tables, title=f"Fixed {matrix} PCA {'subspace ' if subspace else ''}frequency IID-unit bootstrap uncertainty", **attrs))


@c.procedure
@resident_cpu
def pca_fweight_bootstrap(data: Any, columns: list[str], *, weights: str,
                          weight_type: str = "fweight", components: int,
                          matrix: str = "correlation", anchors: list[str] | None = None,
                          replications: int = 199, confidence: float = .95,
                          seed: int = 0, missing: str = "drop") -> TableSet:
    """IID-unit uncertainty for fixed simple frequency-weighted PCA eigenpairs.

    Exact nonnegative integer frequencies represent N independent units.
    Each draw samples N virtual units and recomputes count-weighted moments;
    correlation PCA restandardizes each draw. Fixed descending-root order,
    anchors and the local signed chart must hold in every draw. Any failed
    draw refuses the complete result. Full count vectors, joint covariance,
    marginal percentile intervals and origin/offset centering state persist.
    No clustered/precision/survey weights, selected counts, p-values or
    simultaneous coverage are provided. Finite fourth moments, separated
    population roots and interior moments are assumptions, not sample proofs.
    Measurement and weight missingness uses one listwise drop/raise policy;
    invalid nonmissing frequencies are refused even on otherwise omitted rows.
    """
    return _bootstrap(data, columns, weights=weights, weight_type=weight_type,
                      components=components, matrix=matrix, anchors=anchors,
                      replications=replications, confidence=confidence, seed=seed,
                      missing=missing, subspace=False)


@c.procedure
@resident_cpu
def pca_subspace_fweight_bootstrap(data: Any, columns: list[str], *, weights: str,
                                   weight_type: str = "fweight", components: int,
                                   matrix: str = "correlation", replications: int = 199,
                                   confidence: float = .95, seed: int = 0,
                                   missing: str = "drop") -> TableSet:
    """IID-unit uncertainty for a fixed frequency-weighted PCA projector.

    Exact integer frequencies denote N IID units; each draw contains N units,
    represented by full row counts without expanding their measurements.
    Only the retained/discarded boundary must be separated; roots inside the
    retained or discarded blocks may repeat. Report projector upper entries,
    captured/residual variance and captured trace share with their complete
    joint draws/covariance and marginal percentile intervals. Covariance is
    not inverted; no repeated-root eigenpair inference or confidence ball is
    provided. Any failed refit refuses the entire result. Population regularity
    and finite fourth moments remain assumptions, not certified by guards.
    Measurement and weight missingness uses one listwise drop/raise policy;
    invalid nonmissing frequencies are refused even on otherwise omitted rows.
    """
    return _bootstrap(data, columns, weights=weights, weight_type=weight_type,
                      components=components, matrix=matrix, anchors=None,
                      replications=replications, confidence=confidence, seed=seed,
                      missing=missing, subspace=True)
