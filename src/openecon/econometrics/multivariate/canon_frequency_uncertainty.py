"""Literal-frequency IID bootstrap for fixed CCA roots and coefficient charts.

References: Efron (1979), doi:10.1214/aos/1176344552; Bickel and Freedman
(1981), doi:10.1214/aos/1176345637; Anderson (1999),
doi:10.1006/jmva.1999.1810. The observations represented by the frequencies
must be independent units. Numerical gates do not establish population
regularity or the absence of dependence.
"""
from __future__ import annotations

import json
import math
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu

from . import canon_uncertainty as u
from . import common as c
from . import frequency_bootstrap as f

STATE_SCHEMA = "openecon.canon_fweight_bootstrap.v1"


@c.procedure
def _parameters(moment: Any, x: list[str], y: list[str], components: int,
                target: str, anchors: list[str] | None, point: u._Fit | None = None) -> u._Fit:
    """QR on frequency-scaled physical rows, never on expanded measurements."""
    correlation = f.correlation(moment)
    p = len(x)
    rxx, ryy, rxy = correlation[:p, :p], correlation[p:, p:], correlation[:p, p:]
    conditions = []
    for block in (rxx, ryy):
        roots = torch.linalg.eigvalsh(block)
        u._finite(roots)
        if float(roots[0]) <= 1e-10 * float(roots[-1]):
            raise AnalysisError("singular_matrix", "Both CCA correlation blocks require full conditioned rank in every frequency refit; no repair or deletion is performed.")
        conditions.append(float(roots[-1] / roots[0]))
    # This Gram matrix equals the standardized literal sample covariance.
    z = (moment.centered / moment.sd) * (moment.counts.to(c.FLOAT) / (moment.total - 1)).sqrt()[:, None]
    qx, rx = torch.linalg.qr(z[:, :p], mode="reduced")
    qy, ry = torch.linalg.qr(z[:, p:], mode="reduced")
    left, roots, right = torch.linalg.svd(qx.T @ qy, full_matrices=False)
    u._finite(roots)
    if len(roots) < components or float(roots[components - 1]) <= u.GEOMETRY_TOLERANCE:
        raise AnalysisError("unidentified_correlation", "Every retained fixed correlation must be positive away from numerical zero.")
    if float(roots[0]) >= 1 - u.GEOMETRY_TOLERANCE:
        raise AnalysisError("unidentified_correlation", "CCA frequency uncertainty requires interior roots away from perfect correlation.")
    following = torch.cat((roots[1:], torch.zeros(1, dtype=c.FLOAT)))
    gap = float((roots[:components] - following[:components]).min())
    if gap <= u.GEOMETRY_TOLERANCE:
        raise AnalysisError("unidentified_correlation", "Retained roots must be simple, including separation from the first omitted root.")
    matrices = {}
    fixed, anchor_min, x_cos, y_cos = None, math.nan, math.nan, math.nan
    vector = roots[:components].clone()
    if target == "coefficients":
        a = torch.linalg.solve_triangular(rx, left[:, :components], upper=True)
        b = torch.linalg.solve_triangular(ry, right.T[:, :components], upper=True)
        u._finite(a, b)
        norms = torch.linalg.vector_norm(a, dim=0)
        if anchors is None:
            ordered = a.abs().sort(dim=0, descending=True).values
            if p > 1 and bool(((ordered[0] - ordered[1]) <= u.GEOMETRY_TOLERANCE * norms).any()):
                raise AnalysisError("unidentified_anchor", "A default point sign anchor needs a unique strongest standardized X coefficient; declare anchors.")
            fixed = [x[int(row)] for row in a.abs().argmax(0)]
        else:
            fixed = anchors
        values = a[[x.index(name) for name in fixed], torch.arange(components)]
        anchor_min = float((values.abs() / norms).min())
        if anchor_min <= u.GEOMETRY_TOLERANCE:
            raise AnalysisError("unidentified_anchor", "Every fixed X coefficient sign anchor must remain away from zero relative to its coefficient norm.")
        a, b = a * values.sign(), b * values.sign()
        raw_a, raw_b = a / moment.sd[:p, None], b / moment.sd[p:, None]
        x_cos = y_cos = 1.0
        if point is not None:
            for key, section, raw in (("x_coefficients", slice(0, p), raw_a),
                                      ("y_coefficients", slice(p, None), raw_b)):
                chol = torch.linalg.cholesky(point.correlation[section, section])
                current = chol.T @ (point.sd[section, None] * raw)
                reference = chol.T @ (point.sd[section, None] * point.matrices[key])
                cosine = (current * reference).sum(0) / (torch.linalg.vector_norm(current, dim=0) * torch.linalg.vector_norm(reference, dim=0))
                u._finite(cosine)
                if bool((cosine <= u.MIN_DIRECTION_COSINE).any()):
                    raise AnalysisError("component_crossing", "A signed coefficient direction left its fixed 45-degree point-covariance chart; no permutation or alignment is applied.")
                if key.startswith("x"):
                    x_cos = float(cosine.min())
                else:
                    y_cos = float(cosine.min())
        matrices = {"x_coefficients": raw_a, "y_coefficients": raw_b,
                    "x_standardized_coefficients": a, "y_standardized_coefficients": b,
                    "x_loadings": rxx @ a, "y_loadings": ryy @ b,
                    "x_cross_loadings": rxy @ b, "y_cross_loadings": rxy.T @ a}
        vector = torch.cat([torch.cat([roots[j:j + 1], *(matrix[:, j] for matrix in matrices.values())])
                            for j in range(components)])
    u._finite(vector, *matrices.values())
    diagnostics = torch.tensor([float(roots[components - 1]), float(roots[0]), gap,
                                *conditions, anchor_min, x_cos, y_cos], dtype=c.FLOAT)
    return u._Fit(vector, roots, moment.mean, moment.sd, moment.covariance,
                  correlation, diagnostics, fixed, matrices)


@c.procedure
@resident_cpu
def canon_fweight_bootstrap(data: Any, x: list[str], y: list[str], *, weights: str,
                           weight_type: str = "fweight", components: int,
                           target: str = "correlations", anchors: list[str] | None = None,
                           replications: int = 199, confidence: float = .95,
                           seed: int = 0, missing: str = "drop") -> TableSet:
    """Marginal literal-frequency IID uncertainty for a fixed CCA functional.

    Nonnegative integer frequencies represent counts of actual independent
    observed units. Every draw samples exactly N=sum(frequencies) copy ranks
    and stores their complete-row counts, without expanding measurement rows.
    The joint complete-case sample is fixed before drawing; complete zero-count
    rows remain in saved source tables and receive zero counts in every draw.
    Point and replicate means, covariance (N-1 divisor), scales and CCA are
    re-estimated. Bootstrap covariance uses B-1; intervals are marginal linear
    percentile intervals. P-values and inference df are undefined.
    Invalid frequencies are refused even where a measurement is missing;
    missing frequencies follow the declared common listwise drop/raise policy.

    Retained correlations must be positive, simple and interior, with the
    retained/omitted boundary separated. The coefficients target includes raw
    and standardized X/Y coefficients and within/cross loadings, with unit
    canonical-variate variance and positive pair correlation. Fixed X anchors
    determine signs; each signed replicate direction stays within the fixed
    45-degree point-covariance chart. No relabelling or alignment is performed.

    Fixed dimensions/count, IID units, finite fourth moments, interior block
    moments and population identification are assumptions, not conclusions of
    numerical checks. First-order coordinate coverage also requires nonzero
    first-order variance. These frequencies are not arbitrary importance,
    analytic or survey weights, and correlated copies are unsupported. No
    exact/familywise coverage, selected-count inference or vendor equivalence
    is claimed. Any failed refit refuses the whole result without replacement.
    """
    if any(isinstance(block, (list, tuple)) and len(block) > u.MAX_VARIABLES for block in (x, y)):
        raise AnalysisError("workspace_limit", "CCA frequency bootstrap admits at most 16 total variables.")
    xs, ys = c.name_list(x, "x"), c.name_list(y, "y")
    if set(xs) & set(ys):
        raise AnalysisError("invalid_spec", "CCA variable blocks must be disjoint.")
    names = xs + ys
    target = c.check_choice(target, "target", ("correlations", "coefficients"))
    components = c.check_count(components, "components", maximum=min(4, len(xs), len(ys)))
    if anchors is not None:
        if target != "coefficients":
            raise AnalysisError("invalid_option", "anchors apply only to coefficient uncertainty.")
        if not isinstance(anchors, (tuple, list)) or len(anchors) != components \
                or any(not isinstance(name, str) or name not in xs for name in anchors):
            raise AnalysisError("invalid_spec", "Supply one ordered X-variable anchor per component.")
        anchors = list(anchors)
    order, labels = u._parameter_order(xs, ys, components, target)
    sample = f.prepare(data, names, weights=weights, weight_type=weight_type,
                       replications=replications, confidence=confidence, seed=seed,
                       missing=missing, parameters=len(labels),
                       minimum_units=max(20, 2 * len(names) + 1), label="fixed CCA frequency bootstrap")
    point_moment = f.moments(sample.values, sample.counts)
    point = _parameters(point_moment, xs, ys, components, target, anchors)
    b, p, d, s = sample.replications, len(names), len(labels), min(len(xs), len(ys))
    generator = torch.Generator(device="cpu").manual_seed(sample.seed)
    draws = torch.empty((b, d), dtype=c.FLOAT)
    counts = torch.empty((b, len(sample.values)), dtype=torch.int64)
    diagnostics = torch.empty((b, len(u._DIAGNOSTICS)), dtype=c.FLOAT)
    means, sds, origins, offsets = [torch.empty((b, p), dtype=c.FLOAT) for _ in range(4)]
    covariances, roots = torch.empty((b, p * p), dtype=c.FLOAT), torch.empty((b, s), dtype=c.FLOAT)
    failures = []
    for i in range(b):
        counts[i] = f.draw_counts(sample, generator)
        try:
            moment = f.moments(sample.values, counts[i])
            fit = _parameters(moment, xs, ys, components, target, point.anchors,
                              point if target == "coefficients" else None)
            draws[i], diagnostics[i], means[i], sds[i] = fit.vector, fit.diagnostics, moment.mean, moment.sd
            origins[i], offsets[i], covariances[i], roots[i] = moment.origin, moment.offset, moment.covariance.flatten(), fit.roots
        except (AnalysisError, torch.linalg.LinAlgError) as error:
            failures.append({"replication": i + 1, "code": getattr(error, "code", "numerical_failure"), "message": str(error)})
    if failures:
        error = AnalysisError("bootstrap_failure", f"{len(failures)} of {b} frequency CCA refits failed; no replicate is replaced or dropped.")
        error.failures, error.replications_attempted, error.successful_replications = failures, b, b - len(failures)
        raise error
    covariance, se, bias, lower, upper = f.joint(point.vector, draws, sample.confidence)
    undefined = torch.full((d,), math.nan, dtype=c.FLOAT)
    ri, axes = range(1, b + 1), c.numbered("Can", components)
    tables = f.source_tables(sample)
    tables.update({
        "estimates": c.frame(torch.stack((point.vector, se, lower, upper, bias, undefined, undefined), 1), columns=["estimate", "std_error", "ci_lower", "ci_upper", "bootstrap_bias", "p_value", "df"], index=labels),
        "covariance": c.frame(covariance, columns=labels, index=labels),
        "replicates": c.frame(draws, columns=labels, index=ri),
        "replicate_counts": table(counts.numpy(), columns=sample.positions, index=ri),
        "replicate_diagnostics": c.frame(diagnostics, columns=u._DIAGNOSTICS, index=ri),
        "replicate_means": c.frame(means, columns=names, index=ri),
        "replicate_standard_deviations": c.frame(sds, columns=names, index=ri),
        "replicate_origins": c.frame(origins, columns=names, index=ri),
        "replicate_mean_offsets": c.frame(offsets, columns=names, index=ri),
        "replicate_covariances": c.frame(covariances, columns=[json.dumps([a, b]) for a in names for b in names], index=ri),
        "replicate_correlations": c.frame(roots, columns=c.numbered("Can", s), index=ri),
        "point_correlations": c.frame(point.roots[:, None], columns=["correlation"], index=c.numbered("Can", s)),
        "point_covariance": c.frame(point.covariance, columns=names, index=names),
        "point_correlation": c.frame(point.correlation, columns=names, index=names),
        "point_diagnostics": c.frame(point.diagnostics[None], columns=u._DIAGNOSTICS, index=["point"]),
        "point_origin": c.frame(point_moment.origin[None], columns=names, index=["origin"]),
        "point_mean_offset": c.frame(point_moment.offset[None], columns=names, index=["offset"]),
        "descriptives": c.frame(torch.stack((point.mean, point.sd), 1), columns=["mean", "std_dev"], index=names),
        "parameter_order": table(order, columns=["kind", "component", "block", "variable"], index=labels),
    })
    for key, matrix in point.matrices.items():
        tables["point_" + key] = c.frame(matrix, columns=axes, index=xs if key.startswith("x") else ys)
    attrs = dict(sample.attrs)
    attrs.update(procedure="canon_fweight_bootstrap", state_schema=STATE_SCHEMA, target=target,
                 components=components, fixed_count=True, x=xs, y=ys, variables=names,
                 n=sample.total, precision="float64", device="cpu",
                 sign_anchors=point.anchors, parameter_dimension=d, parameter_labels=labels,
                 parameter_order=order, successful_replications=b, failed_replications=[],
                 geometry_tolerance=u.GEOMETRY_TOLERANCE, maximum_block_correlation_condition=1e10,
                 minimum_direction_cosine=u.MIN_DIRECTION_COSINE if target == "coefficients" else None,
                 coefficient_normalization="unit frequency sample variance; positive canonical pair correlation",
                 axis_identity="fixed decreasing simple roots; fixed X signs and point-covariance charts; no permutation or alignment" if target == "coefficients" else "fixed decreasing simple roots only",
                 moment_covariance_divisor=sample.total - 1, covariance_divisor=b - 1,
                 quantile_interpolation="linear", reestimate_moments_every_draw=True,
                 restandardize_every_draw=True, full_replicate_covariance_order="row-major declared variables",
                 inference_target=f"fixed simple canonical {target} functional under literal-frequency IID units",
                 inferential_assumptions="IID complete-case observed units; frequencies are actual unit counts; fixed dimension/count; finite fourth moments; nonsingular population blocks; positive simple interior roots and retained/omitted gap; fixed nonzero signs/local coefficient charts; coordinate coverage additionally requires nonzero first-order variance",
                 uncertainty="first-order IID marginal percentile bootstrap", familywise_intervals=False,
                 p_values_available=False, inference_df_available=False,
                 notes=["No exact, familywise, selected-count or zero-root inference; no arbitrary-weight, correlated-copy or survey interpretation.",
                        "All requested refits must pass; any failure refuses the whole result without replacement.",
                        "Saved counts reference the fixed complete physical sample, including complete zero-frequency rows; full semantic replay requires refitting those counts."])
    result = TableSet(tables, title=f"Fixed CCA {target} literal-frequency bootstrap uncertainty", **attrs)
    return f.seal(result)
