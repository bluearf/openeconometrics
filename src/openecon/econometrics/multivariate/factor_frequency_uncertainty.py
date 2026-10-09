"""Literal-frequency IID uncertainty for two fixed principal-factor functionals.

Principal factors means one SMC-reduced correlation eigendecomposition, not
an ML latent-factor parameter estimator. Bootstrap references: Efron (1979),
doi:10.1214/aos/1176344552; Bickel and Freedman (1981),
doi:10.1214/aos/1176345637.
"""
from __future__ import annotations

import json
import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu

from . import common as c
from . import extraction as ex
from . import factor_uncertainty as u
from . import frequency_bootstrap as f

STATE_SCHEMA = "openecon.factor_multifactor_fweight_bootstrap.v1"


@c.procedure
def _parameters(moment: Any, names: list[str], factors: int,
                target: Tensor | None, anchors: list[str] | None) -> u._Fit:
    correlation = f.correlation(moment)
    spectrum = torch.linalg.eigvalsh(correlation)
    if not bool(torch.isfinite(spectrum).all()) or float(spectrum[0]) <= u.GEOMETRY_TOLERANCE * float(spectrum[-1]):
        raise AnalysisError("singular_matrix", "PF frequency bootstrap requires a positive-definite conditioned correlation in every fit; no repair or deletion is performed.")
    smc = ex.squared_multiple_correlations(correlation)
    extracted = ex.principal_factors(correlation, smc, factors)
    loadings, rotation, fixed, gap, anchor, cross_min, cross_condition = u._orientation(
        extracted.loadings, extracted.factored, names, target, anchors)
    uniqueness = 1 - loadings.square().sum(1)
    if not bool(torch.isfinite(uniqueness).all()) or bool((uniqueness <= u.GEOMETRY_TOLERANCE).any()):
        raise AnalysisError("heywood_case", "PF frequency bootstrap refuses boundary or nonpositive uniqueness in every fit.")
    diagnostics = torch.tensor([float(extracted.factored[factors - 1]), gap,
                                float(uniqueness.min()), anchor, cross_min, cross_condition,
                                float(spectrum[-1] / spectrum[0])], dtype=c.FLOAT)
    return u._Fit(loadings, uniqueness, extracted.factored, smc, correlation,
                  moment.mean, moment.sd, rotation, fixed, diagnostics)


@c.procedure
@resident_cpu
def factor_multifactor_fweight_bootstrap(data: Any, columns: list[str], *, weights: str,
                                       weight_type: str = "fweight", factors: int,
                                       target: Any = None, anchors: list[str] | None = None,
                                       replications: int = 199, confidence: float = .95,
                                       seed: int = 0, missing: str = "drop") -> TableSet:
    """Marginal IID frequency uncertainty for a fixed two-to-four-factor PF.

    Frequencies count actual independent observed units. Each replicate draws
    N=sum(frequencies) literal copy ranks and retains complete physical-row
    counts without allocating expanded measurement rows. Point/every draw
    recomputes weighted means, N-1 covariance, scales, correlation and SMCs.
    Fixed-count loadings and uniquenesses have full joint B-1 covariance, SE,
    bias and marginal linear percentile intervals; tests and df are undefined.
    Invalid frequencies are refused even where a measurement is missing;
    missing frequencies follow the declared common listwise drop/raise policy.

    Unrotated axes require positive simple retained reduced roots, including
    the retained/omitted gap, with fixed loading sign anchors away from zero.
    A complete independently caller-declared target fixes orthogonal
    Procrustes orientation without Kaiser normalization. It remains unchanged
    in every draw; its loading cross product requires a unique conditioned
    polar factor. Internal repeated retained roots are allowed for this
    target functional, while its retained/omitted boundary stays separated.
    Reflections are allowed; arbitrary extraction rotations are not inferred.

    The inferential target is the smooth PF estimator functional, not latent
    model parameters or ML loading SEs. IID units, fixed dimension/count,
    finite fourth moments, interior population correlation/uniqueness and
    the stated identification are assumptions; finite-sample gates do not
    establish them. First-order marginal coverage additionally requires
    nonzero first-order coordinate variance. Arbitrary analytic/survey weights,
    correlated copies, selection, other extractions/rotations and exact or
    familywise coverage are unsupported. Any failed replicate refuses the
    whole result without dropping or replacing draws.
    """
    if isinstance(columns, (list, tuple)) and len(columns) > u.MAX_VARIABLES:
        raise AnalysisError("workspace_limit", "PF frequency bootstrap admits at most 16 variables.")
    names = c.name_list(columns, "columns", minimum=3)
    factors = c.check_count(factors, "factors", minimum=2, maximum=4)
    if factors >= len(names):
        raise AnalysisError("invalid_spec", "Retain fewer fixed factors than analysed variables.")
    fixed_anchors = u._anchors(anchors, names, factors, target)
    d = len(names) * (factors + 1)
    sample = f.prepare(data, names, weights=weights, weight_type=weight_type,
                       replications=replications, confidence=confidence, seed=seed,
                       missing=missing, parameters=d, minimum_units=max(20, 2 * len(names) + 1),
                       label="fixed principal-factor frequency bootstrap")
    fixed_target = u._target(target, names, factors)
    point_moment = f.moments(sample.values, sample.counts)
    point = _parameters(point_moment, names, factors, fixed_target, fixed_anchors)
    b, p = sample.replications, len(names)
    generator = torch.Generator(device="cpu").manual_seed(sample.seed)
    draws = torch.empty((b, d), dtype=c.FLOAT)
    counts = torch.empty((b, len(sample.values)), dtype=torch.int64)
    diagnostics = torch.empty((b, len(u._DIAGNOSTICS)), dtype=c.FLOAT)
    means, sds, origins, offsets = [torch.empty((b, p), dtype=c.FLOAT) for _ in range(4)]
    covariances = torch.empty((b, p * p), dtype=c.FLOAT)
    eigenvalues, smcs = torch.empty((b, p), dtype=c.FLOAT), torch.empty((b, p), dtype=c.FLOAT)
    failures = []
    for i in range(b):
        counts[i] = f.draw_counts(sample, generator)
        try:
            moment = f.moments(sample.values, counts[i])
            fit = _parameters(moment, names, factors, fixed_target, point.anchors)
            draws[i], diagnostics[i], means[i], sds[i] = fit.vector(), fit.diagnostics, moment.mean, moment.sd
            origins[i], offsets[i], covariances[i] = moment.origin, moment.offset, moment.covariance.flatten()
            eigenvalues[i], smcs[i] = fit.eigenvalues, fit.smc
        except (AnalysisError, torch.linalg.LinAlgError) as error:
            failures.append({"replication": i + 1, "code": getattr(error, "code", "numerical_failure"), "message": str(error)})
    if failures:
        error = AnalysisError("bootstrap_failure", f"{len(failures)} of {b} frequency PF refits failed; no replicate is replaced or dropped.")
        error.failures, error.replications_attempted, error.successful_replications = failures, b, b - len(failures)
        raise error
    covariance, se, bias, lower, upper = f.joint(point.vector(), draws, sample.confidence)
    axes, ri = c.numbered("Factor", factors), range(1, b + 1)
    labels = [f"loading:{name}:{axis}" for name in names for axis in axes] + [f"uniqueness:{name}" for name in names]
    undefined = torch.full((d,), math.nan, dtype=c.FLOAT)
    tables = f.source_tables(sample)
    tables.update({
        "estimates": c.frame(torch.stack((point.vector(), se, lower, upper, bias, undefined, undefined), 1), columns=["estimate", "std_error", "ci_lower", "ci_upper", "bootstrap_bias", "p_value", "df"], index=labels),
        "covariance": c.frame(covariance, columns=labels, index=labels),
        "replicates": c.frame(draws, columns=labels, index=ri),
        "replicate_counts": table(counts.numpy(), columns=sample.positions, index=ri),
        "replicate_diagnostics": c.frame(diagnostics, columns=u._DIAGNOSTICS, index=ri),
        "replicate_means": c.frame(means, columns=names, index=ri),
        "replicate_standard_deviations": c.frame(sds, columns=names, index=ri),
        "replicate_origins": c.frame(origins, columns=names, index=ri),
        "replicate_mean_offsets": c.frame(offsets, columns=names, index=ri),
        "replicate_covariances": c.frame(covariances, columns=[json.dumps([a, b]) for a in names for b in names], index=ri),
        "replicate_eigenvalues": c.frame(eigenvalues, columns=c.numbered("Root", p), index=ri),
        "replicate_smc": c.frame(smcs, columns=names, index=ri),
        "point_loadings": c.frame(point.loadings, columns=axes, index=names),
        "point_uniqueness": c.frame(point.uniqueness[:, None], columns=["uniqueness"], index=names),
        "point_eigenvalues": c.frame(point.eigenvalues[:, None], columns=["reduced_eigenvalue"], index=c.numbered("Root", p)),
        "point_smc": c.frame(point.smc[:, None], columns=["smc"], index=names),
        "point_covariance": c.frame(point_moment.covariance, columns=names, index=names),
        "point_correlation": c.frame(point.correlation, columns=names, index=names),
        "point_rotation_matrix": c.frame(point.rotation, columns=axes, index=axes),
        "point_diagnostics": c.frame(point.diagnostics[None], columns=u._DIAGNOSTICS, index=["point"]),
        "point_origin": c.frame(point_moment.origin[None], columns=names, index=["origin"]),
        "point_mean_offset": c.frame(point_moment.offset[None], columns=names, index=["offset"]),
        "descriptives": c.frame(torch.stack((point.mean, point.std), 1), columns=["mean", "std_dev"], index=names),
    })
    if fixed_target is not None:
        tables["target"] = c.frame(fixed_target, columns=axes, index=names)
    attrs = dict(sample.attrs)
    attrs.update(procedure="factor_multifactor_fweight_bootstrap", state_schema=STATE_SCHEMA,
                 method="pf", factors=factors, fixed_count=True, variables=names,
                 n=sample.total, precision="float64", device="cpu",
                 rotate="target" if fixed_target is not None else None, kaiser=False,
                 sign_anchors=point.anchors, target=None if fixed_target is None else fixed_target.tolist(),
                 target_policy="independently caller declared; identical in every refit" if fixed_target is not None else None,
                 internal_repeated_roots_allowed=fixed_target is not None,
                 point_unrotated_axes_identified=bool(((point.eigenvalues[:factors - 1] - point.eigenvalues[1:factors]) > u.GEOMETRY_TOLERANCE * max(1., abs(float(point.eigenvalues[0])))).all()),
                 point_rotation_matrix_basis="nuisance extraction basis to declared loading axes; not an inferential parameter",
                 orientation="fixed full-target orthogonal Procrustes" if fixed_target is not None else "fixed ordered reduced eigenaxes with fixed signs",
                 geometry_tolerance=u.GEOMETRY_TOLERANCE, parameter_dimension=d, parameter_labels=labels,
                 parameter_order="variable-major loadings, then variable-order uniquenesses",
                 successful_replications=b, failed_replications=[],
                 moment_covariance_divisor=sample.total - 1, covariance_divisor=b - 1,
                 quantile_interpolation="linear", reestimate_moments_every_draw=True,
                 restandardize_every_draw=True, full_replicate_covariance_order="row-major declared variables",
                 inference_target="fixed full-target PF estimator functional under literal-frequency IID units" if fixed_target is not None else "fixed ordered unrotated PF estimator functional under literal-frequency IID units",
                 inferential_assumptions="IID complete-case observed units; actual integer unit counts; fixed p/count; finite fourth moments; interior nonsingular population correlation and uniqueness; positive retained roots and retained/omitted gap; unique conditioned fixed-target polar factor or simple ordered roots with fixed nonzero loading signs; coordinate coverage additionally requires nonzero first-order variance",
                 uncertainty="first-order IID marginal percentile bootstrap", familywise_intervals=False,
                 p_values_available=False, inference_df_available=False,
                 notes=["PF means one SMC-reduced eigendecomposition; no latent-model parameter or ML loading standard-error claim.",
                        "No exact/familywise, selected-count, arbitrary-weight, correlated-copy or survey inference.",
                        "Any refit failure refuses all uncertainty; source values, original frequencies and every draw count persist for independent semantic replay."])
    result = TableSet(tables, title="Fixed principal-factor literal-frequency bootstrap uncertainty", **attrs)
    return f.seal(result)
