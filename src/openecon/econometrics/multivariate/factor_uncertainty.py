"""IID uncertainty for two identified, fixed multi-factor PF functionals.

Principal factoring is the single spectral decomposition of the sample
correlation with its diagonal replaced by re-estimated squared multiple
correlations. This module does not estimate latent-model ML standard errors.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.postest import index_codec
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.resources import plan_workspace

from . import common as c
from . import extraction as ex
from . import uncertainty as u
from .rotation import target_matrix

MAX_ROWS = 10_000
MAX_VARIABLES = 16
MAX_FACTORS = 4
MAX_REPLICATIONS = 1_999
MAX_WORK = 250_000_000
MAX_EXPORT_BYTES = 32 * 1024 * 1024
MAX_LABEL_BYTES = 4096
MAX_IDENTITY_BYTES = 2 * 1024 * 1024
MAX_IDENTITY_DEPTH = 32
MAX_IDENTITY_NODES = 100_000
GEOMETRY_TOLERANCE = 1e-8
STATE_SCHEMA = "openecon.factor_multifactor_bootstrap.v1"
_DIAGNOSTICS = ["minimum_retained_eigenvalue", "minimum_required_eigenvalue_gap",
                "minimum_uniqueness", "minimum_anchor_loading",
                "target_cross_minimum_singular_value", "target_cross_condition",
                "correlation_condition"]


@dataclass
class _Fit:
    loadings: Tensor
    uniqueness: Tensor
    eigenvalues: Tensor
    smc: Tensor
    correlation: Tensor
    mean: Tensor
    std: Tensor
    rotation: Tensor
    anchors: list[str] | None
    diagnostics: Tensor

    def vector(self) -> Tensor:
        return torch.cat((self.loadings.flatten(), self.uniqueness))


def _anchors(value: Any, names: list[str], factors: int, target: Any) -> list[str] | None:
    if value is None:
        return None
    if target is not None:
        raise AnalysisError("invalid_option", "anchors apply only to unrotated factors; a full target fixes axis order and sign.")
    if not isinstance(value, (list, tuple)) or len(value) != factors \
            or any(not isinstance(item, str) or item not in names for item in value):
        raise AnalysisError("invalid_spec", "anchors must supply one analysed variable name per factor in eigenvalue order; repeated anchor variables are allowed.")
    return list(value)


def _target(value: Any, names: list[str], factors: int) -> Tensor | None:
    if value is None:
        return None
    if isinstance(value, Tensor) and value.device.type != "cpu":
        raise AnalysisError("unsupported_device", "The fixed target must reside on the CPU.")
    if isinstance(value, pd.DataFrame):
        labels = [f"Factor{i + 1}" for i in range(factors)]
        if list(value.index) != names or list(value.columns) != labels:
            raise AnalysisError("invalid_spec", "A labelled target must have the analysed variables in order as rows and Factor1,...,FactorM in order as columns.")
        value = value.to_numpy()
    try:
        shape = tuple(value.shape) if hasattr(value, "shape") else (len(value), len(value[0]))
    except (TypeError, IndexError) as exc:
        raise AnalysisError("invalid_spec", "target must be a finite complete p-by-factors matrix.") from exc
    if shape != (len(names), factors):
        raise AnalysisError("invalid_spec", "target must be a finite complete p-by-factors matrix in the caller's variable and factor order.")
    kind = getattr(getattr(value, "dtype", None), "kind", None)
    if isinstance(value, Tensor):
        real = not value.is_complex() and value.dtype != torch.bool and value.layout == torch.strided
    elif kind is not None:
        real = kind in "iuf"
    else:
        try:
            real = all(isinstance(item, Real) and not isinstance(item, bool)
                       for row in value for item in row)
        except TypeError:
            real = False
    if not real:
        raise AnalysisError("invalid_spec", "target must contain real numeric values; Boolean, complex and string values are unsupported.")
    try:
        result = torch.as_tensor(value, dtype=c.FLOAT).detach().clone()
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise AnalysisError("invalid_spec", "target must be a finite complete p-by-factors matrix.") from exc
    if tuple(result.shape) != shape or not bool(torch.isfinite(result).all()) \
            or float(result.abs().max()) > c.LARGEST:
        raise AnalysisError("invalid_spec", "target must contain finite real values of a numerically representable magnitude.")
    return result


def _identity(value: Any, *, counter: list[int] | None = None) -> str:
    # Bound traversal and scalar allocation before the recursive typed codec.
    counter = [0] if counter is None else counter
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        counter[0] += 1
        if depth > MAX_IDENTITY_DEPTH or counter[0] > MAX_IDENTITY_NODES:
            raise AnalysisError("export_limit", "Source identities exceed bounded tuple depth or traversal work.")
        if isinstance(item, (str, bytes)) and len(item) > MAX_LABEL_BYTES:
            raise AnalysisError("export_limit", "Each source row identity or axis name must fit within 4096 text/byte units.")
        if isinstance(item, int) and item.bit_length() > 4 * MAX_LABEL_BYTES \
                or isinstance(item, Decimal) and item.__sizeof__() > MAX_LABEL_BYTES:
            raise AnalysisError("export_limit", "An encoded numeric source identity exceeds bounded scalar storage.")
        if isinstance(item, tuple):
            if len(item) > MAX_IDENTITY_NODES - counter[0]:
                raise AnalysisError("export_limit", "Source identities exceed bounded tuple traversal work.")
            pending.extend((child, depth + 1) for child in item)
    try:
        code = index_codec.encode(value)
    except (TypeError, ValueError, OverflowError, RecursionError, AnalysisError) as exc:
        raise AnalysisError("invalid_index", "Source row labels and axis names need supported lossless scalar/tuple identities.") from exc
    if len(code.encode()) > MAX_LABEL_BYTES:
        raise AnalysisError("export_limit", "Each source row identity or axis name must fit within 4096 encoded bytes.")
    return code


def _identities(data: Any, rows: int):
    counter = [0]
    if isinstance(data, pd.DataFrame):
        index = data.index
        index_names = [_identity(name, counter=counter) for name in index.names]
        column_names = [_identity(name, counter=counter) for name in data.columns.names]
        multi = isinstance(index, pd.MultiIndex)
    else:
        index, index_names, column_names, multi = range(rows), [_identity(None, counter=counter)], [_identity(None, counter=counter)], False
    codes, size = [], 0
    for value in index:
        code = _identity(value, counter=counter)
        size += len(code.encode())
        if size > MAX_IDENTITY_BYTES:
            raise AnalysisError("export_limit", "Complete source row identities exceed the 2 MiB portable identity budget.")
        codes.append(code)
    return codes, index_names, column_names, multi, size


def _export_admission(names: list[str], rows: int, factors: int, replications: int,
                      identity_bytes: int, axis_names: list[str], has_target: bool) -> int:
    if any(len(name.encode()) > MAX_LABEL_BYTES for name in names):
        raise AnalysisError("export_limit", "Each analysed variable name must fit within 4096 bytes.")
    p, d = len(names), len(names) * (factors + 1)
    # Conservative complete tables/attributes/LaTeX allowance. Every numeric
    # cell and row has 96 bytes, with extra copies for escaped labels and IDs.
    cells = 7 * d + d * d + replications * (d + len(_DIAGNOSTICS)) + len(_DIAGNOSTICS) \
        + p * factors + 3 * p + p * p + factors * factors + 2 * p + rows * p \
        + (p * factors if has_target else 0)
    table_rows = 2 * d + 2 * replications + 7 * p + factors + rows + 1
    parameter_bytes = sum(len(f"loading:{name}:Factor{i + 1}".encode())
                          for name in names for i in range(factors)) \
        + sum(len(f"uniqueness:{name}".encode()) for name in names)
    estimate = 96 * (cells + table_rows) + 32 * parameter_bytes \
        + 32 * sum(len(name.encode()) for name in names + axis_names) + 2 * identity_bytes + 65536
    if estimate > MAX_EXPORT_BYTES:
        raise AnalysisError("export_limit", "The complete bootstrap tables, attributes and portable export exceed the 32 MiB budget.")
    return estimate


def _orientation(loadings: Tensor, eigenvalues: Tensor, names: list[str],
                 target: Tensor | None, anchors: list[str] | None):
    """Identify ordered axes, or the gauge-invariant full-target loading matrix."""
    m = loadings.shape[1]
    scale = max(1.0, abs(float(eigenvalues[0])))
    threshold = GEOMETRY_TOLERANCE * scale
    if float(eigenvalues[m - 1]) <= threshold:
        raise AnalysisError("unidentified_factor", "Every fixed retained reduced-matrix eigenvalue must be positive away from zero; no factor count is selected or reduced.")
    # A fixed retained cluster must be separated from the discarded cluster.
    gaps = eigenvalues[:m] - eigenvalues[1:m + 1]
    required_gap = float(gaps[-1] if target is not None else gaps.min())
    if required_gap <= threshold:
        raise AnalysisError("unidentified_factor", "The required retained reduced-matrix eigenvalue gap is too small to identify the fixed factor functional.")
    if target is None:
        fixed = anchors or [names[int(row)] for row in loadings.abs().argmax(0)]
        values = loadings[[names.index(name) for name in fixed], torch.arange(m)]
        minimum = float(values.abs().min())
        if minimum <= GEOMETRY_TOLERANCE * max(1.0, float(loadings.abs().max())):
            raise AnalysisError("unidentified_factor", "Every fixed factor sign anchor must have a loading away from numerical zero.")
        signs = torch.where(values < 0, -torch.ones_like(values), torch.ones_like(values))
        return loadings * signs, torch.diag(signs), fixed, required_gap, minimum, math.nan, math.nan
    cross = loadings.T @ target
    if not bool(torch.isfinite(cross).all()):
        raise AnalysisError("numerical_failure", "The loading-target cross product is non-finite; rescale the target.")
    singular = torch.linalg.svdvals(cross)
    minimum, maximum = float(singular[-1]), float(singular[0])
    if minimum <= GEOMETRY_TOLERANCE * maximum or maximum <= 0:
        raise AnalysisError("unidentified_target", "The fixed full target does not identify a unique, numerically conditioned orthogonal polar factor.")
    rotation = target_matrix(loadings, target)
    rotated = loadings @ rotation
    if not bool(torch.isfinite(rotated).all()):
        raise AnalysisError("numerical_failure", "Full-target factor rotation produced non-finite loadings.")
    return rotated, rotation, None, required_gap, math.nan, minimum, maximum / minimum


@c.procedure
def _parameters(x: Tensor, names: list[str], factors: int,
                target: Tensor | None, anchors: list[str] | None) -> _Fit:
    # Referencing one observed row before centring avoids cancellation from a
    # large common origin; correlations must not depend on physical units.
    shifted = x - x[0]
    centred, shift_mean = c.centre(shifted)
    mean = x[0] + shift_mean
    sscp = centred.T @ centred
    sscp = (sscp + sscp.T) / 2
    diagonal = sscp.diagonal()
    if not bool(torch.isfinite(sscp).all()) or bool((diagonal <= 0).any()):
        raise AnalysisError("constant_column", "Every analysed variable must have finite positive sample variation in every fit.")
    if bool((diagonal / (len(x) - 1) < torch.finfo(c.FLOAT).tiny).any()):
        raise AnalysisError("numerical_failure", "A sample variance underflows normal float64 precision; rescale the measurements before principal-factor uncertainty.")
    r = sscp / torch.outer(diagonal.sqrt(), diagonal.sqrt())
    r = (r + r.T) / 2
    r.diagonal().fill_(1.0)
    roots = torch.linalg.eigvalsh(r)
    if not bool(torch.isfinite(roots).all()) \
            or float(roots[0]) <= GEOMETRY_TOLERANCE * float(roots[-1]):
        raise AnalysisError("singular_matrix", "Principal-factor bootstrap requires a positive-definite correlation matrix away from singularity in every fit; no repair or variable deletion is performed.")
    smc = ex.squared_multiple_correlations(r)
    fitted = ex.principal_factors(r, smc, factors)
    loadings, rotation, fixed, gap, anchor, cross_min, cross_condition = _orientation(
        fitted.loadings, fitted.factored, names, target, anchors)
    uniqueness = 1 - loadings.square().sum(1)
    if not bool(torch.isfinite(uniqueness).all()) \
            or bool((uniqueness <= GEOMETRY_TOLERANCE).any()):
        raise AnalysisError("heywood_case", "Principal-factor bootstrap refuses boundary or nonpositive uniquenesses in every fit.")
    diagnostics = torch.tensor([float(fitted.factored[factors - 1]), gap,
                                float(uniqueness.min()), anchor, cross_min,
                                cross_condition, float(roots[-1] / roots[0])], dtype=c.FLOAT)
    return _Fit(loadings, uniqueness, fitted.factored, smc, r, mean,
                (diagonal / (len(x) - 1)).sqrt(), rotation, fixed, diagnostics)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _state_digest(result: TableSet) -> str:
    """Integrity receipt for the complete portable result, with undefined cells null."""
    payload = {"title": result.title,
               "attrs": {key: value for key, value in result.attrs.items()
                         if key != "state_content_sha256"},
               "tables": {name: {"index": list(frame.index), "columns": list(frame.columns),
                                 "index_name": frame.index.name, "column_name": frame.columns.name,
                                 "data": frame.to_numpy().tolist(), "attrs": frame.attrs}
                          for name, frame in result.items()}}
    encoded = json.dumps(_json_value(payload), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_EXPORT_BYTES:
        raise AnalysisError("export_limit", "The complete portable bootstrap result exceeds 32 MiB.")
    return hashlib.sha256(encoded).hexdigest()


@c.procedure
@resident_cpu
def factor_multifactor_bootstrap(data: Any, columns: list[str], *, factors: int,
                                target: Any = None, anchors: list[str] | None = None,
                                replications: int = 199, confidence: float = .95,
                                seed: int = 0, missing: str = "drop") -> TableSet:
    """Marginal iid intervals for a fixed two-to-four-factor PF functional.

    Each complete-row bootstrap refit recomputes sample means, scales,
    correlation and squared multiple correlations, then retains exactly the
    caller's fixed factor count. Loadings and uniquenesses have full joint
    covariance (divisor B-1), standard errors, bias and marginal percentile
    intervals with linear interpolation. P-values and inference df are undefined.

    With ``target=None``, retained axes stay in descending reduced-eigenvalue
    order. Every retained root and the retained/discarded boundary must be
    separated. One sign anchor per axis is fixed across all replicates; supplied
    ``anchors`` name variables in factor order, or the strongest absolute point
    loading is recorded once per axis. Axes are never permuted or mixed.

    A finite complete p-by-factors ``target`` defines an orthogonal Procrustes
    orientation in its declared column order and signs, without Kaiser
    normalization. The caller must declare this target independently of this
    sample; it is never optimized or re-estimated from data or replicates.
    The loading-target cross product must have full, conditioned rank in every
    fit. This unique polar factor removes arbitrary internal retained-space
    orientation; repeated internal retained roots are allowed, while the
    retained/discarded boundary remains separated. Reflections are allowed.

    The target is the smooth principal-factor estimator functional, not latent
    model parameters or an ML loading SE. IID complete-case rows, fixed p/count,
    finite fourth moments, interior population correlation and uniqueness,
    and the stated population identification conditions are assumed; finite
    sample checks do not establish them. Percentile intervals have first-order
    marginal bootstrap justification, with no familywise or exact coverage
    claim. Selection, other extractions/rotations, weights, clusters, Dataset,
    summary input and non-CPU devices are unsupported. Every replicate must
    pass; all failures are collected and refuse the entire uncertainty result.
    The raw complete sample, all parameter draws, identification settings,
    diagnostic matrices and RNG/version persist for independent saved replay.
    """
    names = c.name_list(columns, "columns", minimum=3)
    factors = c.check_count(factors, "factors", minimum=2, maximum=MAX_FACTORS)
    if factors >= len(names):
        raise AnalysisError("invalid_spec", "A fixed multi-factor bootstrap needs more analysed variables than retained factors.")
    replications = c.check_count(replications, "replications", minimum=19, maximum=MAX_REPLICATIONS)
    confidence = c.check_number(confidence, "confidence", minimum=0, maximum=1, exclusive=True)
    if confidence == 1:
        raise AnalysisError("invalid_option", "confidence must be strictly below one.")
    if (replications + 1) * (1 - confidence) / 2 < 1 - 1e-12:
        raise AnalysisError("insufficient_replications", "Each requested percentile tail needs at least one expected bootstrap order statistic; raise replications or lower confidence.")
    seed = c.check_count(seed, "seed", minimum=0, maximum=2**63 - 1)
    c.check_choice(missing, "missing", ("drop", "raise"))
    fixed_anchors = _anchors(anchors, names, factors, target)
    p, rows = len(names), u._input_rows(data, names)
    if p > MAX_VARIABLES or rows > MAX_ROWS:
        raise AnalysisError("workspace_limit", f"Multi-factor bootstrap supports at most {MAX_VARIABLES} variables and {MAX_ROWS} resident physical rows.")
    d = p * (factors + 1)
    work = (replications + 1) * (rows * p * p + 64 * p**3) + replications * d * d
    if work > MAX_WORK:
        raise AnalysisError("work_limit", "Requested bootstrap refits and full joint covariance exceed the multi-factor-bootstrap work budget.")
    plan = plan_workspace("fixed multi-factor iid bootstrap", {
        "selected_and_refit_blocks": rows * p * 64,
        "row_indices_and_masks": rows * 64,
        "factor_matrix_and_target_workspace": 64 * p * p * 8,
        "full_replicate_vectors_and_diagnostics": replications * (d + len(_DIAGNOSTICS)) * 32,
        "joint_covariance_and_quantiles": 64 * d * d * 8,
    }).record()
    source_ids, index_names, column_names, multi_index, identity_bytes = _identities(data, rows)
    export_bytes = _export_admission(names, rows, factors, replications, identity_bytes,
                                    index_names + column_names, target is not None)
    fixed_target = _target(target, names, factors)
    sample, keep, dropped = c.select(u._selected_input(data, names), names, missing=missing)
    if any(pd.api.types.is_bool_dtype(sample[name].dtype)
           or pd.api.types.is_complex_dtype(sample[name].dtype) for name in names):
        raise AnalysisError("non_numeric_column", "Bootstrap variables must be real numeric observations; Boolean and complex columns are unsupported.")
    n = len(sample)
    if n < max(20, 2 * p + 1):
        raise AnalysisError("insufficient_observations", "Multi-factor bootstrap needs at least max(20, 2*p+1) complete iid rows.")
    x = c.matrix(sample, names)
    point = _parameters(x, names, factors, fixed_target, fixed_anchors)
    fixed_anchors = point.anchors
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.empty((replications, d), dtype=c.FLOAT)
    diagnostics = torch.empty((replications, len(_DIAGNOSTICS)), dtype=c.FLOAT)
    failures = []
    for replication in range(replications):
        indices = torch.randint(n, (n,), generator=generator, device="cpu")
        try:
            fitted = _parameters(x[indices], names, factors, fixed_target, fixed_anchors)
            draws[replication] = fitted.vector()
            diagnostics[replication] = fitted.diagnostics
        except AnalysisError as exc:
            failures.append({"replication": replication + 1, "code": exc.code, "message": str(exc)})
    if failures:
        error = AnalysisError("bootstrap_failure", f"{len(failures)} of {replications} iid bootstrap refits failed. No uncertainty result is returned and no failed replicate is dropped or replaced.")
        error.failures = failures
        error.replications_attempted = replications
        error.successful_replications = replications - len(failures)
        raise error
    centred = draws - draws.mean(0)
    covariance = centred.T @ centred / (replications - 1)
    covariance = (covariance + covariance.T) / 2
    se = covariance.diagonal().clamp_min(0).sqrt()
    tails = torch.tensor([(1 - confidence) / 2, (1 + confidence) / 2], dtype=c.FLOAT)
    intervals = torch.quantile(draws, tails, dim=0, interpolation="linear")
    labels = [f"Factor{i + 1}" for i in range(factors)]
    parameters = [f"loading:{name}:{axis}" for name in names for axis in labels] \
        + [f"uniqueness:{name}" for name in names]
    positions = [i for i, kept in enumerate(keep.tolist()) if kept]
    source_digest = hashlib.sha256(json.dumps({"variables": names, "positions": positions,
                                              "shape": list(x.shape), "physical_rows": rows,
                                              "index_codes": [source_ids[i] for i in positions],
                                              "index_names": index_names, "column_names": column_names,
                                              "multi_index": multi_index}, separators=(",", ":")).encode())
    source_digest.update(x.numpy().tobytes())
    undefined = torch.full((d,), math.nan, dtype=c.FLOAT)
    tables = {
        "estimates": c.frame(torch.stack((point.vector(), se, intervals[0], intervals[1],
                                           draws.mean(0) - point.vector(), undefined, undefined), dim=1),
                             columns=["estimate", "std_error", "ci_lower", "ci_upper", "bootstrap_bias", "p_value", "df"], index=parameters),
        "covariance": c.frame(covariance, columns=parameters, index=parameters),
        "replicates": c.frame(draws, columns=parameters, index=list(range(1, replications + 1))),
        "replicate_diagnostics": c.frame(diagnostics, columns=_DIAGNOSTICS, index=list(range(1, replications + 1))),
        "point_diagnostics": c.frame(point.diagnostics[None, :], columns=_DIAGNOSTICS, index=["point"]),
        "point_loadings": c.frame(point.loadings, columns=labels, index=names),
        "point_uniqueness": c.frame(point.uniqueness[:, None], columns=["uniqueness"], index=names),
        "point_eigenvalues": c.frame(point.eigenvalues[:, None], columns=["reduced_eigenvalue"], index=[f"Root{i + 1}" for i in range(p)]),
        "point_smc": c.frame(point.smc[:, None], columns=["smc"], index=names),
        "point_correlation": c.frame(point.correlation, columns=names, index=names),
        "point_rotation_matrix": c.frame(point.rotation, columns=labels, index=labels),
        "descriptives": c.frame(torch.stack((point.mean, point.std), dim=1), columns=["mean", "std_dev"], index=names),
        "sample": c.frame(x, columns=names, index=positions),
    }
    if fixed_target is not None:
        tables["target"] = c.frame(fixed_target, columns=labels, index=names)
    rotation = "target" if fixed_target is not None else None
    orientation = "fixed full-target orthogonal Procrustes" if rotation else "fixed ordered reduced eigenaxes with fixed sign anchors"
    notes = ["Marginal percentile intervals for the fixed principal-factor estimator functional; no familywise or exact finite-sample coverage.",
             "IID complete-case rows, fixed p/count, finite fourth moments and interior nonsingular population correlation/uniqueness are assumed; finite sample checks do not verify these population conditions.",
             "No loading ML standard errors, p-values or inference df; no selected factor count, weights, clusters or other rotations; every requested replicate must pass the same geometry checks."]
    if rotation:
        notes.append("The complete target is caller declared independently of this sample and remains identical in every replicate; no Kaiser normalization. Internal repeated roots do not identify unrotated axes, but the target-oriented loading functional is invariant to their arbitrary basis.")
    result = TableSet(tables, title="Fixed multi-factor principal-factor iid bootstrap uncertainty",
                      procedure="factor_multifactor_bootstrap", state_schema=STATE_SCHEMA,
                      method="pf", factors=factors, fixed_count=True, rotate=rotation,
                      kaiser=False, orientation=orientation, sign_anchors=fixed_anchors,
                      anchor_selection=None if rotation else "strongest absolute point loading per axis" if anchors is None else "caller declared per axis",
                      target_policy="caller declared independently of this sample; held fixed in every refit" if rotation else None,
                      target=None if fixed_target is None else fixed_target.tolist(),
                      internal_repeated_roots_allowed=rotation is not None,
                      point_unrotated_axes_identified=bool(((point.eigenvalues[:factors - 1] - point.eigenvalues[1:factors])
                                                            > GEOMETRY_TOLERANCE * max(1., abs(float(point.eigenvalues[0])))).all()),
                      point_rotation_matrix_basis="nuisance extraction basis to declared loading axes; not an inferential parameter",
                      geometry_tolerance=GEOMETRY_TOLERANCE,
                      n=n, n_missing=dropped, physical_rows=rows, variables=names,
                      replications=replications, successful_replications=replications,
                      failed_replications=[], confidence=confidence, seed=seed,
                      missing="listwise", missing_policy=missing, sample_positions=positions,
                      original_index_encoded=[source_ids[i] for i in positions],
                      original_index_names_encoded=index_names, source_column_index_names_encoded=column_names,
                      original_index_is_multi=multi_index,
                      source_content_sha256=source_digest.hexdigest(), precision="float64", device="cpu",
                      rng="torch.Generator CPU randint", rng_version=torch.__version__,
                      covariance_divisor=replications - 1, quantile_interpolation="linear",
                      parameter_order="variable-major loadings, then variable-order uniquenesses",
                      parameter_dimension=d, parameter_labels=parameters,
                      uncertainty="iid nonparametric marginal percentile bootstrap",
                      inference_target="fixed full-target orthogonal principal-factor estimator functional" if rotation else "fixed ordered unrotated multi-factor principal-factor estimator functional",
                      inferential_assumptions="iid complete-case rows; fixed p and factor count; finite fourth moments; population correlation positive definite away from singularity; positive retained roots; retained/discarded spectral gap; interior positive uniqueness; fixed conditioned full-target polar factor" if rotation else "iid complete-case rows; fixed p and factor count; finite fourth moments; population correlation positive definite away from singularity; simple positive retained roots with retained/discarded spectral gap; fixed sign anchors away from zero; interior positive uniqueness",
                      p_values_available=False, inference_df_available=False,
                      resource_plan=plan, estimated_work=work, estimated_export_bytes=export_bytes,
                      portable_export_limit_bytes=MAX_EXPORT_BYTES, notes=notes)
    result.attrs["state_content_sha256"] = _state_digest(result)
    return result
