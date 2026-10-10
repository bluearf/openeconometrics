"""Fixed independent-query score functionals of complete spectral bootstraps.

The original point and every seeded training refit are replayed before reuse.
Intervals propagate training-estimator uncertainty, not observation noise or
individual latent posterior variation. See the preregistered method guide.
"""
from __future__ import annotations

import hashlib
from numbers import Integral
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.postest.index_codec import decode, encode
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary, summary_state
from openecon.resources import plan_workspace

from . import canon_frequency_uncertainty as cf
from . import canon_uncertainty as cu
from . import common as c
from . import factor_frequency_uncertainty as ff
from . import factor_uncertainty as fu
from . import frequency_bootstrap as f
from . import pca_frequency_uncertainty as pf
from . import pca_score_uncertainty as ps
from .pca_subspace import _measurement_admission, _numeric_identity_admission

MAX_QUERY_ROWS = 128
MAX_PARAMETERS = 128
MAX_WORK = 250_000_000
MAX_EXPORT_BYTES = 32 * 1024**2
_PROCEDURES = {
    "pca": {"pca_fweight_bootstrap"},
    "canon": {"canon_bootstrap", "canon_fweight_bootstrap"},
    "factor": {"factor_multifactor_bootstrap", "factor_multifactor_fweight_bootstrap"},
}


def _invalid(message="The complete saved training sample, settings or numerical state are invalid."):
    raise AnalysisError("invalid_state", message)


def _integer(value, low, high):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        _invalid()
    return int(value)


def _admit(result, kind):
    """Bound all stored dimensions and metadata before encoding or copying."""
    if not isinstance(result, TableSet) or result.attrs.get("procedure") not in _PROCEDURES[kind]:
        _invalid("Pass the complete supported training bootstrap result, including its original sample.")
    a = result.attrs
    if a.get("state_schema") != "openecon." + a["procedure"] + ".v1":
        _invalid()
    budget = [0, 0]
    ps._metadata_bounds([result.title, a], budget)
    names = c.name_list(a.get("variables"), "saved variables", minimum=2)
    p = len(names)
    if p > 16 or any(len(name.encode()) > 4096 for name in names):
        _invalid()
    physical = _integer(a.get("physical_rows"), 20, 10_000)
    b = _integer(a.get("replications"), 19, 1999)
    _integer(a.get("estimated_work"), 1, MAX_WORK)
    _integer(a.get("seed"), 0, 2**63 - 1)
    _integer(a.get("estimated_export_bytes" if a["procedure"] == "factor_multifactor_bootstrap"
                   else "estimated_complete_export_bytes"), 1, MAX_EXPORT_BYTES)
    n = _integer(a.get("n_units" if "fweight" in a["procedure"] else "n"), 20,
                 100_000 if "fweight" in a["procedure"] else 10_000)
    m = _integer(a.get("factors" if kind == "factor" else "components"),
                 2 if kind == "factor" else 1, 4 if kind != "pca" else p)
    if kind == "canon" and a.get("target") != "coefficients":
        _invalid("CCA query scores require the identified coefficient target, not roots alone.")
    if kind == "factor" and a.get("method") != "pf":
        _invalid("Only the fixed multifactor principal-factor regression-score functional is admitted.")
    if not isinstance(result.title, str) or not 1 <= len(result) <= 32:
        _invalid()
    cells = 0
    for frame in result.values():
        if not isinstance(frame, pd.DataFrame) or frame.ndim != 2:
            _invalid()
        cells += frame.size
        if cells > 500_000 or max(frame.shape, default=0) > 100_000:
            _invalid("Saved tables exceed the bounded complete score-replay geometry.")
        if any(pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype)
               for dtype in frame.dtypes) and frame is not result.get("source_accounting"):
            _invalid("Saved numerical tables cannot contain boolean or complex geometry.")
    # Collection sizes are now bounded. No to_numpy/data serialization occurs
    # before this pass over axes and potentially untrusted object metadata.
    for frame in result.values():
        ps._metadata_bounds([list(frame.index), list(frame.columns),
                             list(frame.index.names), list(frame.columns.names), frame.attrs], budget)
        for name in frame:
            if pd.api.types.is_object_dtype(frame[name].dtype) or pd.api.types.is_string_dtype(frame[name].dtype):
                for value in frame[name]:
                    ps._metadata_bounds(value, budget)
    sample = result.get("sample")
    if sample is None or list(sample.columns) != names or not 1 <= len(sample) <= physical:
        _invalid()
    if kind == "canon":
        xs, ys = c.name_list(a.get("x"), "saved X"), c.name_list(a.get("y"), "saved Y")
        if xs + ys != names or len(set(names)) != p or m > min(len(xs), len(ys)):
            _invalid()
    if kind == "factor" and m >= p:
        _invalid()
    r = len(sample)
    weighted = "fweight" in a["procedure"]
    d = m * (1 + 2 * p) if kind == "pca" else m * (1 + 4 * p) if kind == "canon" else p * (m + 1)
    shapes = {"sample": (r, p), "estimates": (d, 7), "covariance": (d, d), "replicates": (b, d),
              "descriptives": (p, 2), "replicate_diagnostics": (b, 4 if kind == "pca" else 8 if kind == "canon" else 7)}
    if kind == "pca":
        shapes.update(point_eigenvalues=(p, 1), point_covariance=(p, p), point_matrix=(p, p),
                      point_eigenvectors=(p, m), point_loadings=(p, m), parameter_order=(d, 3))
    elif kind == "canon":
        px, py = len(a["x"]), len(a["y"])
        shapes.update(point_correlations=(min(px, py), 1), point_covariance=(p, p), point_correlation=(p, p),
                      point_diagnostics=(1, 8), parameter_order=(d, 4), replicate_correlations=(b, min(px, py)))
        for block, size in (("x", px), ("y", py)):
            for role in ("coefficients", "standardized_coefficients", "loadings", "cross_loadings"):
                shapes[f"point_{block}_{role}"] = (size, m)
    else:
        shapes.update(point_loadings=(p, m), point_uniqueness=(p, 1), point_eigenvalues=(p, 1),
                      point_smc=(p, 1), point_correlation=(p, p), point_rotation_matrix=(m, m), point_diagnostics=(1, 7))
        if a.get("target") is not None:
            shapes["target"] = (p, m)
    if weighted:
        point_shape = (p, 1) if kind == "pca" else (1, p)
        shapes.update(source_frequencies=(r, 1), source_index=(r, 2), source_accounting=(physical, 5),
                      replicate_counts=(b, r), replicate_origins=(b, p), replicate_mean_offsets=(b, p),
                      point_origin=point_shape, point_mean_offset=point_shape)
        if kind != "pca":
            shapes["replicate_covariances"] = (b, p * p)
        if kind == "factor":
            shapes.update(replicate_eigenvalues=(b, p), replicate_smc=(b, p), point_covariance=(p, p))
    elif kind == "canon":
        shapes["resample_indices"] = (b, n)
    if weighted or kind == "canon":
        shapes.update(replicate_means=(b, p), replicate_standard_deviations=(b, p))
        settings = result.get("settings")
        if settings is None or settings.shape[1] != 2 or not len(a) - 2 <= len(settings) <= len(a):
            _invalid()
        shapes["settings"] = settings.shape
    if set(result) != set(shapes) or any(result[key].shape != shape for key, shape in shapes.items()):
        _invalid("Saved table names and shapes differ from the bounded complete training geometry.")
    # Derive replay work from geometry; a forged small stored estimate cannot
    # bypass admission before the canonical refit detects it.
    replay_work = (b + 1) * (physical * p * p + 64 * p**3) + 2 * b * d * d + 100_000
    if weighted:
        replay_work += b * n * ((r - 1).bit_length() + 4)
    if replay_work > MAX_WORK:
        raise AnalysisError("work_limit", "Complete source numerical replay exceeds the combined work domain.")
    return names, p, physical, n, m, b, cells, replay_work


def _decoded(codes):
    if not isinstance(codes, list):
        _invalid()
    values = []
    for code in codes:
        if not isinstance(code, str) or len(code.encode()) > 4096:
            _invalid()
        try:
            value = decode(code)
            if encode(value) != code:
                _invalid("Source typed identities must be canonical and lossless.")
        except (ValueError, TypeError, RecursionError, OverflowError):
            _invalid()
        values.append(value)
    return values


def _source(result, names, physical):
    a = result.attrs
    weighted = "fweight" in a["procedure"]
    if weighted:
        accounting = result["source_accounting"]
        if accounting.shape != (physical, 5) or list(accounting["source_position"]) != list(range(physical)):
            _invalid()
        labels = _decoded(accounting["index_code"].tolist())
        index_names = _decoded(a["index_names_json"])
        column_names = _decoded(a["column_axis_names_json"])
        positions = result["source_index"]["source_position"].tolist()
        multi = len(index_names) > 1
    else:
        factor = a["procedure"] == "factor_multifactor_bootstrap"
        positions = a["sample_positions"]
        codes = a["original_index_encoded" if factor else "sample_index_codes"]
        index_names = _decoded(a["original_index_names_encoded" if factor else "source_index_names"])
        column_names = _decoded(a["source_column_index_names_encoded" if factor else "source_column_names"])
        multi = a["original_index_is_multi"] if factor else a["source_index_nlevels"] > 1
        labels = [(f"__dropped_{i}__",) * len(index_names) if multi else f"__dropped_{i}__"
                  for i in range(physical)]
        kept_labels = _decoded(codes)
        if len(kept_labels) != len(positions):
            _invalid()
        for position, label in zip(positions, kept_labels, strict=True):
            _integer(position, 0, physical - 1)
            labels[position] = label
    if not isinstance(positions, list) or len(positions) != len(result["sample"]) \
            or positions != sorted(set(positions)) or any(type(i) is not int or not 0 <= i < physical for i in positions) \
            or not 1 <= len(index_names) <= 32 or len(column_names) != 1:
        _invalid()
    index = (pd.MultiIndex.from_tuples(labels, names=index_names) if multi else
             pd.Index(labels, dtype=object, tupleize_cols=False, name=index_names[0]))
    source = pd.DataFrame(float("nan"), index=index, columns=names)
    source.iloc[positions] = result["sample"].to_numpy(dtype=float)
    if weighted:
        weight = a["weights"]
        if not isinstance(weight, str) or weight in names or len(weight.encode()) > 4096:
            _invalid()
        values = accounting["frequency_json"].tolist()
        if any(not isinstance(value, str) or len(value) > 16 for value in values):
            _invalid()
        source[weight] = pd.array([None if value == "null" else int(value) for value in values], dtype=object)
    source.columns.name = column_names[0]
    return source


def _replay(result, kind, names, physical):
    """Re-run the original public fit and compare every portable cell/setting."""
    a = result.attrs
    weighted = "fweight" in a["procedure"]
    try:
        if weighted:
            f.validate_saved(result)
        elif kind == "canon":
            cu._validate_saved(result)
        elif a.get("state_content_sha256") != fu._state_digest(result):
            _invalid()
        source = _source(result, names, physical)
        options = dict(replications=a["replications"], confidence=a["confidence"], seed=a["seed"],
                       missing=a["missing"] if weighted else a["missing_policy"])
        if weighted:
            options.update(weights=a["weights"], weight_type="fweight")
        anchors = a.get("sign_anchors")
        selection = a.get("anchor_selection")
        if selection is not None and selection != "caller declared" and selection != "caller declared per axis":
            anchors = None
        if kind == "pca":
            canonical = pf.pca_fweight_bootstrap(source, names, components=a["components"],
                                                matrix=a["matrix"], anchors=anchors, **options)
        elif kind == "canon":
            function = cf.canon_fweight_bootstrap if weighted else cu.canon_bootstrap
            canonical = function(source, a["x"], a["y"], components=a["components"],
                                 target="coefficients", anchors=anchors, **options)
        else:
            function = ff.factor_multifactor_fweight_bootstrap if weighted else fu.factor_multifactor_bootstrap
            canonical = function(source, names, factors=a["factors"], target=a["target"], anchors=anchors, **options)
        # A fresh plan has already admitted replay in this environment. Its
        # recorded historical budget is not a fitted numerical parameter.
        canonical.attrs["resource_plan"] = a["resource_plan"]
        if kind == "factor" and not weighted:
            # Legacy PF retains complete-case identities, but its historical
            # export estimate also charged unsaved dropped-row labels. Those
            # labels do not enter the fit; retain this bounded resource receipt
            # while independently charging all present replay/output geometry.
            canonical.attrs["estimated_export_bytes"] = a["estimated_export_bytes"]
        if weighted:
            f.seal(canonical)
        else:
            digest = cu._state_digest if kind == "canon" else fu._state_digest
            canonical.attrs["state_content_sha256"] = digest(canonical)
        original, replayed = summary_state(result), summary_state(canonical)
        if original != replayed:
            _invalid("Saved training state differs from the canonical point and every seeded bootstrap refit.")
        return canonical, original
    except AnalysisError as error:
        if error.code in ("workspace_limit", "resource_limit", "work_limit", "export_limit"):
            raise
        _invalid("Complete training bootstrap numerical replay failed: " + str(error))
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, RecursionError):
        _invalid()


def _iid_moments(values):
    origin = values[0]
    centered, offset = c.centre(values - origin)
    sscp = centered.T @ centered
    sscp = (sscp + sscp.T) / 2
    sd = (sscp.diagonal() / (len(values) - 1)).sqrt()
    correlation = sscp / torch.outer(sscp.diagonal().sqrt(), sscp.diagonal().sqrt())
    correlation = (correlation + correlation.T) / 2
    correlation.diagonal().fill_(1)
    return origin, offset, sd, correlation


def _geometry(fit, kind, p, m, b):
    a = fit.attrs
    source = torch.tensor(fit["sample"].to_numpy(dtype=float), dtype=c.FLOAT)
    vector = torch.tensor(fit["replicates"].to_numpy(dtype=float), dtype=c.FLOAT)
    weighted = "fweight" in a["procedure"]
    origins, offsets, sds = [torch.empty((b, p), dtype=c.FLOAT) for _ in range(3)]
    correlations = torch.empty((b, p, p), dtype=c.FLOAT) if kind == "factor" else None
    if weighted:
        origins[:] = torch.tensor(fit["replicate_origins"].to_numpy(), dtype=c.FLOAT)
        offsets[:] = torch.tensor(fit["replicate_mean_offsets"].to_numpy(), dtype=c.FLOAT)
        sds[:] = torch.tensor(fit["replicate_standard_deviations"].to_numpy(), dtype=c.FLOAT)
        point_origin = torch.tensor(fit["point_origin"].to_numpy(), dtype=c.FLOAT).flatten()
        point_offset = torch.tensor(fit["point_mean_offset"].to_numpy(), dtype=c.FLOAT).flatten()
        if kind == "factor":
            covariance = torch.tensor(fit["replicate_covariances"].to_numpy(), dtype=c.FLOAT).reshape(b, p, p)
            correlations[:] = covariance / (sds[:, :, None] * sds[:, None, :])
            for correlation in correlations:
                correlation.diagonal().fill_(1)
    else:
        generator = torch.Generator(device="cpu").manual_seed(a["seed"])
        for i in range(b):
            indices = torch.randint(len(source), (len(source),), generator=generator)
            origin, offset, sd, correlation = _iid_moments(source[indices])
            origins[i], offsets[i], sds[i] = origin, offset, sd
            if kind == "factor":
                correlations[i] = correlation
        point_origin, point_offset, _, _ = _iid_moments(source)
    point_sd = torch.tensor(fit["descriptives"]["std_dev"].to_numpy(), dtype=c.FLOAT)
    if kind == "pca":
        axes = c.numbered("Comp", m)
        point_coefficients = torch.tensor(fit["point_eigenvectors"].to_numpy(), dtype=c.FLOAT)
        coefficients = vector.reshape(b, m, 1 + 2 * p)[:, :, 1:1 + p].transpose(1, 2)
        standardized = a["matrix"] == "correlation"
    elif kind == "canon":
        px = len(a["x"])
        axes = [f"{block}:Can{i + 1}" for block in ("X", "Y") for i in range(m)]
        point_coefficients = torch.zeros((p, 2 * m), dtype=c.FLOAT)
        point_coefficients[:px, :m] = torch.tensor(fit["point_x_coefficients"].to_numpy(), dtype=c.FLOAT)
        point_coefficients[px:, m:] = torch.tensor(fit["point_y_coefficients"].to_numpy(), dtype=c.FLOAT)
        raw = vector.reshape(b, m, 1 + 4 * p)[:, :, 1:1 + p].transpose(1, 2)
        coefficients = torch.zeros((b, p, 2 * m), dtype=c.FLOAT)
        coefficients[:, :px, :m], coefficients[:, px:, m:] = raw[:, :px], raw[:, px:]
        standardized = False
    else:
        axes = c.numbered("Factor", m)
        loadings = vector[:, :p * m].reshape(b, p, m)
        coefficients = torch.cholesky_solve(loadings, torch.linalg.cholesky(correlations))
        point_r = torch.tensor(fit["point_correlation"].to_numpy(), dtype=c.FLOAT)
        point_l = torch.tensor(fit["point_loadings"].to_numpy(), dtype=c.FLOAT)
        point_coefficients = torch.cholesky_solve(point_l, torch.linalg.cholesky(point_r))
        standardized = True
    cu._finite(origins, offsets, sds, coefficients, point_coefficients)
    return axes, coefficients, point_coefficients, origins, offsets, sds, point_origin, point_offset, point_sd, standardized


def _scores(result, data, *, kind, procedure, missing):
    c.check_choice(missing, "missing", ("drop", "raise"))
    names, p, physical, n, m, b, cells, replay_work = _admit(result, kind)
    rows = cu._rows(data, names)
    k = 2 * m if kind == "canon" else m
    q = rows * k
    if rows > MAX_QUERY_ROWS or q > MAX_PARAMETERS:
        raise AnalysisError("workspace_limit", "Score uncertainty admits 128 physical query rows and 128 joint score coordinates.")
    # Charge replay, moment recovery and complete query/source output before
    # source decoding, sample conversion, fitting or new query allocation.
    work = 2 * replay_work + b * (n * p * p + q * q + rows * p * k + 64 * p**3)
    export = 64 * (cells + b * (q + p * k + 3 * p) + q * q + rows * p + 16 * q) + 4 * 1024**2
    if work > MAX_WORK:
        raise AnalysisError("work_limit", "Complete source replay and joint query scores exceed 250 million work units.")
    if export > MAX_EXPORT_BYTES:
        raise AnalysisError("export_limit", "Complete source and query score state exceed the conservative 32 MiB domain.")
    plan = plan_workspace("fixed multivariate query-score bootstrap", {
        "saved_state_and_replay": 64 * cells,
        "source_and_moment_recovery": 128 * (physical * p + b * p * p),
        "all_score_draws_and_coefficients": 64 * b * (q + p * k + 3 * p),
        "joint_covariance_and_quantiles": 64 * (q * q + b * q),
        "complete_portable_source_and_query": export,
    }).record()
    _measurement_admission(data, names)
    selected_input = cu._selected(data, names)
    if isinstance(selected_input, pd.DataFrame) and any(
            cu._nonreal_measurement(value) for name in names for value in selected_input[name]):
        raise AnalysisError("non_numeric_column", "Query measurements exclude boolean and complex values, including mixed columns.")
    fit, source_state = _replay(result, kind, names, physical)
    selected, keep, dropped = c.select(selected_input, names, missing=missing)
    query = c.matrix(selected, names)
    cu._finite(query)
    positions = [i for i, flag in enumerate(keep.tolist()) if flag]
    column_names = list(data.columns.names) if isinstance(data, pd.DataFrame) else [None]
    _numeric_identity_admission(keep.index, [True] * rows, column_names)
    codes, index_names, column_codes, _ = cu._identities(keep.index, [True] * rows, column_names)
    geometry = _geometry(fit, kind, p, m, b)
    axes, coefficients, point_coeff, origins, offsets, sds, origin, offset, sd, standardized = geometry
    centered = (query[None] - origins[:, None]) - offsets[:, None]
    point_input = (query - origin) - offset
    if standardized:
        centered = centered / sds[:, None]
        point_input = point_input / sd
    draws = torch.bmm(centered, coefficients).reshape(b, -1)
    point = (point_input @ point_coeff).flatten()
    covariance, se, bias, lower, upper = f.joint(point, draws, fit.attrs["confidence"])
    order = [[position, axis] for position in positions for axis in axes]
    labels = [f"query[{position}]:{axis}" for position, axis in order]
    q = len(labels)
    undefined = torch.full((q,), float("nan"), dtype=c.FLOAT)
    full_query = torch.full((rows, p), float("nan"), dtype=c.FLOAT)
    full_query[positions] = query
    full_point = torch.full((rows, k), float("nan"), dtype=c.FLOAT)
    full_point[positions] = point.reshape(len(positions), k)
    ri = range(1, b + 1)
    out = TableSet({
        "point_scores": c.frame(full_point, columns=axes),
        "estimates": c.frame(torch.stack((point, se, lower, upper, bias, undefined, undefined), 1),
                             columns=["estimate", "std_error", "ci_lower", "ci_upper", "bootstrap_bias", "p_value", "df"], index=labels),
        "covariance": c.frame(covariance, columns=labels, index=labels),
        "replicates": c.frame(draws, columns=labels, index=ri),
        "point_score_coefficients": c.frame(point_coeff, columns=axes, index=names),
        "replicate_score_coefficients": c.frame(coefficients.reshape(b, -1), index=ri,
                                                columns=[f"{name}:{axis}" for name in names for axis in axes]),
        "replicate_centering_origins": c.frame(origins, columns=names, index=ri),
        "replicate_centering_offsets": c.frame(offsets, columns=names, index=ri),
        "replicate_scales": c.frame(sds, columns=names, index=ri),
        "point_centering": c.frame(torch.stack((origin, offset, sd), 1), columns=["origin", "offset", "std_dev"], index=names),
        "parameter_order": table(order, columns=["query_position", "score"], index=labels),
        "query": c.frame(full_query, columns=names),
        "query_identities": table({"typed_index_code": codes, "complete": keep.tolist()}),
        **{f"fit__{name}": frame.copy(deep=True) for name, frame in result.items()},
    }, title=f"Fixed-query {kind.upper()} score bootstrap uncertainty",
        procedure=procedure, state_schema=f"openecon.{procedure}.v1", variables=names,
        axes=axes, physical_query_rows=rows, n_query=len(positions), n_missing_query=dropped,
        missing_policy=missing, query_positions=positions, query_index_codes=codes,
        query_index_names=index_names, query_column_names=column_codes,
        query_index_nlevels=keep.index.nlevels, parameter_order=order, parameter_labels=labels,
        parameter_dimension=q, replications=b, confidence=fit.attrs["confidence"], seed=fit.attrs["seed"],
        covariance_divisor=b - 1, quantile_interpolation="linear", fit_attrs=fit.attrs.copy(), fit_title=fit.title,
        source_state_sha256=hashlib.sha256(source_state.encode()).hexdigest(),
        saved_fit_validation="complete canonical point and every seeded bootstrap refit",
        query_contract="caller fixed independently of training; estimator-score functional, no future observation/noise or latent posterior interval",
        scoring="regression R^-1 L" if kind == "factor" else "raw paired canonical coefficients" if kind == "canon" else "identified PCA eigenvectors",
        stable_centering="(query-training_origin)-training_offset in every draw", standardized_scores=standardized,
        inferential_assumptions=fit.attrs["inferential_assumptions"],
        uncertainty="first-order IID marginal percentile bootstrap", familywise_intervals=False,
        p_values_available=False, inference_df_available=False, precision="float64", device="cpu",
        resource_plan=plan, estimated_work=work, estimated_complete_export_bytes=export,
        notes=["Full covariance includes all query rows and both CCA blocks; no independent-row interval shortcut.",
               "Missing query rows preserve physical positions and typed identities without inferred scores.",
               "All original fit tables and settings persist; source numerical replay is distinct from checksum validation."])
    saved_summary(out)
    out.attrs["state_content_sha256"] = f._digest(out)
    if len(summary_state(out).encode()) > MAX_EXPORT_BYTES:
        raise AnalysisError("export_limit", "Complete portable score state exceeds 32 MiB.")
    return out


@c.procedure
@resident_cpu
def pca_fweight_bootstrap_scores(result: TableSet, data: Any, *, missing: str = "drop") -> TableSet:
    """Fixed-query covariance/correlation PCA scores from a full frequency bootstrap.

    Queries are independent of training. Each accepted draw re-centers and,
    for correlation PCA, re-standardizes by its own training moments. Full
    joint percentile uncertainty describes estimator variation only.
    """
    return _scores(result, data, kind="pca", procedure="pca_fweight_bootstrap_scores", missing=missing)


@c.procedure
@resident_cpu
def canon_bootstrap_scores(result: TableSet, data: Any, *, missing: str = "drop") -> TableSet:
    """Joint paired X/Y fixed-query scores from an IID or frequency CCA bootstrap.

    Require target='coefficients', fixed simple identified roots and all X/Y
    query columns. Full covariance retains cross-block and cross-query terms.
    Intervals concern training-estimator variation, not future observations.
    """
    return _scores(result, data, kind="canon", procedure="canon_bootstrap_scores", missing=missing)


@c.procedure
@resident_cpu
def factor_bootstrap_scores(result: TableSet, data: Any, *, missing: str = "drop") -> TableSet:
    """Fixed-query regression factor scores from a complete multifactor PF bootstrap.

    IID and literal-frequency unrotated/full-target orthogonal fits are
    supported. Every draw uses its own correlation, accepted loadings,
    R^-1 L regression coefficients, centering and scales. These are score
    functionals, not individual latent-posterior or observation intervals.
    """
    return _scores(result, data, kind="factor", procedure="factor_bootstrap_scores", missing=missing)
