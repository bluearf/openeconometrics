"""Fixed-query score functionals of a complete, identified PCA bootstrap.

Scores use the training means (and correlation scales) of every fitted draw.
They describe training-estimator uncertainty, not a future observation/noise
interval. The saved source is checked by a complete deterministic refit before
reuse; generic JSON restoration alone is not a numerical acceptance check.

Projection definition: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prcomp.html
Bootstrap PCA scores: https://arxiv.org/abs/1405.0922
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.postest.index_codec import decode
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary, summary_state
from openecon.resources import plan_workspace

from . import common as c
from . import pca_uncertainty as pu
from .pca_subspace import _measurement_admission, _numeric_identity_admission

MAX_QUERY_ROWS = 128
MAX_PARAMETERS = 128


def _invalid(message="The complete saved PCA sample, settings or geometry are invalid."):
    raise AnalysisError("invalid_pca_state", message)


def _integer(value, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        _invalid()
    return int(value)


def _metadata_bounds(value, budget=None):
    if budget is None:
        budget = [0, 0]
    pending = [(value, 0)]
    nodes, size = budget
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > pu.MAX_INDEX_DEPTH or nodes > pu.MAX_INDEX_NODES:
            _invalid("Saved metadata exceeds bounded traversal depth or work.")
        if isinstance(item, (dict, list, tuple)):
            if len(item) > pu.MAX_INDEX_NODES-nodes:
                _invalid("Saved metadata has an unbounded collection.")
            if isinstance(item, dict):
                if any(not isinstance(key, str) for key in item):
                    _invalid()
                pending.extend((child, depth+1) for pair in item.items() for child in pair)
            else:
                pending.extend((child, depth+1) for child in item)
        elif isinstance(item, str):
            if len(item) > 4*pu.MAX_INDEX_BYTES:
                _invalid("Saved metadata contains an oversized text field.")
            size += len(item.encode())
        elif type(item) is int and item.bit_length() > 16384:
            _invalid("Saved metadata contains an oversized integer before serialization.")
        elif type(item) not in (int, float, bool, type(None)) \
                or isinstance(item, float) and not math.isfinite(item):
            _invalid("Saved metadata must contain bounded finite JSON values.")
        if size > 4*pu.MAX_INDEX_BYTES:
            _invalid("Saved metadata exceeds four MiB before serialization.")
    budget[:] = [nodes, size]


def _admission(result, *, validate_numerics=False):
    """Derive the original public refit bounds without trusting saved estimates."""
    if not isinstance(result, TableSet) or result.attrs.get("procedure") != "pca_bootstrap" \
            or result.attrs.get("state_schema") != "openecon.pca_bootstrap.v1":
        _invalid("Pass a complete saved pca_bootstrap result with identified axes.")
    a = result.attrs
    metadata = [0, 0]
    if not isinstance(result.title, str):
        _invalid()
    _metadata_bounds([a, result.title], metadata)
    try:
        names = c.name_list(a["variables"], "variables", minimum=2)
        p = len(names)
        n = _integer(a["n"], 20, pu.MAX_ROWS)
        physical = _integer(a["physical_rows"], n, pu.MAX_ROWS)
        _integer(a["n_missing"], 0, physical)
        _integer(a["source_column_nlevels"], 1, 1)
        m = _integer(a["components"], 1, p)
        b = _integer(a["replications"], 19, pu.MAX_REPLICATIONS)
        q = m*(1+2*p)
        if p > pu.MAX_VARIABLES or q > pu.MAX_PARAMETERS \
                or any(len(json.dumps(name).encode()) > 4096 for name in names):
            _invalid()
        work = (b+1)*(physical*p*p+64*p**3)+2*b*q*q+pu.MAX_INDEX_NODES
        cells = physical*p+b*(q+2*p+4)+q*q+8*q+6*p*p
        export = 32*cells+8*b*physical+64*(physical+b)+4*pu.MAX_INDEX_BYTES
        export += 16*q*sum(len(json.dumps(name).encode()) for name in names)
        if work > pu.MAX_WORK or export > pu.MAX_EXPORT_BYTES \
                or _integer(a["estimated_work"], 1, pu.MAX_WORK) != work \
                or _integer(a["estimated_complete_export_bytes"], 1, pu.MAX_EXPORT_BYTES) != export:
            _invalid("Saved PCA refit work/output admission does not match its actual declared dimensions.")
        shapes = {"estimates": (q, 7), "covariance": (q, q), "replicates": (b, q),
                  "resample_indices": (b, n), "replicate_diagnostics": (b, 4),
                  "replicate_means": (b, p), "replicate_standard_deviations": (b, p),
                  "point_eigenvalues": (p, 1), "point_eigenvectors": (p, m),
                  "point_loadings": (p, m), "point_covariance": (p, p), "point_matrix": (p, p),
                  "descriptives": (p, 2), "sample": (n, p), "parameter_order": (q, 3),
                  "settings": (len(a), 2)}
        if set(result) != set(shapes) or any(not isinstance(result[key], pd.DataFrame)
              or result[key].shape != shape for key, shape in shapes.items()):
            _invalid("Saved PCA table names/shapes do not match the bounded complete state.")
        for key, frame in result.items():
            _metadata_bounds([list(frame.columns), list(frame.index), list(frame.index.names), list(frame.columns.names)], metadata)
            if key in ("parameter_order", "settings"):
                _metadata_bounds(frame.to_numpy().tolist(), metadata)
            else:
                if any(pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype)
                       for dtype in frame.dtypes):
                    _invalid("Saved numeric table booleans/complex values are not admitted.")
                # Public admission checks bounded shapes, metadata and dtypes
                # only. Numeric conversion/copies occur once, after its
                # workspace plan, during complete saved-fit validation.
                if not validate_numerics:
                    continue
                for dtype in frame.dtypes:
                    if not pd.api.types.is_numeric_dtype(dtype):
                        for item in frame.to_numpy().flat:
                            if item is not None and (isinstance(item, bool) or not isinstance(item, Real)):
                                _invalid("Saved numeric tables contain a nonnumeric object payload.")
                        break
                numeric = frame.to_numpy(dtype=float, na_value=float("nan"))
                required = numeric[:, :5] if key == "estimates" else numeric
                if not bool(torch.isfinite(torch.tensor(required, dtype=c.FLOAT)).all()):
                    _invalid("Required saved numerical geometry must be finite.")
                if key == "estimates" and not bool(pd.isna(numeric[:, 5:]).all()):
                    _invalid("PCA p-values and inferential df must remain unavailable.")
        return names, n, physical, m, b, work, export
    except AnalysisError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, RecursionError):
        _invalid()


def _codes(values, length):
    if not isinstance(values, list) or len(values) != length \
            or any(not isinstance(v, str) for v in values) \
            or sum(len(v.encode()) for v in values) > pu.MAX_INDEX_BYTES:
        _invalid("Saved source identity metadata exceeds its declared portable domain.")
    decoded = [decode(v) for v in values]
    # Re-encode to check type syntax, traversal depth and canonical identities.
    index = pd.Index(decoded, dtype=object, tupleize_cols=False)
    _numeric_identity_admission(index, [True]*length, [None])
    codes, _, _ = pu._index_codes(index,
                                  [True]*length, [None])
    if codes != values:
        _invalid("Saved source identities are not canonical lossless typed codes.")
    return decoded


def _validated(result):
    """Reconstruct kept source rows and replay every planned bootstrap fit."""
    admitted = _admission(result, validate_numerics=True)
    a = result.attrs
    try:
        names, n, physical, m, b, _, _ = admitted
        p = len(names)
        levels = _integer(a["source_index_nlevels"], 1, pu.MAX_INDEX_DEPTH)
        positions = a["sample_positions"]
        if p > pu.MAX_VARIABLES or not isinstance(positions, list) or len(positions) != n \
                or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < physical for x in positions) \
                or positions != sorted(set(positions)) or a["n_missing"] != physical-n:
            _invalid()
        # Bound both original state and refit before decoding or allocating.
        full = summary_state(result)
        state_bytes = len(full.encode())
        _integer(a["estimated_complete_export_bytes"], 1, pu.MAX_EXPORT_BYTES)
        sample = result["sample"]
        if sample.shape != (n, p) or list(sample.columns) != names or list(sample.index) != positions:
            _invalid()
        values = c.matrix(sample, names)
        pu._finite(values)
        labels = _codes(a["sample_index_codes"], n)
        index_names = _codes(a["source_index_names"], levels)
        column_names = _codes(a["source_column_names"], 1)
        if a["source_column_nlevels"] != 1:
            _invalid()
        index_values = [(f"__missing_source_{i}__",)*levels if levels > 1 else f"__missing_source_{i}__"
                        for i in range(physical)]
        for position, label in zip(positions, labels, strict=True):
            if levels > 1 and (not isinstance(label, tuple) or len(label) != levels):
                _invalid()
            index_values[position] = label
        index = (pd.MultiIndex.from_tuples(index_values, names=index_names) if levels > 1
                 else pd.Index(index_values, dtype=object, tupleize_cols=False, name=index_names[0]))
        source = pd.DataFrame(float("nan"), index=index, columns=names)
        source.columns.name = column_names[0]
        source.iloc[positions] = values.numpy()
        canonical = pu.pca_bootstrap(source, names, components=m, matrix=a["matrix"],
                                      anchors=a["sign_anchors"], replications=b,
                                      confidence=a["confidence"], seed=a["seed"], missing=a["missing_policy"])
        selection = a["anchor_selection"]
        if selection not in ("caller declared", "strongest point entry"):
            _invalid()
        if selection == "strongest point entry":
            point = canonical["point_eigenvectors"].to_numpy()
            expected = [names[int(abs(point[:, j]).argmax())] for j in range(m)]
            if expected != a["sign_anchors"]:
                _invalid()
        # Environment workspace budgets do not alter fitted geometry. Retain
        # the original recorded plan, while the refit performs fresh admission.
        canonical.attrs.update(anchor_selection=selection, resource_plan=a["resource_plan"])
        saved_summary(canonical)
        if summary_state(canonical) != full:
            _invalid("Saved PCA state differs from the complete canonical point and every seeded draw refit.")
        return canonical, state_bytes
    except AnalysisError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, RecursionError):
        _invalid()


@c.procedure
@resident_cpu
def pca_bootstrap_scores(result: TableSet, data: Any, *, missing: str = "drop") -> TableSet:
    """Joint marginal percentile uncertainty for caller-fixed PCA query scores.

    For covariance PCA each draw uses (query - draw_mean) @ draw_axes.
    Correlation PCA additionally divides by that draw's own sample SDs.
    The component count, signs, regularity assumptions and confidence level
    are inherited from a complete simple-axis pca_bootstrap result. All saved
    source fits are numerically replayed before any query inference is used.

    Queries must be fixed independently of the training sample; conditioning
    on fitted/adaptively chosen queries has no coverage claim. Intervals
    describe an estimator score functional, not future-observation variability.
    Complete query positions/typed identities, including missing rows, persist;
    table row ordinals disambiguate duplicate or complex source index labels.
    No exact, familywise, selected-axis, weighted or dependent-row inference.
    """
    c.check_choice(missing, "missing", ("drop", "raise"))
    names, source_n, _, m, b, source_work, existing_bound = _admission(result)
    rows = pu._rows(data, names)
    q_max = rows*m
    if rows > MAX_QUERY_ROWS or q_max > MAX_PARAMETERS:
        raise AnalysisError("workspace_limit", "Fixed PCA query uncertainty admits 128 physical query rows and 128 joint query-axis parameters.")
    new_bytes = 64*(b*q_max+q_max*q_max+12*q_max+rows*len(names)+2*b*len(names))+8*pu.MAX_INDEX_BYTES
    if existing_bound+new_bytes > pu.MAX_EXPORT_BYTES:
        raise AnalysisError("resource_limit", "The complete source and query bootstrap state exceed the conservative 32 MiB portable domain.")
    work = b*(q_max*q_max+rows*m*len(names)+4*source_n*len(names))
    if work+source_work > pu.MAX_WORK:
        raise AnalysisError("work_limit", "Saved-fit replay and complete joint query work exceed 250 million operations.")
    plan = plan_workspace("fixed PCA query score bootstrap", {
        "one_saved_numeric_validation_block": 24*max(b*source_n, b*m*(1+2*len(names)), source_n*len(names), (m*(1+2*len(names)))**2),
        "all_query_and_score_draws": 64*(rows*len(names)+b*q_max),
        "one_stable_training_centering_block_and_all_offsets": 64*(source_n*len(names)+2*b*len(names)),
        "complete_score_covariance_and_quantiles": 64*(q_max*q_max+b*q_max),
        "portable_query_and_source_state": existing_bound+new_bytes,
    }).record()
    _measurement_admission(data, names)
    fit, source_bytes = _validated(result)
    sample, keep, dropped = c.select(pu._selected(data, names), names, missing=missing)
    if any(is_bool_dtype(sample[name].dtype) for name in names):
        raise AnalysisError("non_numeric_column", "Fixed PCA queries must contain real measurements, excluding booleans.")
    query = c.matrix(sample, names)
    pu._finite(query)
    positions = [i for i, flag in enumerate(keep.tolist()) if flag]
    _numeric_identity_admission(keep.index, [True]*rows,
                       list(data.columns.names) if isinstance(data, pd.DataFrame) else [None])
    index_codes, index_names, column_names = pu._index_codes(keep.index, [True]*rows,
                       list(data.columns.names) if isinstance(data, pd.DataFrame) else [None])
    q = len(sample)*m
    sds = torch.tensor(fit["replicate_standard_deviations"].to_numpy(), dtype=c.FLOAT)
    source = torch.tensor(fit["sample"].to_numpy(), dtype=c.FLOAT)
    origins = torch.empty((b, len(names)), dtype=c.FLOAT)
    offsets = torch.empty_like(origins)
    for ordinal, indices in enumerate(fit["resample_indices"].to_numpy()):
        training_draw = source[torch.tensor(indices, dtype=torch.int64)]
        origins[ordinal] = training_draw[0]
        _, offsets[ordinal] = c.centre(training_draw-training_draw[0])
    parameters = torch.tensor(fit["replicates"].to_numpy(), dtype=c.FLOAT).reshape(b, m, 1+2*len(names))
    axes = parameters[:, :, 1:1+len(names)].transpose(1, 2)
    centred = (query[None, :, :]-origins[:, None, :])-offsets[:, None, :]
    if fit.attrs["matrix"] == "correlation":
        centred = centred/sds[:, None, :]
    draws = torch.bmm(centred, axes).reshape(b, q)
    point_sd = torch.tensor(fit["descriptives"]["std_dev"].to_numpy(), dtype=c.FLOAT)
    point_axes = torch.tensor(fit["point_eigenvectors"].to_numpy(), dtype=c.FLOAT)
    _, point_offset = c.centre(source-source[0])
    point_input = (query-source[0])-point_offset
    if fit.attrs["matrix"] == "correlation":
        point_input = point_input/point_sd
    point = (point_input@point_axes).reshape(q)
    centred_draws = draws-draws.mean(0)
    covariance = centred_draws.T@centred_draws/(b-1)
    covariance = (covariance+covariance.T)/2
    if bool(((centred_draws.abs().amax(0)>0)&(covariance.diagonal()<torch.finfo(c.FLOAT).tiny)).any()):
        raise AnalysisError("numerical_failure", "A nonconstant query score variance underflows normal float64 precision; rescale the queries.")
    se = covariance.diagonal().sqrt()
    confidence = fit.attrs["confidence"]
    bounds = torch.quantile(draws, torch.tensor([(1-confidence)/2, (1+confidence)/2], dtype=c.FLOAT),
                            dim=0, interpolation="linear")
    pu._finite(point, draws, covariance, se, bounds)
    order = [[position, f"Comp{j+1}"] for position in positions for j in range(m)]
    labels = [f"query[{position}]:{axis}" for position, axis in order]
    undefined = torch.full_like(point, float("nan"))
    full_point = torch.full((rows, m), float("nan"), dtype=c.FLOAT)
    full_point[positions] = point.reshape(len(sample), m)
    full_query = torch.full((rows, len(names)), float("nan"), dtype=c.FLOAT)
    full_query[positions] = query
    output = TableSet({
        "point_scores": c.frame(full_point, columns=c.numbered("Comp", m), index=range(rows)),
        "estimates": c.frame(torch.stack((point, se, bounds[0], bounds[1], draws.mean(0)-point,
                                           undefined, undefined), 1),
                             columns=["estimate", "std_error", "ci_lower", "ci_upper", "bootstrap_bias", "p_value", "df"], index=labels),
        "covariance": c.frame(covariance, columns=labels, index=labels),
        "replicates": c.frame(draws, columns=labels, index=range(1, b+1)),
        "replicate_centering_origins": c.frame(origins, columns=names, index=range(1, b+1)),
        "replicate_centering_offsets": c.frame(offsets, columns=names, index=range(1, b+1)),
        "point_centering": c.frame(torch.stack((source[0], point_offset), 1), columns=["origin", "offset"], index=names),
        "parameter_order": table(order, columns=["query_position", "component"], index=labels),
        "query": c.frame(full_query, columns=names, index=range(rows)),
        "query_identities": table([[code, bool(flag)] for code, flag in zip(index_codes, keep.tolist(), strict=True)],
                                   columns=["typed_index_code", "complete"], index=range(rows)),
        **{f"fit__{name}": frame for name, frame in fit.items()},
    }, title=f"Fixed-query {fit.attrs['matrix']} PCA score uncertainty",
        procedure="pca_bootstrap_scores", state_schema="openecon.pca_bootstrap_scores.v1",
        variables=names, matrix=fit.attrs["matrix"], components=m, physical_query_rows=rows,
        n_query=len(sample), n_missing_query=dropped, missing_policy=missing,
        query_positions=positions, query_index_codes=index_codes, query_index_names=index_names,
        query_index_nlevels=keep.index.nlevels, query_column_names=column_names,
        parameter_order=order, parameter_labels=labels, parameter_dimension=q,
        replications=b, confidence=confidence, seed=fit.attrs["seed"], covariance_divisor=b-1,
        quantile_interpolation="linear", source_content_sha256=fit.attrs["source_content_sha256"],
        source_state_sha256=hashlib.sha256(summary_state(fit).encode()).hexdigest(),
        source_state_bytes=source_bytes, fit_attrs=fit.attrs, fit_title=fit.title,
        saved_fit_validation="complete deterministic refit of original kept sample, point and every seeded draw",
        query_contract="caller-fixed independently of training data; estimator functional only, no future-observation/noise interval",
        inference_target="ordered signed PCA score functional at each fixed query",
        inferential_assumptions=fit.attrs["inferential_assumptions"], uncertainty="first-order IID marginal percentile bootstrap",
        p_values_available=False, inference_df_available=False, familywise_intervals=False,
        reestimate_means_every_draw=True, restandardize_every_draw=fit.attrs["matrix"]=="correlation",
        stable_centering="(query-draw_origin)-two_pass_mean_of_shifted_draw; original rounded mean levels are not subtracted",
        precision="float64", device="cpu", resource_plan=plan, estimated_additional_work=work,
        estimated_complete_export_bytes=existing_bound+new_bytes,
        notes=["Missing query rows retain their physical row positions and typed identities but have no inferred target.",
               "The full original accepted fit is retained in fit__ tables and fit_attrs, independently of previews."])
    return saved_summary(output)
