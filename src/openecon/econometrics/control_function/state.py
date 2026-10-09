"""Bounded, semantic replay of complete generated-control estimating equations.

The digest detects corruption; it does not authenticate a document. Restoration
reconstructs the original sample and the entire joint sandwich without refitting.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Integral, Real

import pandas as pd
import torch

from openecon.analysis import _frame_hasher, _numeric, _position_bytes
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import kernel_call
from openecon.econometrics.registry import cluster_columns, validate_spec
from openecon.econometrics.mi.common import _decode_index, _encode_index, _index_envelope, _label, _unlabel
from openecon.engines.inference import critical_value, two_sided_p_values
from openecon.models import ResultBundle
from openecon.resources import plan_workspace

SCHEMA = "openecon.control_function.v1"
KINDS = {"gaussian", "logit", "probit", "cloglog", "poisson", "gamma", "inverse_gaussian", "fractional_logit"}
MAX_ROWS = 5000
MAX_COLUMNS = 33
MAX_JOINT_PARAMETERS = 36
MAX_WORK = 10_000_000_000
DT = torch.float64
ARRAYS = ("z", "x", "d", "y", "gamma", "beta", "residual", "design", "fitted",
          "row_scores", "bread", "meat", "joint_covariance", "scalar_score", "scalar_derivative")


def fail(message, code="invalid_state"):
    raise AnalysisError(code, message)


def digest(state):
    """Canonical complete-state checksum, excluding only its checksum field."""
    try:
        raw = json.dumps({k: v for k, v in state.items() if k != "integrity_sha256"},
                         sort_keys=True, allow_nan=False, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise AnalysisError("invalid_state", "Control-function state must contain finite bounded JSON.") from exc
    return hashlib.sha256(raw.encode()).hexdigest()


def work_limit(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= MAX_WORK:
        fail("max_work must be an integer in [1,10000000000].", "invalid_option")
    return int(value)


def _shape(value, depth=0):
    if not isinstance(value, (list, tuple)):
        return ()
    if depth >= 2 or len(value) > MAX_ROWS:
        fail("Saved numerical arrays exceed their two-dimensional envelope.")
    if not value:
        return (0,)
    child = _shape(value[0], depth + 1)
    if any(_shape(v, depth + 1) != child for v in value[1:]):
        fail("Saved numerical arrays must be rectangular.")
    return (len(value), *child)


def _metadata_size(value):
    stack, size, nodes = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > 32 or nodes > 10_000_000:
            fail("Saved control-function metadata exceeds its nesting/node budget.")
        if isinstance(item, Mapping):
            if any(not isinstance(k, str) or len(k) > 256 for k in item):
                fail("Saved metadata keys must be bounded strings.")
            size += 256 + 96 * len(item)
            stack.extend((v, depth + 1) for pair in item.items() for v in pair)
        elif isinstance(item, (list, tuple)):
            size += 64 + 16 * len(item)
            stack.extend((v, depth + 1) for v in item)
        elif isinstance(item, str):
            if len(item) > 65_536:
                fail("Saved metadata strings exceed their bounded length.")
            size += 64 + 4 * len(item)
        elif isinstance(item, int) and not isinstance(item, bool):
            if item.bit_length() > 1024:
                fail("Saved integer values exceed their bounded scalar envelope.")
            size += 32
        elif item is None or isinstance(item, bool):
            size += 32
        elif isinstance(item, float) and math.isfinite(item):
            size += 32
        else:
            fail("Saved state contains an unsupported or nonfinite JSON scalar.")
    return size


def _names(value, size=None):
    if (not isinstance(value, (list, tuple)) or (size is not None and len(value) != size)
            or any(not isinstance(v, str) or not v.strip() or len(v) > 256 for v in value)
            or len(set(value)) != len(value)):
        fail("Saved source/design terms must be unique bounded nonempty strings.")
    return list(value)


def _real(value):
    try:
        return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _source_label(value):
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > 16 or nodes > 4096:
            fail("Typed source/index/cluster labels exceed bounded nesting.", "unsupported_index")
        if isinstance(item, tuple):
            if len(item) > 16:
                fail("Typed tuple labels have at most 16 components.", "unsupported_index")
            pending.extend((v, depth + 1) for v in item)
        elif isinstance(item, str) and len(item) > 65536:
            fail("Typed labels exceed bounded string length.", "unsupported_index")
        elif isinstance(item, int) and item.bit_length() > 1024:
            fail("Typed labels exceed bounded integer width.", "unsupported_index")
    return _label(value)


def _admit(state, max_work):
    """Resource/type envelope before JSON hashing, pandas copies or tensors."""
    from .stream_state import is_stream_state, admit_stream_state
    if is_stream_state(state):
        return admit_stream_state(state, max_work)
    if not isinstance(state, Mapping) or state.get("schema") != SCHEMA:
        fail("Supply complete openecon.control_function.v1 state.")
    kind = state.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        fail("Saved state requires a supported conditional outcome kind.")
    fields = {*ARRAYS, "schema", "kind", "spec", "n_original", "sample_positions", "source_index",
              "source_columns", "data_hash", "sample_hash", "z_terms", "x_terms", "parameter_order",
              "cluster_codes", "cluster_count", "criterion", "optimizer", "working_dispersion",
              "conditional_mean_only", "generated_control_correction", "structural_effects_identified",
              "weak_instrument_robust", "finite_cluster_exact", "stata_parity_validated"}
    if set(state) not in (fields, fields | {"integrity_sha256"}):
        fail("Saved state must retain exactly the declared complete schema.")
    shape = _shape(state.get("z"))
    xshape = _shape(state.get("x"))
    if (len(shape) != 2 or len(xshape) != 2 or shape[0] != xshape[0]
            or not 1 <= shape[0] <= MAX_ROWS or not 1 <= shape[1] <= MAX_COLUMNS
            or not 1 <= xshape[1] <= MAX_COLUMNS):
        fail("Saved state needs 1..5000 rows and bounded columns in each design.")
    n, kz, kx = shape[0], shape[1], xshape[1]
    k = kz + kx + 1
    if k > MAX_JOINT_PARAMETERS or n <= k:
        fail("Saved joint state needs at most 36 coefficients and more rows than coefficients.")
    dims = {"z": (n, kz), "x": (n, kx), "d": (n,), "y": (n,),
            "gamma": (kz,), "beta": (kx + 1,), "residual": (n,),
            "design": (n, kx + 1), "fitted": (n,), "row_scores": (n, k),
            "bread": (k, k), "meat": (k, k), "joint_covariance": (k, k),
            "scalar_score": (n,), "scalar_derivative": (n,)}
    if any(_shape(state.get(name)) != target for name, target in dims.items()):
        fail("Saved full joint array dimensions disagree.")
    for name, shape in dims.items():
        rows = state[name] if len(shape) == 2 else (state[name],)
        if any(not _real(v) for row in rows for v in row):
            fail(f"Saved {name} must contain finite real float64 values without booleans.")
    if not _real(state.get("criterion")):
        fail("Saved working criterion must be finite real.")
    original = state.get("n_original")
    if type(original) is not int or not n <= original <= MAX_ROWS:
        fail("Saved original row count exceeds its supported envelope.")
    _names(state.get("z_terms"), kz)
    _names(state.get("x_terms"), kx)
    columns = state.get("source_columns")
    if not isinstance(columns, (list, tuple)) or not 1 <= len(columns) <= 2 * MAX_COLUMNS + 3:
        fail("Saved source column envelope is invalid.")
    for column in columns:
        if (not isinstance(column, Mapping) or set(column) != {"name", "dtype", "values", "categorical_index"}
                or not isinstance(column.get("dtype"), str) or len(column["dtype"]) > 256
                or not isinstance(column.get("values"), (list, tuple)) or len(column["values"]) != original):
            fail("Saved source values/dtypes must preserve every original row.")
    _names([v.get("name") for v in columns])
    positions = state.get("sample_positions")
    if (not isinstance(positions, (list, tuple)) or len(positions) != n
            or any(type(p) is not int or not 0 <= p < original for p in positions)
            or list(positions) != sorted(set(positions))):
        fail("Saved physical row positions must be unique, ordered and within the original sample.")
    codes = state.get("cluster_codes")
    if codes is not None and (not isinstance(codes, (list, tuple)) or len(codes) != n
                             or any(type(v) is not int or not 0 <= v < n for v in codes)):
        fail("Saved cluster codes must be complete nonnegative integer row codes.")
    max_work = work_limit(max_work)
    from .kernels import certificate_work
    work = 32 * (n * k * k + k**3 + n * (kz + kx + 1)**2)
    work += certificate_work(n, kx + 1, kind)
    if work > max_work:
        fail(f"Saved joint replay needs {work:,} work units, exceeding max_work={max_work:,}.", "resource_limit")
    metadata = _metadata_size(state)
    plan = plan_workspace("control-function complete semantic replay", {
        "complete state and bounded JSON serialization": 4 * metadata,
        "source reconstruction, designs, joint scores and algebra": 256 * (n * k + k * k + original * len(columns)),
    })
    return n, kz, kx, work, plan.record()


def _tensor(value, name):
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        else:
            try:
                valid = _real(item)
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                fail(f"Saved {name} must contain finite real numeric values without booleans.")
    try:
        tensor = torch.tensor(value, dtype=DT, device="cpu")
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise AnalysisError("invalid_state", f"Cannot reconstruct saved {name} as float64.") from exc
    if not bool(torch.isfinite(tensor).all()):
        fail(f"Saved {name} exceeds finite float64 range.")
    return tensor


def _source(state):
    records = {}
    for column in state["source_columns"]:
        name, dtype = column["name"], column["dtype"]
        if dtype == "category":
            encoded = column["categorical_index"]
            _index_envelope(encoded, state["n_original"])
            values = _decode_index(encoded)
            if len(values) != state["n_original"] or [_label(v) for v in values] != list(column["values"]):
                fail("Saved category values and source category geometry disagree.")
            records[name] = pd.Series(values.array)
        else:
            if column["categorical_index"] is not None:
                fail("A noncategorical source must not carry category geometry.")
            values = [_unlabel(v) for v in column["values"]]
            records[name] = pd.Series(values, dtype=dtype)
    return pd.DataFrame(records)


def _design(source, terms, intercept):
    names = terms[1:] if intercept else terms
    if intercept and terms[0] not in {"Intercept", "_cons"}:
        fail("The saved intercept term is not the declared constant.")
    if any(name not in source for name in names):
        fail("Saved numeric design terms do not identify source columns.")
    parts = [torch.ones((len(source), 1), dtype=DT, device="cpu")] if intercept else []
    parts.extend(_numeric(source[name], name)[:, None] for name in names)
    return torch.cat(parts, 1)


def _equal(actual, expected, name, *, tolerance=2e-11):
    if isinstance(actual, torch.Tensor):
        target = _tensor(expected, name) if not isinstance(expected, torch.Tensor) else expected
        if actual.shape != target.shape:
            fail(f"Saved {name} disagrees with the complete estimating-equation replay.")
        if "covariance" in name:
            scale = actual.diagonal().sqrt()
            bound = tolerance * scale[:, None] * scale[None, :]
        else:
            bound = tolerance * actual.abs().amax(dim=0)
        if not bool(((actual - target).abs() <= bound).all()):
            fail(f"Saved {name} disagrees with the complete estimating-equation replay.")
    elif isinstance(actual, Real) and not isinstance(actual, bool):
        if not isinstance(expected, Real) or isinstance(expected, bool) or not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=0):
            fail(f"Saved {name} disagrees with semantic replay.")
    elif actual != expected:
        fail(f"Saved {name} disagrees with semantic replay.")


def parameter_order(z_terms, x_terms):
    return ([{"term": "first_stage:" + name, "equation": "first_stage"} for name in z_terms]
            + [{"term": "outcome:" + name, "equation": "outcome"} for name in x_terms]
            + [{"term": "outcome:ControlResidual", "equation": "outcome"}])


def typed_cluster_codes(series):
    """First-appearance CPU codes with exact typed scalar/tuple group identity."""
    if not isinstance(series, pd.Series) or not 1 <= len(series) <= MAX_ROWS:
        fail("Cluster labels must be a complete resident bounded Series.", "invalid_clusters")
    plan_workspace("control-function typed cluster labels", {
        "typed labels, canonical identity and dense codes": max(512 * len(series), 8 * int(series.memory_usage(index=False, deep=True))),
    })
    identities, codes = {}, []
    for label in series:
        encoded = _source_label(label)
        # Missing scalar labels and missing tuple components are not cluster IDs.
        pending = [encoded]
        while pending:
            item = pending.pop()
            if item["type"] in {"nat", "missing", "nan"} or (item["type"] == "scalar" and item["value"] is None):
                fail("Cluster identifiers must be complete.", "invalid_clusters")
            if item["type"] == "tuple":
                pending.extend(item["value"])
        key = json.dumps(encoded, sort_keys=True, allow_nan=False, separators=(",", ":"))
        if key not in identities:
            identities[key] = len(identities)
        codes.append(identities[key])
    return torch.tensor(codes, dtype=torch.int64, device="cpu"), len(identities)


def capture_state(frame, zdesign, xdesign, d, y, fit, kind, *, clusters=None, original_index=None):
    """Capture all source rows and the complete joint Gamma/Beta sandwich."""
    n, original = frame.n, len(frame.original)
    if not 1 <= n <= original <= MAX_ROWS or max(len(zdesign.terms), len(xdesign.terms)) > MAX_COLUMNS:
        fail("Control-function saved-state support is limited to 5000 resident rows and bounded designs.", "resource_limit")
    index = pd.RangeIndex(original) if original_index is None else original_index
    if not isinstance(index, pd.Index) or len(index) != original:
        fail("The source index must preserve all original physical rows.")
    plan_workspace("control-function state capture", {
        "typed original source and complete serialized joint arrays": max(
            1536 * (original * len(frame.original.columns) + n * (len(zdesign.terms) + len(xdesign.terms) + 2)),
            8 * (int(frame.original.memory_usage(index=False, deep=True).sum()) + int(index.memory_usage(deep=True)))),
        "joint factors and copied inference": 512 * (len(zdesign.terms) + len(xdesign.terms) + 1)**2,
    })
    for label in index:
        _source_label(label)
    source_columns = [{"name": name, "dtype": str(frame.original[name].dtype),
                       "values": [_source_label(v) for v in frame.original[name]],
                       "categorical_index": _encode_index(pd.CategoricalIndex(frame.original[name]))
                       if isinstance(frame.original[name].dtype, pd.CategoricalDtype) else None}
                      for name in frame.original]
    input_hash = _frame_hasher(frame.original)
    sample_hash = input_hash.copy() if frame.sample is frame.original else _frame_hasher(frame.sample)
    sample_hash.update(_position_bytes(frame.positions))
    state = {"schema": SCHEMA, "kind": kind, "spec": frame.spec.model_dump(mode="json"),
             "n_original": original, "sample_positions": frame.positions,
             "source_index": _encode_index(index), "source_columns": source_columns,
             "data_hash": input_hash.hexdigest(), "sample_hash": sample_hash.hexdigest(),
             "z_terms": list(zdesign.terms), "x_terms": list(xdesign.terms),
             "parameter_order": parameter_order(zdesign.terms, xdesign.terms),
             "cluster_codes": None if clusters is None else clusters.tolist(),
             "cluster_count": fit["cluster_count"], "criterion": float(fit["criterion"]),
             "optimizer": fit["optimizer"], "working_dispersion": 1.0,
             "conditional_mean_only": True, "generated_control_correction": "full stacked estimating equations",
             "structural_effects_identified": False, "weak_instrument_robust": False,
             "finite_cluster_exact": False, "stata_parity_validated": False}
    values = {"z": zdesign.x, "x": xdesign.x, "d": d, "y": y, **fit}
    for name in ARRAYS:
        state[name] = values[name].tolist()
    _admit(state, frame.spec.options.get("max_work", MAX_WORK))
    state["integrity_sha256"] = digest(state)
    return state


@torch.no_grad()
def validate_state(bundle, *, max_work=MAX_WORK):
    """Strong state, source, full covariance and coefficient inference validation."""
    from .stream_state import is_stream_state, validate_stream_state
    if isinstance(bundle, ResultBundle) and is_stream_state(bundle.extra.get("control_function_state")):
        return validate_stream_state(bundle, max_work=max_work)
    with torch.device("cpu"):
        return _validate_state(bundle, max_work=max_work)


def _validate_state(bundle, *, max_work):
    if not isinstance(bundle, ResultBundle) or not isinstance(bundle.extra, Mapping):
        fail("Supply a saved control-function ResultBundle.", "invalid_result")
    state = bundle.extra.get("control_function_state")
    n, _, _, _, _ = _admit(state, max_work)
    try:
        if digest(state) != state.get("integrity_sha256"):
            fail("Control-function state checksum differs.")
        if bundle.provenance.get("control_function_state_sha256") != state["integrity_sha256"]:
            fail("ResultBundle and control-function state integrity identities differ.")
        validate_spec(bundle.spec)
        from . import KINDS as ESTIMATORS
        if (state["spec"] != bundle.spec.model_dump(mode="json") or not isinstance(state["kind"], str)
                or state["kind"] not in KINDS or bundle.spec.estimator not in ESTIMATORS):
            fail("Saved conditional model kind/spec differs from the result.")
        kind = state["kind"]
        if ESTIMATORS[bundle.spec.estimator][0] != kind:
            fail("Saved conditional mean family differs from the declared specification.")
        if any(state.get(name) is not False for name in ("structural_effects_identified", "weak_instrument_robust", "finite_cluster_exact", "stata_parity_validated")) or state.get("conditional_mean_only") is not True or state.get("working_dispersion") != 1.0 or state.get("generated_control_correction") != "full stacked estimating equations":
            fail("Saved conditional model scope/inference declaration is unsupported.")
        order = parameter_order(state["z_terms"], state["x_terms"])
        if state["parameter_order"] != order or len(bundle.coefficients) != len(order):
            fail("Saved joint Gamma/Beta parameter order differs.")
        if bundle.sample_positions != list(state["sample_positions"]) or (bundle.nobs, bundle.nobs_original, bundle.dropped_rows) != (n, state["n_original"], state["n_original"] - n):
            fail("Saved physical sample differs from ResultBundle.")
        if bundle.spec.weights is not None or bundle.spec.categorical or bundle.spec.panel is not None or bundle.spec.time is not None:
            fail("This saved conditional model supports numeric unweighted resident data.")
        source = _source(state)
        keep = ~source.isna().any(axis=1)
        positions = [int(i) for i in keep.to_numpy().nonzero()[0]]
        if positions != list(state["sample_positions"]) or (bundle.spec.missing == "raise" and not bool(keep.all())):
            fail("Saved missing exclusions do not reproduce the declared common sample.")
        original_hash = _frame_hasher(source)
        selected = source if n == len(source) else source.iloc[positions].reset_index(drop=True)
        selected_hash = original_hash.copy() if selected is source else _frame_hasher(selected)
        selected_hash.update(_position_bytes(positions))
        if original_hash.hexdigest() != state["data_hash"] or selected_hash.hexdigest() != state["sample_hash"] or bundle.provenance.get("data_hash") != state["data_hash"] or bundle.provenance.get("sample_hash") != state["sample_hash"] or bundle.provenance.get("input_columns") != list(source.columns):
            fail("Saved original input/sample content hashes differ.")
        _index_envelope(state["source_index"], state["n_original"])
        index = _decode_index(state["source_index"])
        if len(index) != state["n_original"] or _encode_index(index) != state["source_index"]:
            fail("Saved source row index cannot be restored exactly.")
        spec = bundle.spec
        endogenous = spec.columns.get("endogenous")
        if isinstance(endogenous, list) and len(endogenous) == 1:
            endogenous = endogenous[0]
        instruments = spec.columns.get("instruments", [])
        instruments = [instruments] if isinstance(instruments, str) else instruments
        intercept = int(spec.intercept)
        if (not isinstance(endogenous, str) or not 1 <= len(instruments) <= 16 or len(spec.predictors) > 16
                or len(set([spec.outcome, endogenous, *spec.predictors, *instruments])) != 2 + len(spec.predictors) + len(instruments)
                or state["x_terms"][intercept:] != [*spec.predictors, endogenous]
                or state["z_terms"][intercept:] != [*spec.predictors, *instruments]):
            fail("Saved first-stage/outcome source-column geometry differs from the specification.")
        z, x = (_design(selected, state[name + "_terms"], spec.intercept) for name in ("z", "x"))
        d, y = _numeric(selected[endogenous], endogenous), _numeric(selected[spec.outcome], spec.outcome)
        if bool(((d == 0) | (d == 1)).all()):
            fail("Binary endogenous first stages are outside saved continuous-first-stage support.")
        for name, value in (("z", z), ("x", x), ("d", d), ("y", y)):
            _equal(value, state[name], name, tolerance=0)
        codes = state["cluster_codes"]
        if spec.cluster is None:
            if codes is not None or state["cluster_count"] is not None:
                fail("Independent-row state must not introduce cluster geometry.")
        else:
            names = cluster_columns(spec)
            if len(names) != 1 or names[0] not in selected:
                fail("Only one complete cluster column is supported.")
            original_codes, count = typed_cluster_codes(selected[names[0]])
            if original_codes.tolist() != codes or count != state["cluster_count"]:
                fail("Saved cluster geometry differs from the source labels.")
        gamma, beta = _tensor(state["gamma"], "gamma"), _tensor(state["beta"], "beta")
        residual = d - z @ gamma
        scale = max(1.0, float(z.abs().max()) * max(1.0, float(residual.abs().max())) * n)
        tolerance = max(1e-10, float(spec.options.get("tolerance", 1e-9)))
        if float((z.T @ residual).abs().max()) > torch.finfo(DT).eps * scale * 512:
            fail("Saved Gamma does not solve the full-rank first-stage OLS equations.")
        from .kernels import evaluate_joint
        evaluated = kernel_call(evaluate_joint, z, x, d, y, kind, gamma, beta,
                                cluster_codes=None if codes is None else torch.tensor(codes, dtype=torch.int64, device="cpu"),
                                max_work=max_work)
        # The already-admitted full-rank OLS geometry determines Gamma uniquely;
        # this is a semantic linear solve, never an outcome optimization/refit.
        from openecon.engines.linalg import least_squares
        restored_gamma = kernel_call(least_squares, z, d, drop_collinear=False).beta
        _equal(restored_gamma, gamma, "first-stage OLS Gamma")
        if kind == "gaussian":
            restored_beta = kernel_call(least_squares, evaluated["design"], y, drop_collinear=False).beta
            _equal(restored_beta, beta, "Gaussian outcome OLS Beta")
        for name in ARRAYS[6:]:
            _equal(evaluated[name], state[name], name)
        _equal(float(evaluated["criterion"]), state["criterion"], "criterion")
        _equal(evaluated["cluster_count"], state["cluster_count"], "cluster_count")
        scores = evaluated["row_scores"].sum(0)
        stationarity = scores.abs() / evaluated["row_scores"].abs().sum(0).clamp_min(1.0)
        if float(stationarity.max()) > max(5e-7, tolerance * 64):
            fail("Saved Gamma/Beta does not satisfy joint score stationarity.")
        gradient = evaluated["row_scores"][:, len(gamma):].sum(0)
        observed = evaluated["bread"][len(gamma):, len(gamma):]
        remaining_step = torch.linalg.solve(observed, gradient)
        scaled_gradient = float(gradient @ remaining_step)
        if not math.isfinite(scaled_gradient) or scaled_gradient > tolerance + 1e-12:
            fail("Saved outcome parameters fail the declared observed-information Newton score tolerance.")
        optimizer = state["optimizer"]
        if not isinstance(optimizer, Mapping) or optimizer.get("converged") is not True:
            fail("Saved conditional mean optimizer is not converged.")
        if (optimizer.get("max_iterations") != spec.options.get("max_iterations", 100)
                or optimizer.get("tolerance") != spec.options.get("tolerance", 1e-9)
                or optimizer.get("working_dispersion") != 1.0
                or optimizer.get("criterion") != "canonical_working_score_sum"
                or optimizer.get("optimization_criterion") != "negative_half_glm_deviance"
                or bundle.provenance.get("optimizer") != optimizer):
            fail("Saved optimizer provenance/options do not reproduce the declared fit.")
        params, covariance = torch.cat([gamma, beta]), evaluated["joint_covariance"]
        _equal(covariance, bundle.covariance_matrix, "ResultBundle joint covariance")
        se = covariance.diagonal().sqrt()
        statistics = params / se
        p = kernel_call(two_sided_p_values, statistics, None)
        critical = kernel_call(critical_value, spec.alpha, None)
        low, high = params - critical * se, params + critical * se
        for i, coefficient in enumerate(bundle.coefficients):
            if (coefficient.term, coefficient.equation) != (order[i]["term"], order[i]["equation"]):
                fail("Reported joint coefficient term/equation order differs.")
            if coefficient.std_error <= 0:
                fail("Every reported joint coefficient needs strictly positive uncertainty.")
            for key, value in (("estimate", params[i]), ("std_error", se[i]), ("statistic", statistics[i]), ("p_value", p[i]), ("ci_low", low[i]), ("ci_high", high[i])):
                reported = getattr(coefficient, key)
                unit = float(se[i]) if key in {"estimate", "ci_low", "ci_high"} else 0.0
                if isinstance(reported, bool) or not math.isfinite(reported) or abs(float(value) - reported) > 2e-11 * max(abs(float(value)), unit):
                    fail("Reported joint coefficient " + key + " differs from semantic replay.")
        if (bundle.inference.get("use_t") is not False or bundle.inference.get("distribution") != "normal"
                or bundle.inference.get("df_inference") is not None
                or bundle.inference.get("alpha") != spec.alpha):
            fail("Saved coefficient inference must be declared asymptotic normal.")
        if (bundle.inference.get("covariance") != spec.covariance
                or bundle.inference.get("small_sample_correction") != 1.0
                or bundle.inference.get("generated_control_uncertainty") is not True
                or bundle.inference.get("cluster_count") != state["cluster_count"]
                or bundle.inference.get("correction") != ("full stacked-equation HC0" if codes is None else "full stacked-equation CR0")):
            fail("Saved full-stack HC0/CR0 inference declaration differs.")
        if bundle.provenance.get("design_terms") != [v["term"] for v in order]:
            fail("Saved reported-design provenance differs from the complete joint vector.")
        _equal(float(evaluated["criterion"]), bundle.metrics.get("working_criterion"), "reported working criterion")
        if (bundle.metrics.get("first_stage_n_parameters") != len(gamma)
                or bundle.metrics.get("outcome_n_parameters") != len(beta)
                or any(bundle.metrics.get(name) is not None for name in ("log_likelihood", "aic", "bic"))
                or bundle.provenance.get("device") != "cpu" or bundle.provenance.get("precision") != "float64"
                or bundle.provenance.get("estimator") != spec.estimator or bundle.provenance.get("family") != "control_function"):
            fail("Saved model dimensions, working criterion or numerical provenance differ.")
        preview = torch.linspace(0, n - 1, steps=min(n, 400), dtype=DT, device="cpu").to(torch.int64).tolist()
        if [v.get("row") for v in bundle.predictions] != [positions[i] for i in preview]:
            fail("Saved fitted preview must preserve every declared preview position.")
        for prediction in bundle.predictions:
            row = prediction.get("row")
            if type(row) is not int or row not in positions:
                fail("Saved fitted preview row is outside the common estimation sample.")
            j = positions.index(row)
            for name, value in (("observed", y[j]), ("fitted", evaluated["fitted"][j]), ("residual", y[j] - evaluated["fitted"][j])):
                _equal(float(value), prediction.get(name), "fitted preview " + name)
    except AnalysisError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, RuntimeError, OverflowError) as exc:
        raise AnalysisError("invalid_state", "Saved control-function state is malformed or semantically inconsistent.") from exc
    return state
