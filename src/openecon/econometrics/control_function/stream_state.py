"""Compact, source-bound CF records with complete bounded semantic replay.

The record contains fixed-width factors and joint estimating-equation summaries,
never training rows. A digest detects corruption, not authenticity. Restoration
replays the retained source at saved parameters; it never optimizes the outcome.
Physical CSV/Parquet descriptors are location hints and are checked by content,
typed index/cluster identity and retained positions before they are accepted.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.data import DataError
from openecon.dataset import Dataset, scan
from openecon.econometrics.core import kernel_call, wald_test
from openecon.econometrics.registry import cluster_columns, validate_spec
from openecon.engines.inference import critical_value, two_sided_p_values
from openecon.models import ResultBundle
from openecon.resources import plan_workspace, workspace_budget_bytes

from .state import (KINDS, MAX_COLUMNS, MAX_JOINT_PARAMETERS, MAX_WORK,
                    _equal, _names, _real, _tensor, digest,
                    fail, parameter_order, work_limit)

SCHEMA = "openecon.control_function.stream.v2"
MAX_STATE_BYTES = 4 * 1024**2
MAX_RESULT_BYTES = 16 * 1024**2
MATRICES = ("bread", "meat", "joint_covariance", "first_stage_factor", "outcome_factor")
VECTORS = ("gamma", "beta", "score_sum", "score_abs_sum", "first_stage_target", "outcome_target")
_FLAGS = {"conditional_mean_only": True, "working_dispersion": 1.0,
          "generated_control_correction": "full stacked estimating equations",
          "structural_effects_identified": False, "weak_instrument_robust": False,
          "finite_cluster_exact": False, "stata_parity_validated": False}
_FIELDS = {*MATRICES, *VECTORS, *_FLAGS, "schema", "kind", "spec", "n_original", "nobs",
           "z_terms", "x_terms", "parameter_order", "source", "source_binding", "data_hash",
           "sample_hash", "sample_positions_hash", "input_columns", "cluster_count", "criterion",
           "optimizer", "work_estimate", "integrity_sha256"}


def is_stream_state(state):
    return isinstance(state, Mapping) and state.get("schema") == SCHEMA


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _array(value, shape, name):
    if value is None and name in {"outcome_factor", "outcome_target"}:
        return
    if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
        fail(f"Compact {name} has the wrong fixed-width dimensions.")
    rows = value if len(shape) == 2 else (value,)
    if len(shape) == 2 and any(not isinstance(row, (list, tuple)) or len(row) != shape[1] for row in rows):
        fail(f"Compact {name} must be a rectangular fixed-width matrix.")
    if any(not _real(v) for row in rows for v in row):
        fail(f"Compact {name} requires finite real values without booleans.")


def _metadata_size(value, *, operation="control-function reporting admission", max_bytes=None):
    """Visit finite JSON with O(depth) traversal state and early byte admission.

    Collection overhead is counted before visiting children. Iterator frames
    avoid scheduling a caller-owned wide list into another wide temporary list.
    The same scalar, key, depth and node envelopes as resident v1 are retained.
    """
    def children(item, depth):
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str) or len(key) > 256:
                    fail("Saved metadata keys must be bounded strings.")
                yield key, depth + 1
                yield child, depth + 1
        else:
            for child in item:
                yield child, depth + 1

    stack, size, nodes = [iter(((value, 0),))], 0, 0
    budget = workspace_budget_bytes()
    while stack:
        try:
            item, depth = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        nodes += 1
        if depth > 32 or nodes > 10_000_000:
            fail("Saved control-function metadata exceeds its nesting/node budget.")
        container = isinstance(item, (Mapping, list, tuple))
        if isinstance(item, Mapping):
            size += 256 + 96 * len(item)
        elif isinstance(item, (list, tuple)):
            size += 64 + 16 * len(item)
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
        if max_bytes is not None and size > max_bytes:
            fail("Compact CF metadata exceed their fixed-size reporting budget.", "resource_limit")
        # This incremental plan admits the traversal itself before adding a
        # child iterator; complete callers add their other buffers afterwards.
        if 4 * size > budget:
            plan_workspace(operation, {"complete finite JSON and model copies": 4 * size}, budget_bytes=budget)
        if container:
            stack.append(iter(children(item, depth)))
    plan_workspace(operation, {"complete finite JSON and model copies": 4 * size}, budget_bytes=budget)
    return size


def admit_stream_state(state, max_work=MAX_WORK):
    """Fixed-width admission before hashing, model copies or numerical allocation."""
    max_work = work_limit(max_work)
    if not is_stream_state(state) or set(state) != _FIELDS:
        fail("Supply complete openecon.control_function.stream.v2 state.")
    if not isinstance(state["kind"], str) or state["kind"] not in KINDS:
        fail("Compact CF outcome kind is unsupported.")
    z, x = _names(state["z_terms"]), _names(state["x_terms"])
    kz, kx, k = len(z), len(x), len(z) + len(x) + 1
    if not 1 <= kz <= MAX_COLUMNS or not 1 <= kx <= MAX_COLUMNS or k > MAX_JOINT_PARAMETERS:
        fail("Compact CF designs exceed the declared coefficient envelope.")
    n, original = state["nobs"], state["n_original"]
    if any(type(v) is not int for v in (n, original)) or not k < n <= original <= 2**53:
        fail("Compact CF counts must be exact positive integers with more rows than coefficients.")
    _names(state["input_columns"])
    if len(state["input_columns"]) > 2 * MAX_COLUMNS + 3:
        fail("Compact CF source projection is too wide.")
    gaussian = state["kind"] == "gaussian"
    dims = {"gamma": (kz,), "beta": (kx + 1,), "score_sum": (k,), "score_abs_sum": (k,),
            "bread": (k, k), "meat": (k, k), "joint_covariance": (k, k),
            "first_stage_factor": (kz + 1, kz), "first_stage_target": (kz + 1,),
            "outcome_factor": (kx + 1 + int(gaussian), kx + 1),
            "outcome_target": (kx + 2,)}
    for name, shape in dims.items():
        _array(state[name], shape, name)
    if state["outcome_factor"] is None or gaussian != (state["outcome_target"] is not None):
        fail("Every compact state retains outcome rank geometry; only Gaussian states retain an OLS target.")
    binding = state["source_binding"]
    if (not isinstance(binding, Mapping) or set(binding) != {"typed_index_hash", "typed_cluster_hash"}
            or not _hash(binding["typed_index_hash"])
            or (binding["typed_cluster_hash"] is not None and not _hash(binding["typed_cluster_hash"]))):
        fail("Compact CF typed source binding is malformed.")
    if any(not _hash(state[name]) for name in ("data_hash", "sample_hash", "sample_positions_hash", "integrity_sha256")):
        fail("Compact CF content and integrity identities require SHA256 digests.")
    source = state["source"]
    if source is not None and (not isinstance(source, Mapping) or set(source) != {"kind", "path"}
            or not isinstance(source["kind"], str) or source["kind"] not in {"csv", "parquet"} or not isinstance(source["path"], str)
            or not 1 <= len(source["path"]) <= 4096 or "://" in source["path"]
            or not Path(source["path"]).is_absolute()):
        fail("Compact CF location descriptors admit bounded local CSV/Parquet paths only.")
    if not _real(state["criterion"]) or any(state[key] != value or type(state[key]) is not type(value) for key, value in _FLAGS.items()):
        fail("Compact CF scope or working-criterion declaration differs.")
    clusters = state["cluster_count"]
    if clusters is not None and (type(clusters) is not int or not k < clusters <= n):
        fail("Compact CF whole-cluster covariance needs an exact count greater than joint width.")
    work = state["work_estimate"]
    if type(work) is not int or not 1 <= work <= MAX_WORK:
        fail("Compact CF work estimate is malformed.")
    # A saved estimate is evidence, not authority for current admission. The
    # replay engine admits actual source counts/work before its row objectives.
    from .kernels import certificate_work
    replay_work = 20 * n * (kz * kz + (kx + 1)**2 + k * k) + 20 * k**3
    replay_work += certificate_work(n, kx + 1, state["kind"])
    if replay_work > max_work:
        fail("Compact CF semantic replay exceeds max_work.", "resource_limit")
    size = _metadata_size(state, operation="compact CF state admission", max_bytes=MAX_STATE_BYTES)
    if size > MAX_STATE_BYTES:
        fail("Compact CF metadata exceed their fixed-size state budget.", "resource_limit")
    plan = plan_workspace("compact CF state admission", {"finite JSON and copies": 4 * size,
                          "joint factors and parameter inference": 256 * k * k})
    return n, kz, kx, replay_work, plan.record()


def source_descriptor(source):
    """Retain a bounded reopenable location, excluding owned temporary sources."""
    if (not isinstance(source, Dataset) or getattr(source, "_cleanup", None) is not None
            or getattr(source, "_kind", None) not in {"csv", "parquet"}):
        return None
    path = str(getattr(source, "_path", ""))
    return {"kind": source._kind, "path": path} if 1 <= len(path) <= 4096 else None


def bind_source(bundle, source):
    """Attach a transient source handle that is never serialized as training rows."""
    if not isinstance(bundle, ResultBundle) or not isinstance(source, Dataset):
        fail("CF source binding needs a ResultBundle and replayable Dataset.", "invalid_data")
    object.__setattr__(bundle, "_control_function_source", source)
    return bundle


def resolve_source(bundle, source=None):
    if source is not None:
        if not isinstance(source, Dataset):
            fail("Compact CF restoration needs the original replayable Dataset.", "invalid_data")
        return source
    handle = getattr(bundle, "_control_function_source", None)
    if isinstance(handle, Dataset):
        return handle
    state = bundle.extra.get("control_function_state")
    descriptor = state.get("source") if isinstance(state, Mapping) else None
    if descriptor is None:
        fail("This compact CF result needs cf_restore(result=..., data=original Dataset) before evaluation.", "prediction_source_required")
    try:
        source = scan(descriptor["path"])
    except Exception as exc:
        raise AnalysisError("prediction_source_required", "The retained local CF source is unavailable; restore with the original Dataset.") from exc
    if source._kind != descriptor["kind"]:
        fail("The retained local CF source format changed.", "source_changed")
    return source


def _admit_reporting(bundle):
    """Admit complete reporting before Pydantic copies or JSON serialization.

    These dictionaries borrow existing JSON containers. Only the already
    bounded coefficient records and specification facade are newly allocated;
    no full model_dump/deepcopy precedes resource admission.
    """
    if isinstance(bundle, ResultBundle):
        if (len(bundle.coefficients) > MAX_JOINT_PARAMETERS or len(bundle.predictions) > 400
                or len(bundle.covariance_matrix) > MAX_JOINT_PARAMETERS):
            fail("The saved result envelope exceeds declared coefficient/preview bounds.")
        record = {name: getattr(bundle, name) for name in ResultBundle.model_fields}
        record["spec"] = {name: getattr(bundle.spec, name) for name in type(bundle.spec).model_fields}
        record["coefficients"] = [{name: getattr(item, name) for name in type(item).model_fields}
                                  for item in bundle.coefficients]
    elif isinstance(bundle, Mapping):
        record = bundle
    else:
        fail("Supply a saved control-function ResultBundle or complete JSON mapping.", "invalid_result")
    extra = record.get("extra")
    state = extra.get("control_function_state") if isinstance(extra, Mapping) else None
    # Legacy v1 legitimately includes full rows; retain its established
    # workspace admission instead of applying the compact record-size cap.
    _metadata_size(record, operation="control-function complete reporting admission",
                   max_bytes=MAX_RESULT_BYTES if is_stream_state(state) else None)
    return record


def _reporting_hash(bundle):
    """Hash all persisted fields only after complete bounded admission."""
    record = _admit_reporting(bundle)
    try:
        payload = json.dumps(record, sort_keys=True, allow_nan=False, ensure_ascii=True, separators=(",", ":"))
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise AnalysisError("invalid_state", "Compact CF reporting must contain bounded finite JSON.") from exc
    return hashlib.sha256(payload.encode()).hexdigest()


def _physical_file_hash(source):
    # Directory, factory, frame and owned scratch sources always replay fully.
    if (source_descriptor(source) is None or len(getattr(source, "_files", ())) != 1
            or source._path.is_dir()):
        return None
    return source.file_hash()


def _compare_scores(actual, saved, absolute):
    """Cancellation-aware bound for float64 dot products and changed block trees.

    A score sum at an optimum is near zero; its relative magnitude is not a
    numerical error scale. 512 eps times the per-parameter absolute row-score
    sum covers dot-product/reduction roundoff in this <=36-parameter contract.
    Stationarity and observed-information checks remain independent below.
    """
    target = _tensor(saved, "score_sum")
    bound = 512 * torch.finfo(torch.float64).eps * absolute
    if actual.shape != target.shape or not bool(((actual - target).abs() <= bound).all()):
        fail("Saved score_sum differs from full-source floating-point score replay.")


def _refined_stationarity(gradient, information, absolute, refinement):
    """Verify the declared Newton refinement allowing bounded score reduction.

    For componentwise score perturbations |delta g| <= u, the information-norm
    perturbation is at most sqrt(u' |I^-1| u). The triangle inequality adds
    this allowance to the engine's declared Newton-norm threshold. Diagonal
    equilibration preserves that norm and avoids solving in raw column units.
    This bounds reduction roundoff, not arbitrary row-function approximation.
    """
    diagonal = information.diagonal()
    if not bool(torch.isfinite(diagonal).all()) or not bool((diagonal > 0).all()):
        fail("Compact CF refined stationarity needs positive observed information.")
    scale = diagonal.rsqrt()
    normalized = scale[:, None] * information * scale[None, :]
    normalized = .5 * normalized + .5 * normalized.T
    factor, status = torch.linalg.cholesky_ex(normalized)
    if int(status) != 0:
        fail("Compact CF refined stationarity needs positive definite observed information.")
    whitened = torch.linalg.solve_triangular(factor, (scale * gradient)[:, None], upper=False)
    newton_norm = float(torch.linalg.vector_norm(whitened))
    uncertainty = scale * (512 * torch.finfo(torch.float64).eps * absolute)
    roundoff_energy = float(uncertainty @ torch.cholesky_inverse(factor).abs() @ uncertainty)
    bound = math.sqrt(refinement) + math.sqrt(max(0., roundoff_energy))
    if not math.isfinite(newton_norm) or not math.isfinite(roundoff_energy) or newton_norm > bound:
        fail("Compact CF outcome parameters fail their declared strict stationarity refinement.")


def capture_stream_state(sample, solved, kind):
    """Capture fixed-width summaries and the engine's complete source receipt."""
    receipt = sample.provenance()
    source = getattr(sample, "cf_original_source", sample.source)
    state = {"schema": SCHEMA, "kind": kind, "spec": sample.spec.model_dump(mode="json"),
             "n_original": sample.original_count, "nobs": sample.nrows,
             "z_terms": list(solved["z_terms"]), "x_terms": list(solved["x_terms"]),
             "source": source_descriptor(source), "source_binding": dict(solved["source_binding"]),
             "data_hash": receipt["data_hash"], "sample_hash": receipt["sample_hash"],
             "sample_positions_hash": receipt["sample_positions_hash"], "input_columns": list(sample.columns),
             "cluster_count": solved["cluster_count"], "criterion": float(solved["criterion"]),
             "optimizer": dict(solved["optimizer"]), "work_estimate": int(solved["work_estimate"]), **_FLAGS}
    state["parameter_order"] = parameter_order(state["z_terms"], state["x_terms"])
    for name in (*MATRICES, *VECTORS):
        value = solved.get(name)
        state[name] = None if value is None else value.tolist()
    state["integrity_sha256"] = digest(state)
    admit_stream_state(state, sample.spec.options.get("max_work", MAX_WORK))
    return state


def _compare_factor(actual_factor, actual_target, saved_factor, saved_target, name):
    """QR signs and reduction trees may differ; normal geometry must agree."""
    factor = _tensor(saved_factor, name)
    _equal(actual_factor.T @ actual_factor, factor.T @ factor, name + " Gram", tolerance=2e-9)
    if saved_target is not None:
        target = _tensor(saved_target, name + " target")
        _equal(actual_factor.T @ actual_target, factor.T @ target, name + " moment", tolerance=2e-9)


def _reported(bundle, state, covariance):
    order = state["parameter_order"]
    params = torch.cat((_tensor(state["gamma"], "gamma"), _tensor(state["beta"], "beta")))
    _equal(covariance, bundle.covariance_matrix, "ResultBundle joint covariance", tolerance=2e-9)
    se = covariance.diagonal().sqrt()
    statistics = params / se
    p = kernel_call(two_sided_p_values, statistics, None)
    critical = kernel_call(critical_value, bundle.spec.alpha, None)
    for i, coefficient in enumerate(bundle.coefficients):
        if (coefficient.term, coefficient.equation) != (order[i]["term"], order[i]["equation"]):
            fail("Compact CF reported parameter/equation order differs.")
        for key, value in (("estimate", params[i]), ("std_error", se[i]), ("statistic", statistics[i]),
                           ("p_value", p[i]), ("ci_low", params[i] - critical * se[i]), ("ci_high", params[i] + critical * se[i])):
            actual = getattr(coefficient, key)
            unit = float(se[i]) if key in {"estimate", "ci_low", "ci_high"} else 0.0
            if not _real(actual) or abs(float(value) - actual) > 2e-9 * max(abs(float(value)), unit):
                fail("Compact CF reported " + key + " differs from source replay.")
        if coefficient.std_error <= 0:
            fail("Compact CF coefficients require strictly positive uncertainty.")


@torch.no_grad()
def validate_stream_state(bundle, *, source=None, batch_rows=None, max_work=MAX_WORK):
    """Replay full source equations at saved Gamma/Beta without outcome refitting."""
    with torch.device("cpu"):
        try:
            return _validate(bundle, source=source, batch_rows=batch_rows, max_work=max_work)
        except AnalysisError:
            raise
        except DataError as exc:
            raise AnalysisError("source_changed" if exc.code == "SOURCE_CHANGED" else "invalid_data", str(exc)) from exc
        except (KeyError, TypeError, ValueError, AttributeError, RuntimeError, OverflowError) as exc:
            raise AnalysisError("invalid_state", "Compact CF state is malformed or semantically inconsistent.") from exc


def _validate(bundle, *, source, batch_rows, max_work):
    if not isinstance(bundle, ResultBundle) or not isinstance(bundle.extra, Mapping):
        fail("Supply a compact saved CF ResultBundle.", "invalid_result")
    state = bundle.extra.get("control_function_state")
    n, kz, kx, _, _ = admit_stream_state(state, max_work)
    if digest(state) != state["integrity_sha256"] or bundle.provenance.get("control_function_state_sha256") != state["integrity_sha256"]:
        fail("Compact CF state/result integrity identities differ.")
    validate_spec(bundle.spec)
    from . import KINDS as ESTIMATORS
    spec = bundle.spec
    if (state["spec"] != spec.model_dump(mode="json") or spec.estimator not in ESTIMATORS
            or ESTIMATORS[spec.estimator][0] != state["kind"]):
        fail("Compact CF saved family/specification differs from the result.")
    if spec.weights is not None or spec.categorical or spec.panel is not None or spec.time is not None:
        fail("Compact CF supports numeric unweighted non-panel source data.")
    endogenous, instruments = spec.columns["endogenous"], spec.columns["instruments"]
    intercept = ["Intercept"] if spec.intercept else []
    if (state["z_terms"] != [*intercept, *spec.predictors, *instruments]
            or state["x_terms"] != [*intercept, *spec.predictors, endogenous]
            or state["parameter_order"] != parameter_order(state["z_terms"], state["x_terms"])
            or len(bundle.coefficients) != kz + kx + 1):
        fail("Compact CF source/design/parameter geometry differs.")
    if (any(type(v) is not int for v in (bundle.nobs, bundle.nobs_original, bundle.dropped_rows))
            or bundle.sample_positions != [] or (bundle.nobs, bundle.nobs_original, bundle.dropped_rows)
            != (n, state["n_original"], state["n_original"] - n)):
        fail("Compact CF must omit full positions and retain exact full-source counts.")
    if len(bundle.predictions) > 400 or len(bundle.covariance_matrix) > MAX_JOINT_PARAMETERS:
        fail("Compact CF reporting preview exceeds its bounded envelope.")
    _array(bundle.covariance_matrix, (kz + kx + 1, kz + kx + 1), "reported joint covariance")
    optimizer = state["optimizer"]
    if (not isinstance(optimizer, Mapping) or optimizer.get("converged") is not True
            or type(optimizer.get("max_iterations")) is not int
            or optimizer.get("max_iterations") != spec.options.get("max_iterations", 100)
            or optimizer.get("tolerance") != spec.options.get("tolerance", 1e-9)
            or optimizer.get("working_dispersion") != 1.0
            or not _real(optimizer.get("working_dispersion"))
            or optimizer.get("criterion") != "canonical_working_score_sum"
            or optimizer.get("optimization_criterion") != "negative_half_glm_deviance"
            or optimizer.get("dense_observation_matrix") is not False
            or bundle.provenance.get("optimizer") != optimizer):
        fail("Compact CF optimizer declarations differ from the saved specification.")
    iterations = optimizer.get("iterations")
    if (type(iterations) is not int or not 0 <= iterations <= optimizer["max_iterations"]
            or not isinstance(optimizer.get("method"), str) or not 1 <= len(optimizer["method"]) <= 256):
        fail("Compact CF optimizer history exceeds its declared iteration envelope.")
    if state["kind"] == "gaussian":
        if iterations != 0 or optimizer["method"] != "global_augmented_tsqr":
            fail("Compact Gaussian CF state requires the declared closed-form TSQR fit.")
    else:
        tolerance = float(spec.options.get("tolerance", 1e-9))
        refinement = max((32 * torch.finfo(torch.float64).eps)**2 * n * (kx + 1), tolerance**2 / n)
        if (optimizer.get("refined_scaled_gradient_tolerance") != refinement
                or optimizer.get("refined_step_tolerance") != min(tolerance, 1e-12)
                or optimizer.get("stationarity_refinement")
                != "global analytic Newton; dimension-aware float64 floor; within declared iteration budget"):
            fail("Compact CF optimizer refinement does not reproduce the declared float64 formula.")
        if (optimizer.get("concave") is not True or not _real(optimizer.get("scaled_gradient"))
                or not 0 <= optimizer["scaled_gradient"] <= refinement):
            fail("Compact CF optimizer convergence exceeds its declared strict stationarity refinement.")
    execution = bundle.provenance.get("execution")
    if (not isinstance(execution, Mapping)
            or set(execution) != {"requested_device", "device", "factor_devices", "fallbacks",
                                  "reporting_precision", "metal_precision"}
            or execution.get("requested_device") != "cpu" or execution.get("device") != "cpu"
            or execution.get("reporting_precision") != "float64"
            or execution.get("metal_precision") is not None
            or not isinstance(execution.get("fallbacks"), Mapping) or execution["fallbacks"] != {}):
        fail("Compact CF execution provenance must declare CPU float64 factors without accelerator fallback.")
    factors = execution.get("factor_devices")
    if (not isinstance(factors, Mapping) or set(factors) != {"cpu"}
            or type(factors["cpu"]) is not int or factors["cpu"] <= 0):
        fail("Compact CF execution provenance needs a positive integer CPU factor count.")
    # Factor counts record fit history, whose iteration and batch geometry can
    # differ from semantic replay. Their device/precision contract is fixed.
    from openecon.econometrics.streaming_control_function import prepare_stream, evaluate_stream, _work
    source = resolve_source(bundle, source)
    reporting_hash = _reporting_hash(bundle)
    byte_hash = _physical_file_hash(source)
    token = (reporting_hash, byte_hash, id(source), max_work, batch_rows)
    cache = getattr(bundle, "_control_function_semantic_cache", None)
    if (byte_hash is not None and isinstance(cache, Mapping) and cache.get("token") == token
            and cache.get("source") is source):
        return state
    sample = prepare_stream(spec, source, batch_rows=batch_rows)
    original_work = _work(sample, state["kind"], 0 if state["kind"] == "gaussian" else spec.options.get("max_iterations", 100),
                          spec.options.get("max_work", MAX_WORK))
    if original_work != state["work_estimate"]:
        fail("Compact CF declared fit work differs from full-source dimensions/options.")
    receipt = sample.provenance()
    for key in ("data_hash", "sample_hash", "sample_positions_hash"):
        if receipt[key] != state[key] or bundle.provenance.get(key) != state[key]:
            fail("Compact CF estimation source/sample content changed.", "source_changed")
    if (sample.original_count != state["n_original"] or sample.nrows != n
            or list(sample.columns) != state["input_columns"]
            or dict(sample.cf_source_binding) != dict(state["source_binding"])):
        fail("Compact CF typed index/cluster identity or retained sample changed.", "source_changed")
    gamma, beta = _tensor(state["gamma"], "gamma"), _tensor(state["beta"], "beta")
    evaluated = kernel_call(evaluate_stream, sample, gamma, beta, state["kind"], max_work=max_work, collect_preview=True)
    if dict(sample.cf_source_binding) != dict(state["source_binding"]):
        fail("Compact CF typed source changed during semantic replay.", "source_changed")
    for name in ("bread", "meat", "joint_covariance", "score_abs_sum"):
        _equal(evaluated[name], state[name], name, tolerance=2e-9)
    _compare_scores(evaluated["score_sum"], state["score_sum"], evaluated["score_abs_sum"])
    _equal(float(evaluated["criterion"]), state["criterion"], "criterion", tolerance=2e-9)
    _equal(evaluated["cluster_count"], state["cluster_count"], "cluster count")
    _compare_factor(evaluated["first_stage_factor"], evaluated["first_stage_target"],
                    state["first_stage_factor"], state["first_stage_target"], "first-stage OLS")
    from openecon.engines.linalg import least_squares
    restored_gamma = kernel_call(least_squares, evaluated["first_stage_factor"],
                                  evaluated["first_stage_target"], drop_collinear=False).beta
    _equal(restored_gamma, gamma, "first-stage OLS Gamma", tolerance=2e-9)
    _compare_factor(evaluated["outcome_factor"], evaluated["outcome_target"],
                    state["outcome_factor"], state["outcome_target"], "outcome design")
    if state["kind"] == "gaussian":
        restored_beta = kernel_call(least_squares, evaluated["outcome_factor"],
                                     evaluated["outcome_target"], drop_collinear=False).beta
        _equal(restored_beta, beta, "Gaussian outcome OLS Beta", tolerance=2e-9)
    tolerance = float(spec.options.get("tolerance", 1e-9))
    stationarity = evaluated["score_sum"].abs() / evaluated["score_abs_sum"].clamp_min(1.0)
    if float(stationarity.max()) > max(5e-7, tolerance * 64):
        fail("Compact CF parameters do not satisfy the full-source joint score equations.")
    gradient, information = evaluated["score_sum"][kz:], evaluated["bread"][kz:, kz:]
    if state["kind"] == "gaussian":
        remaining = torch.linalg.solve(information, gradient)
        scaled_gradient = float(gradient @ remaining)
        if not math.isfinite(scaled_gradient) or scaled_gradient > tolerance + 1e-12:
            fail("Compact CF outcome parameters fail the observed-information score tolerance.")
    else:
        _refined_stationarity(gradient, information, evaluated["score_abs_sum"][kz:], refinement)
        _refined_stationarity(_tensor(state["score_sum"], "score_sum")[kz:],
                               _tensor(state["bread"], "bread")[kz:, kz:],
                               _tensor(state["score_abs_sum"], "score_abs_sum")[kz:], refinement)
    covariance = evaluated["joint_covariance"]
    _reported(bundle, state, covariance)
    correction = "full stacked-equation HC0" if spec.cluster is None else "full stacked-equation CR0"
    cluster_names = cluster_columns(spec)
    expected = {"use_t": False, "distribution": "normal", "df_inference": None, "alpha": spec.alpha,
                "covariance": spec.covariance, "small_sample_correction": 1.0,
                "generated_control_uncertainty": True, "cluster_count": state["cluster_count"], "correction": correction,
                "n_parameters": kz + kx + 1, "intercept": spec.intercept,
                "confidence_level": 1 - spec.alpha, "cluster_column": cluster_names[0] if cluster_names else None,
                "cluster_df": None, "df_resid": None, "residual_definition": "observed minus fitted response"}
    if (any(bundle.inference.get(key) != value for key, value in expected.items())
            or bundle.inference.get("use_t") is not False
            or bundle.inference.get("generated_control_uncertainty") is not True
            or bundle.inference.get("intercept") is not spec.intercept
            or not _real(bundle.inference.get("small_sample_correction"))):
        fail("Compact CF full-joint asymptotic inference declarations differ.")
    params = torch.cat((gamma, beta))
    expected_test = wald_test(params, covariance, [len(params) - 1])
    reported_test = bundle.tests.get("control_coefficient_zero")
    if (set(bundle.tests) != {"control_coefficient_zero"} or not isinstance(reported_test, Mapping)
            or set(reported_test) != set(expected_test)):
        fail("Compact CF generated-control Wald test declaration differs.")
    for key, value in expected_test.items():
        _equal(value, reported_test[key], "generated-control Wald " + key, tolerance=2e-9)
    if (bundle.provenance.get("design_terms") != [v["term"] for v in state["parameter_order"]]
            or bundle.provenance.get("input_columns") != state["input_columns"]
            or bundle.provenance.get("sample_positions_omitted") is not True
            or bundle.provenance.get("sample_position_count") != n
            or bundle.provenance.get("precision") != "float64" or bundle.provenance.get("device") != "cpu"
            or bundle.provenance.get("estimator") != spec.estimator or bundle.provenance.get("family") != "control_function"):
        fail("Compact CF full-source/numerical provenance differs.")
    _equal(state["criterion"], bundle.metrics.get("working_criterion"), "reported criterion", tolerance=2e-9)
    if (bundle.metrics.get("first_stage_n_parameters") != kz
            or bundle.metrics.get("outcome_n_parameters") != kx + 1
            or not _real(bundle.metrics.get("first_stage_n_parameters"))
            or not _real(bundle.metrics.get("outcome_n_parameters"))
            or any(bundle.metrics.get(key) is not None for key in ("log_likelihood", "aic", "bic"))):
        fail("Compact CF reported working-score dimensions or likelihood scope differ.")
    preview = evaluated["preview"]
    if len(preview) != len(bundle.predictions):
        fail("Compact CF fitted preview length differs from complete source replay.")
    for actual, saved in zip(preview, bundle.predictions, strict=True):
        if set(actual) != set(saved) or actual["row"] != saved["row"]:
            fail("Compact CF fitted preview positions differ from complete source replay.")
        for key in ("observed", "fitted", "residual"):
            _equal(actual[key], saved[key], "fitted preview " + key, tolerance=2e-9)
    bind_source(bundle, source)
    if byte_hash is not None:
        if _physical_file_hash(source) != byte_hash:
            fail("The physical CF source bytes changed during complete semantic replay.", "source_changed")
        if _reporting_hash(bundle) != reporting_hash:
            fail("The saved CF reporting changed during complete semantic replay.")
        object.__setattr__(bundle, "_control_function_semantic_cache", {"token": token, "state": state, "source": source})
    return state


def cf_restore(*, result, data=None, batch_rows=None, max_work=MAX_WORK):
    """Restore and bind saved CF results to their verified original Dataset.

    JSON carries no frame/factory handle. Such results need explicit ``data``;
    retained physical CSV/Parquet locations are reopened and fully replayed.
    """
    from .postest import _restore
    max_work = work_limit(max_work)
    if isinstance(result, str):
        if len(result) > 64 * 1024**2:
            fail("Saved control-function JSON exceeds its bounded input size.", "resource_limit")
        plan_workspace("control-function JSON admission", {"JSON decoding and model restoration": 8 * len(result)})
        try:
            result = json.loads(result)
        except (ValueError, TypeError, RecursionError, OverflowError) as exc:
            raise AnalysisError("invalid_result", "Supply finite saved control-function JSON.") from exc
    # Admit caller-owned object/mapping containers before _restore can validate
    # a Pydantic copy, and before copying all reporting fields below.
    _admit_reporting(result)
    bundle = _restore(result, max_work)
    # Copies reporting JSON only; caller-owned source handles are not deep-copied.
    restored = ResultBundle.model_validate(bundle.model_dump(mode="json"))
    state = restored.extra["control_function_state"]
    if is_stream_state(state):
        source = data if data is not None else getattr(bundle, "_control_function_source", None)
        validate_stream_state(restored, source=source, batch_rows=batch_rows, max_work=max_work)
    else:
        if data is not None:
            fail("Legacy complete CF state restores its embedded source; external source binding is unsupported.", "invalid_data")
        if batch_rows is not None:
            fail("batch_rows applies to compact Dataset restoration only.", "invalid_option")
        from .state import validate_state
        validate_state(restored, max_work=max_work)
    return restored
