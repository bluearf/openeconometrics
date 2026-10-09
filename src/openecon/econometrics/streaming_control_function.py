"""Global, bounded-replay control-function estimating equations.

Only projected blocks, small QR factors and complete stacked score aggregates
survive a pass.  Normalization is numerical conditioning: the final sensitivity,
meat and covariance are evaluated in the original Gamma/Beta coordinates.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import wraps
import hashlib
import json
import math
from numbers import Integral, Real

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines.contracts import KernelError
from openecon.engines import execution as _execution
from openecon.engines.execution import qr_factor
from openecon.engines.linalg import least_squares
from openecon.engines.optimize import maximize_newton
from openecon.engines.separation import certify_separation
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_ols import _CompensatedSum, _ScaledMoments, _TSQRTree, _normalize
from openecon.models import ModelSpec

from . import registry
from .control_function import KINDS
from .control_function.kernels import (
    CERTIFICATE_MAX_PASSES, DEFAULT_MAX_WORK, MAX_CONDITION,
    MAX_EXOG, MAX_INSTRUMENTS, MAX_JOINT_PARAMETERS,
    _criterion, _family_link, _objective, _positive_matrix, _state, _tensor,
    certificate_work,
)
from .control_function.state import _source_label
from .replay_sample import ReplaySample

DT = torch.float64
EPS = torch.finfo(DT).eps
KEY_BYTES = 1024 * 1024


@contextmanager
def _cpu_execution():
    """Override ambient acceleration; share only an already explicit CPU trace."""
    current = _execution._TRACE.get()
    device = "auto" if current is not None and current.requested == "cpu" else "cpu"
    with _execution.execution_scope(device) as trace:
        yield trace


def _error(code, message):
    raise AnalysisError(code, message)


def _label_bytes(value, *, cluster=False):
    # Exact byte spelling of the existing typed JSON contract for common
    # primitive labels; tuple/date/string labels retain its general encoder.
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if type(value) is bool:
        return b'{"type":"scalar","value":true}' if value else b'{"type":"scalar","value":false}'
    if type(value) is int and value.bit_length() <= 1024:
        return b'{"type":"scalar","value":' + str(value).encode() + b'}'
    encoded = _source_label(value)
    if cluster:
        pending = [encoded]
        while pending:
            item = pending.pop()
            if item["type"] in {"nat", "missing", "nan"} or (item["type"] == "scalar" and item["value"] is None):
                _error("invalid_clusters", "Cluster identifiers must be complete, including tuple components.")
            if item["type"] == "tuple":
                pending.extend(item["value"])
    return json.dumps(encoded, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _hash_label(digest, value):
    encoded = _label_bytes(value)
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _typed_mode(values):
    dtype = values.dtype
    # Extension arrays can change their to_numpy representation in a block
    # containing NA. Keep their typed JSON mode fixed across missing/nonmissing
    # blocks; primitive NumPy index/column dtypes use the bulk byte route.
    if (getattr(dtype, "kind", None) in {"i", "u", "b"}
            and type(dtype).__module__.startswith("numpy")):
        return {"i": "signed-little-i8", "u": "unsigned-little-u8", "b": "boolean-u1"}[dtype.kind]
    return "length-prefixed-typed-json"


def _hash_values(digest, values, mode):
    if mode != "length-prefixed-typed-json":
        dtype = {"signed-little-i8": "<i8", "unsigned-little-u8": "<u8", "boolean-u1": "u1"}[mode]
        digest.update(values.to_numpy().astype(dtype, copy=False).tobytes())
    else:
        for value in values:
            _hash_label(digest, value)


def _bound_source(spec, source, reader_rows):
    """Bind typed indices/cluster labels in addition to the projected row hash.

    pandas row hashing intentionally omits the index and can identify True
    with 1.  CF's existing saved-model cluster contract distinguishes them.
    The receipt covers original rows before the common missing filter.
    """
    columns = registry.spec_columns(spec)
    cluster = registry.cluster_columns(spec)
    binding = {}

    def factory():
        index_hash, cluster_hash = hashlib.sha256(), hashlib.sha256()
        # Resolve the admitted producer partition before it projects/parses
        # any rows; splitting an oversized producer block cannot undo its
        # allocation. ReplaySample keeps this reader reservation across passes.
        iterator = source.iter_batches(columns, batch_rows=reader_rows())
        index_geometry = cluster_mode = None
        try:
            for frame in iterator:
                index_mode = _typed_mode(frame.index)
                geometry = {"class": type(frame.index).__name__, "names": [_source_label(v) for v in frame.index.names],
                            "encoding": index_mode, "version": "openecon.cf.typed_source.v1"}
                if index_geometry is None:
                    index_geometry = geometry
                    index_hash.update(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode())
                elif geometry != index_geometry:
                    _error("source_changed", "Typed source index geometry changed within a replay pass.")
                _hash_values(index_hash, frame.index, index_mode)
                if cluster:
                    values = frame[cluster[0]]
                    mode = _typed_mode(values)
                    if cluster_mode is None:
                        cluster_mode = mode
                        cluster_hash.update(("openecon.cf.typed_source.v1:"+mode).encode())
                    elif mode != cluster_mode:
                        _error("source_changed", "Typed cluster representation changed within a replay pass.")
                    _hash_values(cluster_hash, values, mode)
                yield frame
            record = {"typed_index_hash": index_hash.hexdigest(),
                      "typed_cluster_hash": cluster_hash.hexdigest() if cluster else None}
            if binding and binding != record:
                _error("source_changed", "Typed source indices or cluster labels changed between replay passes.")
            binding.update(record)
        finally:
            iterator.close()

    wrapped = Dataset.from_batches(factory, columns, row_count=source.row_count, metadata=source.metadata)
    wrapped._parents = (source,)
    wrapped._provenance = source.provenance
    return wrapped, binding


def _options(spec):
    iterations = spec.options.get("max_iterations", 100)
    tolerance = spec.options.get("tolerance", 1e-9)
    max_work = spec.options.get("max_work", DEFAULT_MAX_WORK)
    if isinstance(iterations, bool) or not isinstance(iterations, Integral) or not 1 <= iterations <= 200:
        _error("invalid_options", "max_iterations must be an integer from 1 to 200.")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not math.isfinite(tolerance) or not 0 < tolerance < 1:
        _error("invalid_options", "tolerance must be finite and in (0, 1).")
    if isinstance(max_work, bool) or not isinstance(max_work, Integral) or not 1 <= max_work <= DEFAULT_MAX_WORK:
        _error("invalid_options", "max_work must be an integer from 1 to 10000000000.")
    return int(iterations), float(tolerance), int(max_work)


def _work(sample, kind, iterations, max_work):
    kz, kq = len(sample.designs["z"].terms), len(sample.designs["x"].terms) + 1
    joint, n = kz + kq, sample.nrows
    work = 20 * n * (kz * kz + kq * kq + joint * joint) + 20 * joint**3
    work += iterations * (128 * n * kq * kq + 256 * kq**3)
    work += certificate_work(n, kq, kind)
    if work > max_work:
        _error("work_limit", f"Conservative control-function work estimate {work} exceeds max_work={max_work}.")
    return work


def prepare_stream(spec: ModelSpec, source: Dataset, *, batch_rows=None) -> ReplaySample:
    """Prepare a globally bound common sample; do not estimate parameters."""
    if not isinstance(spec, ModelSpec) or spec.estimator not in KINDS or not isinstance(source, Dataset):
        _error("invalid_dataset", "Streamed control functions require a CF ModelSpec and replayable Dataset.")
    registry.validate_spec(spec)
    _options(spec)
    if spec.options.get("device", "cpu") != "cpu":
        _error("unsupported_device", "Control functions support explicit CPU float64 only.")
    exog = list(spec.predictors)
    instruments = registry.role_columns(spec, "instruments")
    endogenous = registry.role_columns(spec, "endogenous")
    if len(endogenous) != 1 or not 1 <= len(instruments) <= MAX_INSTRUMENTS or len(exog) > MAX_EXOG:
        _error("dimension_limit", "Use one continuous endogenous variable, at most 16 exogenous predictors and 16 excluded instruments.")
    names = [spec.outcome, endogenous[0], *exog, *instruments]
    if len(names) != len(set(names)):
        _error("invalid_spec", "Outcome, endogenous, exogenous and excluded instrument columns must be distinct.")
    joint = 2 * len(exog) + len(instruments) + 2 * int(spec.intercept) + 2
    if joint > MAX_JOINT_PARAMETERS:
        _error("dimension_limit", "Control functions admit at most 36 joint coefficients.")
    if batch_rows is None:
        batch_rows = spec.options.get("batch_rows")
    if batch_rows is not None and (type(batch_rows) is not int or not 1 <= batch_rows <= 65536):
        _error("invalid_batch_size", "batch_rows must be an integer from 1 to 65536.")
    with torch.no_grad(), torch.device("cpu"), _cpu_execution() as execution:
        # Discovery admission constructs no source rows. Bind the original
        # producer only after its projected reader allocation has been planned.
        sample = ReplaySample(spec, source, batch_rows=batch_rows)
        wrapped, binding = _bound_source(spec, source,
                                         lambda: min(batch_rows or 8192, sample.reader_rows))
        sample.source = wrapped
        z = sample.add_design("z", [*exog, *instruments])
        x = sample.add_design("x", [*exog, endogenous[0]])
        expected_z, expected_x = len(exog) + len(instruments) + int(spec.intercept), len(exog) + 1 + int(spec.intercept)
        sample.prepare()
        if len(z.terms) != expected_z or len(x.terms) != expected_x or sample.notes["omitted_terms"]:
            _error("rank_deficient", "Control-function designs must retain every declared column; silent rank drops are unsupported.")
        if sample.nrows <= joint:
            _error("dimension_limit", "Control functions require more retained observations than joint coefficients.")
        sample.plan_rows("streamed full control-function equations", {
            "joint_derivative_meat_covariance_and_solves": 160 * joint**2,
            "balanced_tsqr_factors": 64 * 8 * ((expected_z+1)**2 + (expected_x+2)**2),
            "typed_key_and_cluster_cache": 8 * 1024**2 if spec.covariance == "cluster" else KEY_BYTES,
        }, 128 * (expected_z + 2*expected_x + joint + 20))
        sample.cf_original_source = source
        sample.cf_source_binding = binding
        sample.cf_endogenous = endogenous[0]
        sample.cf_execution = execution.metadata()
        return sample


def _rank_factor(factor, maxima, name):
    if not bool(torch.isfinite(factor).all()) or not bool(torch.isfinite(maxima).all()) or not bool((maxima > 0).all()):
        _error("rank_deficient", f"{name} has a zero or unresolved column.")
    singular = torch.linalg.svdvals(factor / maxima)
    if not bool(torch.isfinite(singular).all()) or float(singular[-1]) <= 0 or float(singular[0]/singular[-1]) > MAX_CONDITION:
        _error("rank_deficient", f"{name} is singular or numerically ill-conditioned after column scaling.")
    return {"condition_number": float(singular[0]/singular[-1]),
            "condition_limit": MAX_CONDITION, "basis": "original-unit column-maximum-scaled global TSQR factor"}


def _validate_y(y, kind):
    valid = True
    if kind in {"logit", "probit", "cloglog"}:
        valid = bool(((y == 0) | (y == 1)).all())
    elif kind == "fractional_logit":
        valid = bool(((y >= 0) & (y <= 1)).all())
    elif kind == "poisson":
        valid = bool(((y >= 0) & (y == y.floor())).all())
    elif kind in {"gamma", "inverse_gaussian"}:
        valid = bool((y > 0).all())
    if not valid:
        _error("invalid_outcome", "Outcome values are outside the declared control-function support.")


def _first_stage(sample, kind, *, solve=True):
    """Augmented normalized TSQR, retaining equivalent original-unit geometry."""
    design = sample.designs["z"]
    d_design = sample.designs["x"]
    d_scale = d_design.scales[-1]
    d_mean = d_design.means[-1] if sample.spec.intercept else torch.tensor(0., dtype=DT)
    tree = _TSQRTree()
    maxima = torch.zeros(len(design.terms), dtype=DT)
    max_d, all_binary, low_y, high_y = 0., True, math.inf, -math.inf
    for batch in sample.batches():
        z, d, y = sample.raw_design(batch, "z"), batch.numeric(sample.cf_endogenous), batch.numeric(sample.spec.outcome)
        _validate_y(y, kind)
        maxima = torch.maximum(maxima, z.abs().amax(0))
        max_d = max(max_d, float(d.abs().max()))
        all_binary &= bool(((d == 0) | (d == 1)).all())
        low_y, high_y = min(low_y, float(y.min())), max(high_y, float(y.max()))
        # The normalized endogenous column preserves small variation around
        # large offsets and avoids squaring the original response scale.
        target = batch.designs["x"][:, -1]
        tree.add(qr_factor(torch.cat((batch.designs["z"], target[:, None]), 1)))
    if all_binary:
        _error("unsupported_endogenous", "Binary endogenous first stages are outside this continuous-first-stage contract.")
    if sample.spec.intercept and ((kind in {"logit", "probit", "cloglog", "fractional_logit"} and low_y == high_y and low_y in {0., 1.})
                                 or (kind == "poisson" and high_y == 0)):
        _error("separation_detected", "A constant boundary outcome has no finite intercept estimate.")
    factor = tree.finish()
    kz = len(design.terms)
    gamma = None
    if solve:
        theta = least_squares(factor[:, :kz], factor[:, kz], drop_collinear=False).beta
        gamma = design.transform @ theta * d_scale
        if sample.spec.intercept:
            gamma[0] += d_mean
    raw_factor = factor[:, :kz] @ torch.linalg.inv(design.transform)
    raw_target = factor[:, kz] * d_scale
    if sample.spec.intercept:
        raw_target = raw_target + raw_factor[:, 0] * d_mean
    rank = _rank_factor(raw_factor, maxima, "First-stage design")
    return {"gamma": gamma, "first_stage_factor": raw_factor,
            "first_stage_target": raw_target, "maximum_abs_z": maxima,
            "maximum_abs_d": max_d, "first_stage_rank": rank,
            "first_stage_tsqr_depth": tree.depth}


def _raw_parts(sample, batch, gamma):
    z, x = sample.raw_design(batch, "z"), sample.raw_design(batch, "x")
    d = batch.numeric(sample.cf_endogenous)
    fitted = z @ gamma
    residual = d - fitted
    if not bool(torch.isfinite(residual).all()) or not bool(torch.isfinite(fitted).all()):
        _error("numerical_domain", "The generated residual exceeds finite float64 support.")
    return z, torch.cat((x, residual[:, None]), 1), residual, fitted


def _outcome_geometry(sample, gamma, kind):
    kq = len(sample.designs["x"].terms) + 1
    moments = _ScaledMoments(kq)
    max_r = max_fit = max_y = 0.
    response = _CompensatedSum(())
    for batch in sample.batches():
        _, q, residual, fitted = _raw_parts(sample, batch, gamma)
        y = batch.numeric(sample.spec.outcome)
        moments.add(q)
        response.add((y/sample.nrows).sum())
        max_r, max_fit, max_y = max(max_r, float(residual.abs().max())), max(max_fit, float(fitted.abs().max())), max(max_y, float(y.abs().max()))
    max_d = max(abs(float(moments.minimum[-2])), abs(float(moments.maximum[-2])))
    if max_r <= 64 * EPS * max(max_d, max_fit):
        _error("rank_deficient", "The first-stage residual has zero variation to float64 precision.")
    magnitude = torch.where(moments.magnitude > 0, moments.magnitude, torch.ones_like(moments.magnitude))
    centers = moments.mean.value.clone() if sample.spec.intercept else torch.zeros(kq, dtype=DT)
    variance = moments.m2.value/sample.nrows
    if not sample.spec.intercept:
        variance = variance + moments.mean.value.square()
    rms = variance.clamp_min(0).sqrt()
    if sample.spec.intercept:
        centers[0], rms[0] = 0., 1.
    if not bool((rms > 0).all()) or not bool(torch.isfinite(rms).all()):
        _error("rank_deficient", "Residual-inclusion design has a zero or unresolved column.")
    scales = magnitude*rms
    if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        _error("numerical_domain", "Residual-inclusion normalization leaves finite float64 support.")
    transform = torch.diag(scales.reciprocal())
    if sample.spec.intercept:
        transform[0] = -centers/rms
        transform[0, 0] = 1.
    tree = _TSQRTree()
    signs = _CompensatedSum((kq,))
    signed_count = 0
    for batch in sample.batches():
        _, q, _, _ = _raw_parts(sample, batch, gamma)
        normalized = _normalize(q, magnitude, centers, rms)
        y = batch.numeric(sample.spec.outcome)
        # The augmented Gaussian target is only a bounded block. Other
        # outcomes require a factor of Q alone for the same global rank check.
        tree.add(qr_factor(torch.cat((normalized, (y/max(1., max_y))[:, None]), 1) if kind == "gaussian" else normalized))
        if kind in {"logit", "probit", "cloglog", "fractional_logit"}:
            endpoint = (y == 0) | (y == 1)
            signs.add((normalized[endpoint] * (2*(y[endpoint] == 1).to(DT)-1)[:, None]).sum(0))
            signed_count += len(y) + int((~endpoint).sum())
    factor = tree.finish()
    raw_factor = factor[:, :kq] @ torch.linalg.inv(transform)
    maxima = moments.magnitude
    rank = _rank_factor(raw_factor, maxima, "Residual-inclusion design")
    return {"magnitude": magnitude, "centers": centers, "rms": rms,
            "transform": transform, "normalized_factor": factor[:, :kq],
            "outcome_factor": raw_factor,
            "outcome_target": factor[:, kq]*max(1., max_y) if kind == "gaussian" else None,
            "normalized_target": factor[:, kq] if kind == "gaussian" else None,
            "maximum_abs_residual": max_r, "maximum_abs_first_stage_fitted": max_fit,
            "maximum_abs_y": max_y, "mean_y": float(response.value),
            "maximum_abs_q": maxima, "outcome_rank": rank,
            "outcome_tsqr_depth": tree.depth,
            "separation_objective": signs.value/max(1, signed_count)}


def _certificate(sample, gamma, kind, geometry):
    if kind not in {"logit", "probit", "cloglog", "fractional_logit"}:
        return {}

    def replay():
        for batch in sample.batches():
            _, q, _, _ = _raw_parts(sample, batch, gamma)
            q = _normalize(q, geometry["magnitude"], geometry["centers"], geometry["rms"])
            y = batch.numeric(sample.spec.outcome)
            yield q * (2*(y == 1).to(DT)-1)[:, None]
            interior = (y > 0) & (y < 1)
            if bool(interior.any()):
                yield q[interior]
    return certify_separation(replay, geometry["separation_objective"], len(geometry["rms"]), max_passes=CERTIFICATE_MAX_PASSES)


class _OutcomeObjective:
    """Replay native analytic GLM rows, draining even invalid trial passes."""
    def __init__(self, sample, gamma, kind, geometry):
        self.sample, self.gamma, self.kind, self.geometry = sample, gamma, kind, geometry
        self.width = len(geometry["rms"])

    def _native(self, batch):
        _, q, _, _ = _raw_parts(self.sample, batch, self.gamma)
        q = _normalize(q, self.geometry["magnitude"], self.geometry["centers"], self.geometry["rms"])
        return _objective(q, batch.numeric(self.sample.spec.outcome), self.kind)

    def __call__(self, beta):
        value, gradient, hessian = _CompensatedSum(()), _CompensatedSum((self.width,)), _CompensatedSum((self.width, self.width))
        valid = True
        for batch in self.sample.batches():
            v, g, h = self._native(batch)(beta)
            finite = bool(torch.isfinite(v)) and bool(torch.isfinite(g).all()) and bool(torch.isfinite(h).all())
            valid &= finite
            if finite:
                value.add(v)
                gradient.add(g)
                hessian.add(h)
        if not valid:
            return torch.tensor(-math.inf, dtype=DT), torch.zeros_like(beta), torch.zeros((self.width, self.width), dtype=DT)
        return value.value, gradient.value, (hessian.value+hessian.value.T)/2

    def value(self, beta):
        value, valid = _CompensatedSum(()), True
        for batch in self.sample.batches():
            v = self._native(batch).value(beta)
            finite = bool(torch.isfinite(v))
            valid &= finite
            if finite:
                value.add(v)
        return value.value if valid else torch.tensor(-math.inf, dtype=DT)


def _sandwich(bread, meat, kz):
    _positive_matrix(bread[kz:, kz:], "Observed outcome information")
    if not bool(torch.isfinite(bread).all()) or not bool(torch.isfinite(meat).all()):
        _error("numerical_domain", "The full stacked sensitivity or meat exceeds finite float64 support.")
    row_scale = bread.abs().amax(1).reciprocal()
    a = row_scale[:, None]*bread
    col_scale = a.abs().amax(0).reciprocal()
    a = a*col_scale[None, :]
    if not bool(torch.isfinite(a).all()):
        _error("singular_bread", "The full stacked derivative cannot be equilibrated within float64 precision.")
    singular = torch.linalg.svdvals(a)
    if float(singular[-1]) <= 0 or float(singular[0]/singular[-1]) > MAX_CONDITION:
        _error("singular_bread", "The full stacked observed derivative is numerically singular.")
    scaled_meat = row_scale[:, None]*meat*row_scale[None, :]
    left = torch.linalg.solve(a, scaled_meat)
    covariance = torch.linalg.solve(a, left.T).T
    covariance = col_scale[:, None]*covariance*col_scale[None, :]
    covariance = (covariance+covariance.T)/2
    _positive_matrix(covariance, "Full joint sandwich covariance")
    return covariance


def _aggregate(sample, gamma, beta, kind, stage, geometry, work, certificate, *, collect_preview):
    kz, kq = len(gamma), len(beta)
    joint = kz+kq
    bread, meat, score, score_abs, criterion = (_CompensatedSum((joint, joint)), _CompensatedSum((joint, joint)),
                                               _CompensatedSum((joint,)), _CompensatedSum((joint,)), _CompensatedSum(()))
    cluster_name = registry.cluster_columns(sample.spec)
    accumulator = ClusterAccumulator(joint) if sample.spec.covariance == "cluster" else None
    preview, max_s, max_fit, max_score = [], 0., 0., 0.
    cluster_info, groups = {}, None
    with accumulator if accumulator is not None else nullcontext():
        for batch in sample.batches():
            z, q, residual, _ = _raw_parts(sample, batch, gamma)
            y = batch.numeric(sample.spec.outcome)
            native = _objective(q, y, kind)
            fitted, s, h = _state(native, beta)
            rows = torch.cat((z*residual[:, None], q*s[:, None]), 1)
            block = torch.zeros((joint, joint), dtype=DT)
            block[:kz, :kz] = z.T@z
            cross = beta[-1]*(q*h[:, None]).T@z
            cross[-1] += s@z
            block[kz:, :kz], block[kz:, kz:] = cross, -(q*h[:, None]).T@q
            if not bool(torch.isfinite(rows).all()) or not bool(torch.isfinite(block).all()):
                _error("numerical_domain", "The full stacked rows or derivative exceeds finite float64 support.")
            bread.add(block)
            score.add(rows.sum(0))
            score_abs.add(rows.abs().sum(0))
            criterion.add(_criterion(kind, y, fitted))
            max_s, max_fit, max_score = max(max_s, float(s.abs().max())), max(max_fit, float(fitted.mu.abs().max())), max(max_score, float(rows.abs().max()))
            if accumulator is None:
                meat.add(rows.T@rows)
            else:
                # Canonical typed keys are constructed in at most 1 MiB
                # sub-blocks, including Python object/payload allowance.
                keys, start, key_bytes = [], 0, 0
                for offset, label in enumerate(batch.frame[cluster_name[0]]):
                    key = _label_bytes(label, cluster=True)
                    cost = len(key)+128
                    if cost > KEY_BYTES:
                        _error("cluster_key_budget", "One typed cluster key exceeds its bounded workspace.")
                    if keys and key_bytes+cost > KEY_BYTES:
                        accumulator.add(keys, rows[start:offset])
                        keys, start, key_bytes = [], offset, 0
                    keys.append(key)
                    key_bytes += cost
                if keys:
                    accumulator.add(keys, rows[start:])
            if collect_preview and len(preview) < 400:
                take = min(400-len(preview), len(y))
                preview.extend({"row": int(batch.positions[j]), "observed": float(y[j]), "fitted": float(fitted.mu[j]),
                                "residual": float(y[j]-fitted.mu[j])} for j in range(take))
        if accumulator is not None:
            cluster_meat, groups = accumulator.finish()
            if groups <= joint:
                _error("singular_covariance", "A positive full joint CR0 covariance requires more clusters than joint parameters.")
            meat.add(cluster_meat)
            cluster_info = accumulator.diagnostics
    if kind == "gaussian" and max_s <= 64*EPS*max(geometry["maximum_abs_y"], max_fit):
        _error("singular_covariance", "The outcome residual variance is zero to float64 precision.")
    covariance = _sandwich(bread.value, meat.value, kz)
    return {"gamma": gamma.clone(), "beta": beta.clone(), "bread": bread.value, "meat": meat.value,
            "joint_covariance": covariance, "criterion": criterion.value, "cluster_count": groups,
            "score_sum": score.value, "score_abs_sum": score_abs.value,
            "outcome_gradient": score.value[kz:], "outcome_information": bread.value[kz:, kz:],
            "first_stage_factor": stage["first_stage_factor"], "first_stage_target": stage["first_stage_target"],
            "outcome_factor": geometry["outcome_factor"], "outcome_target": geometry["outcome_target"],
            "z_terms": list(sample.designs["z"].terms), "x_terms": list(sample.designs["x"].terms),
            "preview": preview, "observed": torch.tensor([v["observed"] for v in preview], dtype=DT),
            "fitted": torch.tensor([v["fitted"] for v in preview], dtype=DT),
            "n_original": sample.original_count, "nobs": sample.nrows, "resource_plan": sample.adapter_plan.record(),
            "work_estimate": work, "source_binding": dict(sample.cf_source_binding),
            "maximum_abs_z": stage["maximum_abs_z"], "maximum_abs_d": stage["maximum_abs_d"],
            "maximum_abs_residual": geometry["maximum_abs_residual"],
            "maximum_abs_first_stage_fitted": geometry["maximum_abs_first_stage_fitted"],
            "maximum_abs_y": geometry["maximum_abs_y"], "maximum_abs_fitted": max_fit,
            "maximum_abs_scalar_score": max_s, "maximum_abs_score": max_score,
            "rank_diagnostics": {"first_stage": stage["first_stage_rank"], "outcome": geometry["outcome_rank"]},
            "cluster_diagnostics": cluster_info, "optimizer": {"method": "evaluation", "converged": False,
                "iterations": 0, **certificate}}


def _translate_errors(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            with torch.no_grad(), torch.device("cpu"), _cpu_execution() as execution:
                result = fn(*args, **kwargs)
                solved = result[1] if isinstance(result, tuple) else result
                solved["execution"] = execution.metadata()
                return result
        except KernelError as error:
            raise AnalysisError(error.code, str(error)) from error
    return wrapped


@_translate_errors
def evaluate_stream(sample, gamma, beta, kind, *, max_work=None, collect_preview=True):
    """Replay exact stacked equations at supplied parameters without refitting.

    A small first-stage factor/target is reconstructed for verification; no
    first-stage or outcome regression is solved during this replay.
    """
    expected = KINDS[sample.spec.estimator][0]
    if kind != expected:
        _error("invalid_state", "Saved kind differs from the replayed specification.")
    _, _, configured = _options(sample.spec)
    max_work = configured if max_work is None else max_work
    if isinstance(max_work, bool) or not isinstance(max_work, Integral) or not 1 <= max_work <= DEFAULT_MAX_WORK:
        _error("invalid_options", "max_work must be an integer from 1 to 10000000000.")
    _tensor(gamma, "gamma", (len(sample.designs["z"].terms),))
    _tensor(beta, "beta", (len(sample.designs["x"].terms)+1,))
    if not bool(torch.isfinite(gamma).all()) or not bool(torch.isfinite(beta).all()):
        _error("non_finite_values", "Saved parameters must be finite.")
    work = _work(sample, kind, 0, int(max_work))
    stage = _first_stage(sample, kind, solve=False)
    geometry = _outcome_geometry(sample, gamma, kind)
    certificate = _certificate(sample, gamma, kind, geometry)
    return _aggregate(sample, gamma, beta, kind, stage, geometry, work, certificate, collect_preview=collect_preview)


@_translate_errors
def solve_stream(spec, source, *, batch_rows=None):
    """Fit one global OLS first stage and exact native residual-inclusion outcome."""
    sample = prepare_stream(spec, source, batch_rows=batch_rows)
    kind = KINDS[spec.estimator][0]
    iterations, tolerance, max_work = _options(spec)
    work = _work(sample, kind, 0 if kind == "gaussian" else iterations, max_work)
    stage = _first_stage(sample, kind)
    gamma = stage["gamma"]
    geometry = _outcome_geometry(sample, gamma, kind)
    certificate = _certificate(sample, gamma, kind, geometry)
    if kind == "gaussian":
        beta = geometry["transform"] @ least_squares(geometry["normalized_factor"], geometry["normalized_target"], drop_collinear=False).beta
        beta *= max(1., geometry["maximum_abs_y"])
        optimizer = {"method": "global_augmented_tsqr", "iterations": 0, "converged": True,
                     "function_evaluations": 0, "derivative_evaluations": 0, "backtracks": 0,
                     "nonconcave_iterations": 0, "message": "Global closed-form residual-inclusion TSQR fit."}
    else:
        objective = _OutcomeObjective(sample, gamma, kind, geometry)
        start = torch.zeros(len(geometry["rms"]), dtype=DT)
        if spec.intercept:
            _, link_type = _family_link(kind)
            intercept = link_type().link(torch.tensor([geometry["mean_y"]], dtype=DT))
            if bool(torch.isfinite(intercept).all()):
                start[0] = intercept[0]
        # A merely small Newton decrement can leave an absolute generated-
        # control score visible in near-zero cross entries of the full meat.
        # Refine the same global objective within the declared iteration/work
        # budget. The floor is the squared, dimension-aware float64 summation
        # error in normalized coordinates; no score or meat entry is zeroed.
        refinement = max((32*EPS)**2 * sample.nrows * len(start),
                         tolerance**2/sample.nrows)
        run = maximize_newton(objective, start, max_iter=iterations,
                              step_tol=min(tolerance, 1e-12),
                              scaled_gradient_tol=refinement,
                              value_fn=objective.value, raise_on_failure=False)
        if not run.converged:
            _error("nonconvergence", "The global outcome optimizer did not converge within its explicit iteration budget.")
        beta = geometry["transform"] @ run.theta
        optimizer = {"method": run.method, "iterations": run.iterations, "converged": True, **run.diagnostics,
                     "refined_scaled_gradient_tolerance": refinement,
                     "refined_step_tolerance": min(tolerance, 1e-12),
                     "stationarity_refinement": "global analytic Newton; dimension-aware float64 floor; within declared iteration budget"}
    solved = _aggregate(sample, gamma, beta, kind, stage, geometry, work, certificate, collect_preview=True)
    solved["optimizer"] = {**optimizer, **certificate, "max_iterations": iterations,
        "tolerance": tolerance, "working_dispersion": 1., "criterion": "canonical_working_score_sum",
        "optimization_criterion": "negative_half_glm_deviance", "dense_observation_matrix": False}
    return sample, solved
