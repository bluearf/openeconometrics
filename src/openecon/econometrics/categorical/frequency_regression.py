"""Native frequency geometry for bounded descriptive optimal regression.

Counts describe literal repeated cases. No row expansion or ordinary regression
inference is used after learning the predictor or response transformations.
"""

from __future__ import annotations

import copy
import json
import math
from numbers import Real
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import restore_summary, summary_state
from . import frequency as f
from . import optimal as base

METHODS = {
    "catreg_nominal_fweight", "catreg_ordinal_fweight",
    "catreg_nominal_response_fweight", "catreg_ordinal_response_fweight",
}
SOURCES = base.SOURCES[:1] + ["https://arxiv.org/abs/1611.05433"]
STATE_FIELDS = {
    "version", "kind", "method", "variables", "outcome", "frequency",
    "response_scale", "descriptors", "response", "raw", "counts", "positions",
    "input_nobs", "zero_positions", "missing_positions", "beta", "intercept",
    "objective", "condition", "stationarity", "response_roots", "controls",
    "chosen_start", "iterations", "histories", "starts", "missing",
}


def _error(message, code="invalid_spec"):
    raise AnalysisError(code, message)


def _controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device):
    if device != "cpu":
        _error("Frequency CATREG supports resident CPU float64 only.", "unsupported_option")
    controls = {
        "n_starts": base._integer(n_starts, "n_starts", 1, 12),
        "seed": base._integer(seed, "seed", 0, 2**31-1),
        "maxiter": base._integer(maxiter, "maxiter", 1, 1000),
        "tol": base._real(tol, "tol", 1e-12, 1e-3),
        "max_work": base._integer(max_work, "max_work", 1, f.WORK),
        "max_bytes": base._integer(max_bytes, "max_bytes", 1, f.BYTES),
    }
    return controls


def _names(predictors, outcome, frequency):
    names = base._names(predictors)
    if len(names) > 11 or any(len(name) > 128 for name in names):
        _error("CATREG permits at most 11 predictors with names up to 128 characters.")
    if (not isinstance(outcome, str) or not 1 <= len(outcome) <= 128
            or outcome in names or not isinstance(frequency, str)
            or not 1 <= len(frequency) <= 128 or frequency in [outcome]+names):
        _error("Response and frequency must name distinct bounded columns.")
    return names


def _scales(variables, outcome, response_scale, default, scales, orders, outcome_order):
    scales = {name: default for name in variables} if scales is None else scales
    if (not isinstance(scales, dict) or set(scales) != set(variables)
            or any(scale not in ("nominal", "ordinal", "numeric") for scale in scales.values())):
        _error("scales must assign nominal, ordinal or numeric to every predictor.")
    if default == "nominal" and response_scale == "numeric" and "ordinal" in scales.values():
        _error("Use catreg_ordinal_fweight with explicit orders for ordinal predictors.")
    orders = {} if orders is None else orders
    if not isinstance(orders, dict) or set(orders) != {name for name in variables if scales[name] == "ordinal"}:
        _error("orders must explicitly name exactly the ordinal predictors.")
    all_orders = dict(orders)
    if response_scale == "ordinal":
        all_orders[outcome] = outcome_order
    for order in all_orders.values():
        if not isinstance(order, (list, tuple)) or not 2 <= len(order) <= 32:
            _error("Each declared order must contain 2–32 bounded categories.")
        for value in order:
            f._atom(value)
    return {outcome: response_scale, **scales}, all_orders


def _rms(vector, w):
    return float(torch.sqrt((vector.square()*w).sum()/w.sum()))


def _solve(z, y, w):
    weighted = z*w.sqrt()[:, None]
    condition = base._rank(weighted)
    return torch.linalg.lstsq(weighted, y*w.sqrt()).solution, condition


def _sweep(y, z, beta, descriptors, prepared, w):
    fitted = z@beta
    total = w.sum()
    for j, (descriptor, values) in enumerate(zip(descriptors, prepared)):
        residual = y-fitted+beta[j]*z[:, j]
        if descriptor["scale"] == "numeric":
            z[:, j] = values
            beta[j] = (values*residual*w).sum()/total
            contribution = values*beta[j]
        else:
            means, counts = f._means(residual, values, len(descriptor["levels"]), w)
            if descriptor["scale"] == "ordinal":
                increasing = f._pava(means, counts)
                decreasing = -f._pava(-means, counts)
                loss_inc = float(((means-increasing).square()*counts).sum())
                loss_dec = float(((means-decreasing).square()*counts).sum())
                means = increasing if loss_inc <= loss_dec else decreasing
            means = means-(means*counts).sum()/counts.sum()
            contribution = means[values]
            q = f._normalize_categories(means, counts)
            if (descriptor["scale"] == "ordinal" and float(q[-1]) < float(q[0])) or (
                    descriptor["scale"] == "nominal" and float(q[int(torch.argmax(q.abs()))]) < 0):
                q = -q
            z[:, j] = q[values]
            beta[j] = (z[:, j]*contribution*w).sum()/total
        fitted = y-residual+contribution
    return _solve(z, y, w)[0]


def _response_block(fitted, descriptor, codes, w):
    means, counts = f._means(fitted, codes, len(descriptor["levels"]), w)
    if descriptor["scale"] == "ordinal":
        means = f._pava(means, counts)
    q = f._normalize_categories(means, counts)
    return q, q[codes]


def _response_roots(z, codes, counts, w):
    u, _, _ = torch.linalg.svd(z*w.sqrt()[:, None], full_matrices=False)
    category = torch.tensor(counts, dtype=f.DT)
    aggregate = torch.zeros((len(counts), z.shape[1]), dtype=f.DT)
    aggregate.index_add_(0, codes, u*w.sqrt()[:, None])
    aggregate = aggregate/category.sqrt()[:, None]
    constant = torch.sqrt(category/w.sum())
    centered = aggregate-constant[:, None]*(constant@aggregate)[None, :]
    matrix = centered@centered.T
    return torch.linalg.eigvalsh((matrix+matrix.T)/2)


def _typed(value, numeric=False):
    if numeric:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or abs(value) > 1e100:
            _error("Numeric values must be finite reals of magnitude at most 1e100.")
        return ["float", float(value)]
    return base._label(f._atom(value))


def _raw(frame, names, scales):
    columns = [[_typed(value, scales[name] == "numeric") for value in frame[name].tolist()] for name in names]
    return [list(row) for row in zip(*columns)]


def _work(n, p, category_sizes, controls):
    cost = 8*n*p*p+64*n*(p+1)+64*sum(size*size for size in category_sizes)
    return controls["n_starts"]*(controls["maxiter"]+2)*cost+64*max(category_sizes+[0])**3


def _bound(raw, n, p, controls):
    labels = sum(len(json.dumps(value, ensure_ascii=True))+12 for row in raw for value in row)
    # Full raw state, category tables, duplicated attrs/settings, trace and
    # training tables are all retained. Unicode escaping is charged explicitly.
    return 32768+4*labels+512*n*(p+6)+1024*32*(p+1)+1024*controls["n_starts"]*(controls["maxiter"]+1)


def _map(raw, descriptors):
    columns = []
    for j, descriptor in enumerate(descriptors):
        if descriptor["scale"] == "numeric":
            values = torch.tensor([row[j][1] for row in raw], dtype=f.DT)
            columns.append((values-descriptor["mean"])/descriptor["sd"])
        else:
            mapping = {base._key(label): q for label, q in zip(descriptor["levels"], descriptor["quantifications"])}
            try:
                values = [mapping[base._key(row[j])] for row in raw]
            except KeyError as exc:
                raise AnalysisError("unknown_category", "A category is absent from the saved calibration.") from exc
            columns.append(torch.tensor(values, dtype=f.DT))
    return torch.stack(columns, dim=1)


def _tables(state, z, y, fitted):
    categorical = state["response_scale"] != "numeric"
    positions, counts = state["positions"], state["counts"]
    beta_column = "beta_quantified_response" if categorical else "beta_descriptive"
    fitted_columns = ["source_position", "frequency", "observed", "fitted", "residual"]
    observed = y.tolist() if categorical else [row[0][1] for row in state["raw"]]
    if categorical:
        fitted_columns += ["outcome_type", "outcome_category"]
    fitted_rows = []
    for i, (position, count, actual, fit) in enumerate(zip(positions, counts, observed, fitted.tolist())):
        row = [position, count, actual, fit, actual-fit]
        if categorical:
            row += state["raw"][i][0]
        fitted_rows.append(row)
    total = sum(counts)
    sse = sum(count*(actual-fit)**2 for count, actual, fit in zip(counts, observed, fitted.tolist()))
    return {
        "coefficients": table([["_cons", state["intercept"]]]+[[name, value] for name, value in zip(state["variables"], state["beta"])], columns=["term", beta_column]),
        "fit": table([[len(positions), total, sse, 1-state["objective"], state["objective"], state["condition"], state["stationarity"]]], columns=["physical_n", "frequency_n", "weighted_sse", "r_squared", "objective", "design_condition", "block_stationarity"]),
        "fitted": table(fitted_rows, columns=fitted_columns),
        "transformed": table([[position, count]+row for position, count, row in zip(positions, counts, z.tolist())], columns=["source_position", "frequency"]+state["variables"]),
        "iterations": table(state["histories"], columns=["start", "iteration", "objective", "improvement", "response_change", "fitted_change"]),
        "starts": table(state["starts"], columns=["start", "converged", "iterations", "objective", "status"]),
        "sample": table([[i, "active" if i in positions else "zero" if i in state["zero_positions"] else "missing"] for i in range(state["input_nobs"])], columns=["source_position", "status"]),
        **base._mapping_tables([state["response"]]+state["descriptors"]),
    }


def _output(state, z, y, fitted, workspace, input_workspace, declared_work):
    result = TableSet(_tables(state, z, y, fitted), title="Frequency-weighted categorical regression",
        method=state["method"], variables=state["variables"], outcome=state["outcome"],
        response_scale=state["response_scale"], frequency=state["frequency"],
        n=len(state["positions"]), frequency_n=sum(state["counts"]), n_input=state["input_nobs"],
        n_zero=len(state["zero_positions"]), n_missing=len(state["missing_positions"]),
        sample_positions=state["positions"], missing=state["missing"], settings=state["controls"],
        frequency_state=state, state_sha256=f._seal(state), resources=workspace,
        input_preparation=input_workspace, declared_work=declared_work,
        chosen_start=state["chosen_start"], iterations=state["iterations"], converged=True,
        dtype="float64", device="cpu", weight_type="frequency", sources=SOURCES,
        solution="best stationary converged deterministic declared start; local solution only",
        inference="descriptive adaptive quantification; no ordinary covariance, SE, p-value, CI or calibrated class probabilities")
    output = f._save(result)
    summary_state(output)
    return output


def _fit(data, outcome, predictors, frequency, response_scale, default, scales, orders, outcome_order, controls, missing):
    variables = _names(predictors, outcome, frequency)
    all_scales, all_orders = _scales(variables, outcome, response_scale, default, scales, orders, outcome_order)
    sample = f._sample(data, [outcome]+variables, frequency, missing, controls["max_bytes"], controls["max_work"], operation="frequency CATREG", min_rows=len(variables)+3)
    frame, n, p = sample["frame"], len(sample["counts"]), len(variables)
    raw = _raw(frame, [outcome]+variables, all_scales)
    sizes = [len({base._key(row[j]) for row in raw}) if all_scales[name] != "numeric" else 0 for j, name in enumerate([outcome]+variables)]
    if any(size > 32 for size in sizes):
        _error("At most 32 retained levels per categorical variable.", "resource_limit")
    output_bound = _bound(raw, n, p, controls)
    if output_bound > 32*1024**2:
        _error("Full frequency CATREG state exceeds its conservative portable domain.", "resource_limit")
    work = _work(n, p, sizes, controls)+8*sample["input_nobs"]*(p+2)+4*output_bound
    fit_size = sample["workspace"]["estimated_workspace_bytes"]+8*n*(48*p+96)+128*p*p+2*output_bound
    workspace = f._admit("frequency CATREG fit and full portable state", fit_size, work, controls["max_bytes"], controls["max_work"])
    w = f._weights(sample["counts"])
    descriptions, prepared = f._prepare(frame, [outcome]+variables, all_scales, all_orders, default, w)
    for descriptor in descriptions:
        if descriptor["scale"] != "numeric":
            descriptor["levels"] = [base._label(value) for value in descriptor["levels"]]
    response, descriptors = descriptions[0], descriptions[1:]
    response_data, predictor_data = prepared[0], prepared[1:]
    categorical = response_scale != "numeric"
    rng = torch.Generator(device="cpu").manual_seed(controls["seed"])
    histories, starts, candidates = [], [], []
    stationary_tolerance = max(1e-9, math.sqrt(controls["tol"])*.1)
    for start in range(controls["n_starts"]):
        iteration, objective = 0, None
        try:
            initial = base._initial(descriptions, prepared, start, rng)
            y, z = initial[:, 0].clone(), initial[:, 1:].clone()
            beta, _ = _solve(z, y, w)
            objective = _rms(y-z@beta, w)**2
            histories.append([start, 0, objective, 0., 0., 0.])
            converged = False
            for iteration in range(1, controls["maxiter"]+1):
                old, old_y, old_fitted = objective, y.clone(), z@beta
                if categorical:
                    _, y = _response_block(old_fitted, response, response_data, w)
                    beta, _ = _solve(z, y, w)
                beta = _sweep(y, z, beta, descriptors, predictor_data, w)
                fitted = z@beta
                objective = _rms(y-fitted, w)**2
                improvement = old-objective
                y_change, fit_change = _rms(y-old_y, w), _rms(fitted-old_fitted, w)
                histories.append([start, iteration, objective, improvement, y_change, fit_change])
                if improvement < -1e-9:
                    _error("Weighted ALS objective increased beyond roundoff.", "numerical_failure")
                if abs(improvement) <= controls["tol"]*max(1., old) and max(y_change, fit_change) <= stationary_tolerance:
                    converged = True
                    break
            if not converged:
                starts.append([start, False, iteration, objective, "nonconvergence"])
                continue
            starts.append([start, True, iteration, objective, "accepted"])
            candidates.append((objective, start, y.clone(), z.clone(), beta.clone(), iteration))
        except AnalysisError as exc:
            starts.append([start, False, iteration, objective, exc.code])
    if not candidates:
        reasons = {row[-1] for row in starts}
        code = next(iter(reasons)) if len(reasons) == 1 else "nonconvergence"
        _error("No declared start reached a nondegenerate full-rank stationary weighted solution: "+", ".join(sorted(reasons)), code)
    objective, chosen, y, z, beta, iterations = min(candidates, key=lambda row: row[:2])
    response, descriptors = copy.deepcopy(response), copy.deepcopy(descriptors)
    if categorical:
        q, _ = f._means(y, response_data, len(response["levels"]), w)
        if response_scale == "nominal" and float(q[int(torch.argmax(q.abs()))]) < 0:
            q, y, beta = -q, -y, -beta
        response["quantifications"] = q.tolist()
    for j, (descriptor, codes) in enumerate(zip(descriptors, predictor_data)):
        if descriptor["scale"] != "numeric":
            descriptor["quantifications"] = f._means(z[:, j], codes, len(descriptor["levels"]), w)[0].tolist()
    # Persisted tables are generated from the persisted maps. Averaging a
    # repeated constant category score can alter its last binary digit; using
    # that saved map here gives restoration exactly the same matrix.
    z = _map([row[1:] for row in raw], descriptors)
    y = _map([[row[0]] for row in raw], [response])[:, 0]
    beta, _ = _solve(z, y, w)
    fitted = z@beta
    checked_z, checked_beta = z.clone(), beta.clone()
    checked_beta = _sweep(y, checked_z, checked_beta, descriptors, predictor_data, w)
    stationarity = _rms(checked_z@checked_beta-fitted, w)
    roots = []
    if categorical:
        next_y = _response_block(fitted, response, response_data, w)[1]
        stationarity = max(stationarity, _rms(next_y-y, w))
        roots = _response_roots(z, response_data, response["counts"], w).tolist()
        if response_scale == "nominal":
            if roots[-1]-roots[-2] <= 1e-8*max(1., roots[-1]):
                _error("The leading conditional nominal response roots tie.", "unidentified_response")
            if roots[-1]-(1-objective) > max(1e-7, 10*controls["tol"]):
                _error("The nominal response failed its conditional leading-root check.", "nonconvergence")
    if stationarity > 2*stationary_tolerance:
        _error("Weighted transformation blocks are not stationary.", "nonconvergence")
    condition = _solve(z, y, w)[1]
    intercept = 0. if categorical else response["mean"]
    beta_output = beta if categorical else beta*response["sd"]
    fitted_output = fitted if categorical else intercept+z@beta_output
    method = f"catreg_{response_scale}_response_fweight" if categorical else f"catreg_{default}_fweight"
    state = dict(version=1, kind="catreg_fweight", method=method, variables=variables, outcome=outcome,
        frequency=frequency, response_scale=response_scale, descriptors=descriptors, response=response,
        raw=raw, counts=sample["counts"], positions=sample["positions"], input_nobs=sample["input_nobs"],
        zero_positions=sample["zero_positions"], missing_positions=sample["missing_positions"],
        beta=beta_output.tolist(), intercept=intercept, objective=objective, condition=condition,
        stationarity=stationarity, response_roots=roots, controls=controls, chosen_start=chosen,
        iterations=iterations, histories=histories, starts=starts, missing=missing)
    return _output(state, z, y, fitted_output, workspace, sample["workspace"], work)


@resident_cpu
def catreg_nominal_fweight(data: Any, outcome: str, predictors: list[str], *, frequency: str,
    scales: dict | None = None, n_starts: int = 3, seed: int = 0, maxiter: int = 500,
    tol: float = 1e-8, missing: str = "drop", max_work: int = f.WORK,
    max_bytes: int = f.BYTES, device: str = "cpu") -> TableSet:
    """Count-weighted nominal predictor scaling with a numeric response."""
    return _fit(data, outcome, predictors, frequency, "numeric", "nominal", scales, None, None,
        _controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device), missing)


@resident_cpu
def catreg_ordinal_fweight(data: Any, outcome: str, predictors: list[str], *, frequency: str,
    orders: dict, scales: dict | None = None, n_starts: int = 3, seed: int = 0,
    maxiter: int = 500, tol: float = 1e-8, missing: str = "drop", max_work: int = f.WORK,
    max_bytes: int = f.BYTES, device: str = "cpu") -> TableSet:
    """Count-weighted monotone predictors with signed numeric-response effects."""
    return _fit(data, outcome, predictors, frequency, "numeric", "ordinal", scales, orders, None,
        _controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device), missing)


@resident_cpu
def catreg_nominal_response_fweight(data: Any, outcome: str, predictors: list[str], *, frequency: str,
    scales: dict | None = None, orders: dict | None = None, n_starts: int = 4,
    seed: int = 0, maxiter: int = 500, tol: float = 1e-8, missing: str = "drop",
    max_work: int = f.WORK, max_bytes: int = f.BYTES, device: str = "cpu") -> TableSet:
    """Count-weighted jointly normalized nominal response and predictor scaling."""
    return _fit(data, outcome, predictors, frequency, "nominal", "nominal", scales, orders, None,
        _controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device), missing)


@resident_cpu
def catreg_ordinal_response_fweight(data: Any, outcome: str, predictors: list[str], *, frequency: str,
    outcome_order: list, scales: dict | None = None, orders: dict | None = None,
    n_starts: int = 4, seed: int = 0, maxiter: int = 500, tol: float = 1e-8,
    missing: str = "drop", max_work: int = f.WORK, max_bytes: int = f.BYTES,
    device: str = "cpu") -> TableSet:
    """Count-weighted monotone response scaling with explicitly declared order."""
    return _fit(data, outcome, predictors, frequency, "ordinal", "nominal", scales, orders, outcome_order,
        _controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device), missing)


def _finite(value):
    return type(value) in (int, float) and abs(value) <= 1e200 and math.isfinite(value)


def _label_valid(label):
    if not isinstance(label, list) or len(label) != 2 or label[0] not in ("str", "int", "float", "bool"):
        return False
    value = label[1]
    if label[0] == "str":
        return isinstance(value, str) and len(value) <= 256
    if label[0] == "int":
        return type(value) is int and abs(value) <= 2**53-1
    if label[0] == "float":
        return type(value) is float and math.isfinite(value)
    return type(value) is bool


def _descriptor_valid(descriptor, name):
    if not isinstance(descriptor, dict) or not 4 <= len(descriptor) <= 5 or descriptor.get("name") != name:
        return False
    if descriptor.get("scale") == "numeric":
        return (set(descriptor) == {"name", "scale", "mean", "sd"}
                and _finite(descriptor["mean"]) and _finite(descriptor["sd"]) and descriptor["sd"] > 0)
    if (set(descriptor) != {"name", "scale", "levels", "counts", "quantifications"}
            or descriptor.get("scale") not in ("nominal", "ordinal")):
        return False
    levels, counts, q = descriptor["levels"], descriptor["counts"], descriptor["quantifications"]
    return (isinstance(levels, list) and 2 <= len(levels) <= 32 and all(_label_valid(v) for v in levels)
            and len({base._key(v) for v in levels}) == len(levels)
            and isinstance(counts, list) and len(counts) == len(levels)
            and all(type(v) is int and 1 <= v <= f.MAX_TOTAL for v in counts)
            and isinstance(q, list) and len(q) == len(levels) and all(_finite(v) for v in q))


def _state_shape(state):
    """Refuse nested/extra/oversized state before sealing or tensor creation."""
    if (not isinstance(state, dict) or len(state) != len(STATE_FIELDS) or set(state) != STATE_FIELDS or type(state["version"]) is not int
            or state["version"] != 1 or type(state["kind"]) is not str or state["kind"] != "catreg_fweight"
            or type(state["method"]) is not str or state["method"] not in METHODS):
        _error("Unknown closed frequency CATREG state schema.", "invalid_state")
    try:
        variables = _names(state["variables"], state["outcome"], state["frequency"])
        p = len(variables)
        controls = state["controls"]
        if not isinstance(controls, dict) or len(controls) != 6 or set(controls) != {"n_starts", "seed", "maxiter", "tol", "max_work", "max_bytes"}:
            raise ValueError("Controls")
        _controls(**controls, device="cpu")
        counts = state["counts"]
        if not isinstance(counts, list) or not p+2 < len(counts) <= f.MAX_ROWS:
            raise ValueError("Physical rows")
        n = len(counts)
        if any(type(v) is not int or not 1 <= v <= f.MAX_TOTAL for v in counts) or sum(counts) > f.MAX_TOTAL:
            raise ValueError("Counts")
        if type(state["input_nobs"]) is not int or not n <= state["input_nobs"] <= f.MAX_ROWS:
            raise ValueError("Input rows")
        positions = []
        for key in ("positions", "zero_positions", "missing_positions"):
            row = state[key]
            if (not isinstance(row, list) or len(row) > state["input_nobs"]
                    or any(type(v) is not int or not 0 <= v < state["input_nobs"] for v in row)
                    or row != sorted(set(row))):
                raise ValueError("Physical sample")
            positions += row
        if len(state["positions"]) != n or sorted(positions) != list(range(state["input_nobs"])):
            raise ValueError("Physical partition")
        if state["missing"] not in ("drop", "raise") or (state["missing"] == "raise" and state["missing_positions"]):
            raise ValueError("Missing policy")
        if state["response_scale"] not in ("numeric", "nominal", "ordinal"):
            raise ValueError("Response scale")
        expected = {"catreg_nominal_fweight", "catreg_ordinal_fweight"} if state["response_scale"] == "numeric" else {f"catreg_{state['response_scale']}_response_fweight"}
        if state["method"] not in expected:
            raise ValueError("Method")
        descriptors = state["descriptors"]
        if (not isinstance(descriptors, list) or len(descriptors) != p
                or not all(_descriptor_valid(d, name) for d, name in zip(descriptors, variables))
                or not _descriptor_valid(state["response"], state["outcome"])
                or state["response"]["scale"] != state["response_scale"]):
            raise ValueError("Descriptors")
        if (state["method"] == "catreg_nominal_fweight"
                and any(d["scale"] == "ordinal" for d in descriptors)):
            raise ValueError("Nominal predictor domain")
        raw = state["raw"]
        if (not isinstance(raw, list) or len(raw) != n or any(not isinstance(row, list) or len(row) != p+1
                or any(not _label_valid(v) for v in row) for row in raw)):
            raise ValueError("Raw typed sample")
        for j, d in enumerate([state["response"]]+descriptors):
            if d["scale"] == "numeric" and any(row[j][0] != "float" or abs(row[j][1]) > 1e100 for row in raw):
                raise ValueError("Numeric raw type")
        if (not isinstance(state["beta"], list) or len(state["beta"]) != p
                or not all(_finite(v) for v in state["beta"])
                or any(not _finite(state[k]) for k in ("intercept", "objective", "condition", "stationarity"))
                or not 0 <= state["objective"] <= 1+1e-8 or not 1 <= state["condition"] <= 1e10
                or not 0 <= state["stationarity"] <= 1):
            raise ValueError("Scalar results")
        roots = state["response_roots"]
        expected_roots = 0 if state["response_scale"] == "numeric" else len(state["response"]["levels"])
        if (not isinstance(roots, list) or len(roots) != expected_roots
                or any(not _finite(v) or not -1e-8 <= v <= 1+1e-8 for v in roots)):
            raise ValueError("Response roots")
        if (type(state["chosen_start"]) is not int or not 0 <= state["chosen_start"] < controls["n_starts"]
                or type(state["iterations"]) is not int or not 1 <= state["iterations"] <= controls["maxiter"]):
            raise ValueError("Selected start")
        starts, histories = state["starts"], state["histories"]
        if (not isinstance(starts, list) or len(starts) != controls["n_starts"]
                or any(not isinstance(row, list) or len(row) != 5 or type(row[0]) is not int or row[0] != i
                or type(row[1]) is not bool or type(row[2]) is not int or not 0 <= row[2] <= controls["maxiter"]
                or (row[3] is not None and not _finite(row[3])) or not isinstance(row[4], str) or len(row[4]) > 128
                for i, row in enumerate(starts))):
            raise ValueError("Start ledger")
        if (not isinstance(histories, list) or not 1 <= len(histories) <= controls["n_starts"]*(controls["maxiter"]+1)
                or any(not isinstance(row, list) or len(row) != 6 or not all(_finite(v) for v in row)
                or type(row[0]) is not int or not 0 <= row[0] < controls["n_starts"]
                or type(row[1]) is not int or not 0 <= row[1] <= controls["maxiter"] for row in histories)):
            raise ValueError("Iteration trace")
    except (KeyError, TypeError, ValueError, AnalysisError) as exc:
        raise AnalysisError("invalid_state", "Invalid bounded frequency CATREG primitive state.") from exc
    return state


ATTR_FIELDS = {
    "method", "variables", "outcome", "response_scale", "frequency", "n", "frequency_n", "n_input",
    "n_zero", "n_missing", "sample_positions", "missing", "settings", "frequency_state", "state_sha256",
    "resources", "input_preparation", "declared_work", "chosen_start", "iterations", "converged",
    "dtype", "device", "weight_type", "sources", "solution", "inference",
}
TABLE_FIELDS = {"coefficients", "fit", "fitted", "transformed", "iterations", "starts", "sample", "quantifications", "numeric_scaling", "settings"}


def _resource_shape(value):
    if (not isinstance(value, dict) or len(value) != 5 or set(value) != {"operation", "estimated_workspace_bytes", "budget_bytes", "buffers", "scope"}
            or any(not isinstance(value[k], str) or len(value[k]) > 256 for k in ("operation", "scope"))
            or any(type(value[k]) is not int or not 1 <= value[k] <= f.BYTES for k in ("estimated_workspace_bytes", "budget_bytes"))
            or not isinstance(value["buffers"], dict) or not 1 <= len(value["buffers"]) <= 4
            or any(not isinstance(k, str) or len(k) > 128 or type(v) is not int or not 0 <= v <= f.BYTES for k, v in value["buffers"].items())
            or sum(value["buffers"].values()) != value["estimated_workspace_bytes"]
            or value["estimated_workspace_bytes"] > value["budget_bytes"]):
        _error("Saved workspace metadata is outside its bounded schema.", "invalid_state")


def _attrs_shape(attrs):
    if not isinstance(attrs, dict) or len(attrs) != len(ATTR_FIELDS) or set(attrs) != ATTR_FIELDS:
        _error("Saved metadata contains unknown fields.", "invalid_state")
    state = _state_shape(attrs["frequency_state"])
    _resource_shape(attrs["resources"])
    _resource_shape(attrs["input_preparation"])
    expected = {
        "method": state["method"], "variables": state["variables"], "outcome": state["outcome"],
        "response_scale": state["response_scale"], "frequency": state["frequency"], "n": len(state["counts"]),
        "frequency_n": sum(state["counts"]), "n_input": state["input_nobs"], "n_zero": len(state["zero_positions"]),
        "n_missing": len(state["missing_positions"]), "sample_positions": state["positions"], "missing": state["missing"],
        "settings": state["controls"], "chosen_start": state["chosen_start"], "iterations": state["iterations"],
        "converged": True, "dtype": "float64", "device": "cpu", "weight_type": "frequency", "sources": SOURCES,
        "solution": "best stationary converged deterministic declared start; local solution only",
        "inference": "descriptive adaptive quantification; no ordinary covariance, SE, p-value, CI or calibrated class probabilities",
    }
    if any(not _equal_primitive(attrs[k], v) for k, v in expected.items()):
        _error("Saved metadata disagrees with its frequency calibration.", "invalid_state")
    if (not isinstance(attrs["state_sha256"], str) or len(attrs["state_sha256"]) != 64
            or type(attrs["declared_work"]) is not int or not 1 <= attrs["declared_work"] <= f.WORK):
        _error("Saved work/checksum is malformed.", "invalid_state")
    return state


def _equal_primitive(actual, expected):
    """Closed metadata equality without arbitrary array/object coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_equal_primitive(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, dict):
        return len(actual) == len(expected) and set(actual) == set(expected) and all(
            _equal_primitive(actual[k], v) for k, v in expected.items())
    return actual == expected


def _table_cells(cells, rows_limit, columns_limit):
    for i, row in enumerate(cells):
        if i >= rows_limit:
            _error("Saved table rows exceed the declared domain.", "invalid_state")
        if not isinstance(row, (list, tuple)) or len(row) > columns_limit or any(type(v) not in (str, int, float, bool, type(None))
                or (isinstance(v, str) and len(v) > 1024)
                or (type(v) is int and abs(v) > 2**53-1)
                or (type(v) is float and math.isinf(v)) for v in row):
            _error("Saved cells must be bounded finite primitives.", "invalid_state")


def _read_result(result, max_bytes, max_work):
    decode_work = 0
    if isinstance(result, str):
        if len(result) > 32*1024**2:
            _error("Supply complete summary JSON within its portable domain.", "invalid_state")
        decode_work = 4*len(result)
        f._admit("frequency CATREG JSON decode", 12*len(result)+4096, decode_work, max_bytes, max_work)
        try:
            payload = json.loads(result)
            if set(payload) != {"schema", "title", "attrs", "tables"} or payload["schema"] != "openecon.summary.v1":
                raise ValueError("Envelope")
            state = _attrs_shape(payload["attrs"])
            if not isinstance(payload["tables"], dict) or set(payload["tables"]) != TABLE_FIELDS:
                raise ValueError("Table names")
            for name, frame in payload["tables"].items():
                if (not isinstance(frame, dict) or set(frame) != {"columns", "index", "data", "index_names", "column_names"}
                        or not isinstance(frame["data"], list) or not isinstance(frame["columns"], list)
                        or not isinstance(frame["index"], list) or len(frame["index"]) != len(frame["data"])
                        or any(not isinstance(v, str) or len(v) > 128 for v in frame["columns"])
                        or any(type(v) not in (int, float) or not _finite(v) for v in frame["index"])
                        or frame["index_names"] != [None] or frame["column_names"] != [None]):
                    raise ValueError("Table schema")
                _table_cells(frame["data"], max(4096, state["controls"]["n_starts"]*(state["controls"]["maxiter"]+1)), 32)
                if any(not isinstance(row, list) or len(row) != len(frame["columns"]) for row in frame["data"]):
                    raise ValueError("Cell shape")
        except (TypeError, KeyError, ValueError) as exc:
            raise AnalysisError("invalid_state", "Invalid complete frequency CATREG summary envelope.") from exc
        result = restore_summary(result)
    if (not isinstance(result, TableSet) or type(result.title) is not str
            or result.title != "Frequency-weighted categorical regression" or set(result) != TABLE_FIELDS):
        _error("Supply a complete fitted/restored frequency CATREG summary.", "invalid_state")
    state = _attrs_shape(result.attrs)
    for frame in result.values():
        if (not isinstance(frame, pd.DataFrame)
                or len(frame) > max(4096, state["controls"]["n_starts"]*(state["controls"]["maxiter"]+1))
                or len(frame.columns) > 32 or any(not isinstance(v, str) or len(v) > 128 for v in frame.columns)):
            _error("Saved column metadata are malformed.", "invalid_state")
        if list(frame.index) != list(range(len(frame))) or list(frame.index.names) != [None] or list(frame.columns.names) != [None]:
            _error("Saved table indexes must be canonical.", "invalid_state")
        _table_cells(frame.itertuples(index=False, name=None), max(4096, state["controls"]["n_starts"]*(state["controls"]["maxiter"]+1)), 32)
    return result, state, decode_work


def _checked(result, max_bytes, max_work):
    result, state, decode_work = _read_result(result, max_bytes, max_work)
    n, p = len(state["counts"]), len(state["variables"])
    bound = _bound(state["raw"], n, p, state["controls"])
    work = decode_work+4*bound+64*n*p*p+64*max([len(d.get("levels", [])) for d in [state["response"]]+state["descriptors"]])**3+16*len(state["histories"])
    plan = f._admit("frequency CATREG semantic saved validation", 4*bound+8*n*(24*p+64), work, max_bytes, max_work)
    input_size = 256*state["input_nobs"]*(p+3)
    fit_size = input_size+8*n*(48*p+96)+128*p*p+2*bound
    for key, operation, size in (("input_preparation", "frequency CATREG input", input_size),
                                 ("resources", "frequency CATREG fit and full portable state", fit_size)):
        resource = result.attrs[key]
        if (resource["operation"] != operation or resource["estimated_workspace_bytes"] != size
                or resource["buffers"] != {"physical_rows_and_numerical_workspace": size}
                or resource["scope"] != plan["scope"] or resource["budget_bytes"] > state["controls"]["max_bytes"]):
            _error("Saved admission record disagrees with its bounded fit dimensions.", "invalid_state")
    if result.attrs["state_sha256"] != f._seal(state):
        _error("Frequency calibration checksum disagrees.", "invalid_state")
    w = f._weights(state["counts"])
    descriptions = [state["response"]]+state["descriptors"]
    prepared = []
    for j, descriptor in enumerate(descriptions):
        if descriptor["scale"] == "numeric":
            raw = torch.tensor([row[j][1] for row in state["raw"]], dtype=f.DT)
            standardized, mean, sd = f._standardize(raw, w)
            if abs(mean-descriptor["mean"]) > 1e-9*max(1., abs(mean)) or abs(sd-descriptor["sd"]) > 1e-9*sd:
                _error("Saved numeric scaling disagrees with weighted raw moments.", "invalid_state")
            prepared.append(standardized)
        else:
            mapping = {base._key(label): k for k, label in enumerate(descriptor["levels"])}
            try:
                codes = torch.tensor([mapping[base._key(row[j])] for row in state["raw"]], dtype=torch.int64)
            except KeyError as exc:
                raise AnalysisError("invalid_state", "Saved raw category lies outside its map.") from exc
            counts = torch.bincount(codes, weights=w, minlength=len(mapping))
            if counts.tolist() != descriptor["counts"]:
                _error("Saved category totals disagree with frequencies and raw membership.", "invalid_state")
            q = torch.tensor(descriptor["quantifications"], dtype=f.DT)
            if (abs(float((q*counts).sum()/w.sum())) > 1e-8
                    or abs(float((q.square()*counts).sum()/w.sum())-1) > 1e-8
                    or (descriptor["scale"] == "ordinal" and bool((torch.diff(q) < -1e-9).any()))):
                _error("Saved category map violates weighted normalization or order.", "invalid_state")
            prepared.append(codes)
    z = _map([row[1:] for row in state["raw"]], state["descriptors"])
    categorical = state["response_scale"] != "numeric"
    y = _map([[row[0]] for row in state["raw"]], [state["response"]])[:, 0]
    beta = torch.tensor(state["beta"], dtype=f.DT)
    standardized_beta = beta if categorical else beta/state["response"]["sd"]
    solved, condition = _solve(z, y, w)
    tolerance = max(1e-9, math.sqrt(state["controls"]["tol"])*.1)
    expected_intercept = 0. if categorical else state["response"]["mean"]
    fitted = z@standardized_beta
    objective = _rms(y-fitted, w)**2
    if (not torch.allclose(standardized_beta, solved, atol=2e-8, rtol=2e-8)
            or abs(state["intercept"]-expected_intercept) > 1e-10*max(1., abs(expected_intercept))
            or abs(state["condition"]-condition) > 1e-7*condition
            or abs(state["objective"]-objective) > 2e-9):
        _error("Saved coefficients/objective do not solve weighted raw-data regression.", "invalid_state")
    check_z = z.clone()
    check_beta = _sweep(y, check_z, standardized_beta.clone(), state["descriptors"], prepared[1:], w)
    stationarity = _rms(check_z@check_beta-fitted, w)
    if categorical:
        response = state["response"]
        next_y = _response_block(fitted, response, prepared[0], w)[1]
        stationarity = max(stationarity, _rms(next_y-y, w))
        roots = _response_roots(z, prepared[0], response["counts"], w)
        if not torch.allclose(roots, torch.tensor(state["response_roots"], dtype=f.DT), atol=2e-8, rtol=2e-8):
            _error("Saved conditional response roots disagree with calibration.", "invalid_state")
        if response["scale"] == "nominal" and (float(roots[-1]-roots[-2]) <= 1e-8*max(1., float(roots[-1]))
                or float(roots[-1])-(1-objective) > max(1e-7, 10*state["controls"]["tol"])):
            _error("Saved nominal response lacks an identified leading contrast.", "invalid_state")
    if stationarity > 2*tolerance or abs(stationarity-state["stationarity"]) > 2e-8:
        _error("Saved transformation blocks fail weighted stationarity.", "invalid_state")
    accepted = [row for row in state["starts"] if row[1] and row[4] == "accepted"]
    if not accepted or min(accepted, key=lambda row: (row[3], row[0]))[0] != state["chosen_start"]:
        _error("Saved selected start disagrees with the converged ledger.", "invalid_state")
    for start in state["starts"]:
        trace = [row for row in state["histories"] if row[0] == start[0]]
        if trace and (trace[0][1] != 0 or [row[1] for row in trace] != list(range(len(trace)))
                or trace[0][3:] != [0., 0., 0.]
                or any(b[2] > a[2]+1e-9 or abs((a[2]-b[2])-b[3]) > 2e-9
                       for a, b in zip(trace, trace[1:]))
                or any(not 0 <= row[2] <= 1+1e-8 or row[4] < 0 or row[5] < 0 for row in trace)):
            _error("Saved objective traces are incomplete or increasing.", "invalid_state")
        if start[1] and (not trace or start[2] != trace[-1][1] or abs(start[3]-trace[-1][2]) > 1e-10):
            _error("Saved accepted start disagrees with its trace.", "invalid_state")
        if start[1] and (start[4] != "accepted" or len(trace) < 2
                or abs(trace[-1][3]) > state["controls"]["tol"]*max(1., trace[-2][2])
                or max(trace[-1][4:]) > tolerance):
            _error("Saved accepted start fails its declared convergence controls.", "invalid_state")
    chosen = state["starts"][state["chosen_start"]]
    if chosen[2] != state["iterations"] or abs(chosen[3]-state["objective"]) > 2e-9:
        _error("Saved optimum disagrees with its selected trace.", "invalid_state")
    declared = _work(n, p, [len(d.get("levels", [])) for d in descriptions], state["controls"])+8*state["input_nobs"]*(p+2)+4*bound
    if declared != result.attrs["declared_work"]:
        _error("Saved fit work does not match its declared dimensions.", "invalid_state")
    output_fitted = fitted if categorical else state["intercept"]+z@beta
    expected = _output(state, z, y, output_fitted, result.attrs["resources"], result.attrs["input_preparation"], declared)
    if summary_state(expected) != summary_state(result):
        _error("Saved complete tables/metadata disagree with their verified calibration.", "invalid_state")
    return state, plan, work


@resident_cpu
def catreg_fweight_predict(result: TableSet | str, data: Any, *, missing: str = "raise",
    max_work: int = f.WORK, max_bytes: int = f.BYTES, device: str = "cpu") -> TableSet:
    """Apply verified weighted calibration; query rows require no count column.

    Numeric outcomes retain their original units. Learned categorical outcomes
    yield descriptive quantified scores, never class probabilities or labels.
    """
    if device != "cpu":
        _error("Saved frequency prediction supports CPU only.", "unsupported_option")
    state, validation, validation_work = _checked(result, max_bytes, max_work)
    names = state["variables"]
    if not isinstance(data, pd.DataFrame) or not 1 <= len(data) <= f.MAX_ROWS or missing not in ("drop", "raise"):
        _error("Queries require 1–3000 resident DataFrame rows and missing='drop'/'raise'.")
    if any(list(data.columns).count(name) != 1 for name in names):
        _error("Every calibrated predictor must occur exactly once in query data.")
    n_input, p = len(data), len(names)
    query_work = validation_work+64*n_input*p*p
    f._admit("saved frequency CATREG query selection", validation["estimated_workspace_bytes"]+256*n_input*(p+6), query_work, max_bytes, max_work)
    selected = data.loc[:, names]
    absent = selected.isna().any(axis=1)
    if bool(absent.any()) and missing == "raise":
        _error("Query predictors contain missing values.", "missing_data")
    positions = [i for i, keep in enumerate((~absent).tolist()) if keep]
    if not positions:
        _error("No complete query rows remain.", "insufficient_sample")
    frame = selected.loc[~absent].reset_index(drop=True)
    scales = {d["name"]: d["scale"] for d in state["descriptors"]}
    raw = _raw(frame, names, scales)
    query_bound = _bound(raw, len(raw), p, dict(state["controls"], n_starts=1, maxiter=1))
    total_bound = query_bound+_bound(state["raw"], len(state["raw"]), p, state["controls"])
    if total_bound > 32*1024**2:
        _error("Complete prediction calibration/query state exceeds the portable domain.", "resource_limit")
    workspace = f._admit("saved frequency CATREG query and calibration", validation["estimated_workspace_bytes"]+2*query_bound+8*n_input*(p+6), query_work, max_bytes, max_work)
    z = _map(raw, state["descriptors"])
    prediction = state["intercept"]+z@torch.tensor(state["beta"], dtype=f.DT)
    categorical = state["response_scale"] != "numeric"
    column = "quantified_fitted" if categorical else "fitted"
    output = f._save(TableSet({
        "predictions": table([[position, value] for position, value in zip(positions, prediction.tolist())], columns=["source_position", column]),
        "transformed": table([[position]+row for position, row in zip(positions, z.tolist())], columns=["source_position"]+names),
    }, title="Saved frequency CATREG predictions", method="catreg_fweight_predict",
        outcome=state["outcome"], response_scale=state["response_scale"], calibration_state=state,
        state_sha256=f._seal(state), query_raw=raw, n=len(positions), n_input=n_input,
        sample_positions=positions, n_missing=n_input-len(positions), resources=workspace,
        dtype="float64", device="cpu", weight_type="frequency calibration, unweighted query rows",
        inference="descriptive quantified response; no probability, inverse label or inference" if categorical else "descriptive original-unit predictions; no ordinary uncertainty after learned quantification"))
    summary_state(output)
    return output
