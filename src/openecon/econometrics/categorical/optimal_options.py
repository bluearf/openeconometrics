"""Categorical-response scaling and paired full-refit prediction resampling."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import json
import math
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table, _json_safe
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.resources import plan_workspace, workspace_budget_bytes

from . import optimal as base

DTYPE = torch.float64
DEFAULT_WORK = base.DEFAULT_WORK
DEFAULT_BYTES = base.DEFAULT_BYTES
SOURCES = [*base.SOURCES,
    "https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=catreg-analysis-subcommand-command",
    "https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=catreg-define-scale-in-categorical-regression",
]
_RESPONSE_FIELDS = frozenset({
    "version", "kind", "variables", "outcome", "response", "descriptors", "intercept", "beta",
    "training_quantified", "training_raw_predictors", "training_response_codes", "training_response",
    "sample_positions", "n_input", "objective", "stationarity_tolerance", "objective_tolerance",
    "conditional_response_roots",
})


def _finite_scalar(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _valid_label(label):
    if not isinstance(label, list) or len(label) != 2 or type(label[0]) is not str:
        return False
    kind, value = label
    if kind == "str":
        return type(value) is str and len(value) <= 256
    if kind == "bool":
        return type(value) is bool
    if kind == "int":
        return type(value) is int and _finite_scalar(value)
    if kind == "float":
        return type(value) is float and _finite_scalar(value)
    return False


def _size(data, names, maximum=3000):
    """Inspect only required-column lengths, before materializing owned input."""
    if isinstance(data, pd.DataFrame):
        n = len(data)
    elif isinstance(data, Mapping):
        if any(name not in data for name in names):
            base._error("Required columns must occur exactly once.")
        try:
            n = max(len(data[name]) for name in names)
        except (TypeError, ValueError):
            base._error("Required columns must be resident finite sequences.")
    elif isinstance(data, list):
        n = len(data)
    else:
        base._error("Use a resident DataFrame, column mapping or records; Dataset is unsupported.", "unsupported_data")
    if not 1 <= n <= maximum:
        base._error(f"Input admits 1–{maximum} rows before coercion.", "resource_limit")
    return n


def _bounded_orders(orders):
    if orders is None:
        return
    if not isinstance(orders, dict) or len(orders) > 12:
        base._error("orders must be a bounded dictionary of explicitly ordered variables.")
    for order in orders.values():
        if not isinstance(order, (list, tuple)) or not 2 <= len(order) <= 32:
            base._error("Each declared order must contain 2–32 categories.")
        if any(isinstance(value, str) and len(value) > 256 for value in order):
            base._error("Declared categorical labels admit at most 256 characters.", "resource_limit")


def _predictor_sweep(y, z, beta, descriptors, prepared):
    fitted = z @ beta
    for j, (descriptor, values) in enumerate(zip(descriptors, prepared)):
        residual = y - fitted + beta[j]*z[:, j]
        if descriptor["scale"] == "numeric":
            beta[j] = torch.dot(values, residual)/len(y)
            z[:, j] = values
            contribution = beta[j]*values
        else:
            means, counts = base._means(residual, values, len(descriptor["levels"]))
            if descriptor["scale"] == "ordinal":
                inc = base._pava(means, counts)
                dec = -base._pava(-means, counts)
                inc_loss = float(((means-inc).square()*counts).sum())
                dec_loss = float(((means-dec).square()*counts).sum())
                effects = inc if inc_loss <= dec_loss else dec
            else:
                effects = means
            effects = effects - (effects*counts).sum()/counts.sum()
            contribution = effects[values]
            q = base._normalize_categories(effects, counts)
            if descriptor["scale"] == "ordinal":
                if float(q[-1]) < float(q[0]):
                    q = -q
            elif float(q[int(torch.argmax(q.abs()))]) < 0:
                q = -q
            z[:, j] = q[values]
            beta[j] = torch.dot(z[:, j], contribution)/len(y)
        fitted = y-residual+contribution
    base._rank(z)
    return torch.linalg.lstsq(z, y).solution


def _response_block(fitted, descriptor, codes):
    means, counts = base._means(fitted, codes, len(descriptor["levels"]))
    if descriptor["scale"] == "ordinal":
        means = base._pava(means, counts)
    # Do not flip a nominal response alone: its sign and all regression effects
    # must change together. Canonical global orientation is applied at the end.
    q = base._normalize_categories(means, counts)
    return q, q[codes]


def _response_roots(z, codes, counts):
    """Conditional response contrast roots, without an n-by-n projector."""
    u, _, _ = torch.linalg.svd(z, full_matrices=False)
    d = torch.tensor(counts, dtype=DTYPE)
    aggregate = torch.zeros((len(d), z.shape[1]), dtype=DTYPE).index_add_(0, codes, u)
    aggregate = aggregate/d.sqrt()[:, None]
    constant = torch.sqrt(d/d.sum())
    centered = aggregate-constant[:, None]*(constant@aggregate)[None, :]
    matrix = centered@centered.T
    return torch.linalg.eigvalsh((matrix+matrix.T)/2)


def _response_fit(data, outcome, predictors, response_scale, outcome_order,
                  scales, orders, controls, missing):
    variables = base._names(predictors)
    if len(variables) > 11:
        base._error("Categorical-response fits admit at most 11 predictors plus the response.", "resource_limit")
    if not isinstance(outcome, str) or not outcome or outcome in variables:
        base._error("Supply a distinct categorical outcome column.")
    if any(len(name) > 256 for name in [outcome]+variables):
        base._error("Column names admit at most 256 characters.", "resource_limit")
    n_before = _size(data, [outcome]+variables)
    p = len(variables)
    combined_plan = plan_workspace("categorical-response CATREG input and numerical peak", {
        "named_input_selection_masks": 64*n_before*(p+2),
        "design_response_updates_and_solver": 8*n_before*(32*(p+1)+64)+128*(p+1)**2,
        "category_maps_and_response_roots": 8*32*(p+1)*(32+2*(p+1))+8*32*32*8,
        "all_declared_start_traces": 8*controls["n_starts"]*(controls["maxiter"]+1)*8,
    }, budget_bytes=min(controls["max_bytes"], workspace_budget_bytes())).record()
    frame, positions, n_input, input_plan = base._frame(data, [outcome]+variables, missing, controls["max_bytes"])
    if len(frame) <= len(variables)+2:
        base._error("More complete rows than predictors + 2 are required.", "insufficient_sample")
    if scales is None:
        scales = {name: "nominal" for name in variables}
    if not isinstance(scales, dict) or set(scales) != set(variables):
        base._error("scales must name exactly the predictors.")
    if orders is None:
        orders = {}
    if not isinstance(orders, dict) or outcome in orders:
        base._error("orders names predictor columns; outcome_order separately declares the response.")
    if response_scale == "ordinal" and not isinstance(outcome_order, (list, tuple)):
        base._error("Ordinal outcomes require an explicit outcome_order.")
    all_scales = {outcome: response_scale, **scales}
    all_orders = ({outcome: outcome_order} if response_scale == "ordinal" else {}) | orders
    _bounded_orders(all_orders)
    if any(not _valid_label(base._label(value)) for name in [outcome]+variables
           if all_scales[name] != "numeric" for value in frame[name].tolist()):
        base._error("Categorical response options admit finite primitive labels up to 256 characters.", "resource_limit")
    descriptions, prepared, plan, work = base._prepare(
        frame, [outcome]+variables, all_scales, all_orders, "nominal",
        dict(controls, max_work=controls["max_work"]-64*32**3))
    response, descriptors = descriptions[0], descriptions[1:]
    response_codes, predictor_data = prepared[0], prepared[1:]
    rng = torch.Generator(device="cpu").manual_seed(controls["seed"])
    histories, starts, candidates = [], [], []
    stationary_tolerance = max(1e-9, math.sqrt(controls["tol"])*0.1)
    for start in range(controls["n_starts"]):
        objective, iteration = None, 0
        try:
            initial = base._initial(descriptions, prepared, start, rng)
            y, z = initial[:, 0].clone(), initial[:, 1:].clone()
            base._rank(z)
            beta = torch.linalg.lstsq(z, y).solution
            objective = float((y-z@beta).square().mean())
            histories.append([start, 0, objective, 0.0, 0.0, 0.0])
            converged = False
            for iteration in range(1, controls["maxiter"]+1):
                old, old_y, old_fitted = objective, y.clone(), z@beta
                _, y = _response_block(old_fitted, response, response_codes)
                beta = torch.linalg.lstsq(z, y).solution
                beta = _predictor_sweep(y, z, beta, descriptors, predictor_data)
                fitted = z@beta
                objective = float((y-fitted).square().mean())
                improvement = old-objective
                y_change = float(torch.sqrt((y-old_y).square().mean()))
                fit_change = float(torch.sqrt((fitted-old_fitted).square().mean()))
                histories.append([start, iteration, objective, improvement, y_change, fit_change])
                if improvement < -1e-9:
                    base._error("Response/predictor ALS increased its actual normalized objective.", "numerical_failure")
                if (abs(improvement) <= controls["tol"]*max(1.0, old)
                        and max(y_change, fit_change) <= stationary_tolerance):
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
        codes = {row[-1] for row in starts}
        code = next(iter(codes)) if len(codes) == 1 and "nonconvergence" not in codes else "nonconvergence"
        base._error("No declared start reached a nondegenerate, full-rank stationary response solution: " + ", ".join(sorted(codes)) + ".", code)
    objective, chosen, y, z, beta, iterations = min(candidates, key=lambda row: row[:2])
    response = copy.deepcopy(response)
    descriptors = copy.deepcopy(descriptors)
    response_q, _ = base._means(y, response_codes, len(response["levels"]))
    if response_scale == "nominal" and float(response_q[int(torch.argmax(response_q.abs()))]) < 0:
        response_q, y, beta = -response_q, -y, -beta
    response["quantifications"] = response_q.tolist()
    base._finish_maps(descriptors, predictor_data, z)
    fitted = z@beta
    _, next_y = _response_block(fitted, response, response_codes)
    block_stationarity = float(torch.sqrt((next_y-y).square().mean()))
    if block_stationarity > 2*stationary_tolerance:
        base._error("Response map did not reach block stationarity.", "nonconvergence")
    roots = _response_roots(z, response_codes, response["counts"])
    root_gap = float(roots[-1]-roots[-2])
    if response_scale == "nominal":
        if root_gap <= 1e-8*max(1.0, float(roots[-1])):
            base._error("The conditional leading response roots tie; a unique nominal contrast is unidentified.", "unidentified_response")
        if float(roots[-1])-(1-objective) > max(1e-7, 10*controls["tol"]):
            base._error("Nominal response failed its conditional leading-root optimum check.", "nonconvergence")
    state = {
        "version": 1, "kind": "catreg_response", "variables": variables,
        "outcome": outcome, "response": response, "descriptors": descriptors,
        "intercept": 0.0, "beta": beta.tolist(), "training_quantified": z.tolist(),
        "training_raw_predictors": [[base._label(value) for value in row] for row in frame[variables].itertuples(index=False, name=None)],
        "training_response_codes": response_codes.tolist(), "training_response": y.tolist(),
        "sample_positions": positions, "n_input": n_input, "objective": objective,
        "stationarity_tolerance": stationary_tolerance,
        "objective_tolerance": controls["tol"],
        "conditional_response_roots": roots.tolist(),
    }
    result = TableSet({
        "coefficients": table([["_cons", 0.0]] + [[name, value] for name, value in zip(variables, beta.tolist())], columns=["term", "beta_quantified_response"]),
        "fit": table([[len(frame), float((y-fitted).square().sum()), 1-objective, objective,
                       base._rank(z), block_stationarity]], columns=["n", "quantified_sse", "r_squared", "objective", "design_condition", "response_block_stationarity"]),
        "fitted": table([[position, base._label(value)[0], base._label(value)[1], observed, prediction, observed-prediction]
                         for position, value, observed, prediction in zip(positions, frame[outcome].tolist(), y.tolist(), fitted.tolist())],
                        columns=["source_position", "outcome_type", "outcome_category", "quantified_observed", "quantified_fitted", "quantified_residual"]),
        "transformed": table([[position]+row for position, row in zip(positions, z.tolist())], columns=["source_position"]+variables),
        "iterations": table(histories, columns=["start", "iteration", "objective", "improvement", "response_change", "fitted_change"]),
        "starts": table(starts, columns=["start", "converged", "iterations", "objective", "status"]),
        **base._mapping_tables([response]+descriptors),
    }, title="CATREG with jointly quantified categorical response", method=f"catreg_{response_scale}_response",
       outcome=outcome, response_scale=response_scale, variables=variables, n=len(frame), n_input=n_input,
       sample_positions=positions, n_missing=n_input-len(frame), missing=missing, settings=controls,
       chosen_start=chosen, iterations=iterations, converged=True, response_state=state,
       state_sha256=base._seal(state), resources=combined_plan, input_preparation=input_plan, fit_preparation=plan,
       declared_work=work+64*len(response["levels"])**3,
       conditional_response_roots=roots.tolist(), conditional_leading_root_gap=root_gap,
       sources=SOURCES, dtype="float64", device="cpu", weight_type="unweighted",
       solution="best stationary converged declared start; local solution, nominal sign fixed by largest absolute response map entry; ties may remain unidentified",
       inference="descriptive learned-response score; no ordinary coefficient covariance/SE/p/CI, class probabilities or unique inverse categorical prediction")
    return base._saved(result)


@resident_cpu
def catreg_nominal_response(data: Any, outcome: str, predictors: list[str], *, scales: dict | None = None,
                            orders: dict | None = None, n_starts: int = 4, seed: int = 0,
                            maxiter: int = 500, tol: float = 1e-8, missing: str = "drop",
                            max_work: int = DEFAULT_WORK, max_bytes: int = DEFAULT_BYTES,
                            device: str = "cpu", weights: str | None = None) -> TableSet:
    """Joint nominal response/predictor scaling with normalized response scores."""
    controls = base._controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device, weights)
    return _response_fit(data, outcome, predictors, "nominal", None, scales, orders, controls, missing)


@resident_cpu
def catreg_ordinal_response(data: Any, outcome: str, predictors: list[str], *, outcome_order: list,
                            scales: dict | None = None, orders: dict | None = None,
                            n_starts: int = 4, seed: int = 0, maxiter: int = 500,
                            tol: float = 1e-8, missing: str = "drop", max_work: int = DEFAULT_WORK,
                            max_bytes: int = DEFAULT_BYTES, device: str = "cpu",
                            weights: str | None = None) -> TableSet:
    """Joint CATREG with declared monotone outcome order and pooled level ties."""
    controls = base._controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device, weights)
    return _response_fit(data, outcome, predictors, "ordinal", outcome_order, scales, orders, controls, missing)


def _response_state(result, max_work, max_bytes):
    if not isinstance(result, TableSet) or result.attrs.get("method") not in ("catreg_nominal_response", "catreg_ordinal_response"):
        base._error("Supply a fitted or restored categorical-response CATREG result.", "invalid_result")
    state = result.attrs.get("response_state")
    try:
        if (not isinstance(state, dict) or len(state) != len(_RESPONSE_FIELDS) or set(state) != _RESPONSE_FIELDS
                or type(state["version"]) is not int or state["version"] != 1 or state["kind"] != "catreg_response"):
            raise ValueError("State version")
        names = base._names(state["variables"])
        if (len(names) > 11 or not isinstance(state["outcome"], str) or not state["outcome"]
                or any(len(name) > 256 for name in names+[state["outcome"]])):
            raise ValueError("Response/predictor limit")
        p, n = len(names), len(state["sample_positions"])
        if not p+2 < n <= 3000 or len(state["training_quantified"]) != n or len(state["training_response"]) != n or len(state["training_response_codes"]) != n:
            raise ValueError("Training shape")
        if (len(state["training_raw_predictors"]) != n
                or any(not isinstance(row, list) or len(row) != p for row in state["training_quantified"])
                or any(not isinstance(row, list) or len(row) != p for row in state["training_raw_predictors"])):
            raise ValueError("Training shape")
        if state["intercept"] != 0.0 or len(state["beta"]) != p or state["outcome"] in names:
            raise ValueError("Regression shape")
        if (isinstance(state["n_input"], bool) or not isinstance(state["n_input"], int)
                or not n <= state["n_input"] <= 3000
                or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < state["n_input"] for i in state["sample_positions"])
                or state["sample_positions"] != sorted(set(state["sample_positions"]))):
            raise ValueError("Sample positions")
        descriptors = state["descriptors"]
        if len(descriptors) != p or [d["name"] for d in descriptors] != names:
            raise ValueError("Descriptor names")
        response = state["response"]
        scale = "nominal" if result.attrs["method"] == "catreg_nominal_response" else "ordinal"
        if response["name"] != state["outcome"] or response["scale"] != scale:
            raise ValueError("Response scale")
        for d in [response]+descriptors:
            fields = {"name", "scale", "mean", "sd"} if d.get("scale") == "numeric" else {"name", "scale", "levels", "counts", "quantifications"}
            if (not isinstance(d, dict) or len(d) != len(fields) or set(d) != fields
                    or not isinstance(d["name"], str) or not 1 <= len(d["name"]) <= 256):
                raise ValueError("Descriptor fields")
            if d["scale"] == "numeric":
                if d["name"] == state["outcome"] or not _finite_scalar(d["mean"]) or not _finite_scalar(d["sd"]) or d["sd"] <= 0:
                    raise ValueError("Numeric scaling")
                continue
            levels, counts, scores = d["levels"], d["counts"], d["quantifications"]
            if d["scale"] not in ("nominal", "ordinal") or not 2 <= len(levels) <= 32 or len(counts) != len(levels) or len(scores) != len(levels):
                raise ValueError("Map shape")
            if any(not _valid_label(label) for label in levels) or len({base._key(label) for label in levels}) != len(levels):
                raise ValueError("Typed levels")
            if any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in counts) or sum(counts) != n or any(not _finite_scalar(q) for q in scores):
                raise ValueError("Counts/scores")
            if abs(sum(c*q for c, q in zip(counts, scores))/n) > 1e-8 or abs(sum(c*q*q for c, q in zip(counts, scores))/n-1) > 1e-8:
                raise ValueError("Map normalization")
            if d["scale"] == "ordinal" and any(b < a-1e-10 for a, b in zip(scores, scores[1:])):
                raise ValueError("Monotone map")
        if (any(not _finite_scalar(value) for value in state["beta"]+state["training_response"]+[value for row in state["training_quantified"] for value in row])
                or any(not _finite_scalar(state[key]) for key in ("intercept", "objective", "stationarity_tolerance", "objective_tolerance"))
                or len(state["conditional_response_roots"]) != len(response["levels"])
                or any(not _finite_scalar(value) for value in state["conditional_response_roots"])):
            raise ValueError("Nonfinite state")
        codes = state["training_response_codes"]
        if any(type(c) is not int or not 0 <= c < len(response["levels"]) for c in codes):
            raise ValueError("Response codes")
        raw = state["training_raw_predictors"]
        if any(not _valid_label(label) for row in raw for label in row):
            raise ValueError("Raw predictor identities")
        digest = result.attrs["state_sha256"]
        if type(digest) is not str or len(digest) != 64:
            raise ValueError("Integrity shape")
        k = len(response["levels"])
        work = 16*n*p*p+256*n*p+64*k**3
        if work > max_work:
            base._error("Response-state numerical verification exceeds max_work.", "resource_limit")
        plan = plan_workspace("saved categorical response state verification", {
            "training_design_response_solver": 8*(8*n*p+16*n+12*p*p),
            "conditional_response_root_buffers": 8*k*(8*k+4*p),
            "bounded_state_json_envelope": 512*n*(p+2)+12*sum(len(label[1].encode("utf-8")) for row in raw for label in row if label[0] == "str")
                +12*sum(len(label[1].encode("utf-8")) for d in [response]+descriptors for label in d.get("levels", []) if label[0] == "str"),
        }, budget_bytes=min(max_bytes, workspace_budget_bytes())).record()
        if digest != base._seal(state):
            raise ValueError("Integrity")
        z = torch.tensor(state["training_quantified"], dtype=DTYPE)
        y = torch.tensor(state["training_response"], dtype=DTYPE)
        beta = torch.tensor(state["beta"], dtype=DTYPE)
        if [codes.count(i) for i in range(len(response["levels"]))] != response["counts"]:
            raise ValueError("Response counts")
        expected_y = torch.tensor(response["quantifications"], dtype=DTYPE)[torch.tensor(codes, dtype=torch.int64)]
        if not torch.allclose(y, expected_y, atol=1e-10, rtol=1e-10):
            raise ValueError("Response map/codes")
        for j, descriptor in enumerate(descriptors):
            labels = [row[j] for row in state["training_raw_predictors"]]
            if descriptor["scale"] == "numeric":
                if any(label[0] not in ("int", "float") for label in labels):
                    raise ValueError("Raw numeric predictor")
                original = torch.tensor([label[1] for label in labels], dtype=DTYPE)
                expected = (original-descriptor["mean"])/descriptor["sd"]
            else:
                keys = [base._key(label) for label in labels]
                mapping = {base._key(label): value for label, value in zip(descriptor["levels"], descriptor["quantifications"])}
                if [keys.count(base._key(level)) for level in descriptor["levels"]] != descriptor["counts"]:
                    raise ValueError("Predictor category counts")
                expected = torch.tensor([mapping[key] for key in keys], dtype=DTYPE)
            if not torch.allclose(z[:, j], expected, atol=1e-9, rtol=1e-9):
                raise ValueError("Predictor maps/training data")
        if not torch.allclose(z.mean(0), torch.zeros(p, dtype=DTYPE), atol=1e-8, rtol=0) or not torch.allclose(z.square().mean(0), torch.ones(p, dtype=DTYPE), atol=1e-8, rtol=0):
            raise ValueError("Predictor normalization")
        base._rank(z)
        solution = torch.linalg.lstsq(z, y).solution
        if not torch.allclose(beta, solution, atol=1e-8, rtol=1e-8):
            raise ValueError("Coefficient/design stationarity")
        fitted = z@beta
        if not math.isfinite(state["objective"]) or abs(float((y-fitted).square().mean())-state["objective"]) > 1e-9:
            raise ValueError("Objective")
        tolerance = state["stationarity_tolerance"]
        objective_tolerance = state["objective_tolerance"]
        if (not 1e-12 <= objective_tolerance <= 1e-3
                or tolerance != max(1e-9, math.sqrt(objective_tolerance)*0.1)):
            raise ValueError("Stationarity tolerance")
        _, update = _response_block(fitted, response, torch.tensor(codes, dtype=torch.int64))
        if float(torch.sqrt((update-y).square().mean())) > 2*tolerance:
            raise ValueError("Response stationarity")
        roots = _response_roots(z, torch.tensor(codes, dtype=torch.int64), response["counts"])
        stored_roots = state["conditional_response_roots"]
        if len(stored_roots) != len(roots) or not torch.allclose(roots, torch.tensor(stored_roots, dtype=DTYPE), atol=1e-9, rtol=1e-9):
            raise ValueError("Response roots")
        if scale == "nominal" and (float(roots[-1]-roots[-2]) <= 1e-8
                                   or float(roots[-1])-(1-state["objective"]) > max(1e-7, 10*objective_tolerance)):
            raise ValueError("Nominal contrast identification")
        return state, plan, work
    except AnalysisError as exc:
        if exc.code in ("workspace_limit", "resource_limit", "invalid_resource_budget"):
            raise
        raise AnalysisError("invalid_state", "Saved categorical response state failed structural/numerical verification.") from exc
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError("invalid_state", "Saved categorical response state failed structural/numerical verification.") from exc


@resident_cpu
def catreg_outcome_predict(result: TableSet, data: Any, *, missing: str = "raise",
                           max_work: int = DEFAULT_WORK, max_bytes: int = DEFAULT_BYTES,
                           device: str = "cpu") -> TableSet:
    """Apply saved categorical-response maps; return quantified scores, not probabilities."""
    if device != "cpu":
        base._error("Saved response prediction is resident CPU only.", "unsupported_option")
    max_work = base._integer(max_work, "max_work", 1, 2**63-1)
    max_bytes = base._integer(max_bytes, "max_bytes", 1, 2**63-1)
    state, audit, work = _response_state(result, max_work, max_bytes)
    frame, positions, n_input, input_plan = base._frame(data, state["variables"], missing, max_bytes)
    n, p = len(frame), len(state["variables"])
    if work+8*n*p*p > max_work:
        base._error("Saved response prediction exceeds max_work.", "resource_limit")
    plan = plan_workspace("saved quantified-response prediction", {
        "state_verification": audit["estimated_workspace_bytes"],
        "named_prediction_input_and_mask": input_plan["estimated_workspace_bytes"],
        "query_maps_prediction": 8*n*(4*p+12),
    }, budget_bytes=min(max_bytes, workspace_budget_bytes())).record()
    z = base._mapped(frame, state["descriptors"])
    fitted = z@torch.tensor(state["beta"], dtype=DTYPE)
    return base._saved(TableSet({
        "predictions": table([[position, value] for position, value in zip(positions, fitted.tolist())], columns=["source_position", "quantified_fitted"]),
        "transformed": table([[position]+row for position, row in zip(positions, z.tolist())], columns=["source_position"]+state["variables"]),
    }, title="Saved quantified-response predictions", method="catreg_outcome_predict", outcome=state["outcome"],
       response_scale=state["response"]["scale"], n=len(positions), n_input=n_input, sample_positions=positions,
       n_missing=n_input-len(positions), state_sha256=result.attrs["state_sha256"], resources=plan,
       dtype="float64", device="cpu", inference="descriptive quantified response; no calibrated class probabilities or uniquely invertible labels"))


@resident_cpu
def catreg_bootstrap(data: Any, outcome: str, predictors: list[str], *, queries: Any,
                     scales: dict | None = None, orders: dict | None = None,
                     predictor_scale: str = "nominal", reps: int = 39, confidence: float = 0.95,
                     seed: int = 0, n_starts: int = 2, maxiter: int = 100, tol: float = 1e-8,
                     missing: str = "drop", failure: str = "raise", max_work: int = DEFAULT_WORK,
                     max_bytes: int = DEFAULT_BYTES, device: str = "cpu",
                     weights: str | None = None) -> TableSet:
    """Paired iid full-refit numeric-response CATREG prediction bootstrap.

    Explicit fixed queries have joint empirical covariance and percentile CIs
    only if every declared draw succeeds. Record-policy failures withhold all
    inference. Every draw refits transformations, selected starts and effects.
    """
    controls = base._controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device, weights)
    reps = base._integer(reps, "reps", 2, 499)
    confidence = base._real(confidence, "confidence", 0.5, 0.999)
    variables = base._names(predictors)
    if not isinstance(outcome, str) or not outcome or outcome in variables:
        base._error("Supply a distinct numeric outcome.")
    if any(len(name) > 256 for name in [outcome]+variables):
        base._error("Column names admit at most 256 characters.", "resource_limit")
    if predictor_scale not in ("nominal", "ordinal") or failure not in ("raise", "record"):
        base._error("predictor_scale is nominal/ordinal; failure is raise/record.", "invalid_option")
    n_input, nq = _size(data, [outcome]+variables), _size(queries, variables, 64)
    p = len(variables)
    # Admit combined selected data, resampling, all draw outputs and one full
    # fit workspace before either caller input is copied or tensorized.
    plan = plan_workspace("paired full-refit CATREG bootstrap", {
        "named_training_query_selection_and_masks": 64*n_input*(p+1)+64*nq*(p+1),
        "one_fit_and_resampling_peak": 8*n_input*(40*p+96)+8*32*p*(32+2*p)+96*p*p,
        "one_fit_trace_and_start_records": 8*n_starts*(maxiter+1)*8,
        "joint_prediction_draws_covariance_intervals": 8*(reps*(nq+2)+6*nq*nq+16*nq),
    }, budget_bytes=min(max_bytes, workspace_budget_bytes())).record()
    frame, positions, n_input, _ = base._frame(data, [outcome]+variables, missing, max_bytes)
    query, query_positions, query_input, _ = base._frame(queries, variables, "raise", max_bytes)
    _bounded_orders(orders)
    if scales is not None and not isinstance(scales, dict):
        base._error("scales must name the predictors.")
    effective_scales = {name: predictor_scale for name in variables} if scales is None else scales
    if any(isinstance(value, str) and len(value) > 256 for name in variables
           if effective_scales.get(name) != "numeric" for value in frame[name].tolist()):
        base._error("Bootstrap categorical labels admit at most 256 characters.", "resource_limit")
    extra_work = (reps+1)*64*nq*p*p+reps*32*n_input
    per_controls = dict(controls, max_work=(max_work-extra_work)//(reps+1))
    descriptors, _, _, per_work = base._prepare(frame, variables, scales, orders, predictor_scale, per_controls)
    labels_bytes = sum(len(json.dumps(d.get("levels", []), ensure_ascii=False).encode()) for d in descriptors)
    artifact_bound = 2_000_000+reps*(64*n_input+64*n_starts*(maxiter+1)+4096*p+2*labels_bytes+128*nq)
    if artifact_bound > 32*1024**2:
        base._error("Complete bootstrap maps/indices/traces exceed the conservative 32 MiB artifact domain.", "resource_limit")
    if len(frame) <= p+2:
        base._error("More complete rows than predictors + 2 are required.", "insufficient_sample")
    # The same declared category schema is required in every draw. Losing a
    # level is an estimator-domain failure, never a reason to drop that draw.
    expected = {d["name"]: {base._key(label) for label in d["levels"]}
                for d in descriptors if d["scale"] != "numeric"}
    baseline = base._regression(frame, outcome, variables, scales, orders, predictor_scale, per_controls, "raise")
    base_prediction = base.catreg_predict(baseline, query, max_work=max_work, max_bytes=max_bytes)
    point = base_prediction["predictions"]["fitted"].to_numpy(dtype=float).tolist()
    rng = torch.Generator(device="cpu").manual_seed(seed)
    draws, receipts, states, failures, samples = [], [], [], [], []
    for draw in range(reps):
        indices = torch.randint(len(frame), (len(frame),), generator=rng).tolist()
        samples.append([positions[i] for i in indices])
        sample = frame.iloc[indices].reset_index(drop=True)
        draw_controls = dict(per_controls, seed=(seed+104729*(draw+1)) % (2**31-1))
        try:
            for name, levels in expected.items():
                observed = {base._key(base._label(value)) for value in sample[name].tolist()}
                if observed != levels:
                    base._error("A bootstrap sample omits a trained category; full-schema refit is undefined.", "bootstrap_absent_category")
            fit = base._regression(sample, outcome, variables, scales, orders, predictor_scale, draw_controls, "raise")
            prediction = base.catreg_predict(fit, query, max_work=max_work, max_bytes=max_bytes)
            values = prediction["predictions"]["fitted"].to_numpy(dtype=float).tolist()
            draws.append([draw]+values)
            receipts.append([draw, True, fit.attrs["chosen_start"], fit.attrs["iterations"], float(fit["fit"]["standardized_objective"].iloc[0]), "accepted"])
            states.append(_json_safe({"draw": draw, "optimal_state": fit.attrs["optimal_state"],
                           "state_sha256": fit.attrs["state_sha256"], "starts": fit["starts"].to_numpy().tolist(),
                           "iterations": fit["iterations"].to_numpy().tolist(), "predictions": values}))
        except AnalysisError as exc:
            failures.append({"draw": draw, "code": exc.code, "message": str(exc)})
            receipts.append([draw, False, None, None, None, exc.code])
            if failure == "raise":
                error = AnalysisError("bootstrap_failure", f"Draw {draw} failed ({exc.code}); no surviving-draw inference: {exc}")
                error.bootstrap_failures = failures
                error.bootstrap_samples = samples
                raise error from exc
    labels = [f"query_{i}" for i in query_positions]
    output = {
        "baseline_predictions": table([[position, value] for position, value in zip(query_positions, point)], columns=["query_position", "fitted"]),
        "draw_predictions": table(draws, columns=["draw"]+labels),
        "draw_status": table(receipts, columns=["draw", "success", "chosen_start", "iterations", "objective", "status"]),
    }
    inference_available = not failures
    if inference_available:
        values = torch.tensor([row[1:] for row in draws], dtype=DTYPE)
        centered = values-values.mean(0)
        covariance = centered.T@centered/(reps-1)
        alpha = (1-confidence)/2
        intervals = torch.quantile(values, torch.tensor([alpha, 1-alpha], dtype=DTYPE), dim=0)
        output["prediction_covariance"] = table(covariance.tolist(), columns=labels, index=labels)
        output["percentile_intervals"] = table([[position, point[j], float(values[:, j].mean()), float(covariance[j, j].sqrt()),
                                                 float(intervals[0, j]), float(intervals[1, j])]
                                                for j, position in enumerate(query_positions)],
                                               columns=["query_position", "point", "bootstrap_mean", "bootstrap_sd", "percentile_lower", "percentile_upper"])
    state = {
        "version": 1, "kind": "catreg_prediction_bootstrap", "baseline_state": baseline.attrs["optimal_state"],
        "baseline_sha256": baseline.attrs["state_sha256"], "query_positions": query_positions,
        "query_input": query_input, "query_transformed": base_prediction["transformed"].to_numpy().tolist(),
        "sample_positions": positions, "n_input": n_input, "sampled_original_positions": samples,
        "draw_states": states, "failures": failures, "reps": reps, "confidence": confidence,
        "inference_available": inference_available,
    }
    return base._saved(TableSet(output, title="Full-refit CATREG fixed-query prediction bootstrap", method="catreg_bootstrap",
       outcome=outcome, variables=variables, n=len(frame), n_input=n_input, n_missing=n_input-len(frame),
       sample_positions=positions, query_positions=query_positions, reps=reps, successful_draws=len(draws),
       failures=failures, failure_policy=failure, confidence=confidence, inference_available=inference_available,
       settings=controls, predictor_scale=predictor_scale, resources=plan,
       declared_work=(reps+1)*per_work+extra_work, declared_artifact_bound_bytes=artifact_bound,
       bootstrap_state=state, state_sha256=base._seal(state), dtype="float64", device="cpu", weight_type="unweighted",
       sources=SOURCES, inference="paired iid full-estimator empirical prediction covariance and percentile intervals only if all draws succeed; Monte Carlo/coverage limits, no ordinary adaptive coefficient Wald or p-values",
       alignment="fixed raw query fitted means in original numeric-response units; refitted map sign/scale choices cancel in beta-times-map prediction"))
