"""Resident mixed-data CF clustering with explicit bounded policies."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real

import pandas as pd
import torch

from openecon.analysis import _coerce_frame
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary, summary_state, restore_summary
from openecon.resources import plan_workspace, workspace_budget_bytes
from . import kernel
from .selection import select_two_stage as _select_two_stage

DTYPE = torch.float64
DEFAULT_WORK = 300_000_000
DEFAULT_BYTES = 128 * 1024**2


def _error(message, code="invalid_spec"):
    raise AnalysisError(code, message)


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        _error(f"{name} must be an integer in [{low}, {high}].", "invalid_option")
    return int(value)


def _real(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not low <= value <= high:
        _error(f"{name} must be finite in [{low}, {high}].", "invalid_option")
    return float(value)


def _names(values):
    if not isinstance(values, (list, tuple)) or len(values) > 8 or any(not isinstance(x, str) or not x for x in values) or len(set(values)) != len(values):
        _error("Each role needs 0–8 distinct nonempty column names.")
    return list(values)


def _label(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, str) and len(value.encode()) <= 4096:
        return ["str", value]
    if isinstance(value, Integral):
        if abs(value) > 2**53:
            _error("Categorical integers must have magnitude at most 2**53; use string identifiers to preserve larger integer identity.")
        return ["int", int(value)]
    if isinstance(value, Real) and math.isfinite(value) and abs(value) <= 1e100:
        return ["float", float(value)]
    _error("Categories need finite bounded scalar strings, booleans, integers or floats.")


def _key(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _seal(state):
    return hashlib.sha256(json.dumps(state, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _domain(device, weights, max_work, max_bytes):
    if device != "cpu" or weights is not None:
        _error("TwoStep supports unweighted resident CPU float64 only.", "unsupported_option")
    return (_integer(max_work, "max_work", 1, 2**63-1),
            _integer(max_bytes, "max_bytes", 1, 2**63-1))


def _input_size(data, names):
    if isinstance(data, pd.DataFrame):
        size = len(data)
        if data.columns.has_duplicates or any(name not in data for name in names):
            _error("Required columns must occur exactly once.")
    elif isinstance(data, Mapping):
        try:
            if any(name not in data for name in names):
                _error("Required columns must occur exactly once.")
            lengths = [len(data[name]) for name in names]
            size = lengths[0]
            if len(set(lengths)) != 1:
                _error("All selected mapping columns must have equal lengths.")
        except TypeError as exc:
            raise AnalysisError("invalid_spec", "Use finite resident column sequences.") from exc
    elif isinstance(data, list):
        size = len(data)
    else:
        _error("Supply a resident DataFrame, column mapping or record list; Dataset cannot be collected.", "unsupported_data")
    if not 1 <= size <= 2000:
        _error("TwoStep accepts 1–2000 physical input rows before coercion.", "resource_limit")
    return size


def _frame(data, names, missing, max_bytes, categorical=()):
    if missing not in ("raise", "drop"):
        _error("missing must be 'raise' or 'drop'.", "invalid_option")
    size = _input_size(data, names)
    plan = plan_workspace("TwoStep resident selected input", {
        "selected_scalar_copies": 64*size*len(names),
        "positions_missing_and_indices": 64*size,
    }, budget_bytes=min(max_bytes, workspace_budget_bytes()))
    if isinstance(data, pd.DataFrame):
        selected = data.loc[:, names].copy()
    else:
        if isinstance(data, Mapping):
            data = {name: list(data[name]) for name in names}
        else:
            if any(not isinstance(row, Mapping) for row in data):
                _error("Every record must be a column mapping.")
            if any(not any(name in row for row in data) for name in names):
                _error("Required columns must be represented.")
            data = [{name: row.get(name) for name in names} for row in data]
        try:
            selected = _coerce_frame(data).loc[:, names]
            # Retain categorical scalar types supplied in resident sequences;
            # ordinary DataFrame numeric inference would merge 1 and 1.0.
            for name in categorical:
                raw_values = data[name] if isinstance(data, Mapping) else [row[name] for row in data]
                selected[name] = pd.array(list(raw_values), dtype=object)
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_spec", "Cannot form a resident selected table.") from exc
    absent = selected.isna().any(axis=1).tolist()
    if missing == "raise" and any(absent):
        _error("Missing values occur in analysis columns.", "missing_values")
    positions = [i for i, value in enumerate(absent) if not value]
    return selected.iloc[positions].reset_index(drop=True), positions, size, plan.record()


def _numeric(series):
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_complex_dtype(series.dtype):
        _error(f"Continuous column {series.name!r} must have a real numeric dtype.")
    values = series.to_numpy(dtype=float).tolist()
    if any(not math.isfinite(x) or abs(x) > 1e100 for x in values):
        _error("Continuous values must be finite with magnitude at most 1e100.")
    return torch.tensor(values, dtype=DTYPE)


def _prepare(frame, continuous, categorical, descriptors=None, scaling=None):
    numeric, scale, maps, codes = [], [], [], []
    for j, name in enumerate(continuous):
        raw = _numeric(frame[name])
        if scaling is None:
            origin = float(raw[0])
            shifted = raw-origin
            center = math.fsum(shifted.tolist())/len(raw)
            deviations = shifted-center
            sd = float(torch.sqrt(deviations.square().mean()))
            if not math.isfinite(sd) or sd <= 1e-12 * max(float(deviations.abs().max()), 1e-100):
                _error("Continuous columns must have positive nondegenerate population variance.", "degenerate_transform")
            entry = {"name": name, "origin": origin, "center": center, "mean": origin+center, "sd": sd}
        else:
            entry = scaling[j]
        values = ((raw-entry["origin"])-entry["center"])/entry["sd"]
        if not bool(torch.isfinite(values).all()):
            _error("Training scaling produces nonfinite query values.", "invalid_spec")
        numeric.append(values)
        scale.append(entry)
    for j, name in enumerate(categorical):
        values = [_label(value) for value in frame[name].tolist()]
        keys = [_key(value) for value in values]
        if descriptors is None:
            levels = []
            seen = set()
            for value, key in zip(values, keys):
                if key not in seen:
                    levels.append(value)
                    seen.add(key)
                    if len(levels) > 32:
                        _error("At most 32 levels per categorical variable.", "resource_limit")
            if not levels:
                _error("No complete observations remain.", "insufficient_sample")
        else:
            levels = descriptors[j]["levels"]
        mapping = {_key(value): i for i, value in enumerate(levels)}
        try:
            encoded = [mapping[key] for key in keys]
        except KeyError as exc:
            raise AnalysisError("unknown_category", f"Unknown category in {name!r}; training maps are fixed.") from exc
        maps.append({"name": name, "levels": levels, "counts": [encoded.count(i) for i in range(len(levels))]})
        codes.append(torch.tensor(encoded, dtype=torch.int64))
    n = len(frame)
    return (torch.stack(numeric, 1) if numeric else torch.empty((n, 0), dtype=DTYPE),
            torch.stack(codes, 1) if codes else torch.empty((n, 0), dtype=torch.int64), scale, maps)


def _cf(value):
    return kernel.CF(value["count"], torch.tensor(value["mean"], dtype=DTYPE),
                     torch.tensor(value["m2"], dtype=DTYPE),
                     tuple(torch.tensor(x, dtype=torch.int64) for x in value["categorical_counts"]),
                     tuple(value["rows"]))


def _record(cf, positions):
    return {"count": cf.count, "mean": cf.mean.tolist(), "m2": cf.m2.tolist(),
            "categorical_counts": [value.tolist() for value in cf.categorical_counts],
            "rows": sorted(positions[i] for i in cf.rows)}


def _saved(output):
    saved_summary(output)
    for name, frame in list(output.items()):
        values = frame.to_numpy()
        output[name] = table(values.tolist(), columns=list(frame.columns), index=list(frame.index))
    ordered = sorted(output.items())
    output.clear()
    output.update(ordered)
    return output


def _labels(state):
    assigned = {row: i+1 for i, cf in enumerate(state["cuts"][str(state["selected_k"])]) for row in cf["rows"]}
    complete = set(state["positions"])
    return [[i, assigned.get(i, 0) if i in complete else None,
             "cluster" if i in assigned else "noise" if i in complete else "missing"] for i in range(state["input_n"])]


def _output(state):
    from .helpers import profile_tables
    adaptive = state["version"] == 2
    selection = state["controls"].get("selection", "global_min")
    result = TableSet({
        "assignments": table(_labels(state), columns=["position", "cluster", "status"]),
        "criteria": table(state["criteria"], columns=["clusters", "score", "parameters", "bic", "aic"]),
        "merges": table([[i+1, entry.get("distance"), entry.get("left"), entry.get("right")] for i, entry in enumerate(state["merges"])], columns=["step", "distance", "left", "right"]),
    }, title="Mixed-data TwoStep clustering", method="twostep", twostep_state=state,
       state_sha256=_seal(state), n_input=state["input_n"], n_complete=len(state["positions"]),
       n_clustered=len(state["positions"])-len(state["noise_positions"]), n_noise=len(state["noise_positions"]),
       n_missing=len(state["missing_positions"]), n_clusters=state["selected_k"],
       selection=("two-stage change/jump" if selection == "two_stage" else "global-min information criterion") if state["controls"]["n_clusters"] is None else "fixed count",
       inference="descriptive; covariance/SE/df/p/CI are not defined", device="cpu", dtype="float64",
       notes=["Independent normal continuous and multinomial categorical CF score; additive global variance regularizes continuous terms.",
              "Global-min BIC/AIC selection differs from IBM's two-stage change/jump heuristic.",
              "Input order can affect the CF tree; no order invariance or licensed vendor parity is claimed.",
              "Small-leaf noise exclusion uses min_precluster_size; no adaptive rebuild or noise reinsertion."])
    if adaptive:
        result.attrs["notes"] = [
            "Independent normal/multinomial CF score; training population scaling and typed maps are fixed.",
            "Two-stage mode follows Statistics14 change/jump equations, with recorded deterministic boundary extensions.",
            "Adaptive rebuild/reinsertion is bounded, preserves every physical row, and records excluded noise.",
            "Order dependence and descriptive inference remain explicit; licensed vendor parity is not claimed.",
        ]
    result.update(profile_tables(state))
    return _saved(result)


def _validation_cost(result):
    if not isinstance(result, TableSet) or result.attrs.get("method") != "twostep":
        _error("Supply a fitted or restored TwoStep TableSet.", "invalid_result")
    state = result.attrs.get("twostep_state")
    try:
        if not isinstance(state, dict) or not 1 <= len(state["positions"]) <= 2000 or not 1 <= len(state["cuts"]) <= 128 or len(state["continuous"]) > 8 or len(state["categorical"]) > 8:
            raise ValueError("Oversized state")
        levels = sum(len(d["levels"]) for d in state["categorical"])
        if levels > 256:
            raise ValueError("Oversized category universe")
        n, cuts = len(state["positions"]), len(state["cuts"])
        if len(state["merges"]) != cuts-1 or len(state["preclusters"]) > 128:
            raise ValueError("Invalid history dimensions")
        pending, nodes, entries = [state["tree"]], 0, 0
        while pending:
            node = pending.pop()
            nodes += 1
            if nodes > 512 or not isinstance(node, dict) or not isinstance(node["entries"], list) or not 1 <= len(node["entries"]) <= 16:
                raise ValueError("Oversized tree")
            entries += len(node["entries"])
            if entries > 8192:
                raise ValueError("Oversized entries")
            for entry in node["entries"]:
                if not isinstance(entry, dict) or len(entry["cf"]["rows"]) > 2000:
                    raise ValueError("Invalid tree CF")
                if entry["child"] is not None:
                    pending.append(entry["child"])
        work = 8*(n*(cuts+nodes+1)+cuts*cuts)*(len(state["continuous"])+levels+8)
        buffers = 128*(n*(cuts+nodes+1)+cuts*cuts*(len(state["continuous"])+levels+8))
        if state.get("version") == 2 and state["controls"]["rebuild"]:
            replay = state["resources"]["tree_work_bound"]
            if type(replay) is not int or not 1 <= replay <= 2**63-1:
                raise ValueError("Invalid adaptive replay bound")
            traces = state["tree"]["rebuild_trace"]
            if not isinstance(traces, list) or len(traces) > 17:
                raise ValueError("Oversized adaptive history")
            for entry in traces:
                for key in ("before", "after"):
                    if not isinstance(entry[key], list) or len(entry[key]) > 129 or any(len(cf["rows"]) > 2000 for cf in entry[key]):
                        raise ValueError("Oversized adaptive CF history")
            work += replay
            buffers += 128*n*(len(traces)+1)*(len(state["continuous"])+levels+8)
        return work, buffers
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("invalid_state", "TwoStep state dimensions are invalid or exceed the saved domain.") from exc


def _state(result, *, max_work=DEFAULT_WORK, max_bytes=DEFAULT_BYTES):
    if not isinstance(result, TableSet) or result.attrs.get("method") != "twostep":
        _error("Supply a fitted or restored TwoStep TableSet.", "invalid_result")
    state = result.attrs.get("twostep_state")
    work, buffers = _validation_cost(result)
    if work > max_work:
        _error("Complete saved TwoStep validation exceeds max_work.", "resource_limit")
    plan_workspace("Complete TwoStep saved state validation", {"state_moments_and_table_replay": buffers}, budget_bytes=min(max_bytes, workspace_budget_bytes()))
    try:
        if not isinstance(state, dict) or type(state["version"]) is not int or state["version"] not in (1, 2) or state["kind"] != "twostep":
            raise ValueError("Unknown schema")
        continuous, categorical = _names(state["continuous"]), state["categorical"]
        names = _names([x["name"] for x in categorical])
        if not 1 <= len(continuous)+len(names) <= 16 or set(continuous)&set(names):
            raise ValueError("Invalid roles")
        n = _integer(state["input_n"], "input_n", 1, 2000)
        positions = state["positions"]
        if not positions or positions != sorted(set(positions)) or any(type(x) is not int or not 0 <= x < n for x in positions):
            raise ValueError("Invalid sample")
        if state["missing_positions"] != [i for i in range(n) if i not in positions]:
            raise ValueError("Invalid missing alignment")
        noise = state["noise_positions"]
        if noise != sorted(set(noise)) or not set(noise) <= set(positions):
            raise ValueError("Invalid noise sample")
        if sorted(state["order"]) != positions:
            raise ValueError("Invalid insertion order")
        if len(state["scaling"]) != len(continuous) or len(state["global_variance"]) != len(continuous):
            raise ValueError("Invalid continuous dimensions")
        for name, entry, variance in zip(continuous, state["scaling"], state["global_variance"]):
            if entry["name"] != name or any(not math.isfinite(entry[key]) for key in ("mean", "origin", "center", "sd")) or entry["sd"] <= 0 or not math.isfinite(variance) or variance <= 0:
                raise ValueError("Invalid saved scaling")
        levels = []
        for descriptor in categorical:
            labels = descriptor["levels"]
            if not 1 <= len(labels) <= 32 or len({_key(x) for x in labels}) != len(labels) or any(not isinstance(x, list) or len(x) != 2 or _label(x[1]) != x for x in labels):
                raise ValueError("Invalid typed category maps")
            counts = descriptor["counts"]
            if len(counts) != len(labels) or any(type(x) is not int or x < 1 for x in counts) or sum(counts) != len(positions):
                raise ValueError("Invalid category totals")
            levels.append(len(labels))
        if len(state["training_numeric"]) != len(positions) or len(state["training_codes"]) != len(positions):
            raise ValueError("Invalid training dimensions")
        for raw, codes in zip(state["training_numeric"], state["training_codes"]):
            if len(raw) != len(continuous) or any(not math.isfinite(x) for x in raw) or len(codes) != len(levels) or any(type(x) is not int or not 0 <= x < levels[j] for j, x in enumerate(codes)):
                raise ValueError("Invalid training values")
        raw_sample = state["raw_sample"]
        if len(raw_sample) != len(positions) or any(len(row) != len(continuous)+len(names) or any(_label(value[1]) != value for value in row) for row in raw_sample):
            raise ValueError("Invalid typed training sample")
        if state["sample_sha256"] != _seal({"columns": continuous+names, "positions": positions, "input_n": n, "raw": raw_sample}):
            raise ValueError("Training corpus fingerprint mismatch")
        mapped = [{_key(label): i for i, label in enumerate(d["levels"])} for d in categorical]
        for i, row in enumerate(raw_sample):
            for j, entry in enumerate(state["scaling"]):
                value = ((row[j][1]-entry["origin"])-entry["center"])/entry["sd"]
                if row[j][0] not in ("int", "float") or not math.isclose(value, state["training_numeric"][i][j], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("Training numeric preprocessing mismatch")
            if any(mapped[j][_key(row[len(continuous)+j])] != state["training_codes"][i][j] for j in range(len(names))):
                raise ValueError("Training category preprocessing mismatch")
        for j, entry in enumerate(state["scaling"]):
            raw_values = [row[j][1] for row in raw_sample]
            origin = raw_values[0]
            center = math.fsum(x-origin for x in raw_values)/len(raw_values)
            mean = origin+center
            sd = math.sqrt(math.fsum(((x-origin)-center)**2 for x in raw_values)/len(raw_values))
            variance = math.fsum(row[j]**2 for row in state["training_numeric"])/len(positions)
            if origin != entry["origin"] or not math.isclose(center, entry["center"], rel_tol=1e-12, abs_tol=1e-12) or mean != entry["mean"] or not math.isclose(sd, entry["sd"], rel_tol=1e-10, abs_tol=1e-12) or not math.isclose(variance, state["global_variance"][j], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Training population moments mismatch")
        physical = {row: i for i, row in enumerate(positions)}
        cuts = state["cuts"]
        if not isinstance(cuts, dict) or not 1 <= len(cuts) <= 128 or set(cuts) != {str(i) for i in range(1, len(cuts)+1)}:
            raise ValueError("Invalid hierarchy range")
        def validate_cf(cf):
            count, rows = cf["count"], cf["rows"]
            if type(count) is not int or count < 1 or count != len(rows) or rows != sorted(set(rows)) or not set(rows) <= set(positions):
                raise ValueError("Invalid CF rows/count")
            if len(cf["mean"]) != len(continuous) or len(cf["m2"]) != len(continuous) or any(not math.isfinite(x) for x in cf["mean"]) or any(not math.isfinite(x) or x < 0 for x in cf["m2"]):
                raise ValueError("Invalid continuous moments")
            if len(cf["categorical_counts"]) != len(levels) or any(len(row) != levels[j] or any(type(x) is not int or x < 0 for x in row) or sum(row) != count for j, row in enumerate(cf["categorical_counts"])):
                raise ValueError("Invalid CF category counts")
            indices = [physical[row] for row in rows]
            for j in range(len(continuous)):
                values = [state["training_numeric"][i][j] for i in indices]
                mean = math.fsum(values)/count
                m2 = math.fsum((value-mean)**2 for value in values)
                if not math.isclose(mean, cf["mean"][j], rel_tol=1e-9, abs_tol=1e-10) or not math.isclose(m2, cf["m2"][j], rel_tol=1e-9, abs_tol=1e-10):
                    raise ValueError("CF moments do not match saved training rows")
            for j, histogram in enumerate(cf["categorical_counts"]):
                if histogram != [sum(state["training_codes"][i][j] == level for i in indices) for level in range(levels[j])]:
                    raise ValueError("CF categories do not match saved training rows")
        for k, cut in cuts.items():
            if len(cut) != int(k):
                raise ValueError("Invalid cut dimensions")
            rows = []
            for cf in cut:
                validate_cf(cf)
                rows.extend(cf["rows"])
            if sorted(rows) != sorted(set(positions)-set(noise)):
                raise ValueError("Hierarchy does not conserve rows")
        preclusters = state["preclusters"]
        if not 1 <= len(preclusters) <= 128:
            raise ValueError("Invalid precluster count")
        all_rows = []
        for cf in preclusters:
            validate_cf(cf)
            all_rows.extend(cf["rows"])
        extended = state["version"] == 2
        noise_cf = state.get("noise_cf") if extended else None
        excluded_rows = []
        if noise_cf is not None:
            validate_cf(noise_cf)
            excluded_rows = noise_cf["rows"]
        if sorted(all_rows+excluded_rows) != positions or len(set(all_rows+excluded_rows)) != len(positions) or state["selected_k"] not in range(1, len(cuts)+1):
            raise ValueError("Invalid preclusters or selection")
        if type(state["selected_k"]) is not int:
            raise ValueError("Invalid selected count")
        controls = state["controls"]
        selection = controls.get("selection", "global_min")
        rebuild = controls.get("rebuild", False)
        noise_mode = controls.get("noise", "none")
        if extended:
            if selection not in ("global_min", "two_stage") or type(rebuild) is not bool or noise_mode not in ("none", "adaptive"):
                raise ValueError("Invalid adaptive controls")
            _integer(controls["max_rebuilds"], "max_rebuilds", 0, 16)
            _real(controls["noise_fraction"], "noise_fraction", 0, 1)
            if noise_mode == "adaptive" and (not rebuild or controls["noise_fraction"] <= 0 or controls["min_precluster_size"] != 1):
                raise ValueError("Conflicting adaptive noise policy")
            if noise_mode != "adaptive" and excluded_rows:
                raise ValueError("Noise CF without adaptive noise")
        elif selection != "global_min" or rebuild is not False or noise_mode != "none" or "noise_cf" in state:
            raise ValueError("Adaptive fields in legacy schema")
        branch = _integer(controls["branch_factor"], "branch_factor", 2, 16)
        leaf_limit = _integer(controls["max_preclusters"], "max_preclusters", 1, 128)
        node_limit = _integer(controls["max_nodes"], "max_nodes", 1, 512)
        _real(controls["threshold"], "threshold", 0, 1e100)
        if controls["order"] not in ("input", "random") or controls["missing"] not in ("raise", "drop"):
            raise ValueError("Invalid saved order/missing policies")
        if controls["order"] == "random":
            seed = _integer(controls["seed"], "seed", 0, 2**63-1)
            permutation = torch.randperm(len(positions), generator=torch.Generator(device="cpu").manual_seed(seed)).tolist()
            if state["order"] != [positions[i] for i in permutation]:
                raise ValueError("Recorded order differs from private seed")
        elif controls["seed"] is not None or state["order"] != positions:
            raise ValueError("Invalid input order")
        if controls["missing"] == "raise" and state["missing_positions"]:
            raise ValueError("Invalid missing policy")
        if len(preclusters) > leaf_limit or state["tree_local_indices"] is not True:
            raise ValueError("Invalid leaf limit/index convention")
        node_ids, entry_ids, depths, leaves = set(), set(), set(), []
        def validate_node(node, depth):
            if type(node["id"]) is not int or not 0 <= node["id"] < node_limit or node["id"] in node_ids or type(node["leaf"]) is not bool or len(node["entries"]) > branch:
                raise ValueError("Invalid tree node")
            node_ids.add(node["id"])
            rows = []
            for entry in node["entries"]:
                if type(entry["id"]) is not int or not 0 <= entry["id"] < 3*node_limit+leaf_limit or entry["id"] in entry_ids:
                    raise ValueError("Invalid tree entry")
                entry_ids.add(entry["id"])
                local = entry["cf"]
                if any(type(i) is not int or not 0 <= i < len(positions) for i in local["rows"]):
                    raise ValueError("Invalid local tree rows")
                cf = {**local, "rows": [positions[i] for i in local["rows"]]}
                validate_cf(cf)
                rows.extend(cf["rows"])
                if node["leaf"]:
                    if entry["child"] is not None:
                        raise ValueError("Leaf has child")
                    depths.add(depth)
                    leaves.append(cf)
                elif entry["child"] is None or validate_node(entry["child"], depth+1) != cf["rows"]:
                    raise ValueError("Internal summary differs from child")
            if len(rows) != len(set(rows)):
                raise ValueError("Tree entries overlap")
            return sorted(rows)
        if validate_node(state["tree"], 1) != sorted(all_rows) or len(depths) != 1 or _seal(sorted(leaves, key=lambda cf: cf["rows"][0])) != _seal(preclusters):
            raise ValueError("Tree leaves differ from preclusters")
        diagnostics = state["tree_diagnostics"]
        if diagnostics["nodes"] != len(node_ids) or diagnostics["depth"] != next(iter(depths)) or diagnostics["splits"] != len(node_ids)-next(iter(depths)):
            raise ValueError("Invalid tree diagnostics")
        for key in ("distance_evaluations", "roundoff_clamps"):
            if type(diagnostics[key]) is not int or not 0 <= diagnostics[key] <= state["resources"]["work_bound"]:
                raise ValueError("Invalid distance diagnostics")
        if diagnostics["roundoff_rule"] != kernel.ROUNDING_RULE or diagnostics["roundoff_clamps"] > diagnostics["distance_evaluations"]:
            raise ValueError("Invalid rounding diagnostics")
        if rebuild:
            expected_bound = _adaptive_work_bound(state["input_n"], len(continuous), len(names), branch, leaf_limit, node_limit, controls["max_rebuilds"])
            if state["resources"]["tree_work_bound"] != expected_bound:
                raise ValueError("Invalid cumulative rebuild admission bound")
            # Replay the declared transition witness only to authenticate saved
            # topology/noise/history. Returned estimates still use saved cuts;
            # validation never replaces them with a newly fitted result.
            X = torch.tensor(state["training_numeric"], dtype=DTYPE).reshape(len(positions), len(continuous))
            codes = torch.tensor(state["training_codes"], dtype=torch.int64).reshape(len(positions), len(names))
            order = torch.tensor([physical[row] for row in state["order"]], dtype=torch.int64)
            replayed = kernel.build_tree(X, codes, levels, torch.tensor(state["global_variance"], dtype=DTYPE), order,
                                         threshold=controls["threshold"], branch_factor=branch,
                                         max_preclusters=leaf_limit, max_nodes=node_limit, rebuild=True,
                                         max_rebuilds=controls["max_rebuilds"],
                                         noise_fraction=controls["noise_fraction"] if noise_mode == "adaptive" else 0.0,
                                         max_work=expected_bound)
            expected_noise_cf = _record(replayed.noise_cf, positions) if replayed.noise_cf is not None else None
            if _seal(replayed.tree) != _seal(state["tree"]) or _seal(expected_noise_cf) != _seal(noise_cf) or replayed.distance_evaluations != diagnostics["distance_evaluations"] or replayed.roundoff_clamps != diagnostics["roundoff_clamps"]:
                raise ValueError("Adaptive witness differs from retained corpus/control decisions")
        elif state["tree"].get("automatic_rebuild") is not False or state["tree"].get("threshold") != controls["threshold"]:
            raise ValueError("Invalid legacy tree policy")
        minimum = _integer(controls["min_precluster_size"], "min_precluster_size", 1, 2000)
        expected_noise = sorted(excluded_rows+[row for cf in preclusters if cf["count"] < minimum for row in cf["rows"]])
        if noise != expected_noise:
            raise ValueError("Noise sample does not match small-leaf policy")
        if len(cuts) != sum(cf["count"] >= minimum for cf in preclusters):
            raise ValueError("Hierarchy does not start at retained leaves")
        if controls["criterion"] not in ("bic", "aic"):
            raise ValueError("Invalid criterion")
        search = _integer(controls["max_clusters"], "max_clusters", 1, 128)
        expected_criteria = []
        globalvar = torch.tensor(state["global_variance"], dtype=DTYPE)
        # Validate the recorded merge sequence against cached pair distances.
        # This verifies saved results without constructing a new tree/hierarchy.
        retained = [cf for cf in preclusters if cf["count"] >= minimum]
        active = {i: _cf(cf) for i, cf in enumerate(retained)}
        if _seal(cuts[str(len(active))]) != _seal(retained):
            raise ValueError("Highest cut differs from retained leaves")
        distances = {}
        def pair(left, right):
            if (active[left].rows[0], left) > (active[right].rows[0], right):
                left, right = right, left
            a, b = active[left], active[right]
            return (kernel.merge_loss(a, b, globalvar), min(a.rows[0], b.rows[0]), max(a.rows[0], b.rows[0]), left, right)
        for left in active:
            for right in active:
                if left < right:
                    distances[left, right] = pair(left, right)
        for index, entry in enumerate(state["merges"]):
            distance, _, _, left, right = min(distances.values())
            merged_id = len(retained)+index
            if entry != {"left": left, "right": right, "merged": merged_id, "distance": distance,
                         "count": active[left].count+active[right].count, "clusters_after": len(active)-1}:
                raise ValueError("Saved merge differs from closest pair")
            combined = kernel.merge(active[left], active[right])
            del active[left], active[right]
            distances = {key: value for key, value in distances.items() if left not in key and right not in key}
            active[merged_id] = combined
            rebuilt = [_record(cf, list(range(n))) for cf in sorted(active.values(), key=lambda cf: cf.rows[0])]
            if _seal(rebuilt) != _seal(cuts[str(len(active))]):
                raise ValueError("Saved cuts are not the recorded merged partitions")
            for other in active:
                if other != merged_id:
                    distances[other, merged_id] = pair(other, merged_id)
        hierarchy_diagnostics = state["hierarchy_diagnostics"]
        if hierarchy_diagnostics["distance_evaluations"] != (len(retained)-1)**2 or type(hierarchy_diagnostics["roundoff_clamps"]) is not int or not 0 <= hierarchy_diagnostics["roundoff_clamps"] <= (len(retained)-1)**2:
            raise ValueError("Invalid hierarchy diagnostics")
        per_cluster = 2*len(continuous)+sum(level-1 for level in levels)
        for k in range(1, min(search+(selection == "two_stage"), len(cuts))+1):
            score = sum(kernel.xi(_cf(cf), globalvar) for cf in cuts[str(k)])
            parameters = k*per_cluster
            expected_criteria.append([k, score, parameters, -2*score+parameters*math.log(len(positions)-len(noise)), -2*score+2*parameters])
        if len(state["criteria"]) != len(expected_criteria) or any(len(a) != 5 or any(not math.isfinite(x) or not math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10) for x, y in zip(a, b)) for a, b in zip(state["criteria"], expected_criteria)):
            raise ValueError("Criterion trace does not match saved hierarchy")
        fixed = controls["n_clusters"]
        expected_trace = None
        if fixed is not None:
            expected_k = _integer(fixed, "n_clusters", 1, len(cuts))
        elif selection == "two_stage":
            # Scores above have already been independently checked against the
            # retained CFs. Authenticate the stored decision against those
            # validated saved scores: a different platform's last-bit log
            # rounding must not replace the original decision witness.
            expected_k, expected_trace = _select_two_stage(state["criteria"], state["merges"], controls["criterion"], search)
        else:
            expected_k = min(expected_criteria, key=lambda row: (row[3 if controls["criterion"] == "bic" else 4], row[0]))[0]
        if extended and _seal(state["selection_trace"]) != _seal(expected_trace):
            raise ValueError("Two-stage decision trace differs from saved criteria/hierarchy")
        if state["selected_k"] != expected_k:
            raise ValueError("Selection differs from declared criterion/fixed count")
        if result.attrs["state_sha256"] != _seal(state):
            raise ValueError("Integrity mismatch")
        # Displayed labels and criterion trace must match the fitted state.
        expected = _output(state)
        for name in ("assignments", "criteria", "merges", "sizes", "continuous_profiles", "categorical_profiles"):
            if name not in result or not result[name].equals(expected[name]):
                raise ValueError("Displayed result differs from state")
        return state
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OverflowError, RecursionError, AnalysisError) as exc:
        raise AnalysisError("invalid_state", "Saved TwoStep state has invalid structure, alignment or integrity.") from exc


def _adaptive_work_bound(n, p, q, branch, leaves, nodes, retries):
    leaves, nodes = min(n, leaves), min(nodes, 2*n+1)
    dimensions = p + min(32, n)*q + 8
    return 16*dimensions*((retries+2)*(n*(leaves+2*branch)+nodes*(branch**2+2*branch)+leaves**2+4*n)+2*n*leaves)


@resident_cpu
def twostep(*, data, continuous=(), categorical=(), n_clusters=None, criterion="bic", max_clusters=15,
            threshold=0.0, branch_factor=8, max_preclusters=128, max_nodes=512,
            order="input", seed=None, missing="raise", min_precluster_size=1,
            selection="global_min", rebuild=False, max_rebuilds=16, noise="none", noise_fraction=0.25,
            max_work=DEFAULT_WORK, max_bytes=DEFAULT_BYTES, device="cpu", weights=None):
    """Cluster resident mixed data using a likelihood CF tree then agglomeration.

    Continuous columns are standardized using training population moments.
    ``threshold`` is an absolute CF merge loss, not a geometric radius.
    ``selection='two_stage'`` uses the Statistics14 change/jump equations;
    legacy ``global_min`` remains the default. ``rebuild=True`` permits bounded
    aggregate-CF threshold rebuilds. ``noise='adaptive'`` adds sparse-leaf
    removal/reinsertion and training log-volume query noise classification;
    it requires rebuild and min_precluster_size=1. Legacy small-leaf exclusion
    remains available. Recorded boundary/rebuild policies are explicit local
    extensions. Private seeded order, unweighted resident CPU float64 only;
    descriptive inference, no licensed executable parity or GPU/Dataset route.
    """
    max_work, max_bytes = _domain(device, weights, max_work, max_bytes)
    continuous, categorical = _names(continuous), _names(categorical)
    if not continuous and not categorical or set(continuous)&set(categorical):
        _error("Use nonoverlapping continuous/categorical roles with at least one column.")
    max_clusters = _integer(max_clusters, "max_clusters", 1, 128)
    if n_clusters is not None:
        n_clusters = _integer(n_clusters, "n_clusters", 1, 128)
    if criterion not in ("bic", "aic") or order not in ("input", "random"):
        _error("criterion is 'bic' or 'aic'; order is 'input' or 'random'.", "invalid_option")
    if order == "random":
        seed = _integer(seed, "seed", 0, 2**63-1)
    elif seed is not None:
        _error("seed is only meaningful with order='random'.", "invalid_option")
    threshold = _real(threshold, "threshold", 0, 1e100)
    branch_factor = _integer(branch_factor, "branch_factor", 2, 16)
    max_preclusters = _integer(max_preclusters, "max_preclusters", 1, 128)
    max_nodes = _integer(max_nodes, "max_nodes", 1, 512)
    min_precluster_size = _integer(min_precluster_size, "min_precluster_size", 1, 2000)
    if selection not in ("global_min", "two_stage") or type(rebuild) is not bool or noise not in ("none", "adaptive"):
        _error("selection is 'global_min' or 'two_stage', rebuild is bool, noise is 'none' or 'adaptive'.", "invalid_option")
    max_rebuilds = _integer(max_rebuilds, "max_rebuilds", 0, 16)
    noise_fraction = _real(noise_fraction, "noise_fraction", 0, 1)
    if noise == "adaptive" and (not rebuild or noise_fraction <= 0 or min_precluster_size != 1):
        _error("Adaptive noise requires rebuild=True, positive noise_fraction and min_precluster_size=1.", "invalid_option")
    if noise == "none" and noise_fraction != 0.25:
        _error("noise_fraction is meaningful only for noise='adaptive'.", "invalid_option")
    if not rebuild and max_rebuilds != 16:
        _error("max_rebuilds is meaningful only with rebuild=True.", "invalid_option")
    extended = selection == "two_stage" or rebuild
    # Conservative admission before selected input coercion. Covers all tree
    # distance evaluations, naive complete hierarchy and saved cuts/row lists.
    columns = continuous+categorical
    worst_work = 16 * (2000*max_preclusters + max_nodes*(branch_factor**2+2*branch_factor) + max_preclusters**2) * (len(continuous)+32*len(categorical)+1)
    tree_work = None
    trace_bytes = 0
    if rebuild:
        n_hint = _input_size(data, columns)
        tree_work = _adaptive_work_bound(n_hint, len(continuous), len(categorical), branch_factor, max_preclusters, max_nodes, max_rebuilds)
        worst_work = tree_work + 16*(len(continuous)+min(32, n_hint)*len(categorical)+8)*(min(n_hint, max_preclusters)**2+n_hint*min(n_hint, max_preclusters))
        trace_bytes = 128*n_hint*(max_rebuilds+1)*(len(continuous)+min(32, n_hint)*len(categorical)+8)
    if worst_work > max_work:
        _error(f"Declared TwoStep work {worst_work} exceeds max_work; lower max_preclusters.", "resource_limit")
    workspace = plan_workspace("TwoStep CF tree, hierarchy and complete state", {
        "bounded_input_and_training_state": 128*2000*(len(columns)+1),
        "tree_cf_and_all_cuts": 128*max_preclusters**2*(len(continuous)+32*len(categorical)+1),
        "complete_cut_row_lists_and_json": 128*2000*max_preclusters,
        "nodes_and_distance_buffers": 128*(max_nodes*branch_factor+max_preclusters**2),
        "adaptive_history_and_validation": trace_bytes,
    }, budget_bytes=min(max_bytes, workspace_budget_bytes()))
    frame, positions, input_n, input_plan = _frame(data, columns, missing, max_bytes, categorical)
    if len(frame) < 2:
        _error("At least two complete observations are required.", "insufficient_sample")
    X, codes, scaling, descriptors = _prepare(frame, continuous, categorical)
    globalvar = X.square().mean(0)
    levels = [len(d["levels"]) for d in descriptors]
    insertion = torch.arange(len(frame)) if order == "input" else torch.randperm(len(frame), generator=torch.Generator(device="cpu").manual_seed(seed))
    tree = kernel.build_tree(X, codes, levels, globalvar, insertion, threshold=threshold,
                             branch_factor=branch_factor, max_preclusters=max_preclusters, max_nodes=max_nodes,
                             rebuild=rebuild, max_rebuilds=max_rebuilds,
                             noise_fraction=noise_fraction if noise == "adaptive" else 0.0,
                             max_work=tree_work or max_work)
    kept = [cf for cf in tree.preclusters if cf.count >= min_precluster_size]
    if not kept:
        _error("Every precluster is excluded by min_precluster_size.", "insufficient_sample")
    noise = sorted(positions[i] for cf in tree.preclusters if cf.count < min_precluster_size for i in cf.rows)
    if tree.noise_cf is not None:
        noise = sorted(noise+[positions[i] for i in tree.noise_cf.rows])
    hierarchy = kernel.agglomerate(kept, globalvar)
    cuts = {str(k): [_record(cf, positions) for cf in cut] for k, cut in hierarchy.cuts.items()}
    N = len(positions)-len(noise)
    per_cluster = 2*len(continuous)+sum(level-1 for level in levels)
    criteria = []
    for k in range(1, min(max_clusters+(selection == "two_stage"), len(cuts))+1):
        score = sum(kernel.xi(cf, globalvar) for cf in hierarchy.cuts[k])
        parameters = k*per_cluster
        criteria.append([k, score, parameters, -2*score+parameters*math.log(N), -2*score+2*parameters])
    if n_clusters is not None and str(n_clusters) not in cuts:
        _error("Fixed n_clusters exceeds the non-noise precluster count.", "invalid_option")
    selection_trace = None
    if n_clusters is not None:
        selected = n_clusters
    elif selection == "two_stage":
        selected, selection_trace = _select_two_stage(criteria, list(hierarchy.merges), criterion, max_clusters)
    else:
        selected = min(criteria, key=lambda row: (row[3 if criterion == "bic" else 4], row[0]))[0]
    controls = dict(n_clusters=n_clusters, criterion=criterion, max_clusters=max_clusters, threshold=threshold,
                    branch_factor=branch_factor, max_preclusters=max_preclusters, max_nodes=max_nodes,
                    order=order, seed=seed, missing=missing, min_precluster_size=min_precluster_size)
    if extended:
        controls.update(selection=selection, rebuild=rebuild, max_rebuilds=max_rebuilds,
                        noise="adaptive" if noise_fraction and tree.tree.get("noise_fraction") else "none",
                        noise_fraction=noise_fraction)
    # Fingerprint same physical corpus before tree insertion; category identity
    # includes scalar types. This binds stability comparisons to the same rows.
    raw = [[_label(float(frame[name].iloc[i])) for name in continuous]
           + [_label(frame[name].iloc[i]) for name in categorical] for i in range(len(frame))]
    state = dict(version=2 if extended else 1, kind="twostep", continuous=continuous, categorical=descriptors,
                 scaling=scaling, global_variance=globalvar.tolist(), input_n=input_n,
                 positions=positions, missing_positions=[i for i in range(input_n) if i not in positions],
                 noise_positions=noise, order=[positions[i] for i in insertion.tolist()],
                 training_numeric=X.tolist(), training_codes=codes.tolist(),
                 raw_sample=raw,
                 sample_sha256=_seal({"columns": columns, "positions": positions, "input_n": input_n, "raw": raw}),
                 preclusters=[_record(cf, positions) for cf in tree.preclusters],
                 tree=tree.tree, tree_local_indices=True,
                 tree_diagnostics=dict(nodes=tree.node_count, depth=tree.depth, splits=tree.split_count,
                                       distance_evaluations=tree.distance_evaluations,
                                       roundoff_clamps=tree.roundoff_clamps, roundoff_rule=kernel.ROUNDING_RULE),
                 hierarchy_diagnostics=dict(distance_evaluations=hierarchy.distance_evaluations,
                                            roundoff_clamps=hierarchy.roundoff_clamps),
                 cuts=cuts, merges=list(hierarchy.merges), criteria=criteria, selected_k=selected,
                 controls=controls, resources=dict(work_bound=worst_work, workspace=workspace.record(), input=input_plan))
    if extended:
        state.update(noise_cf=_record(tree.noise_cf, positions) if tree.noise_cf is not None else None,
                     selection_trace=selection_trace)
        if rebuild:
            state["resources"]["tree_work_bound"] = tree_work
    return _output(state)


@resident_cpu
def twostep_cut(result, n_clusters, *, max_work=DEFAULT_WORK, max_bytes=DEFAULT_BYTES, device="cpu", weights=None):
    """Replay any saved non-noise hierarchy cut without collecting or refitting."""
    max_work, max_bytes = _domain(device, weights, max_work, max_bytes)
    validation_work, _ = _validation_cost(result)
    hint = result.attrs["twostep_state"]
    replay_work = 16*len(hint["positions"])*len(hint["cuts"])
    if validation_work+replay_work > max_work:
        _error("Combined saved validation and cut replay exceed max_work.", "resource_limit")
    state = _state(result, max_work=max_work, max_bytes=max_bytes)
    n_clusters = _integer(n_clusters, "n_clusters", 1, len(state["cuts"]))
    if 16*len(state["positions"])*len(state["cuts"]) > max_work:
        _error("Saved hierarchy replay exceeds max_work.", "resource_limit")
    plan_workspace("TwoStep saved cut replay", {"complete_saved_state_and_copy": 4*len(json.dumps(state).encode())}, budget_bytes=min(max_bytes, workspace_budget_bytes()))
    copied = copy.deepcopy(state)
    copied["selected_k"] = n_clusters
    copied["controls"]["n_clusters"] = n_clusters
    if copied["version"] == 2:
        copied["selection_trace"] = None
    return _output(copied)


@resident_cpu
def twostep_assign(result, *, data, missing="raise", max_work=DEFAULT_WORK, max_bytes=DEFAULT_BYTES, device="cpu", weights=None):
    """Assign new rows by nearest saved CF merge loss and training category maps.

    New rows do not update training CF, scaling or maps. Adaptive noise uses
    the saved training log-volume cutoff: nearest non-noise cluster distance
    must be strictly below it. Legacy results always assign complete queries.
    """
    max_work, max_bytes = _domain(device, weights, max_work, max_bytes)
    validation_work, _ = _validation_cost(result)
    hint = result.attrs["twostep_state"]
    selected = _integer(hint["selected_k"], "selected_k", 1, len(hint["cuts"]))
    work = 16*2000*selected*(len(hint["continuous"])+sum(len(d["levels"]) for d in hint["categorical"])+1)
    if validation_work+work > max_work:
        _error("Combined saved validation and assignment exceed max_work.", "resource_limit")
    state = _state(result, max_work=max_work, max_bytes=max_bytes)
    continuous = state["continuous"]
    categorical = [d["name"] for d in state["categorical"]]
    cut = state["cuts"][str(state["selected_k"])]
    work = 16*2000*len(cut)*(len(continuous)+sum(len(d["levels"]) for d in state["categorical"])+1)
    if work > max_work:
        _error("Saved assignment work exceeds max_work.", "resource_limit")
    plan = plan_workspace("TwoStep saved assignment", {"saved_cf_and_query": 128*2000*(len(continuous)+len(categorical)+1)+4*len(json.dumps(state))}, budget_bytes=min(max_bytes, workspace_budget_bytes()))
    frame, positions, size, input_plan = _frame(data, continuous+categorical, missing, max_bytes, categorical)
    X, codes, _, _ = _prepare(frame, continuous, categorical, state["categorical"], state["scaling"])
    globalvar = torch.tensor(state["global_variance"], dtype=DTYPE)
    clusters = [_cf(cf) for cf in cut]
    levels = [len(d["levels"]) for d in state["categorical"]]
    rows = {i: [i, None, None, "missing"] for i in range(size)}
    for j, position in enumerate(positions):
        singleton = kernel.CF(1, X[j], torch.zeros_like(X[j]),
                              tuple(torch.nn.functional.one_hot(codes[j, c], levels[c]).to(torch.int64) for c in range(len(levels))), (state["input_n"]+j,))
        distances = [kernel.merge_loss(singleton, cf, globalvar) for cf in clusters]
        label = min(range(len(distances)), key=lambda i: (distances[i], i))
        query_noise = state["controls"].get("noise") == "adaptive" and distances[label] >= state["tree"]["noise_cutoff"]
        rows[position] = [position, 0 if query_noise else label+1, distances[label], "noise" if query_noise else "cluster"]
    return _saved(TableSet({"assignments": table(list(rows.values()), columns=["position", "cluster", "merge_loss", "status"])},
                           title="Saved TwoStep assignment", method="twostep_assignment", training_state_sha256=result.attrs["state_sha256"],
                           n_input=size, n_complete=len(positions), n_missing=size-len(positions),
                           policy="nearest saved merge loss; strict training log-volume noise cutoff" if state["controls"].get("noise") == "adaptive" else "nearest saved merge loss; no query noise classification", device="cpu", dtype="float64",
                           resources=dict(work_bound=work, workspace=plan.record(), input=input_plan)))


@resident_cpu
def twostep_save(result, *, max_work=DEFAULT_WORK, max_bytes=DEFAULT_BYTES):
    """Export every TwoStep table, typed mapping, moment and hierarchy as JSON."""
    max_work, max_bytes = _domain("cpu", None, max_work, max_bytes)
    _state(result, max_work=max_work, max_bytes=max_bytes)
    return summary_state(result)


@resident_cpu
def twostep_load(state, *, max_work=DEFAULT_WORK, max_bytes=DEFAULT_BYTES):
    """Restore and validate complete saved TwoStep state without refitting."""
    max_work, max_bytes = _domain("cpu", None, max_work, max_bytes)
    if not isinstance(state, str):
        _error("Supply complete saved TwoStep JSON.", "invalid_state")
    plan_workspace("TwoStep JSON restore", {"complete_json_and_object_buffers": 8*len(state.encode())}, budget_bytes=min(max_bytes, workspace_budget_bytes()))
    result = restore_summary(state)
    _state(result, max_work=max_work, max_bytes=max_bytes)
    return result
