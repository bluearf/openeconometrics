"""Known calibrated banks and exact Bayes on a declared finite latent model."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import math
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, _json_safe, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.resources import plan_workspace, workspace_budget_bytes

DTYPE = torch.float64
DEFAULT_BYTES = 128 * 1024**2
DEFAULT_WORK = 300_000_000
STATE_LIMIT = 8 * 1024**2
SOURCES = ["https://www.stata.com/manuals/irtirtnrm.pdf", "https://www.stata.com/manuals/irtirtpcm.pdf",
           "https://philchalmers.github.io/mirt/reference/mirt.html"]


def _error(message, code="invalid_spec"):
    raise AnalysisError(code, message)


def _integer(value, name, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        _error(f"{name} must be a primitive integer in [{lo},{hi}].")
    return value


def _number(value, name, lo, hi):
    if type(value) not in (int, float) or not lo <= value <= hi or not math.isfinite(value):
        _error(f"{name} must be a finite primitive number in [{lo},{hi}].")
    return float(value)


def _names(items):
    if type(items) is not list or not 1 <= len(items) <= 16 or any(type(x) is not str or not 1 <= len(x) <= 256 for x in items) or len(set(items)) != len(items):
        _error("items must be 1..16 unique primitive names of 1..256 characters.")
    return items


def _list(value, name, n):
    if type(value) is not list or len(value) != n:
        _error(f"{name} must be a primitive list of length {n}.")
    return value


def _primitive(value):
    if value is None or type(value) is bool:
        return value
    if type(value) is str and len(value) <= 256:
        return value
    if type(value) is int and abs(value) <= 2**53-1:
        return value
    if type(value) is float and math.isfinite(value) and abs(value) <= 2**53-1:
        return value
    _error("Row index labels must be finite primitive scalar values; strings <=256 characters.", "invalid_index")


def _seal(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _budget(max_bytes):
    return min(_integer(max_bytes, "max_bytes", 1, 2**63-1), workspace_budget_bytes())


def _plan(operation, buffers, max_bytes):
    return plan_workspace(operation, buffers, budget_bytes=_budget(max_bytes)).record()


def _escaped_size(value):
    # All callers first validate bounded primitive scalars. Account for the
    # default full-summary JSON writer's Unicode escaping, not Python len().
    return len(json.dumps(value, ensure_ascii=True, allow_nan=False))


def _bank_buffers(state):
    return {"bank_definitions_tables_serialization": 65536+4096*len(state["items"])+24*sum(_escaped_size(name) for name in state["items"])}


def _saved(result):
    result.attrs = dict(sorted(result.attrs.items()))
    saved_summary(result)
    for name, frame in list(result.items()):
        result[name] = table(frame.to_numpy().tolist(), columns=list(frame.columns), index=list(frame.index))
    ordered = sorted(result.items())
    result.clear()
    result.update(ordered)
    return result


def _validate_bank(state):
    if type(state) is not dict or set(state) != {"version", "kind", "items", "definitions"} or state["version"] != 1 or type(state["version"]) is not int or type(state["kind"]) is not str or state["kind"] != "irt_calibrated_bank":
        _error("Invalid calibrated-bank state schema.", "invalid_state")
    items = _names(state["items"])
    definitions = _list(state["definitions"], "definitions", len(items))
    for item in definitions:
        if type(item) is not dict:
            _error("Item definitions must be primitive dictionaries.", "invalid_state")
        family = item.get("family")
        if type(family) is not str:
            _error("Item families must be primitive strings.", "invalid_state")
        k = _integer(item.get("K"), "K", 2, 6)
        scores = _list(item.get("scores"), "scores", k)
        for score in scores:
            _integer(score, "category score", 0, 10)
        common = {"family", "K", "scores"}
        if family == "binary":
            if k != 2 or set(item) != common | {"discrimination", "difficulty", "guessing", "upper"} or scores != [0, 1]:
                _error("Invalid binary bank definition.", "invalid_state")
            _number(item["discrimination"], "discrimination", .1, 5)
            _number(item["difficulty"], "difficulty", -8, 8)
            c = _number(item["guessing"], "guessing", 0, .4)
            u = _number(item["upper"], "upper", .6, 1)
            if c >= u:
                _error("Known binary asymptotes must satisfy guessing < upper.")
        elif family in ("grm", "gpcm"):
            if set(item) != common | {"discrimination", "thresholds"}:
                _error("Invalid polytomous bank definition.", "invalid_state")
            _number(item["discrimination"], "discrimination", .1, 5)
            steps = _list(item["thresholds"], "thresholds", k-1)
            for value in steps:
                _number(value, "threshold/step difficulty", -8, 8)
            if family == "grm" and any(b <= a for a, b in zip(steps, steps[1:])):
                _error("GRM thresholds must be strictly increasing; tied thresholds are unsupported.")
        elif family == "nrm":
            if set(item) != common | {"slopes", "intercepts"}:
                _error("Invalid nominal bank definition.", "invalid_state")
            slopes = _list(item["slopes"], "slopes", k)
            intercepts = _list(item["intercepts"], "intercepts", k)
            for value in slopes:
                _number(value, "category slope", -5, 5)
            for value in intercepts:
                _number(value, "category intercept", -12, 12)
            if slopes[0] != 0 or intercepts[0] != 0 or len(set(slopes)) < 2:
                _error("NRM requires exactly zero baseline slope/intercept and nonconstant slopes.")
        else:
            _error("Known banks support binary, grm, gpcm and nrm only.", "unsupported_option")
    return state


def _bank_tables(state):
    parameters, scores = [], []
    for name, definition in zip(state["items"], state["definitions"]):
        for key in ("discrimination", "difficulty", "guessing", "upper"):
            if key in definition:
                parameters.append([name, key, definition[key]])
        for key in ("thresholds", "slopes", "intercepts"):
            for i, value in enumerate(definition.get(key, [])):
                parameters.append([name, f"{key}_{i if key != 'thresholds' else i+1}", value])
        scores.extend([name, i, value] for i, value in enumerate(definition["scores"]))
    return {
        "items": table([[name, d["family"], d["K"]] for name, d in zip(state["items"], state["definitions"])], columns=["item", "family", "categories"]),
        "parameters": table(parameters, columns=["item", "parameter", "known_value"]),
        "category_scores": table(scores, columns=["item", "category_code", "score"]),
    }


def _assemble_bank(state, max_bytes):
    _validate_bank(state)
    resource = _plan("known calibrated bank and portable tables", _bank_buffers(state), max_bytes)
    return _saved(TableSet(_bank_tables(state), title="Known calibrated IRT item bank", method="irt_calibrated_bank",
        bank_state=copy.deepcopy(state), state_sha256=_seal(state), resources=resource, sources=list(SOURCES),
        dtype="float64", device="cpu", calibration="parameters supplied and treated as known; no fitting or parameter SE",
        latent_model="no latent distribution assigned until explicit support/masses are supplied",
        scope="unidimensional local independence; category codes and scoring values are distinct"))


@resident_cpu
def irt_bank_binary(items: list[str], discrimination: list, difficulty: list, *, guessing: list | None = None,
                    upper: list | None = None, max_bytes: int = DEFAULT_BYTES, device: str = "cpu") -> TableSet:
    """Known binary c+(u-c)sigmoid(a(theta-b)); no calibration estimation."""
    if device != "cpu":
        _error("Known banks support resident CPU only.", "unsupported_option")
    items = _names(items)
    n = len(items)
    a, b = _list(discrimination, "discrimination", n), _list(difficulty, "difficulty", n)
    c = [0.0]*n if guessing is None else _list(guessing, "guessing", n)
    u = [1.0]*n if upper is None else _list(upper, "upper", n)
    state = {"version": 1, "kind": "irt_calibrated_bank", "items": items, "definitions": [
        {"family": "binary", "K": 2, "discrimination": _number(ai, "discrimination", .1, 5),
         "difficulty": _number(bi, "difficulty", -8, 8), "guessing": _number(ci, "guessing", 0, .4),
         "upper": _number(ui, "upper", .6, 1), "scores": [0, 1]} for ai, bi, ci, ui in zip(a, b, c, u)]}
    return _assemble_bank(state, max_bytes)


@resident_cpu
def irt_bank_polytomous(items: list[str], *, family: str | list[str], thresholds: list | None = None,
                         discrimination: list | None = None, slopes: list | None = None,
                         intercepts: list | None = None, scores: list | None = None,
                         max_bytes: int = DEFAULT_BYTES, device: str = "cpu") -> TableSet:
    """Known GRM, GPCM step difficulties or baseline-zero NRM category logits.

    Family may be a per-item list. Irrelevant parameter rows must be None.
    NRM scoring values are explicit integers 0..10, with gaps/ties allowed.
    """
    if device != "cpu":
        _error("Known banks support resident CPU only.", "unsupported_option")
    items = _names(items)
    n = len(items)
    families = [family]*n if type(family) is str else _list(family, "family", n)
    arguments = {name: [None]*n if value is None else _list(value, name, n) for name, value in
                 (("thresholds", thresholds), ("discrimination", discrimination), ("slopes", slopes), ("intercepts", intercepts), ("scores", scores))}
    definitions = []
    for j, kind in enumerate(families):
        if type(kind) is not str:
            _error("Polytomous family entries must be primitive strings.")
        if kind in ("grm", "gpcm"):
            steps = arguments["thresholds"][j]
            if type(steps) is not list or not 1 <= len(steps) <= 5 or arguments["slopes"][j] is not None or arguments["intercepts"][j] is not None:
                _error("GRM/GPCM require 1..5 threshold/step values and no nominal parameters.")
            definition = {"family": kind, "K": len(steps)+1,
                          "discrimination": _number(1 if arguments["discrimination"][j] is None else arguments["discrimination"][j], "discrimination", .1, 5),
                          "thresholds": [_number(x, "threshold/step", -8, 8) for x in steps]}
        elif kind == "nrm":
            a, d = arguments["slopes"][j], arguments["intercepts"][j]
            if type(a) is not list or not 2 <= len(a) <= 6 or type(d) is not list or len(d) != len(a) or arguments["thresholds"][j] is not None or arguments["discrimination"][j] is not None:
                _error("NRM requires 2..6 matching slopes/intercepts and no ordinal parameters.")
            definition = {"family": kind, "K": len(a), "slopes": [_number(x, "slope", -5, 5) for x in a], "intercepts": [_number(x, "intercept", -12, 12) for x in d]}
        else:
            _error("Polytomous family must be grm, gpcm or nrm.", "unsupported_option")
        score = arguments["scores"][j]
        if kind == "nrm" and score is None:
            _error("NRM requires explicit scoring values; category order is not inferred.")
        definition["scores"] = list(range(definition["K"])) if score is None else _list(score, "scores", definition["K"])
        definitions.append(definition)
    return _assemble_bank({"version": 1, "kind": "irt_calibrated_bank", "items": items, "definitions": definitions}, max_bytes)


def _input(result, key, max_bytes):
    _budget(max_bytes)
    if isinstance(result, TableSet):
        if type(result.attrs) is not dict:
            _error("Calibrated summary metadata must be a dictionary.", "invalid_state")
        return result.attrs.get(key), result.attrs, result
    if type(result) is not str or len(result) > STATE_LIMIT:
        _error("Supply a complete bank/posterior TableSet or summary JSON <=8 MiB.", "invalid_state")
    encoded = result.encode()
    if len(encoded) > STATE_LIMIT:
        _error("Complete calibrated summary JSON exceeds 8 MiB.", "resource_limit")
    _plan("calibrated summary JSON admission", {"encoded_JSON_and_primitive_parser": 12*len(encoded)}, max_bytes)
    try:
        payload = json.loads(result, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
        if set(payload) != {"schema", "title", "attrs", "tables"} or payload["schema"] != "openecon.summary.v1" or type(payload["attrs"]) is not dict or type(payload["tables"]) is not dict:
            raise ValueError("Bad summary schema")
        return payload["attrs"].get(key), payload["attrs"], payload
    except (ValueError, TypeError, RecursionError) as exc:
        raise AnalysisError("invalid_state", "Malformed complete calibrated summary.") from exc


def _canonical_tables(source, attrs, tables, title):
    expected = _saved(TableSet(tables, title=title, **attrs))
    if isinstance(source, TableSet):
        if source.title != title or set(source) != set(expected) or any(not isinstance(source[name], pd.DataFrame) or source[name].shape != expected[name].shape for name in expected):
            _error("Calibrated summary dimensions changed before numerical table copies.", "invalid_state")
        actual = {name: {"columns": list(frame.columns), "index": list(frame.index), "data": frame.to_numpy().tolist()} for name, frame in source.items()}
        actual_title = source.title
    else:
        actual, actual_title = source["tables"], source["title"]
    wanted = {name: {"columns": list(frame.columns), "index": list(frame.index), "data": frame.to_numpy().tolist()} for name, frame in expected.items()}
    if actual_title != title or set(actual) != set(wanted):
        _error("Calibrated summary table identities/title changed.", "invalid_state")
    for name in wanted:
        frame = actual[name]
        if type(frame) is not dict or set(frame) != {"columns", "index", "data"} or type(frame["data"]) is not list or type(frame["columns"]) is not list or type(frame["index"]) is not list or len(frame["columns"]) != len(wanted[name]["columns"]) or len(frame["index"]) != len(wanted[name]["index"]) or len(frame["data"]) != len(wanted[name]["data"]) or any(type(row) is not list or len(row) != len(wanted[name]["columns"]) for row in frame["data"]):
            _error("Calibrated summary table dimensions changed before hashing.", "invalid_state")
        # TableSet cells are untrusted even when dimensions match. Admit only
        # primitive scalars before _json_safe can expand a tensor/object cell.
        for values in (frame["columns"], frame["index"], *frame["data"]):
            for value in values:
                if value is None:
                    continue
                if type(value) is str and len(value) <= 512:
                    continue
                if type(value) in (bool, int) and -2**63 <= value <= 2**63:
                    continue
                if type(value) is float and (math.isnan(value) or math.isfinite(value) and abs(value) <= 2**63):
                    continue
                _error("Canonical table labels/cells require bounded primitive scalars.", "invalid_state")
        if json.dumps(_json_safe(frame), sort_keys=True, allow_nan=False) != json.dumps(_json_safe(wanted[name]), sort_keys=True, allow_nan=False):
            _error("Calibrated summary tables disagree with complete state.", "invalid_state")
    return expected


def _canonical(source, attrs, tables, title):
    try:
        return _canonical_tables(source, attrs, tables, title)
    except AnalysisError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise AnalysisError("invalid_state", "Malformed canonical calibrated table state.") from exc


def _attrs(attrs, posterior=False):
    keys = {"method", "state_sha256", "resources", "sources", "dtype", "device", "latent_model"}
    keys |= {"posterior_state", "n", "items", "sample_positions", "indices", "declared_work", "missing", "uncertainty", "quantiles"} if posterior else {"bank_state", "calibration", "scope"}
    if type(attrs) is not dict or set(attrs) != keys:
        _error("Unknown calibrated summary metadata fields.", "invalid_state")
    pending = [(value, 0) for key, value in attrs.items() if key not in ("bank_state", "posterior_state")]
    count, text_units = 0, 0
    while pending:
        value, depth = pending.pop()
        count += 1
        if count > (10000 if posterior else 128) or depth > 8:
            _error("Calibrated metadata complexity exceeded bounds.", "invalid_state")
        if type(value) is dict:
            if len(value) > 128 or any(type(key) is not str or len(key) > 256 for key in value):
                _error("Calibrated metadata dictionary bounds changed.", "invalid_state")
            pending.extend((child, depth+1) for child in value.values())
        elif type(value) is list:
            if len(value) > 1000:
                _error("Calibrated metadata sequence bounds changed.", "invalid_state")
            pending.extend((child, depth+1) for child in value)
        elif value is not None and (type(value) not in (str, int, float, bool) or type(value) is str and len(value) > 2048 or type(value) in (int, float) and not -2**63 <= value <= 2**63):
            _error("Calibrated metadata requires bounded primitive values.", "invalid_state")
        if type(value) is str:
            text_units += len(value)
            if text_units > (300000 if posterior else 2048):
                _error("Calibrated metadata text exceeds admitted portable bounds.", "invalid_state")


@resident_cpu
def _bank(result, *, max_bytes=DEFAULT_BYTES):
    state, attrs, source = _input(result, "bank_state", max_bytes)
    try:
        _validate_bank(state)
        _attrs(attrs)
        _plan("calibrated bank semantic validation", _bank_buffers(state), max_bytes)
        if attrs.get("method") != "irt_calibrated_bank" or attrs.get("dtype") != "float64" or attrs.get("device") != "cpu" or type(attrs.get("state_sha256")) is not str or len(attrs["state_sha256"]) != 64 or _seal(state) != attrs["state_sha256"]:
            _error("Calibrated bank identity/integrity changed.", "invalid_state")
        _canonical(source, attrs, _bank_tables(state), "Known calibrated IRT item bank")
        return copy.deepcopy(state)
    except AnalysisError as exc:
        if exc.code in ("workspace_limit", "resource_limit", "invalid_state"):
            raise
        raise AnalysisError("invalid_state", "Invalid calibrated bank parameters/schema.") from exc


@resident_cpu
def irt_bank_restore(result: TableSet | str, *, max_bytes: int = DEFAULT_BYTES, device: str = "cpu") -> TableSet:
    """Restore and check an own known bank, never reinterpret an MML fitted model."""
    if device != "cpu":
        _error("Bank restoration supports resident CPU only.", "unsupported_option")
    state = _bank(result, max_bytes=max_bytes)
    _, attrs, source = _input(result, "bank_state", max_bytes)
    return _canonical(source, attrs, _bank_tables(state), "Known calibrated IRT item bank")


@resident_cpu
def _logp(state, theta):
    if not isinstance(theta, torch.Tensor) or theta.device.type != "cpu" or theta.dtype != DTYPE or theta.ndim != 1 or not 1 <= len(theta) <= 101 or not bool(torch.isfinite(theta).all()) or bool((theta.abs() > 12).any()):
        _error("theta must be a resident CPU float64 vector, length1..101, bounded to [-12,12].")
    result = []
    for definition in state["definitions"]:
        kind = definition["family"]
        if kind == "binary":
            z = definition["discrimination"]*(theta-definition["difficulty"])
            c, u = definition["guessing"], definition["upper"]
            difference = math.log(u-c)
            success = torch.logaddexp(torch.full_like(z, math.log(c) if c else -math.inf), difference+F.logsigmoid(z))
            failure = torch.logaddexp(torch.full_like(z, math.log1p(-u) if u < 1 else -math.inf), difference+F.logsigmoid(-z))
            result.append(torch.stack([failure, success], dim=1))
        elif kind == "grm":
            a, thresholds = definition["discrimination"], definition["thresholds"]
            z = a*(theta[:, None]-torch.tensor(thresholds, dtype=DTYPE)[None, :])
            columns = [F.logsigmoid(-z[:, 0])]
            for k in range(len(thresholds)-1):
                # Compute the gap from thresholds, avoiding theta cancellation
                # and underflow even for distinct subnormal thresholds.
                log_gap = math.log(a)+math.log(thresholds[k+1]-thresholds[k])
                log_difference = log_gap if log_gap < -36 else math.log(-math.expm1(-math.exp(log_gap)))
                columns.append(F.logsigmoid(z[:, k])+F.logsigmoid(-z[:, k+1])+log_difference)
            columns.append(F.logsigmoid(z[:, -1]))
            result.append(torch.stack(columns, dim=1))
        elif kind == "gpcm":
            steps = torch.tensor(definition["thresholds"], dtype=DTYPE)
            increments = definition["discrimination"]*(theta[:, None]-steps[None, :])
            logits = torch.cat([torch.zeros((len(theta), 1), dtype=DTYPE), increments.cumsum(1)], dim=1)
            result.append(F.log_softmax(logits, dim=1))
        else:
            logits = theta[:, None]*torch.tensor(definition["slopes"], dtype=DTYPE)[None, :]+torch.tensor(definition["intercepts"], dtype=DTYPE)[None, :]
            result.append(F.log_softmax(logits, dim=1))
    return result


def _response_size(data, state, *, max_rows=1000):
    _integer(max_rows, "max_rows", 1, 1000)
    items = state["items"]
    if isinstance(data, pd.DataFrame):
        n = len(data)
        if any(list(data.columns).count(name) != 1 for name in items):
            _error("Each bank item must occur exactly once in response columns.")
    elif isinstance(data, Mapping):
        if any(name not in data for name in items):
            _error("Response mapping is missing bank items.")
        if any(isinstance(data[name], (str, bytes, Mapping)) for name in items):
            _error("Response columns require resident sequences, not scalar strings or mappings.", "unsupported_data")
        try:
            sizes = [len(data[name]) for name in items]
        except TypeError as exc:
            raise AnalysisError("unsupported_data", "Responses require finite resident column sequences.") from exc
        if len(set(sizes)) != 1:
            _error("Selected response columns must have equal length.")
        n = sizes[0]
    elif type(data) is list:
        n = len(data)
        if n <= max_rows and any(not isinstance(row, Mapping) or any(name not in row for name in items) for row in data):
            _error("Each response record must explicitly include all bank items.")
    else:
        _error("Responses require resident DataFrame, column mapping or explicit records; no Dataset/device transfer.", "unsupported_data")
    if not 1 <= n <= max_rows:
        _error(f"Responses admit1..{max_rows} people before selection/coercion.", "resource_limit")
    return n


def _response_indices(data, n):
    indices = []
    for label in data.index if isinstance(data, pd.DataFrame) else range(n):
        if isinstance(label, torch.Tensor):
            _error("Tensor row labels require an explicit primitive conversion.", "unsupported_data")
        indices.append(_primitive(label.item() if hasattr(label, "item") else label))
    return indices


@resident_cpu
def _responses(data, state, *, max_rows=1000, max_bytes=DEFAULT_BYTES):
    n = _response_size(data, state, max_rows=max_rows)
    items = state["items"]
    _plan("calibrated response admission", {"selected_response_copies_codes_and_indices": 64*n*(len(items)+2)}, max_bytes)
    indices = _response_indices(data, n)
    try:
        if isinstance(data, pd.DataFrame):
            source_rows = data.loc[:, items].itertuples(index=False, name=None)
        elif isinstance(data, Mapping):
            if any(isinstance(data[name], torch.Tensor) for name in items):
                _error("Response tensor columns are unsupported; use resident primitive sequences.", "unsupported_data")
            # Mappings are positional sequences. Do not align Series indexes
            # and silently expand beyond the admitted row count.
            source_rows = zip(*(list(data[name]) for name in items))
        else:
            source_rows = ([row[name] for name in items] for row in data)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError("invalid_categories", "Malformed resident response columns/records.") from exc
    rows = []
    for row in source_rows:
        codes = []
        for value, definition in zip(row, state["definitions"]):
            if value is None or value is pd.NA or value is pd.NaT or type(value) is float and math.isnan(value):
                codes.append(-1)
                continue
            if isinstance(value, torch.Tensor):
                _error("Response tensor cells require an explicit resident primitive conversion.", "unsupported_data")
            if hasattr(value, "item"):
                value = value.item()
            if type(value) is float and math.isnan(value):
                codes.append(-1)
                continue
            if type(value) not in (int, float) or not 0 <= value < definition["K"] or not math.isfinite(value) or value != int(value):
                _error("Observed responses must be integer category codes0..K-1; booleans/-1/infinity are not missing.", "invalid_categories")
            codes.append(int(value))
        rows.append(codes)
    return torch.tensor(rows, dtype=torch.int64), indices


def _prior(support, masses):
    if type(support) is not list or not 1 <= len(support) <= 101:
        _error("Finite support must be a primitive list with1..101 points.")
    _list(masses, "masses", len(support))
    for value in support:
        _number(value, "support", -8, 8)
    if any(b <= a for a, b in zip(support, support[1:])):
        _error("Finite support must be sorted and unique.")
    for value in masses:
        _number(value, "prior mass", 0, 1)
        if value <= 0:
            _error("Every finite prior mass must be strictly positive.")
    if abs(math.fsum(masses)-1) > 1e-12:
        _error("Finite masses must sum to one within1e-12; no implicit normalization.")


def _posterior_plan(state, max_bytes, max_work, extra_work=0, extra_buffers=None):
    n, q, j = len(state["responses"]), len(state["support"]), len(state["bank"]["items"])
    work = 128*n*q*j+128*q*sum(d["K"] for d in state["bank"]["definitions"])+128*n*q
    _integer(max_work, "max_work", 1, 2**63-1)
    _integer(extra_work, "extra_work", 0, 2**63-1)
    if work+extra_work > max_work:
        _error("Complete finite posterior plus downstream work exceeds max_work before tensor allocation.", "resource_limit")
    label_bytes = sum(_escaped_size(index) for index in state.get("indices", range(n)))
    name_bytes = sum(_escaped_size(name) for name in state["bank"]["items"])
    portable_bound = 32768+28*n*(2*q+2*j+24)+5*label_bytes+4*name_bytes+512*q
    if portable_bound > STATE_LIMIT:
        _error("Complete finite posterior's escaped full-summary bound exceeds the 8 MiB portable domain before numerical allocation.", "resource_limit")
    buffers = {"posterior_numerical_work": 8*n*(6*q+3*j+24),
               "full_state_tables_and_serialization": 64*n*(3*q+3*j+32),
               "item_probability_workspace": 32*q*sum(d["K"] for d in state["bank"]["definitions"]),
               "escaped_full_summary_bound": portable_bound}
    if extra_buffers is not None:
        if type(extra_buffers) is not dict or len(extra_buffers) > 32 or any(type(k) is not str or not 1 <= len(k) <= 256 or k in buffers for k in extra_buffers):
            _error("Downstream buffer declarations must be a bounded distinct-name dictionary.")
        buffers.update({k: _integer(v, "extra buffer bytes", 0, 2**63-1) for k, v in extra_buffers.items()})
    return _plan("exact finite calibrated posterior plus downstream workspace", buffers, max_bytes), work+extra_work


def _bayes(state):
    responses = torch.tensor(state["responses"], dtype=torch.int64)
    theta = torch.tensor(state["support"], dtype=DTYPE)
    masses = torch.tensor(state["masses"], dtype=DTYPE)
    likelihood = torch.zeros((len(responses), len(theta)), dtype=DTYPE)
    for j, logp in enumerate(_logp(state["bank"], theta)):
        observed = responses[:, j] >= 0
        if bool(observed.any()):
            likelihood[observed] += logp[:, responses[observed, j]].T
    joint = likelihood+masses.log()[None, :]
    evidence = torch.logsumexp(joint, dim=1)
    posterior = (joint-evidence[:, None]).exp()
    return posterior, evidence


def _posterior_tables(state):
    posterior = torch.tensor(state["posterior"], dtype=DTYPE)
    theta = torch.tensor(state["support"], dtype=DTYPE)
    mean = posterior@theta
    sd = (posterior*(theta[None, :]-mean[:, None]).square()).sum(1).sqrt()
    entropy = -torch.special.xlogy(posterior, posterior).sum(1)
    cdf = posterior.cumsum(1)
    cdf[:, -1] = 1.0
    quantiles = torch.stack([theta[(cdf < value).sum(1)] for value in (.025, .5, .975)], dim=1)
    rows, probabilities, responses = [], [], []
    for i, index in enumerate(state["indices"]):
        observed = sum(code >= 0 for code in state["responses"][i])
        rows.append([i, index, observed, float(mean[i]), float(sd[i]), float(entropy[i]), state["log_evidence"][i], *quantiles[i].tolist()])
        probabilities.append([i, index, *state["posterior"][i]])
        responses.append([i, index, *[None if code < 0 else code for code in state["responses"][i]]])
    return {
        "summary": table(rows, columns=["source_position", "index", "observed_items", "posterior_mean", "posterior_sd", "entropy", "log_evidence", "q025", "q500", "q975"]),
        "posterior": table(probabilities, columns=["source_position", "index"]+[f"mass_{i}" for i in range(len(theta))]),
        "prior": table([[x, p] for x, p in zip(state["support"], state["masses"])], columns=["theta", "mass"]),
        "responses": table(responses, columns=["source_position", "index"]+state["bank"]["items"]),
    }


def _assemble_posterior(state, resource, work):
    return _saved(TableSet(_posterior_tables(state), title="Exact finite calibrated IRT posterior", method="irt_posterior",
        posterior_state=copy.deepcopy(state), state_sha256=_seal(state), n=len(state["responses"]), items=state["bank"]["items"],
        sample_positions=list(range(len(state["responses"]))), indices=state["indices"], resources=resource, declared_work=work,
        dtype="float64", device="cpu", sources=list(SOURCES), missing="each observed item contributes; all-missing people retain the declared prior",
        latent_model="the supplied finite support and masses are the actual discrete prior; no continuous quadrature approximation",
        uncertainty="exact finite conditional posterior given known parameters; no parameter-calibration uncertainty",
        quantiles="inverse discrete CDF at .025,.5,.975; smallest support value reaching the probability"))


@resident_cpu
def irt_posterior(bank: TableSet | str, *, data: Any, support: list, masses: list,
                   max_work: int = DEFAULT_WORK, max_bytes: int = DEFAULT_BYTES, device: str = "cpu") -> TableSet:
    """Exact finite-support Bayes with partial and all-missing people retained."""
    if device != "cpu":
        _error("Finite posterior supports resident CPU float64 only.", "unsupported_option")
    bank_state = _bank(bank, max_bytes=max_bytes)
    _prior(support, masses)
    n = _response_size(data, bank_state)
    _plan("finite posterior primitive index admission", {"source_indices": 64*n}, max_bytes)
    skeleton = {"bank": bank_state, "support": support, "responses": [None]*n, "indices": _response_indices(data, n)}
    resource, work = _posterior_plan(skeleton, max_bytes, max_work)
    responses, indices = _responses(data, bank_state, max_bytes=max_bytes)
    state = {"version": 1, "kind": "irt_finite_posterior", "bank": bank_state,
             "responses": responses.tolist(), "indices": indices, "support": [float(x) for x in support],
             "masses": [float(x) for x in masses]}
    posterior, evidence = _bayes(state)
    state.update(posterior=posterior.tolist(), log_evidence=evidence.tolist())
    return _assemble_posterior(state, resource, work)


def _posterior_input(result, *, max_bytes=DEFAULT_BYTES):
    state, attrs, _ = _input(result, "posterior_state", max_bytes)
    try:
        if type(state) is not dict or set(state) != {"version", "kind", "bank", "responses", "indices", "support", "masses", "posterior", "log_evidence"} or type(state["version"]) is not int or state["version"] != 1 or type(state["kind"]) is not str or state["kind"] != "irt_finite_posterior":
            _error("Invalid finite posterior schema.", "invalid_state")
        _validate_bank(state["bank"])
        _attrs(attrs, posterior=True)
        _prior(state["support"], state["masses"])
        responses = state["responses"]
        if type(responses) is not list or not 1 <= len(responses) <= 1000:
            _error("Finite posterior responses require1..1000 people.", "invalid_state")
        n, j, q = len(responses), len(state["bank"]["items"]), len(state["support"])
        for row in responses:
            _list(row, "response row", j)
            for code, definition in zip(row, state["bank"]["definitions"]):
                _integer(code, "saved response code", -1, definition["K"]-1)
        for index in _list(state["indices"], "indices", n):
            _primitive(index)
        for row in _list(state["posterior"], "posterior", n):
            for value in _list(row, "posterior row", q):
                _number(value, "posterior mass", 0, 1)
            if abs(math.fsum(row)-1) > 1e-10:
                _error("Saved posterior rows must sum to one.", "invalid_state")
        for value in _list(state["log_evidence"], "log_evidence", n):
            _number(value, "log evidence", -20000, 1e-10)
        return state
    except AnalysisError as exc:
        if exc.code in ("workspace_limit", "resource_limit", "invalid_state"):
            raise
        raise AnalysisError("invalid_state", "Finite posterior primitive schema/prior/response state is invalid.") from exc


@resident_cpu
def _posterior(result, *, max_bytes=DEFAULT_BYTES, max_work=DEFAULT_WORK, extra_work=0, extra_buffers=None):
    state = _posterior_input(result, max_bytes=max_bytes)
    _posterior_plan(state, max_bytes, max_work, extra_work, extra_buffers)
    _, attrs, source = _input(result, "posterior_state", max_bytes)
    if attrs.get("method") != "irt_posterior" or attrs.get("device") != "cpu" or attrs.get("dtype") != "float64" or type(attrs.get("state_sha256")) is not str or len(attrs["state_sha256"]) != 64 or _seal(state) != attrs["state_sha256"]:
        _error("Finite posterior integrity/identity changed.", "invalid_state")
    posterior, evidence = _bayes(state)
    saved = torch.tensor(state["posterior"], dtype=DTYPE)
    saved_evidence = torch.tensor(state["log_evidence"], dtype=DTYPE)
    if not torch.allclose(saved, posterior, atol=2e-12, rtol=2e-11) or not torch.allclose(saved_evidence, evidence, atol=2e-10, rtol=2e-11):
        _error("Saved finite posterior/evidence is not coherent exact Bayes.", "invalid_state")
    n = len(state["responses"])
    if attrs.get("n") != n or attrs.get("items") != state["bank"]["items"] or attrs.get("indices") != state["indices"] or attrs.get("sample_positions") != list(range(n)):
        _error("Finite posterior sample/index metadata changed.", "invalid_state")
    _canonical(source, attrs, _posterior_tables(state), "Exact finite calibrated IRT posterior")
    return copy.deepcopy(state)


@resident_cpu
def irt_posterior_restore(result: TableSet | str, *, max_work: int = DEFAULT_WORK,
                           max_bytes: int = DEFAULT_BYTES, device: str = "cpu") -> TableSet:
    """Guard full finite Bayes state and canonical tables, without fitting."""
    if device != "cpu":
        _error("Finite posterior restoration supports resident CPU only.", "unsupported_option")
    state = _posterior(result, max_bytes=max_bytes, max_work=max_work)
    _, attrs, source = _input(result, "posterior_state", max_bytes)
    return _canonical(source, attrs, _posterior_tables(state), "Exact finite calibrated IRT posterior")
