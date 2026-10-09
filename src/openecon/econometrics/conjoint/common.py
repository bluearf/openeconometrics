"""Explicit geometry, bounded CPU work and complete integrity-checked artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet, _json_safe, table
from openecon.econometrics.multivariate import common as c
from openecon.resources import plan_workspace

PROCEDURES = (
    "conjoint_plan", "conjoint_orthogonal", "conjoint_diagnostics", "conjoint_fit",
    "conjoint_predict", "conjoint_holdout", "conjoint_importance", "conjoint_simulate",
)
MAX_ARTIFACT = 32 * 1024**2


def canonical(value):
    return json.dumps(_json_safe(value), sort_keys=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def label(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (str, int, float)) or (
        isinstance(value, float) and not math.isfinite(value)
    ) or (isinstance(value, str) and (not value or len(value) > 128)):
        raise AnalysisError("invalid_label", "IDs/levels must be nonempty short strings or finite numbers; no booleans.")
    return value


def key(value):
    value = label(value)
    # Numeric scalar equality survives pandas' int/float column promotion.
    # Strings retain distinct identity; original labels remain in saved state.
    return canonical(int(value) if isinstance(value, float) and value.is_integer() else value)


def name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_]\w{0,47}", value):
        raise AnalysisError("invalid_spec", "Names must be Python identifiers up to 48 characters.")
    return value


def definition(attributes, factors=None):
    if not isinstance(attributes, dict) or not 1 <= len(attributes) <= 16:
        raise AnalysisError("invalid_spec", "Declare 1..16 ordered attributes and their finite level lists.")
    result = {}
    for attr, levels in attributes.items():
        name(attr)
        if not isinstance(levels, (list, tuple)) or not 2 <= len(levels) <= 32:
            raise AnalysisError("invalid_spec", "Each attribute needs 2..32 distinct declared levels.")
        values = [label(v) for v in levels]
        if len(set(values)) != len(values):
            raise AnalysisError("invalid_spec", "Attribute levels must be distinct, including numerically equal values.")
        result[attr] = values
    if factors is None:
        factors = {}
    if not isinstance(factors, dict) or set(factors) - set(result):
        raise AnalysisError("invalid_spec", "Factor specifications must name declared attributes.")
    modes = {attr: factors.get(attr, "discrete") for attr in result}
    for attr, mode in modes.items():
        c.check_choice(mode, "factor", ("discrete", "linear", "ideal"))
        if mode != "discrete":
            if not all(isinstance(v, (int, float)) and abs(v) <= 1e6 for v in result[attr]):
                raise AnalysisError("invalid_spec", "Linear/quadratic levels must be finite numeric values within +/-1e6.")
            if mode == "ideal" and len(result[attr]) < 3:
                raise AnalysisError("invalid_spec", "An ideal/anti-ideal quadratic needs at least three declared levels.")
    terms = 1 + sum(len(result[a]) - 1 if m == "discrete" else 1 if m == "linear" else 2
                    for a, m in modes.items())
    if terms > 128:
        raise AnalysisError("resource_limit", "Conjoint designs support at most 128 coefficients.")
    return result, modes, terms


def guard(operation, n, p, subjects=1, *, device="cpu", weights=None, max_work=100_000_000, work=None, extra_buffers=None):
    if device != "cpu":
        raise AnalysisError("unsupported_device", "Conjoint procedures require resident CPU float64.")
    if weights is not None:
        raise AnalysisError("unsupported_weights", "Conjoint observation/respondent weights are unsupported.")
    c.check_count(max_work, "max_work")
    if n > 4096 or p > 128 or subjects > 128 or n * subjects > 100000:
        raise AnalysisError("resource_limit", "Domain: <=4096 profiles, 128 coefficients/subjects and 100000 response cells.")
    planned_work = work if work is not None else subjects * (n * p * p + p**3)
    if planned_work > max_work:
        raise AnalysisError("work_limit", f"Planned {planned_work} work exceeds max_work={max_work}.")
    budget = plan_workspace(operation, {
        "design_and_qr": 8 * (8 * n * p + 8 * p * p),
        "subject_state_and_covariances": 8 * subjects * (6 * p * p + 8 * p + 8 * n),
        "complete_tables_and_json": 256 * subjects * (p * p + 16 * n + 16 * p),
        **(extra_buffers or {}),
    }).record()
    return {"resource_plan": budget, "planned_work": planned_work, "max_work": int(max_work)}


def source(data):
    if isinstance(data, Dataset):
        raise AnalysisError("unsupported_dataset", "Conjoint needs an explicitly resident table.")
    frame = c.source(data)
    if len(frame) > 100000 or len(frame.columns) > 128:
        raise AnalysisError("resource_limit", "Resident input exceeds the conjoint row/column domain.")
    return frame


def _tables(output):
    return {n: {"columns": list(f.columns), "index": list(f.index), "data": f.to_numpy().tolist()}
            for n, f in output.items()}


def seal(procedure, tables, state, **attrs):
    output = TableSet(tables, title=procedure.replace("_", " ").title(),
                      procedure=procedure, state=state, precision="float64", device="cpu", **attrs)
    output.attrs["integrity_sha256"] = digest({"attrs": output.attrs, "tables": _tables(output),
                                                "table_order": list(output)})
    return output


def intact(output, procedure=None):
    if not isinstance(output, TableSet) or output.attrs.get("procedure") not in PROCEDURES or (
        procedure is not None and output.attrs["procedure"] != procedure
    ):
        raise AnalysisError("invalid_result", "Supply a complete conjoint result of the required procedure.")
    attrs = {k: v for k, v in output.attrs.items() if k != "integrity_sha256"}
    if digest({"attrs": attrs, "tables": _tables(output), "table_order": list(output)}) != output.attrs.get("integrity_sha256"):
        raise AnalysisError("invalid_result", "Conjoint tables/state were modified after creation.")
    return output.attrs["state"]


def conjoint_save(result, path=None):
    """Export every conjoint table, full covariance, coding and state with integrity checks."""
    intact(result)
    payload = {"schema": "openecon.conjoint.v1", "title": result.title,
               "attrs": result.attrs, "tables": _tables(result), "table_order": list(result)}
    artifact = json.loads(canonical({"payload": payload, "sha256": digest(payload)}))
    encoded = canonical(artifact)
    if len(encoded.encode()) > MAX_ARTIFACT:
        raise AnalysisError("resource_limit", "Complete conjoint JSON exceeds 32 MiB.")
    if path is not None:
        Path(path).write_text(encoded + "\n", encoding="utf-8")
    return artifact


def conjoint_load(artifact):
    """Restore a complete checksummed conjoint result; numerical state is never refitted."""
    try:
        if isinstance(artifact, (str, Path)):
            path = Path(artifact)
            if path.stat().st_size > MAX_ARTIFACT:
                raise ValueError("Artifact exceeds 32 MiB")
            artifact = json.loads(path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        if len(canonical(artifact).encode()) > MAX_ARTIFACT:
            raise ValueError("Artifact exceeds 32 MiB")
        payload = artifact["payload"]
        if payload["schema"] != "openecon.conjoint.v1" or digest(payload) != artifact["sha256"]:
            raise ValueError("Checksum/schema mismatch")
        order = payload["table_order"]
        if not isinstance(order, list) or not order or any(not isinstance(n, str) for n in order) or (
            len(set(order)) != len(order) or set(order) != set(payload["tables"])
        ):
            raise ValueError("Malformed table order")
        tables = {}
        for n in order:
            f = payload["tables"][n]
            if not isinstance(n, str) or len(f["data"]) != len(f["index"]) or any(
                len(row) != len(f["columns"]) for row in f["data"]
            ):
                raise ValueError("Malformed table")
            tables[n] = table(f["data"], columns=f["columns"], index=f["index"])
        output = TableSet(tables, title=payload["title"], **payload["attrs"])
        intact(output)
        return output
    except (KeyError, TypeError, ValueError, OSError, AnalysisError, RecursionError) as exc:
        raise AnalysisError("invalid_saved_result", "Conjoint artifact schema/checksum/tables/state are invalid.") from exc


def profiles(plan, attributes=None, *, profile="profile_id"):
    name(profile)
    if isinstance(plan, TableSet):
        state = intact(plan)
        if plan.attrs["procedure"] not in ("conjoint_plan", "conjoint_orthogonal"):
            raise AnalysisError("invalid_result", "A plan must come from conjoint_plan/orthogonal or be an explicit resident table.")
        if attributes is None:
            attributes = {a: state["attributes"][a] for a in state["attribute_order"]}
        if profile != state["profile_column"]:
            raise AnalysisError("invalid_spec", "The profile column must match the plan declaration.")
        plan = plan["profiles"]
    declared, _, _ = definition(attributes)
    if profile in declared:
        raise AnalysisError("invalid_spec", "The profile ID column cannot also be an attribute.")
    data = source(plan)
    names = [profile, *declared]
    if set(names) - set(data.columns) or data[names].isna().any().any():
        raise AnalysisError("invalid_plan", "All profile IDs/attribute values must be present and nonmissing.")
    ids = [label(v) for v in data[profile]]
    if len({key(v) for v in ids}) != len(ids):
        raise AnalysisError("invalid_plan", "Profile IDs must be unique typed scalar labels.")
    if len(data) > 4096:
        raise AnalysisError("resource_limit", "Conjoint plans contain at most 4096 profiles.")
    for a, levels in declared.items():
        if any(key(v) not in {key(level) for level in levels} for v in data[a]):
            raise AnalysisError("unknown_level", f"Attribute '{a}' contains an undeclared level.")
    return data[names].copy(), declared


def coding(data, attributes, modes, *, anchors=None):
    """Effects coding and centred/scaled numeric columns; record all transformations."""
    columns = [torch.ones(len(data), dtype=torch.float64)]
    terms, mapping, computed = ["_cons"], [], {}
    for attr in attributes:
        levels, mode = attributes[attr], modes[attr]
        start = len(columns)
        if mode == "discrete":
            values = [key(v) for v in data[attr]]
            for v in levels[:-1]:
                columns.append(torch.tensor([1.0 if x == key(v) else -1.0 if x == key(levels[-1]) else 0.0
                                             for x in values], dtype=torch.float64))
                terms.append(f"{attr}:{key(v)}")
        else:
            x = torch.tensor(data[attr].tolist(), dtype=torch.float64)
            if anchors is None:
                center = float(x.mean())
                scale = float(torch.sqrt(torch.mean((x - center)**2))) or 1.0
            else:
                center, scale = anchors[attr]["center"], anchors[attr]["scale"]
            computed[attr] = {"center": center, "scale": scale}
            z = (x - center) / scale
            columns.append(z)
            terms.append(f"{attr}:linear")
            if mode == "ideal":
                columns.append(z**2)
                terms.append(f"{attr}:quadratic")
        mapping.append({"attribute": attr, "mode": mode, "start": start, "stop": len(columns)})
    x = torch.stack(columns, dim=1)
    transform = torch.eye(len(columns), dtype=torch.float64)
    for item in mapping:
        if item["mode"] == "discrete":
            continue
        j = item["start"]
        mean, scale = computed[item["attribute"]]["center"], computed[item["attribute"]]["scale"]
        transform[0, j], transform[j, j] = -mean / scale, 1 / scale
        if item["mode"] == "ideal":
            transform[0, j+1] = mean**2 / scale**2
            transform[j, j+1] = -2 * mean / scale**2
            transform[j+1, j+1] = 1 / scale**2
    return x, terms, mapping, computed, transform


def level_vector(item, value, levels, p):
    vector = torch.zeros(p, dtype=torch.float64)
    start = item["start"]
    if item["mode"] == "discrete":
        if key(value) == key(levels[-1]):
            vector[start:item["stop"]] = -1
        else:
            vector[start + [key(v) for v in levels].index(key(value))] = 1
    else:
        vector[start] = float(value)
        if item["mode"] == "ideal":
            vector[start+1] = float(value)**2
    return vector
