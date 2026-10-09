"""Resident samples, early resource plans and complete ordered saved results."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet, _json_safe, table
from openecon.econometrics.multivariate import common as c
from openecon.resources import plan_workspace

PROCEDURES = (
    "firth_logit",
    "firth_predict",
    "firth_profile",
    "firth_test",
    "exact_logistic",
    "exact_poisson_rate",
    "lts",
    "lts_predict",
)
MAX_ARTIFACT = 32 * 1024**2


def canonical(value):
    return json.dumps(_json_safe(value), sort_keys=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def label(value):
    if hasattr(value, "item"):
        value = value.item()
    if (
        isinstance(value, bool)
        or not isinstance(value, (str, int, float))
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        raise AnalysisError(
            "invalid_label", "Row labels must be finite numbers or strings, not booleans."
        )
    return value


def confidence(level):
    level = c.check_number(level, "level", minimum=0, maximum=1, exclusive=True)
    if level == 1:
        raise AnalysisError("invalid_option", "level must be strictly between zero and one.")
    return level


def domain(device, weights):
    if device != "cpu":
        raise AnalysisError(
            "unsupported_device", "Finite regression requires resident CPU float64."
        )
    if weights is not None:
        raise AnalysisError(
            "unsupported_weights", "Observation weights are unsupported for finite regression."
        )


def guard(operation, n, p, *, work, max_work, records=0):
    c.check_count(max_work, "max_work")
    if work > max_work:
        raise AnalysisError(
            "work_limit", f"Planned {work} structural work exceeds max_work={max_work}."
        )
    resource = plan_workspace(
        operation,
        {
            "design_qr_autograd_and_information": 8 * (32 * n * max(p, 1) + 64 * p * p + 32 * n),
            "complete_state_and_tables": 256 * (8 * n * max(p, 1) + 8 * p * p + 64),
            "complete_candidate_or_profile_records": 512 * records * (p + 8),
        },
    ).record()
    return {"resource_plan": resource, "planned_work": work, "max_work": int(max_work)}


def sample(data, names, missing, limit):
    if isinstance(data, Dataset):
        raise AnalysisError("unsupported_dataset", "Supply an explicitly resident numeric table.")
    c.check_choice(missing, "missing", ("raise", "drop"))
    frame = c.source(data)
    if len(frame) > limit or len(frame.columns) > 128:
        raise AnalysisError(
            "resource_limit", f"Input requires at most {limit} rows and 128 columns."
        )
    c.require_numeric(frame, names)
    if any(pd.api.types.is_bool_dtype(frame[n].dtype) for n in names):
        raise AnalysisError(
            "non_numeric_column",
            "Boolean columns are unsupported; declare numeric values explicitly.",
        )
    keep = ~frame[names].isna().any(axis=1)
    positions = [i for i, v in enumerate(keep) if v]
    dropped = [i for i, v in enumerate(keep) if not v]
    if dropped and missing == "raise":
        raise AnalysisError(
            "missing_values", "Missing analysed values require explicit missing='drop'."
        )
    if not positions:
        raise AnalysisError("empty_sample", "No complete rows remain.")
    labels = [label(v) for v in frame.index]
    state = {
        "original_n": len(frame),
        "sample_positions": positions,
        "sample_labels": [labels[i] for i in positions],
        "dropped_positions": dropped,
        "dropped_labels": [labels[i] for i in dropped],
        "missing": missing,
    }
    return frame.iloc[positions][names].copy(), state


def design(
    data, y, x, intercept, missing, limit, max_p, operation, work_factor, max_work, records=0
):
    c.check_flag(intercept, "intercept")
    names = c.name_list(x, "x", minimum=0)
    if "_cons" in names or y in names or (y is not None and y == "_cons"):
        raise AnalysisError(
            "invalid_spec",
            "Outcome/predictors must have distinct roles and cannot use reserved _cons.",
        )
    if y is not None:
        c.check_name(y, "y")
    p = len(names) + int(intercept)
    if not 1 <= p <= max_p:
        raise AnalysisError(
            "resource_limit", f"Declare 1..{max_p} numeric coefficients including any intercept."
        )
    frame, state = sample(data, ([y] if y is not None else []) + names, missing, limit)
    n = len(frame)
    settings = guard(
        operation, n, p, work=n * p**3 * work_factor, max_work=max_work, records=records
    )
    raw = c.matrix(frame, names)
    if raw.numel() and float(raw.abs().max()) > 1e6:
        raise AnalysisError(
            "non_finite_values", "Predictors must lie within +/-1e6; rescale explicitly."
        )
    # With no intercept centring would change the model, so scale only.
    center = raw.mean(0) if intercept else torch.zeros(len(names), dtype=torch.float64)
    scale = torch.sqrt(torch.mean((raw - center) ** 2, dim=0))
    if bool((scale <= 1e-12).any()):
        raise AnalysisError("unidentified_design", "Constant or numerically degenerate predictor.")
    z = (raw - center) / scale
    matrix = torch.cat([torch.ones((n, 1), dtype=torch.float64), z], dim=1) if intercept else z
    condition(matrix)
    transform = torch.eye(p, dtype=torch.float64)
    start = int(intercept)
    for j in range(len(names)):
        transform[start + j, start + j] = 1 / scale[j]
        if intercept:
            transform[0, start + j] = -center[j] / scale[j]
    response = c.column(frame, y) if y is not None else None
    state.update(
        x=names,
        y=y,
        intercept=intercept,
        terms=(["_cons"] if intercept else []) + names,
        center=center.tolist(),
        scale=scale.tolist(),
        transform=transform.tolist(),
        design_scaled=matrix.tolist(),
        raw_predictors=raw.tolist(),
        response=response.tolist() if response is not None else None,
    )
    return matrix, response, transform, state, settings


def condition(matrix):
    s = torch.linalg.svdvals(matrix)
    if (
        len(s) < matrix.shape[1]
        or float(s[-1]) <= 1e-12 * float(s[0])
        or float(s[0] / s[-1]) > 1e10
    ):
        raise AnalysisError(
            "unidentified_design", "A design/subset is rank deficient or too ill-conditioned."
        )
    return float(s[0] / s[-1])


def prediction(data, state, missing, operation, max_work):
    frame, rows = sample(data, state["x"], missing, 2000)
    p = len(state["terms"])
    settings = guard(operation, len(frame), p, work=len(frame) * p * p, max_work=max_work)
    raw = c.matrix(frame, state["x"])
    if raw.numel() and float(raw.abs().max()) > 1e6:
        raise AnalysisError("non_finite_values", "Predictors must lie within +/-1e6.")
    z = (raw - torch.tensor(state["center"], dtype=torch.float64)) / torch.tensor(
        state["scale"], dtype=torch.float64
    )
    matrix = (
        torch.cat([torch.ones((len(frame), 1), dtype=torch.float64), z], dim=1)
        if state["intercept"]
        else z
    )
    return matrix, rows, settings


def _tables(output):
    return {
        n: {"columns": list(f.columns), "index": list(f.index), "data": f.to_numpy().tolist()}
        for n, f in output.items()
    }


def seal(procedure, tables, state, **attrs):
    output = TableSet(
        tables,
        title=procedure.replace("_", " ").title(),
        procedure=procedure,
        state=state,
        precision="float64",
        device="cpu",
        **attrs,
    )
    output.attrs["integrity_sha256"] = digest(
        {"attrs": output.attrs, "tables": _tables(output), "table_order": list(output)}
    )
    return output


def intact(output, procedure=None):
    if (
        not isinstance(output, TableSet)
        or output.attrs.get("procedure") not in PROCEDURES
        or (procedure is not None and output.attrs["procedure"] != procedure)
    ):
        raise AnalysisError(
            "invalid_result",
            "Supply a complete finite regression result of the required procedure.",
        )
    attrs = {k: v for k, v in output.attrs.items() if k != "integrity_sha256"}
    if digest(
        {"attrs": attrs, "tables": _tables(output), "table_order": list(output)}
    ) != output.attrs.get("integrity_sha256"):
        raise AnalysisError(
            "invalid_result", "Finite regression tables/state were modified after creation."
        )
    return output.attrs["state"]


def finite_save(result, path=None):
    """Save all ordered finite-regression tables, covariance, samples and numerical state."""
    intact(result)
    payload = {
        "schema": "openecon.finite.v1",
        "title": result.title,
        "attrs": result.attrs,
        "tables": _tables(result),
        "table_order": list(result),
    }
    artifact = json.loads(canonical({"payload": payload, "sha256": digest(payload)}))
    encoded = canonical(artifact)
    if len(encoded.encode()) > MAX_ARTIFACT:
        raise AnalysisError("resource_limit", "Complete finite-regression artifact exceeds 32 MiB.")
    if path is not None:
        Path(path).write_text(encoded + "\n", encoding="utf-8")
    return artifact


def finite_load(artifact):
    """Restore a complete checksummed finite-regression result without numerical refitting."""
    try:
        if isinstance(artifact, (str, Path)):
            path = Path(artifact)
            if path.stat().st_size > MAX_ARTIFACT:
                raise ValueError("Artifact exceeds 32 MiB")
            artifact = json.loads(
                path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v))
            )
        if len(canonical(artifact).encode()) > MAX_ARTIFACT:
            raise ValueError("Artifact exceeds 32 MiB")
        payload = artifact["payload"]
        if payload["schema"] != "openecon.finite.v1" or digest(payload) != artifact["sha256"]:
            raise ValueError("Checksum/schema mismatch")
        order = payload["table_order"]
        if (
            not isinstance(order, list)
            or not order
            or any(not isinstance(n, str) for n in order)
            or (len(set(order)) != len(order) or set(order) != set(payload["tables"]))
        ):
            raise ValueError("Malformed table order")
        tables = {}
        for n in order:
            f = payload["tables"][n]
            if len(f["data"]) != len(f["index"]) or any(
                len(row) != len(f["columns"]) for row in f["data"]
            ):
                raise ValueError("Malformed table")
            tables[n] = table(f["data"], columns=f["columns"], index=f["index"])
        output = TableSet(tables, title=payload["title"], **payload["attrs"])
        intact(output)
        return output
    except (KeyError, TypeError, ValueError, OSError, AnalysisError, RecursionError) as exc:
        raise AnalysisError(
            "invalid_saved_result", "Finite artifact schema/checksum/tables/state are invalid."
        ) from exc
