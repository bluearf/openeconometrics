"""Resident design/sample contracts and complete checksummed causal tables."""

from __future__ import annotations

import hashlib
import json
import math
from functools import wraps
from numbers import Integral, Real
from pathlib import Path

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet, _json_safe
from openecon.econometrics.multivariate import common as c
from openecon.resources import plan_workspace

PROCEDURES = (
    "ebalance",
    "cem",
    "balance",
    "rosenbaum_bounds",
    "paired_randomization",
    "treatment_cdf",
    "treatment_quantile",
    "treatment_rmst",
    "ovb_sensitivity",
    "evalue",
    "manski_ate",
    "lee_bounds",
    "randomization_test",
    "stratified_randomization",
    "rosenbaum_rank_bounds",
    "bias_sensitivity",
    "ovb_benchmark",
    "ovb_robustness",
    "treatment_cdf_ipw",
    "treatment_quantile_ipw",
    "treatment_cdf_aipw",
    "treatment_survival_ipcw",
    "treatment_rmst_ipcw",
    "treatment_rmst_aipw",
    "cluster_randomization",
    "bernoulli_randomization",
    "neyman_ate",
    "stratified_neyman_ate",
    "cluster_neyman_ate",
    "paired_neyman_ate",
    "manski_ate_inference",
    "stratified_lee_bounds",
    "randomization_confidence_set",
    "paired_randomization_confidence_set",
    "cluster_randomization_confidence_set",
    "bernoulli_neyman_ate",
    "multiarm_neyman_ate",
    "stratified_multiarm_neyman_ate",
    "treatment_effect_cdf_bounds",
    "treatment_effect_quantile_bounds",
)
frame = c.frame


def procedure(function):
    wrapped = c.procedure(function)

    @wraps(function)
    def cpu(*args, **kwargs):
        with torch.device("cpu"):
            return wrapped(*args, **kwargs)

    return cpu


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _label(value):
    if isinstance(value, Real) and not isinstance(value, Integral) and not math.isfinite(value):
        return {"type": "nonfinite_index_label", "value": str(value)}
    return c.label(value)


def options(device, weights, max_work, level=0.95):
    if device != "cpu":
        raise AnalysisError(
            "unsupported_device", "Causal design procedures support CPU float64 only."
        )
    if weights is not None:
        raise AnalysisError(
            "unsupported_weights",
            "This procedure has no generic observation-weight route; use a documented balancing-weight role when available.",
        )
    c.check_count(max_work, "max_work")
    c.check_number(level, "level", minimum=0, maximum=1, exclusive=True)
    if level >= 1:
        raise AnalysisError("invalid_option", "level must be strictly between zero and one.")


def work(units, max_work, label):
    c.check_count(max_work, "max_work")
    if units > max_work:
        raise AnalysisError(
            "work_budget_exceeded",
            f"Complete {label} exceeds max_work; no partial result is returned.",
        )


def sample(data, columns, *, numeric=(), missing="raise", max_work=100_000_000, cost=1, minimum=2):
    if isinstance(data, Dataset):
        raise AnalysisError(
            "unsupported_dataset",
            "Causal design requires a resident table; no automatic collection is performed.",
        )
    names = c.name_list(columns, "columns", minimum=1)
    c.check_choice(missing, "missing", ("raise", "drop"))
    source = c.source(data)
    n, p = len(source), len(names)
    if n > 100000 or p > 64:
        raise AnalysisError(
            "resource_limit",
            "Causal design input is limited to 100000 rows and 64 selected columns.",
        )
    absent = [name for name in names if name not in source]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {absent}.")
    c.require_numeric(source, numeric)
    work(n * max(cost, p), max_work, "sample and declared computation")
    plan = plan_workspace(
        "resident causal design sample",
        {
            "selected_table_and_numeric_copies": 96 * n * p,
            "positions_labels_and_state": 256 * n,
        },
    )
    selected = source.loc[:, names]
    keep = selected.notna().all(axis=1)
    if missing == "raise" and not bool(keep.all()):
        raise AnalysisError(
            "missing_values",
            "Selected causal inputs contain missing values; choose missing='drop' explicitly.",
        )
    positions = [i for i, value in enumerate(keep) if value]
    selected = selected.loc[keep].reset_index(drop=True).copy()
    if len(selected) < minimum:
        raise AnalysisError(
            "insufficient_observations", f"At least {minimum} complete observations are required."
        )
    for name in numeric:
        tensor(selected, name)
    metadata = dict(
        n=len(selected),
        n_input=n,
        n_missing=n - len(selected),
        positions=positions,
        unit_labels=[_label(source.index[i]) for i in positions],
        columns=names,
        missing=missing,
        device="cpu",
        dtype="float64",
        max_work=max_work,
        resource_plan=plan.record(),
    )
    schema = {name: str(source[name].dtype) for name in names}
    try:
        payload = pd.util.hash_pandas_object(selected, index=False).values.tobytes()
    except OverflowError:
        # Pandas factorization can cast a bounded, exact integer identity to
        # float while constructing object categories. Preserve the original
        # typed rows instead; ordinary samples retain their existing encoding.
        rows = list(selected.itertuples(index=False, name=None))
        if not any(
            isinstance(value, Integral) and int(value).bit_length() > 1023
            for row in rows for value in row
        ):
            raise
        payload = canonical([
            [{"type": type(value).__name__, "value": _label(value)} for value in row]
            for row in rows
        ]).encode()
        metadata["sample_hash_encoding"] = "typed-canonical-rows-v1"
    metadata["sample_sha256"] = hashlib.sha256(
        canonical(schema).encode() + payload + canonical(positions).encode()
    ).hexdigest()
    return selected, metadata


def tensor(selected, name):
    # A caller's global default device must not override this CPU-only route.
    with torch.device("cpu"):
        return c.column(selected, name)


def binary(selected, name):
    if pd.api.types.is_bool_dtype(selected[name].dtype):
        raise AnalysisError(
            "invalid_treatment", "Treatment must use numeric 0/1, not Boolean labels."
        )
    value = tensor(selected, name)
    if not bool(((value == 0) | (value == 1)).all()) or value.min() == value.max():
        raise AnalysisError(
            "invalid_treatment", "Treatment must contain both numeric 0 and 1, and no other value."
        )
    return value


def _tables(tables):
    for table in tables.values():
        for row in table.itertuples(index=False, name=None):
            if any(isinstance(value, Real) and not isinstance(value, Integral) and math.isinf(value) for value in row):
                raise AnalysisError(
                    "numerical_failure",
                    "A computed table cell overflowed; infinity is not converted to an undefined saved cell.",
                )
    return {
        name: dict(
            index=_json_safe(list(table.index)),
            columns=_json_safe(list(table.columns)),
            dtypes=[str(dtype) for dtype in table.dtypes],
            data=_json_safe(list(table.itertuples(index=False, name=None))),
        )
        for name, table in tables.items()
    }


def result(procedure, tables, metadata, settings, state, notes=()):
    if procedure not in PROCEDURES:
        raise AnalysisError("invalid_result", "Unknown causal design procedure.")

    def finite(value):
        if isinstance(value, Real) and not isinstance(value, Integral) and not math.isfinite(value):
            raise AnalysisError(
                "numerical_failure",
                "Computed scientific state contains a non-finite number; no silent saved-value replacement is performed.",
            )
        if isinstance(value, dict):
            for entry in value.values():
                finite(entry)
        elif isinstance(value, (list, tuple)):
            for entry in value:
                finite(entry)

    finite(dict(settings=settings, sample=metadata, **state))
    saved = _json_safe(
        dict(
            schema="openecon.causal_design.v1",
            procedure=procedure,
            settings=settings,
            sample=metadata,
            **state,
        )
    )
    # Do not use random IDs/timestamps: complete artifacts can be compared across
    # source, wheel and frozen execution without excluding scientific fields.
    canonical(saved)
    payload = _tables(tables)
    canonical(payload)
    return TableSet(
        tables,
        title=f"Causal design: {procedure}",
        procedure=procedure,
        **metadata,
        state=saved,
        state_sha256=digest(saved),
        tables_sha256=digest(payload),
        notes=list(notes),
        stata_parity_validated=False,
    )


def _consistent(attrs):
    state = attrs.get("state")
    if not isinstance(state, dict) or attrs.get("procedure") != state.get("procedure"):
        return False
    sample = state.get("sample")
    if not isinstance(sample, dict) or any(key not in attrs for key in sample):
        return False
    try:
        # Python equality conflates True/1 and integer/floating labels. Compare
        # the same JSON representation used by the integrity checks so every
        # duplicated sample value retains its saved type as well as its value.
        return canonical({key: attrs[key] for key in sample}) == canonical(sample)
    except (TypeError, ValueError):
        return False


def causal_design_save(result: TableSet, path=None) -> dict:
    """Save all tables, sample positions, assumptions and scientific state with checksums."""
    if (
        not isinstance(result, TableSet)
        or result.attrs.get("procedure") not in PROCEDURES
        or not _consistent(result.attrs)
        or digest(result.attrs.get("state")) != result.attrs.get("state_sha256")
        or digest(_tables(result)) != result.attrs.get("tables_sha256")
    ):
        raise AnalysisError("invalid_result", "Supply an intact causal design result.")
    _validate_scientific_certificates(result)
    payload = json.loads(
        canonical(
            dict(
                schema="openecon.causal_design.artifact.v1",
                title=result.title,
                attrs=result.attrs,
                tables=_tables(result),
                table_order=list(result),
            )
        )
    )
    artifact = dict(payload=payload, sha256=digest(payload))
    if path is not None:
        Path(path).write_text(canonical(artifact) + "\n", encoding="utf-8")
    return artifact


def causal_design_load(artifact) -> TableSet:
    """Verify and restore a complete causal-design JSON artifact without refitting."""
    try:
        if isinstance(artifact, (str, Path)):
            artifact = json.loads(Path(artifact).read_text(encoding="utf-8"))
        payload = artifact["payload"]
        attrs = payload["attrs"]
        if (
            payload["schema"] != "openecon.causal_design.artifact.v1"
            or digest(payload) != artifact["sha256"]
            or attrs["procedure"] not in PROCEDURES
            or not _consistent(attrs)
            or attrs["state"].get("schema") != "openecon.causal_design.v1"
            or attrs["state"].get("procedure") != attrs["procedure"]
            or digest(attrs["state"]) != attrs["state_sha256"]
            or digest(payload["tables"]) != attrs["tables_sha256"]
        ):
            raise ValueError("artifact/state/table checksum or schema mismatch")
        order = payload["table_order"]
        if (
            not isinstance(order, list)
            or len(order) != len(set(order))
            or set(order) != set(payload["tables"])
        ):
            raise ValueError("saved table order is incomplete")
        from openecon.frame import DataFrame

        tables = {
            name: DataFrame(
                payload["tables"][name]["data"],
                dtype=object,
                columns=payload["tables"][name]["columns"],
                index=payload["tables"][name]["index"],
            )
            for name in order
        }
        for name, table in tables.items():
            value = payload["tables"][name]
            if len(value["dtypes"]) != len(table.columns):
                raise ValueError("saved table dtype schema mismatch")
            for column, dtype in zip(table.columns, value["dtypes"], strict=True):
                table[column] = table[column].astype(dtype)
        restored = TableSet(tables, title=payload["title"], **attrs)
        if digest(_tables(restored)) != attrs["tables_sha256"]:
            raise ValueError("restored table differs from saved table")
        _validate_scientific_certificates(restored)
        return restored
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise AnalysisError(
            "invalid_saved_result", f"Causal design artifact could not be restored: {exc}."
        ) from exc


def _validate_scientific_certificates(result):
    """Check the new discrete coupling proofs without rerunning optimization."""
    if result.attrs["procedure"] in (
        "treatment_effect_cdf_bounds", "treatment_effect_quantile_bounds",
    ):
        from .effect_distribution import validate_effect_distribution_result

        validate_effect_distribution_result(result)
