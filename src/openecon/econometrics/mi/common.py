"""Bounded resident admission and immutable, replayable multiple-imputation state.

pandas supplies tabular I/O and index identity; numerical work uses CPU float64 Torch.
No row is dropped and no streaming/Dataset source is materialized implicitly.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, Literal

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace

MAX_ROWS = 10_000
MAX_COLUMNS = 16
MAX_IMPUTATIONS = 100
DEFAULT_MAX_WORK = 100_000_000


def integer(value: Any, name: str, *, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise AnalysisError("invalid_option", f"{name} must be an integer in [{minimum}, {maximum}].")
    return int(value)


def check_seed(seed: Any) -> int:
    return integer(seed, "seed")


class _FrozenDict(dict):
    """JSON-compatible dictionary with recursively immutable values."""
    def _deny(self, *args, **kwargs):
        raise TypeError("MI result metadata is immutable.")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _deny

    def __deepcopy__(self, memo):
        return self


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Metadata keys must be strings.")
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("Metadata must contain finite JSON values only.")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _label(value: Any) -> dict[str, Any]:
    if value is pd.NaT:
        return {"type": "nat", "value": None}
    if value is pd.NA:
        return {"type": "missing", "value": None}
    if isinstance(value, tuple):
        if len(value) > MAX_COLUMNS:
            raise AnalysisError("unsupported_index", "MI tuple row labels are limited to 16 components.")
        return {"type": "tuple", "value": [_label(v) for v in value]}
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return {"type": "timestamp", "value": value.isoformat()}
    if isinstance(value, (pd.Timedelta, dt.timedelta)):
        return {"type": "timedelta", "value": pd.Timedelta(value).value}
    if isinstance(value, dt.date):
        return {"type": "date", "value": value.isoformat()}
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return {"type": "scalar", "value": value}
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "nan", "value": None}
        if math.isfinite(value):
            return {"type": "scalar", "value": value}
    raise AnalysisError("unsupported_index", "MI row labels and index names must be JSON scalars, dates, timedeltas, or tuples of these.")


def _unlabel(label: Mapping) -> Any:
    kind, value = label["type"], label["value"]
    if kind == "scalar":
        if value is not None and not isinstance(value, (str, bool, int, float)):
            raise ValueError("Invalid scalar index label.")
        return value
    if kind == "tuple":
        return tuple(_unlabel(v) for v in value)
    if kind == "timestamp":
        return pd.Timestamp(value)
    if kind == "timedelta":
        return pd.Timedelta(value, unit="ns")
    if kind == "date":
        return dt.date.fromisoformat(value)
    if kind == "nan":
        return float("nan")
    if kind == "missing":
        return pd.NA
    if kind == "nat":
        return pd.NaT
    raise ValueError("Invalid MI index label encoding.")


def _multi_index_sortorder(value: Any, nlevels: int) -> int | None:
    """Validate the saved declaration without sorting or changing any row code."""
    if value is not None and (type(value) is not int or not 0 <= value <= nlevels):
        raise ValueError("MI MultiIndex sortorder must be an integer in [0, nlevels] or None.")
    return value


def _encode_index(index: pd.Index) -> dict[str, Any]:
    if len(index) > MAX_ROWS:
        raise AnalysisError("unsupported_index", "MI index levels/categories may contain at most 10000 entries.")
    if isinstance(index, pd.MultiIndex):
        if index.nlevels > MAX_COLUMNS:
            raise AnalysisError("unsupported_index", "MI supports at most 16 row-index levels.")
        result = {"kind": "multi", "names": [_label(n) for n in index.names],
                  "levels": [_encode_index(level) for level in index.levels],
                  "codes": [code.tolist() for code in index.codes]}
        order = _multi_index_sortorder(index.sortorder, index.nlevels)
        # An absent declaration retains the exact historical descriptor/digest.
        if order is not None:
            result["sortorder"] = order
        return result
    if isinstance(index, pd.RangeIndex):
        return {"kind": "range", "name": _label(index.name), "start": index.start,
                "stop": index.stop, "step": index.step}
    if isinstance(index, pd.CategoricalIndex):
        return {"kind": "categorical", "name": _label(index.name),
                "categories": _encode_index(index.categories), "ordered": index.ordered,
                "codes": index.codes.tolist()}
    if isinstance(index, pd.DatetimeIndex):
        return {"kind": "datetime", "name": _label(index.name), "dtype": str(index.dtype),
                "values": [_label(v) for v in index], "freq": index.freqstr}
    if isinstance(index, pd.TimedeltaIndex):
        return {"kind": "timedelta", "name": _label(index.name), "dtype": str(index.dtype),
                "values": [_label(v) for v in index], "freq": index.freqstr}
    if type(index) is not pd.Index:
        raise AnalysisError("unsupported_index", f"MI does not serialize {type(index).__name__}; provide an ordinary index explicitly.")
    return {"kind": "index", "name": _label(index.name), "dtype": str(index.dtype),
            "values": [_label(v) for v in index]}


def _index_envelope(state: Any, n: int, *, depth: int = 0) -> None:
    """Bound saved descriptors before constructing any pandas index."""
    if not isinstance(state, Mapping) or depth > MAX_COLUMNS:
        raise ValueError("MI index descriptor nesting exceeds supported bounds.")
    kind = state.get("kind")
    if kind == "multi":
        levels, codes, names = (state.get(k, ()) for k in ("levels", "codes", "names"))
        if not 1 <= len(levels) <= MAX_COLUMNS or len(codes) != len(levels) or len(names) != len(levels):
            raise ValueError("MI multi-index dimensions disagree.")
        _multi_index_sortorder(state.get("sortorder"), len(levels))
        if any(len(c) != n for c in codes):
            raise ValueError("MI multi-index code length disagrees with rows.")
        for level in levels:
            _index_envelope(level, MAX_ROWS, depth=depth + 1)
    elif kind == "categorical":
        if len(state.get("codes", ())) > n:
            raise ValueError("MI categorical index exceeds its row envelope.")
        _index_envelope(state.get("categories"), MAX_ROWS, depth=depth + 1)
    elif kind == "range":
        start, stop, step = (state.get(k) for k in ("start", "stop", "step"))
        if any(type(v) is not int for v in (start, stop, step)) or step == 0:
            raise ValueError("Invalid MI range index.")
        if len(range(start, stop, step)) > n:
            raise ValueError("MI range index exceeds its row envelope.")
    elif kind in ("index", "datetime", "timedelta"):
        if not isinstance(state.get("values"), (list, tuple)) or len(state["values"]) > n:
            raise ValueError("MI saved index exceeds its row envelope.")
    else:
        raise ValueError("Unsupported MI saved index kind.")


def _digest(state: Mapping) -> str:
    """Corruption/tampering detection for recorded state; not authentication."""
    canonical = json.dumps(_thaw(state), sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _metadata_bytes(value: Any) -> int:
    """Estimate saved JSON object memory before index restoration or freezing."""
    stack, size, nodes = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > 64 or nodes > 2_000_000:
            raise ValueError("MI metadata exceeds supported size/nesting bounds.")
        if isinstance(item, Mapping):
            size += 256 + 96 * len(item)
            stack.extend((v, depth + 1) for pair in item.items() for v in pair)
        elif isinstance(item, (list, tuple)):
            size += 64 + 16 * len(item)
            stack.extend((v, depth + 1) for v in item)
        elif isinstance(item, str):
            size += 64 + 4 * len(item)
        else:
            size += 32
    return size


def _decode_index(state: Mapping) -> pd.Index:
    kind = state["kind"]
    if kind == "multi":
        order = _multi_index_sortorder(state.get("sortorder"), len(state["levels"]))
        return pd.MultiIndex(levels=[_decode_index(v) for v in state["levels"]],
                             codes=[list(v) for v in state["codes"]],
                             names=[_unlabel(v) for v in state["names"]],
                             sortorder=order, verify_integrity=True)
    name = _unlabel(state["name"])
    if kind == "range":
        return pd.RangeIndex(state["start"], state["stop"], state["step"], name=name)
    if kind == "categorical":
        values = pd.Categorical.from_codes(list(state["codes"]),
                    categories=_decode_index(state["categories"]), ordered=state["ordered"])
        return pd.CategoricalIndex(values, name=name)
    values = [_unlabel(v) for v in state["values"]]
    if kind == "datetime":
        # DatetimeIndex dtype retains the timezone, including an empty index.
        return pd.DatetimeIndex(values, dtype=state["dtype"], freq=state["freq"], name=name)
    if kind == "timedelta":
        return pd.TimedeltaIndex(values, dtype=state["dtype"], freq=state["freq"], name=name)
    if kind == "index":
        return pd.Index(values, dtype=state["dtype"], name=name, tupleize_cols=False)
    raise ValueError("Invalid MI index kind.")


def _resident(data: Any, columns: list[str]) -> tuple[int, Any]:
    """Determine selected shape without constructing a frame or tensor."""
    if isinstance(data, pd.DataFrame):
        if data.columns.has_duplicates:
            raise AnalysisError("duplicate_columns", "MI data must have unique column names.")
        absent = [name for name in columns if name not in data.columns]
        if absent:
            raise AnalysisError("missing_columns", f"MI columns are absent: {', '.join(absent)}.")
        return len(data), data
    if isinstance(data, Mapping):
        absent = [name for name in columns if name not in data]
        if absent:
            raise AnalysisError("missing_columns", f"MI columns are absent: {', '.join(absent)}.")
        selected = {name: data[name] for name in columns}
        lengths = []
        for value in selected.values():
            if not isinstance(value, (Sequence, pd.Series)) or isinstance(value, (str, bytes)):
                raise AnalysisError("invalid_data", "MI mappings require resident, sized column sequences.")
            lengths.append(len(value))
        if len(set(lengths)) != 1:
            raise AnalysisError("invalid_data", "MI columns must have consistent lengths.")
        series = [v for v in selected.values() if isinstance(v, pd.Series)]
        if series and any(not v.index.equals(series[0].index) for v in series):
            raise AnalysisError("invalid_data", "MI Series columns must have identical row indexes; align them explicitly.")
        return lengths[0], selected
    if isinstance(data, (list, tuple)):
        if any(not isinstance(row, Mapping) for row in data):
            raise AnalysisError("invalid_data", "MI row records must be resident mappings.")
        return len(data), data
    raise AnalysisError("unsupported_data", "MI requires a resident DataFrame, mapping, or list of records; Dataset and replay sources are not supported.")


def admit(data: Any, columns: Sequence[str], *, m: int = 5, iterations: int = 1,
          max_work: int = DEFAULT_MAX_WORK) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor, dict]:
    """Admit a bounded numeric panel before materializing its selected buffers."""
    if isinstance(columns, (str, bytes)) or not isinstance(columns, (list, tuple)):
        raise AnalysisError("invalid_spec", "columns must be a list of distinct column names.")
    names = list(columns)
    if not 1 <= len(names) <= MAX_COLUMNS or any(not isinstance(v, str) or not v for v in names) or len(set(names)) != len(names):
        raise AnalysisError("invalid_spec", f"MI needs 1..{MAX_COLUMNS} distinct nonempty column names.")
    m = integer(m, "m", minimum=1, maximum=MAX_IMPUTATIONS)
    iterations = integer(iterations, "iterations", minimum=1, maximum=100_000)
    max_work = integer(max_work, "max_work", minimum=1)
    n, resident = _resident(data, names)
    p = len(names)
    if not 1 <= n <= MAX_ROWS:
        raise AnalysisError("mi_shape_limit", f"MI requires 1..{MAX_ROWS} resident rows; no rows are dropped.")
    work = iterations * (n * p**3 + n * p**2 + p**3) + m * n * p
    if work > max_work:
        raise AnalysisError("mi_work_limit", f"MI estimated work {work:,} exceeds max_work={max_work:,}; reduce rows, columns, or iterations.")
    # Python/JSON output is explicitly included in this conservative plan, in addition
    # to all retained completions, working matrices, masks, and small factors.
    plan = plan_workspace("multiple imputation", {
        "selected and working matrices": 8 * n * p * 8,
        "missing and grouping masks": n * p * 4 + 8 * n * 4,
        "retained completed tensors": m * n * p * 8,
        "regression and factor workspace": 4 * n * (p + 1) * 8 + 64 * p * p * 8,
        "immutable result and JSON state": (m + 2) * n * p * 160 + n * 4096,
        "encoded row identity": 6 * int(resident.index.memory_usage(deep=True)) if isinstance(resident, pd.DataFrame) else n * 512,
    })
    frame = resident.loc[:, names].copy() if isinstance(resident, pd.DataFrame) else pd.DataFrame(resident, columns=names)
    if len(frame) != n:
        raise AnalysisError("invalid_data", "MI input construction changed row alignment; provide a DataFrame explicitly.")
    for name in names:
        dtype = frame[name].dtype
        if not (is_numeric_dtype(dtype) or is_bool_dtype(dtype)) or is_complex_dtype(dtype):
            raise AnalysisError("non_numeric_column", f"MI column '{name}' must contain real numeric values.")
    index = _encode_index(frame.index)
    values = torch.tensor(frame.to_numpy(dtype="float64", na_value=float("nan")), dtype=torch.float64, device="cpu")
    missing = torch.isnan(values)
    observed = values[~missing]
    if not bool(torch.isfinite(observed).all()):
        raise AnalysisError("non_finite_values", "MI observed values must be finite; infinity is not a missing value.")
    if observed.numel() and float(observed.abs().max()) > 1e140:
        raise AnalysisError("non_finite_values", "MI values are too large for stable double precision products; rescale columns.")
    for j, name in enumerate(names):
        present = values[~missing[:, j], j]
        if present.numel() and 0 < float(present.abs().max()) < 1e-100:
            raise AnalysisError("non_finite_values", f"MI column '{name}' must be rescaled for double precision products.")
    metadata = {"schema_version": "1", "n": n, "p": p, "m": m,
                "engine": "torch_cpu_float64", "columns": names, "index": index,
                "converged": False, "stata_parity_validated": False,
                "sample": {"nobs_original": n, "nobs": n, "dropped_rows": 0,
                           "positions": list(range(n))},
                "missing_mask": missing.tolist(), "missing_counts": missing.sum(0).tolist(),
                "missing_total": int(missing.sum()), "work_estimate": work,
                "max_work": max_work, "resource_plan": plan.record()}
    return frame, values, missing, metadata


class MIResult(BaseModel):
    """Immutable full-state MI result; JSON loading rechecks all integrity fields."""
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    method: Literal["mi_mvn", "mi_monotone", "mi_chained"]
    columns: tuple[str, ...]
    original: tuple[tuple[float | None, ...], ...]
    completed_matrices: tuple[tuple[tuple[float, ...], ...], ...]
    seed: int
    metadata: dict[str, Any]
    integrity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def _bounded_state(cls, state):
        if isinstance(state, Mapping):
            columns, original, completed = (state.get(k, ()) for k in ("columns", "original", "completed_matrices"))
            if not 1 <= len(columns) <= MAX_COLUMNS or not 1 <= len(original) <= MAX_ROWS or not 1 <= len(completed) <= MAX_IMPUTATIONS:
                raise ValueError("MI state exceeds its admitted dimensions.")
            for matrix in (original, *completed):
                if len(matrix) != len(original) or any(len(row) != len(columns) for row in matrix):
                    raise ValueError("MI matrix shapes disagree.")
                for row in matrix:
                    if any(v is not None and (isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)) for v in row):
                        raise ValueError("MI state matrices need finite numeric values or original missing nulls.")
            check_seed(state.get("seed"))
            metadata = state.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError("MI requires declared metadata.")
            _index_envelope(metadata.get("index"), len(original))
            # Include Python tuple state, temporary validated copies and canonical
            # JSON bytes; this is an estimated workspace bound, not process RSS.
            plan_workspace("MI saved result restoration", {
                "validated matrices and canonical JSON": (len(completed) + 2) * len(original) * len(columns) * 128,
                "row identity and metadata": max(len(original) * 2048, _metadata_bytes(metadata)),
            })
        return state

    @field_validator("metadata")
    @classmethod
    def _immutable_metadata(cls, value):
        return _freeze(value)

    @field_serializer("metadata")
    def _metadata_json(self, value):
        return _thaw(value)

    @model_validator(mode="after")
    def _integrity(self):
        n, p, m = len(self.original), len(self.columns), len(self.completed_matrices)
        meta = self.metadata
        if len(set(self.columns)) != p or any(not name for name in self.columns):
            raise ValueError("MI columns must be distinct nonempty names.")
        for key, expected in (("schema_version", "1"), ("n", n), ("p", p), ("m", m),
                              ("engine", "torch_cpu_float64"), ("columns", self.columns),
                              ("seed", self.seed), ("method", self.method)):
            if meta.get(key) != expected:
                raise ValueError(f"MI metadata {key} disagrees with result state.")
        if any(type(meta.get(key)) is not int for key in ("n", "p", "m", "seed")):
            raise ValueError("MI dimensions and root seed must be integer metadata.")
        if meta.get("converged") is not False or meta.get("stata_parity_validated") is not False:
            raise ValueError("MI generation does not establish finite-chain convergence or blanket Stata parity.")
        convergence = meta.get("convergence", {})
        if self.method in ("mi_mvn", "mi_chained") and isinstance(convergence, Mapping) and any(convergence.get(k) is True for k in ("assessed", "converged")):
            raise ValueError("Finite-chain MI convergence must remain unassessed.")
        if meta.get("convergence_claim") is True or (isinstance(meta.get("logit_sampler"), Mapping) and meta["logit_sampler"].get("stationarity_claim") is True):
            raise ValueError("Finite-chain MI cannot declare convergence or stationarity from its iteration count.")
        work, limit = meta.get("work_estimate"), meta.get("max_work")
        if type(work) is not int or type(limit) is not int or not 1 <= work <= limit:
            raise ValueError("MI work estimate must be positive and within max_work.")
        for plan_name in ("resource_plan", "chained_resource_plan"):
            if plan_name not in meta and plan_name != "resource_plan":
                continue
            plan = meta.get(plan_name, {})
            buffers = plan.get("buffers", {})
            budget, estimate = plan.get("budget_bytes"), plan.get("estimated_workspace_bytes")
            if not isinstance(buffers, Mapping) or not buffers or any(type(size) is not int or size < 0 for size in buffers.values()):
                raise ValueError("MI resource plans require nonnegative named buffer estimates.")
            if type(budget) is not int or type(estimate) is not int or not 1 <= estimate <= budget or estimate != sum(buffers.values()) or not isinstance(plan.get("operation"), str) or not plan["operation"] or not isinstance(plan.get("scope"), str) or not plan["scope"]:
                raise ValueError("MI resource plan estimates and budget disagree.")
            if plan_name == "resource_plan":
                expected_buffers = {
                    "selected and working matrices": 8 * n * p * 8,
                    "missing and grouping masks": n * p * 4 + 8 * n * 4,
                    "retained completed tensors": m * n * p * 8,
                    "regression and factor workspace": 4 * n * (p + 1) * 8 + 64 * p * p * 8,
                    "immutable result and JSON state": (m + 2) * n * p * 160 + n * 4096,
                }
                if plan["operation"] != "multiple imputation" or any(buffers.get(key) != size for key, size in expected_buffers.items()) or type(buffers.get("encoded row identity")) is not int or buffers["encoded row identity"] <= 0:
                    raise ValueError("MI admission resource buffers disagree with matrix dimensions.")
        mask = tuple(tuple(v is None for v in row) for row in self.original)
        if meta.get("missing_mask") != mask or any(type(v) is not bool for row in meta["missing_mask"] for v in row):
            raise ValueError("MI missing geometry disagrees with original values.")
        counts = tuple(sum(row[j] for row in mask) for j in range(p))
        if meta.get("missing_counts") != counts or meta.get("missing_total") != sum(counts) or any(type(v) is not int for v in meta.get("missing_counts", ())) or type(meta.get("missing_total")) is not int:
            raise ValueError("MI missing counts disagree with original values.")
        sample = meta.get("sample", {})
        if any(sample.get(k) != v for k, v in (("nobs_original", n), ("nobs", n), ("dropped_rows", 0), ("positions", tuple(range(n))))):
            raise ValueError("MI sample identity disagrees with matrix rows.")
        if any(type(sample.get(k)) is not int for k in ("nobs_original", "nobs", "dropped_rows")) or any(type(v) is not int for v in sample.get("positions", ())):
            raise ValueError("MI sample identities must be integers.")
        try:
            if len(_decode_index(meta["index"])) != n:
                raise ValueError("MI index length disagrees with matrix rows.")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("Invalid declared MI row index.") from error
        if meta.get("imputation_ids") != tuple(range(1, m + 1)) or any(type(v) is not int for v in meta.get("imputation_ids", ())):
            raise ValueError("MI imputation IDs must be the ordered integers 1..m.")
        seeds = meta.get("imputation_seeds", ())
        if len(seeds) != m:
            raise ValueError("MI must declare one seed per imputation.")
        for seed in seeds:
            check_seed(seed)
        for completed in self.completed_matrices:
            for source, target in zip(self.original, completed):
                if any(not math.isfinite(v) for v in target):
                    raise ValueError("MI completed matrices must be finite.")
                if any(a is not None and a != b for a, b in zip(source, target)):
                    raise ValueError("MI must preserve every observed cell unchanged.")
        if _digest(self.model_dump(exclude={"integrity_sha256"})) != self.integrity_sha256:
            raise ValueError("MI integrity checksum disagrees with the recorded full state.")
        return self

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False):
        """Validate updates instead of Pydantic's unchecked copy shortcut."""
        if not update:
            return self
        state = self.model_dump()
        state.update(update)
        return type(self).model_validate(state)

    def dataset(self, imputation: int = 1):
        """A new complete DataFrame for the 1-based imputation ID, retaining row index."""
        from openecon.frame import DataFrame
        imputation = integer(imputation, "imputation", minimum=1, maximum=len(self.completed_matrices))
        return DataFrame(self.completed_matrices[imputation - 1], columns=list(self.columns),
                         index=_decode_index(self.metadata["index"]))

    @property
    def table(self):
        from openecon.econometrics.core import table
        n = len(self.original)
        return table([{"column": name, "observed": n - count, "missing": count,
                       "imputations": len(self.completed_matrices)}
                      for name, count in zip(self.columns, self.metadata["missing_counts"])],
                     method=self.method, seed=self.seed, nobs=n)

    @property
    def latex(self):
        return self.table.latex

    def to_latex(self, **options):
        return self.table.to_latex(**options)


def make_result(method: str, frame: pd.DataFrame, values: torch.Tensor, missing: torch.Tensor,
                completed: Sequence[torch.Tensor], seed: int, metadata: Mapping,
                imputation_seeds: Sequence[int] | None = None) -> MIResult:
    """Construct the common result after a native sampler has filled missing cells."""
    seed = check_seed(seed)
    meta = dict(metadata)
    m = len(completed)
    meta.update({"method": method, "seed": seed, "imputation_ids": list(range(1, m + 1)),
                 "imputation_seeds": list(imputation_seeds) if imputation_seeds is not None else [seed] * m})
    original = tuple(tuple(None if absent else float(v) for v, absent in zip(row, mask))
                     for row, mask in zip(values.tolist(), missing.tolist()))
    matrices = tuple(tuple(tuple(float(v) for v in row) for row in matrix.tolist()) for matrix in completed)
    state = {"method": method, "columns": tuple(frame.columns), "original": original,
             "completed_matrices": matrices, "seed": seed, "metadata": meta}
    return MIResult(**state, integrity_sha256=_digest(state))
