"""Replayable treatment coding without retaining observations or group maps.

Categorical dictionaries come from the original projected inputs before missing
rows are removed. Pandas Categorical declarations retain their order and unused
levels. Arrow dictionary columns follow the existing dense API: only observed
values determine their sorted levels. Numeric designs need no discovery pass.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timedelta
from numbers import Integral, Real
import sys
from typing import Any
from uuid import UUID

import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.models import ModelSpec

MAX_PARAMETERS = 384
WORKING_BYTES = 128 * 1024 * 1024
MAX_CATEGORY_BYTES = 2 * 1024 * 1024
MAX_CLUSTER_BYTES = 6 * 1024 * 1024
MAX_KEY_DEPTH = 8
SAMPLE_LIMIT = 400


def _scalar(value: Any) -> Any:
    kind = getattr(getattr(value, "dtype", None), "kind", None)
    if kind == "M":
        return pd.Timestamp(value)
    if kind == "m":
        return pd.Timedelta(value)
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    return value


def _json_scalar(value: Any) -> str | int | float | bool:
    value = _scalar(value)
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(value):
        return float(value)
    raise AnalysisError("unsupported_category", "Categorical levels must be finite strings, numbers, or booleans.")


def _integer_bytes(value: int) -> bytes:
    magnitude = abs(value)
    width = max(1, (magnitude.bit_length() + 7) // 8)
    return bytes([int(value < 0)]) + width.to_bytes(8, "big") + magnitude.to_bytes(width, "big")


def _key_guard(size: int, budget: int) -> None:
    if size > budget:
        raise AnalysisError("cluster_key_budget", "Cluster labels exceed the bounded batch budget; use smaller batches or shorter labels.")


def _label(value: Any, *, budget: int = MAX_CLUSTER_BYTES, depth: int = 0,
           typed_temporal: bool = False) -> bytes:
    """Injective scalar equality key; numerically equal types share a key."""
    if depth > MAX_KEY_DEPTH:
        raise AnalysisError("cluster_key_budget", "Cluster labels exceed the supported nesting depth.")
    if getattr(getattr(value, "dtype", None), "kind", None) in {"M", "m"}:
        raise AnalysisError("invalid_clusters", "Use pandas Timestamp or Timedelta labels instead of object-valued NumPy temporal scalars.")
    value = _scalar(value)
    if value is None or value is pd.NA or value is pd.NaT:
        raise AnalysisError("invalid_clusters", "Cluster labels must not contain missing values.")
    if isinstance(value, str):
        width = 1
        try:
            for start in range(0, len(value), 1024):
                width += len(value[start:start + 1024].encode("utf-8"))
                _key_guard(width, budget)
            return b"S" + value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise AnalysisError("invalid_clusters", "Cluster strings must contain valid UTF-8 text.") from exc
    if isinstance(value, bytes):
        _key_guard(len(value) + 1, budget)
        return b"B" + value
    if isinstance(value, (tuple, frozenset)):
        size = 9 + 8 * len(value)
        _key_guard(size, budget)
        parts = []
        for member in value:
            encoded = _label(member, budget=budget - size, depth=depth + 1)
            size += len(encoded)
            _key_guard(size, budget)
            parts.append(encoded)
        if isinstance(value, frozenset):
            parts.sort()
        return (b"Q" if isinstance(value, tuple) else b"F") + len(parts).to_bytes(8, "big") + b"".join(
            len(part).to_bytes(8, "big") + part for part in parts)
    if isinstance(value, (datetime, pd.Timestamp)):
        if (not (isinstance(value, pd.Timestamp) and typed_temporal) and value.tzinfo is not None
                and value.utcoffset() != value.replace(fold=1 - value.fold).utcoffset()):
            raise AnalysisError("invalid_clusters", "Ambiguous local datetime labels need a typed pandas datetime Series to preserve group identity.")
        timestamp = pd.Timestamp(value)
        aware = timestamp.tzinfo is not None
        if aware:
            timestamp = timestamp.tz_convert("UTC")
        fields = (timestamp.year, timestamp.month, timestamp.day, timestamp.hour,
                  timestamp.minute, timestamp.second, timestamp.microsecond, timestamp.nanosecond)
        encoded = b"T" + bytes([aware]) + b"".join(_integer_bytes(field) for field in fields)
        _key_guard(len(encoded), budget)
        return encoded
    if isinstance(value, date):
        encoded = b"D" + b"".join(_integer_bytes(field) for field in (value.year, value.month, value.day))
        _key_guard(len(encoded), budget)
        return encoded
    if isinstance(value, (timedelta, pd.Timedelta)):
        fields = (value.days, value.seconds, value.microseconds, getattr(value, "nanoseconds", 0))
        encoded = b"L" + b"".join(_integer_bytes(field) for field in fields)
        _key_guard(len(encoded), budget)
        return encoded
    if isinstance(value, UUID):
        _key_guard(17, budget)
        return b"U" + value.bytes
    if isinstance(value, Integral):
        numerator, denominator = int(value), 1
    elif isinstance(value, Real):
        ratio = getattr(value, "as_integer_ratio", None)
        if not callable(ratio):
            raise AnalysisError("invalid_clusters", "Cluster labels must be strings, finite real numbers, booleans, or bytes.")
        try:
            numerator, denominator = ratio()
        except (ValueError, OverflowError) as exc:
            raise AnalysisError("non_finite_values", "Cluster labels must contain finite values.") from exc
    else:
        raise AnalysisError("invalid_clusters", "Cluster labels must be strings, finite real numbers, booleans, or bytes.")
    numerator, denominator = int(numerator), int(denominator)
    if denominator <= 0:
        raise AnalysisError("invalid_clusters", "Cluster labels must contain valid finite real numbers.")
    _key_guard(19 + max(1, (numerator.bit_length() + 7) // 8) + max(1, (denominator.bit_length() + 7) // 8), budget)
    return b"N" + _integer_bytes(numerator) + _integer_bytes(denominator)


def encode_cluster_labels(series: pd.Series) -> list[bytes]:
    """Encode one retained batch, preserving pandas factorize equality.

    Strings and numeric types are separate; True, 1 and 1.0 are equal, as are
    negative and positive zero. Finite numbers use exact integer ratios, so
    adjacent large integers never collapse through a float64 conversion.
    Ordinary temporal values, UUIDs, tuples and frozensets are framed without
    digest collisions. Ambiguous object datetimes and object-valued NumPy
    temporals are rejected; typed pandas datetime Series preserve DST instants.
    """
    labels: list[bytes] = []
    payload_size = 0
    size = sys.getsizeof(labels)
    typed_temporal = getattr(series.dtype, "kind", None) == "M"
    for value in series:
        available = MAX_CLUSTER_BYTES - size - sys.getsizeof(b"") - 8
        encoded = _label(value, budget=available, typed_temporal=typed_temporal)
        payload_size += sys.getsizeof(encoded)
        labels.append(encoded)
        size = payload_size + sys.getsizeof(labels)
        if size > MAX_CLUSTER_BYTES:
            raise AnalysisError("cluster_key_budget", "Cluster labels exceed the bounded batch budget; use smaller batches.")
    return labels


def numeric_values(series: pd.Series, name: str) -> torch.Tensor:
    if not (is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype)):
        raise AnalysisError("non_numeric_column", f"Column '{name}' must be numeric; declare a categorical predictor explicitly.")
    if is_complex_dtype(series.dtype):
        raise AnalysisError("complex_values", f"Column '{name}' must contain real numbers.")
    buffer = series.to_numpy(dtype="float64", na_value=float("nan"))
    if not buffer.flags.writeable:
        buffer = buffer.copy()
    values = torch.as_tensor(buffer, dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(values).all()):
        raise AnalysisError("non_finite_values", f"Column '{name}' contains non-finite values.")
    return values


def row_hash_bytes(projected: pd.DataFrame) -> bytes:
    """Existing streaming hash contract, independent of batch boundaries."""
    large_dictionaries = [name for name in projected.columns
                          if isinstance(projected[name].dtype, pd.CategoricalDtype)
                          and len(projected[name].cat.categories) > MAX_PARAMETERS + 1]
    if large_dictionaries:
        # Cluster columns can have enormous unused dictionaries although only
        # one bounded set of row values is relevant. Categorical.to_numpy takes
        # those row codes before casting; hashing the full dictionary would
        # allocate in proportion to unused levels rather than batch rows.
        projected = projected.copy(deep=False)
        for name in large_dictionaries:
            projected[name] = pd.Series(projected[name].to_numpy(dtype=object),
                                        index=projected.index, dtype=object)
    try:
        hashed = pd.util.hash_pandas_object(projected, index=False, categorize=True)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("unsupported_values", "Model inputs must contain scalar values.") from exc
    return hashed.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes()


class StreamingDesign:
    def __init__(self, source: Dataset, spec: ModelSpec, rows_per_batch: int):
        if isinstance(rows_per_batch, bool) or not isinstance(rows_per_batch, int) or rows_per_batch < 1:
            raise AnalysisError("invalid_batch_size", "A positive integer reader batch size is required.")
        if spec.cluster is not None and not isinstance(spec.cluster, str):
            raise AnalysisError("unsupported_cluster_dimensions", "Streaming estimation supports one cluster column.")
        self.source, self.spec = source, spec
        self.rows_per_batch = rows_per_batch
        self.columns = list(dict.fromkeys([spec.outcome, *spec.predictors,
                                          *([spec.cluster] if spec.cluster else [])]))
        self.terms: list[str] = []
        self.categorical_encoding: dict[str, dict[str, Any]] = {}
        self.baseline: dict[str, Any] | None = None
        self.sample_rows: list[int] = []
        self._levels: dict[str, dict[bytes, Any]] = {name: {} for name in spec.categorical}
        self._declared: dict[str, tuple[tuple[bytes, ...], bool] | None] = {}
        self._category_bytes = 0
        self._prepared = False

    def _width(self) -> int:
        return int(self.spec.intercept) + len(self.spec.predictors) - len(self.spec.categorical) + sum(
            max(0, len(levels) - 1) for levels in self._levels.values())

    def _guard_width(self) -> None:
        k = self._width()
        if k > MAX_PARAMETERS or 80 * (k + 1) ** 2 * 8 >= WORKING_BYTES:
            raise AnalysisError("model_too_wide", "The expanded model exceeds the bounded factor-memory budget; reduce the number of predictors or category levels.")

    def _add_level(self, name: str, value: Any) -> None:
        value = _scalar(value)
        _json_scalar(value)
        if sys.getsizeof(value) + 128 > MAX_CATEGORY_BYTES:
            raise AnalysisError("category_metadata_too_large", "Categorical metadata exceeds its bounded memory budget; shorten labels or reduce category levels.")
        key = _label(value)
        if key in self._levels[name]:
            return
        size = sys.getsizeof(value) + sys.getsizeof(key) + 128
        if self._category_bytes + size > MAX_CATEGORY_BYTES:
            raise AnalysisError("category_metadata_too_large", "Categorical metadata exceeds its bounded memory budget; shorten labels or reduce category levels.")
        self._levels[name][key] = value
        self._category_bytes += size
        self._guard_width()

    def _check_declaration(self, name: str, series: pd.Series, *, discover: bool) -> None:
        if isinstance(series.dtype, pd.CategoricalDtype):
            categories = series.cat.categories
            # Check before copying the categorical Index into Python objects.
            if len(categories) - 1 + int(self.spec.intercept) + len(self.spec.predictors) - len(self.spec.categorical) > MAX_PARAMETERS:
                raise AnalysisError("model_too_wide", "Declared categorical levels exceed the bounded factor-memory budget.")
            keys = []
            for value in categories:
                _json_scalar(value)
                if discover:
                    self._add_level(name, value)
                keys.append(_label(value))
            declaration = (tuple(keys), bool(series.cat.ordered))
        else:
            declaration = None
        if name in self._declared and declaration != self._declared[name]:
            raise AnalysisError("source_changed", f"Categorical declaration for '{name}' changed between batches or passes.")
        if discover:
            self._declared[name] = declaration

    def validate_batch(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Select complete observations; cluster missingness participates."""
        projected = frame.loc[:, self.columns]
        if self._prepared:
            for name in self.spec.categorical:
                self._check_declaration(name, projected[name], discover=False)
        missing = projected.isna().any(axis=1)
        if bool(missing.any()) and self.spec.missing == "raise":
            raise AnalysisError("missing_values", "Model inputs contain missing observations; choose missing='drop' explicitly to exclude them.")
        return projected.loc[~missing]

    def prepare(self) -> StreamingDesign:
        if self._prepared:
            return self
        absent = [name for name in self.columns if name not in self.source.columns]
        if absent:
            raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
        self._guard_width()
        if self.spec.categorical:
            self._discover()
        self.terms = ["Intercept"] if self.spec.intercept else []
        for name in self.spec.predictors:
            if name in self.spec.categorical:
                declaration = self._declared[name]
                raw_levels = list(self._levels[name].values())
                if declaration is None:
                    try:
                        raw_levels = sorted(raw_levels)
                    except (TypeError, ValueError) as exc:
                        raise AnalysisError("ambiguous_categories", f"Column '{name}' has mixed category types; use a pandas Categorical with explicit levels.") from exc
                safe = [_json_scalar(value) for value in raw_levels]
                ordered = declaration[1] if declaration is not None else False
                self._levels[name] = {_label(value): value for value in raw_levels}
                self.categorical_encoding[name] = {
                    "levels": safe, "reference": safe[0], "ordered": ordered,
                    "coding": "treatment_drop_first", "levels_source": "original_model_inputs_before_missing_filter",
                }
                self.terms.extend(f"{name}[{value}]" for value in safe[1:])
            else:
                self.terms.append(name)
        if len(self.terms) != len(set(self.terms)):
            raise AnalysisError("duplicate_terms", "Predictor names collide with generated design terms; rename the affected columns.")
        self._prepared = True
        return self

    def _discover(self) -> None:
        digest = hashlib.sha256()
        digest.update(json.dumps({"columns": self.columns, "missing": self.spec.missing},
                                 sort_keys=True, separators=(",", ":")).encode())
        position_digest = hashlib.sha256()
        original_count = used_count = 0
        sample_rows: list[int] = []
        numeric_names = [self.spec.outcome, *[name for name in self.spec.predictors if name not in self.spec.categorical]]
        first: dict[str, float] = {}
        varied = dict.fromkeys(numeric_names, False)
        observed = {name: set() for name in self.spec.categorical}
        invalid_binary = False
        iterator = self.source.iter_batches(self.columns, batch_rows=self.rows_per_batch)
        self.source.assert_unchanged()
        try:
            for frame in iterator:
                projected = frame.loc[:, self.columns]
                retained = self.validate_batch(projected)
                for name in self.spec.categorical:
                    self._check_declaration(name, projected[name], discover=True)
                    if self._declared[name] is None:
                        for value in projected[name].dropna():
                            self._add_level(name, value)
                # Declared category dictionaries are checked before pandas
                # hashing, which otherwise hashes unused dictionary levels.
                digest.update(row_hash_bytes(projected))
                keep = ~projected.isna().any(axis=1)
                offsets = torch.from_numpy(keep.to_numpy()).nonzero().flatten().to(torch.int64)
                positions = offsets + original_count
                position_digest.update(positions.numpy().astype("<i8", copy=False).tobytes())
                sample_rows.extend(positions[:max(0, SAMPLE_LIMIT - len(sample_rows))].tolist())
                original_count += len(projected)
                used_count += len(retained)
                for name in self.spec.categorical:
                    if len(observed[name]) < 2:
                        for value in retained[name]:
                            observed[name].add(_label(value))
                            if len(observed[name]) == 2:
                                break
                if retained.empty:
                    continue
                for name in numeric_names:
                    values = numeric_values(retained[name], name)
                    if name not in first:
                        first[name] = float(values[0])
                    varied[name] = varied[name] or bool(torch.any(values != first[name]))
                    if name == self.spec.outcome and self.spec.estimator != "ols":
                        invalid_binary = invalid_binary or not bool(((values == 0) | (values == 1)).all())
                if self.spec.cluster:
                    start = 0
                    while start < len(retained):
                        width = min(1024, len(retained) - start)
                        while True:
                            try:
                                encode_cluster_labels(retained[self.spec.cluster].iloc[start:start + width])
                                break
                            except AnalysisError as exc:
                                if exc.code != "cluster_key_budget" or width == 1:
                                    raise
                                width = max(1, width // 2)
                        start += width
        finally:
            iterator.close()
        self.source.assert_unchanged()
        if original_count == 0:
            raise AnalysisError("empty_data", "The dataset contains no observations.")
        if used_count == 0:
            raise AnalysisError("empty_sample", "No complete observations remain for estimation.")
        if not varied[self.spec.outcome]:
            raise AnalysisError("constant_outcome", "The outcome must contain at least two distinct values.")
        if invalid_binary:
            raise AnalysisError("invalid_binary_outcome", "Logit and probit require an outcome containing both 0 and 1 (or booleans).")
        for name in self.spec.predictors:
            if name in self.spec.categorical:
                if len(self._levels[name]) < 2 or len(observed[name]) < 2:
                    raise AnalysisError("constant_predictor", f"Categorical predictor '{name}' has fewer than two observed levels.")
            elif not varied[name]:
                raise AnalysisError("constant_predictor", f"Predictor '{name}' is constant; use the intercept option instead.")
        if used_count <= self._width():
            raise AnalysisError("insufficient_observations", f"Estimation requires more observations ({used_count}) than design columns ({self._width()}).")
        self.baseline = {"original": original_count, "used": used_count,
                         "data_hash": digest.hexdigest(), "positions_hash": position_digest.hexdigest()}
        self.sample_rows = sample_rows

    def encode(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._prepared:
            raise AnalysisError("unprepared_design", "Prepare the streaming design before encoding batches.")
        k = len(self.terms)
        scratch = max(1, k) * 8 * 32
        if len(frame) * scratch + 80 * (k + 1) ** 2 * 8 > WORKING_BYTES:
            raise AnalysisError("batch_too_large", "The encoded batch exceeds the numerical working-memory budget; split retained rows into smaller batches.")
        if bool(frame.loc[:, self.columns].isna().any(axis=None)):
            raise AnalysisError("missing_values", "Encoding requires retained observations without missing model inputs.")
        y = numeric_values(frame[self.spec.outcome], self.spec.outcome)
        pieces = [torch.ones((len(frame), 1), dtype=torch.float64)] if self.spec.intercept else []
        for name in self.spec.predictors:
            if name not in self.spec.categorical:
                pieces.append(numeric_values(frame[name], name)[:, None])
                continue
            self._check_declaration(name, frame[name], discover=False)
            levels = list(self._levels[name].values())
            encoded = pd.Categorical(frame[name], categories=levels,
                                     ordered=self.categorical_encoding[name]["ordered"])
            if bool((encoded.codes < 0).any()):
                raise AnalysisError("source_changed", f"Categorical predictor '{name}' gained values after discovery.")
            dummies = pd.get_dummies(pd.Series(encoded), drop_first=True, dtype="float64")
            pieces.append(torch.as_tensor(dummies.to_numpy(), dtype=torch.float64))
        return torch.cat(pieces, dim=1), y
