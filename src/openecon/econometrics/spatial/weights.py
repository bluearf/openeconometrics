"""Keyed, nonnegative, zero-diagonal sparse weights; no network inference."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from collections.abc import Sized

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace


def _weight_plan(n, entries, operation, *, key_characters=0):
    # Include owned Python indexing/sorting/COO copies as well as numeric buffers.
    # This remains a conservative buffer estimate, not a process-RSS promise.
    return plan_workspace(
        operation,
        {
            "key_indices_and_owned_copies": 160 * n + 12 * key_characters,
            "edge_dictionary_sorting_COO_and_payload_copies": 384 * entries,
        },
    )


def _key(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise AnalysisError(
            "invalid_spatial_keys",
            "Spatial identities must be nonmissing strings or integers (not booleans or floats).",
        )
    if isinstance(value, str) and not value:
        raise AnalysisError("invalid_spatial_keys", "Spatial identities must not be empty.")
    return int(value) if isinstance(value, Integral) else value


def _keys(values):
    if isinstance(values, (str, bytes)):
        raise AnalysisError(
            "invalid_spatial_keys", "Spatial keys must be a sequence of unique strings or integers."
        )
    reserved = len(values) if isinstance(values, Sized) else 0
    _weight_plan(reserved, 0, "spatial key validation")
    out = []
    characters = 0
    for value in values:
        value = _key(value)
        characters += len(value) if isinstance(value, str) else 0
        if len(out) == reserved:
            reserved = max(64, 2 * reserved)
        _weight_plan(reserved, 0, "spatial key validation", key_characters=characters)
        out.append(value)
    if not out or len(out) != len(set(out)):
        raise AnalysisError("invalid_spatial_keys", "Spatial keys must be nonempty and unique.")
    return tuple(out)


@dataclass(frozen=True, init=False)
class SpatialWeights:
    """Immutable effective COO weights, always constructed through validated payloads.

    Row i receives the weighted value of column j. An isolate is a zero row;
    ``isolates='zero'`` retains it in every sample and in Moran's N.
    """

    keys: tuple
    rows: tuple[int, ...]
    cols: tuple[int, ...]
    values: tuple[float, ...]
    normalization: str
    diagonal: str
    isolates: str

    @classmethod
    def _create(cls, keys, rows, cols, values, normalization, diagonal, isolates):
        result = object.__new__(cls)
        for name, value in zip(
            ("keys", "rows", "cols", "values", "normalization", "diagonal", "isolates"),
            (keys, tuple(rows), tuple(cols), tuple(values), normalization, diagonal, isolates),
            strict=True,
        ):
            object.__setattr__(result, name, value)
        return result

    @property
    def n(self):
        return len(self.keys)

    @property
    def nnz(self):
        return len(self.values)

    def to_payload(self):
        _weight_plan(
            self.n,
            self.nnz,
            "spatial weight payload",
            key_characters=sum(len(k) for k in self.keys if isinstance(k, str)),
        )
        return {
            "schema": "openecon.spatial_weights.v1",
            "keys": list(self.keys),
            "rows": list(self.rows),
            "cols": list(self.cols),
            "values": list(self.values),
            "normalization": self.normalization,
            "diagonal": self.diagonal,
            "isolates": self.isolates,
        }

    @classmethod
    def from_payload(cls, payload):
        if isinstance(payload, cls):
            return payload
        expected = {
            "schema",
            "keys",
            "rows",
            "cols",
            "values",
            "normalization",
            "diagonal",
            "isolates",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload["schema"] != "openecon.spatial_weights.v1"
        ):
            raise AnalysisError(
                "invalid_spatial_weights",
                "Provide a SpatialWeights object or its complete versioned JSON payload.",
            )
        rows, cols, values = payload["rows"], payload["cols"], payload["values"]
        if not all(isinstance(part, (list, tuple)) for part in (rows, cols, values)) or not len(
            rows
        ) == len(cols) == len(values):
            raise AnalysisError(
                "invalid_spatial_weights", "COO rows, cols and values must have equal lengths."
            )
        if not isinstance(payload["keys"], (list, tuple)):
            raise AnalysisError(
                "invalid_spatial_keys", "Persisted spatial keys must be a JSON sequence."
            )
        _weight_plan(len(payload["keys"]), len(values), "spatial weight payload validation")
        keys = _keys(payload["keys"])
        if any(
            isinstance(i, bool) or not isinstance(i, Integral) or i < 0 or i >= len(keys)
            for positions in (rows, cols)
            for i in positions
        ):
            raise AnalysisError(
                "invalid_spatial_weights",
                "COO positions must be integer indices in the key domain.",
            )
        if any(i == j and value != 0 for i, j, value in zip(rows, cols, values, strict=True)):
            raise AnalysisError(
                "spatial_diagonal", "Persisted effective spatial weights must have a zero diagonal."
            )
        # Validate *effective* row-normalized payloads without silently changing a persisted W.
        result = spatial_weights(
            keys,
            ((keys[i], keys[j], value) for i, j, value in zip(rows, cols, values, strict=True)),
            normalization="none",
            diagonal=payload["diagonal"],
            isolates=payload["isolates"],
        )
        normalization = payload["normalization"]
        if normalization not in {"none", "row"}:
            raise AnalysisError("invalid_spatial_weights", "normalization must be 'none' or 'row'.")
        if normalization == "row":
            sums = [0.0] * len(keys)
            for i, value in zip(result.rows, result.values, strict=True):
                sums[i] += value
            if any(
                value and not math.isclose(value, 1.0, rel_tol=1e-12, abs_tol=1e-12)
                for value in sums
            ):
                raise AnalysisError(
                    "invalid_spatial_weights",
                    "Persisted row-normalized weights must have unit nonzero row sums.",
                )
        return cls._create(
            keys,
            result.rows,
            result.cols,
            result.values,
            normalization,
            payload["diagonal"],
            payload["isolates"],
        )

    def align(self, keys, *, subset=False):
        """Reorder by identities; induced subsetting is explicit and re-normalizes rows."""
        keys = _keys(keys)
        if keys == self.keys:
            return self
        _weight_plan(self.n + len(keys), self.nnz, "spatial keyed weight alignment")
        if not set(keys).issubset(self.keys) or (not subset and set(keys) != set(self.keys)):
            raise AnalysisError(
                "spatial_key_mismatch",
                "Table and spatial-weight key domains must agree exactly before missing-data filtering.",
            )
        retained = set(keys)
        edges = (
            (self.keys[i], self.keys[j], value)
            for i, j, value in zip(self.rows, self.cols, self.values, strict=True)
            if self.keys[i] in retained and self.keys[j] in retained
        )
        return spatial_weights(
            keys,
            edges,
            normalization=self.normalization,
            diagonal=self.diagonal,
            isolates=self.isolates,
        )

    def sparse_tensor(self):
        plan_workspace(
            "spatial COO tensor", {"COO_and_temporary_indices": 48 * self.nnz + 16 * self.n}
        )
        indices = torch.tensor([self.rows, self.cols], dtype=torch.int64).reshape(2, self.nnz)
        return torch.sparse_coo_tensor(
            indices,
            torch.tensor(self.values, dtype=torch.float64),
            (self.n, self.n),
            dtype=torch.float64,
            check_invariants=True,
        ).coalesce()

    def dense(self, *, max_n=512):
        if isinstance(max_n, bool) or not isinstance(max_n, int) or max_n < 1 or self.n > max_n:
            raise AnalysisError(
                "spatial_dense_limit",
                f"Exact dense spatial computation permits at most max_n={max_n} observations; supplied {self.n}.",
            )
        plan_workspace(
            "spatial dense matrix", {"dense_W_and_COO": 8 * self.n**2 + 48 * self.nnz + 16 * self.n}
        )
        return self.sparse_tensor().to_dense()

    def summary(self):
        _weight_plan(
            self.n,
            self.nnz,
            "spatial weight summary",
            key_characters=sum(len(k) for k in self.keys if isinstance(k, str)),
        )
        row_sums = [0.0] * self.n
        for i, value in zip(self.rows, self.values, strict=True):
            row_sums[i] += value
        digest = hashlib.sha256(
            json.dumps(
                self.to_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        return {
            "n": self.n,
            "nnz": self.nnz,
            "normalization": self.normalization,
            "diagonal": "zero",
            "diagonal_policy": self.diagonal,
            "isolate_policy": self.isolates,
            "isolate_keys": [
                key for key, value in zip(self.keys, row_sums, strict=True) if value == 0
            ],
            "row_sum_max": max(row_sums),
            "sum_weights": sum(row_sums),
            "sha256": digest,
            "orientation": "row i receives column j",
            "precision": "float64",
        }


def spatial_weights(keys, edges, *, normalization="row", diagonal="raise", isolates="raise"):
    """Construct keyed COO W from (receiving_key, source_key, nonnegative_weight).

    Duplicate ordered edges fail; there is no implicit network symmetrization,
    diagonal removal, isolate dropping or data truncation.
    """
    keys = _keys(keys)
    if (
        normalization not in {"none", "row"}
        or diagonal not in {"raise", "drop"}
        or isolates not in {"raise", "zero"}
    ):
        raise AnalysisError(
            "invalid_spatial_weights",
            "Choose normalization='row'/'none', diagonal='raise'/'drop' and isolates='raise'/'zero'.",
        )
    reserved = len(edges) if isinstance(edges, Sized) else 0
    characters = sum(len(k) for k in keys if isinstance(k, str))
    _weight_plan(
        len(keys), reserved, "spatial sparse weight construction", key_characters=characters
    )
    index = {key: i for i, key in enumerate(keys)}
    entries = {}
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 3:
            raise AnalysisError(
                "invalid_spatial_weights", "Each spatial edge must contain two keys and one weight."
            )
        source, target = _key(edge[0]), _key(edge[1])
        if source not in index or target not in index:
            raise AnalysisError(
                "spatial_key_mismatch", "Spatial edges refer to identities absent from keys."
            )
        value = edge[2]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value < 0
        ):
            raise AnalysisError(
                "invalid_spatial_weights", "Spatial weights must be finite nonnegative numbers."
            )
        pair = (index[source], index[target])
        if pair in entries:
            raise AnalysisError(
                "duplicate_spatial_edges",
                "Duplicate ordered spatial edges must be resolved explicitly.",
            )
        if source == target and value != 0:
            if diagonal == "raise":
                raise AnalysisError(
                    "spatial_diagonal",
                    "Spatial weights require a zero diagonal; pass diagonal='drop' explicitly to remove it.",
                )
            continue
        if len(entries) == reserved:
            reserved = max(64, 2 * reserved)
            _weight_plan(
                len(keys),
                reserved,
                "spatial sparse iterable edge ingestion",
                key_characters=characters,
            )
        entries[pair] = float(value)
    sums = [0.0] * len(keys)
    for (i, j), value in entries.items():
        sums[i] += value
    if not all(math.isfinite(value) for value in sums) or not math.isfinite(sum(sums)):
        raise AnalysisError(
            "invalid_spatial_weights", "Spatial row sums and total weight must be finite."
        )
    if any(value == 0 for value in sums) and isolates == "raise":
        raise AnalysisError(
            "spatial_isolates",
            "Spatial weights contain zero-row isolates; pass isolates='zero' explicitly to retain them.",
        )
    positions = sorted(pair for pair, value in entries.items() if value != 0)
    rows, cols = [pair[0] for pair in positions], [pair[1] for pair in positions]
    values = [
        entries[pair] / sums[pair[0]] if normalization == "row" else entries[pair]
        for pair in positions
    ]
    return SpatialWeights._create(keys, rows, cols, values, normalization, diagonal, isolates)
