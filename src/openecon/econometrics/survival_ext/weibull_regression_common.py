"""Early primitive, source and lifecycle admission for interval Weibull regression."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
import hashlib
import json
import math
import struct

import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi.common import _decode_index as _decode_index
from openecon.econometrics.mi.common import _encode_index, _thaw
from openecon.econometrics.mi.common import _freeze as _freeze
from openecon.econometrics.mi.common import _label, _unlabel
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.resources import plan_workspace

MAX_ROWS, MAX_COLUMNS = 4096, 10
MAX_JSON, MAX_INDEX = 32 * 1024**2, 1024**2


def fail(message, code="invalid_state"):
    raise AnalysisError(code, message)


def checked(fn):
    @wraps(fn)
    @resident_cpu
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except AnalysisError:
            raise
        except (
            ValueError,
            TypeError,
            KeyError,
            OverflowError,
            RuntimeError,
            RecursionError,
        ) as exc:
            fail(f"Complete interval Weibull regression admission/replay failed: {exc}")

    return wrapped


def integer(value, label, low, high):
    if type(value) is not int or not low <= value <= high:
        fail(f"{label} requires an integer in {low}..{high}.", "invalid_argument")
    return value


def real(value, label, *, positive=False):
    if type(value) not in (float, int) or type(value) is int and value.bit_length() > 512:
        fail(f"{label} requires a finite real primitive.", "invalid_argument")
    value = float(value)
    if not math.isfinite(value) or positive and value <= 0:
        fail(
            f"{label} requires a finite {'positive ' if positive else ''}real.", "invalid_argument"
        )
    return value


def name(value):
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 64
        or value == "Intercept"
        or value in ("log_sigma", "log_shape")
    ):
        fail(
            "Column names require 1..64 characters, excluding Intercept and the M: namespace.",
            "invalid_argument",
        )
    return value


def names(value, *, empty=False):
    if not isinstance(value, (list, tuple)) or not int(not empty) <= len(value) <= MAX_COLUMNS:
        fail("Declare at most 16 ordered column names.", "invalid_argument")
    out = [name(v) for v in value]
    if len(set(out)) != len(out):
        fail("Column names must be distinct.", "invalid_argument")
    return out


def keys(value, expected, label):
    if (
        not isinstance(value, Mapping)
        or len(value) != len(expected)
        or any(key not in value for key in expected)
    ):
        fail(f"{label} requires exactly the declared schema keys.")


def measure(value, *, maximum=MAX_JSON, indent=0):
    """Streaming bound before escaped JSON, checksum, parser or nested-copy allocation."""
    encoded = resident = nodes = 0
    pending = [(iter((value,)), 0)]
    while pending:
        iterator, depth = pending[-1]
        try:
            item = next(iterator)
        except StopIteration:
            pending.pop()
            continue
        nodes += 1
        if depth > 64 or nodes > 2000000:
            fail("interval Weibull state exceeds bounded metadata traversal.", "state_limit")
        if isinstance(item, Mapping):
            if any(not isinstance(k, str) for k in item):
                fail("Saved mapping keys require strings.")
            encoded += 2 + 4 * len(item) + indent * ((depth + 1) * len(item) + depth)
            resident += 256 + 96 * len(item)
            pending.append((iter(v for pair in item.items() for v in pair), depth + 1))
        elif isinstance(item, (list, tuple)):
            encoded += 2 + 2 * len(item) + indent * ((depth + 1) * len(item) + depth)
            resident += 64 + 16 * len(item)
            pending.append((iter(item), depth + 1))
        elif isinstance(item, str):
            encoded += 2 + 12 * len(item)
            resident += 64 + 4 * len(item)
        elif type(item) is int:
            if item.bit_length() > 512:
                fail("Saved integers exceed primitive bounds.")
            encoded += max(24, item.bit_length() // 3 + 2)
            resident += 32 + item.bit_length() // 8
        elif item is None or type(item) is bool or type(item) is float and math.isfinite(item):
            encoded += 24
            resident += 32
        else:
            fail("Saved state requires finite JSON primitives.")
        if encoded > maximum or resident > 128 * 1024**2:
            fail("interval Weibull state exceeds its encoded/resident envelope.", "state_limit")
        if nodes % 4096 == 0 or isinstance(item, (str, list, tuple, Mapping)) and len(item) > 1024:
            plan_workspace(
                "interval Weibull early metadata",
                {"complete resident copies": 4 * resident, "encoded copies": 3 * encoded},
            )
    plan_workspace(
        "interval Weibull complete metadata",
        {"complete resident copies": 4 * resident, "encoded copies": 3 * encoded},
    )
    return encoded, resident


def json_input(value):
    if not isinstance(value, (str, bytes, bytearray)) or len(value) > MAX_JSON:
        fail("interval Weibull JSON exceeds 32 MiB.", "state_limit")
    size = 0
    if isinstance(value, str):
        for pos in range(0, len(value), 4096):
            size += len(value[pos : pos + 4096].encode())
            if size > MAX_JSON:
                fail("interval Weibull UTF-8 JSON exceeds 32 MiB.", "state_limit")
    else:
        size = len(value)
    plan_workspace("interval Weibull before JSON parser", {"encoded and parsed record": 64 * size})
    text = value if isinstance(value, str) else value.decode("utf-8")
    depth, quoted, escaped = 0, False, False
    for ch in text:
        if quoted:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
        elif ch == '"':
            quoted = True
        elif ch in "[{":
            depth += 1
            if depth > 64:
                fail("interval Weibull JSON nesting exceeds 64.", "state_limit")
        elif ch in "]}":
            depth -= 1
    return text


def load(value):
    if isinstance(value, (str, bytes, bytearray)):
        value = json.loads(json_input(value))
    measure(value)
    if not isinstance(value, Mapping):
        fail("Supply complete interval Weibull state or JSON.")
    return _thaw(value)


def digest(value):
    measure(value)
    return hashlib.sha256(
        json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def seal(value):
    encoded, resident = measure(value)
    plan_workspace(
        "interval Weibull transport-ready result",
        {"readback parser envelope": 64 * encoded, "immutable result": resident},
    )
    return dict(value, sha256=digest(value))


def equal(actual, expected, label, *, rtol=1e-11):
    """No dimensionful absolute floor; exact expected zeros and primitive types."""
    if isinstance(expected, Mapping):
        keys(actual, expected, label)
        for key in expected:
            equal(actual[key], expected[key], f"{label}.{key}", rtol=rtol)
    elif isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            fail(f"{label} dimensions disagree.")
        for a, b in zip(actual, expected):
            equal(a, b, label, rtol=rtol)
    elif type(expected) is float:
        a = real(actual, label)
        if a != expected and (expected == 0 or abs(a / expected - 1) > rtol):
            fail(f"{label} disagrees with its complete numerical replay.")
    elif type(actual) is not type(expected) or actual != expected:
        fail(f"{label} disagrees with its typed source semantics.")


def vector(value, length, label, *, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        fail(f"{label} dimensions disagree.")
    return [real(v, label, positive=positive) for v in value]


def matrix(value, n, p, label, *, nullable=False):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        fail(f"{label} row dimensions disagree.")
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != p:
            fail(f"{label} column dimensions disagree.")
        for cell in row:
            if cell is None and nullable:
                continue
            real(cell, label)


def dtype(value):
    if not isinstance(value, str) or len(value) > 32:
        fail("Numeric source dtype requires a bounded string.")
    parsed = pd.api.types.pandas_dtype(value)
    if not (is_float_dtype(parsed) or is_integer_dtype(parsed)) or parsed.itemsize > 8:
        fail("interval Weibull source dtypes require real float/integer widths at most 64 bits.")
    return parsed


def label_header(value, depth=0):
    keys(value, ("type", "value"), "index label")
    kind, raw = value["type"], value["value"]
    if depth > 16:
        fail("Index label nesting exceeds 16.")
    if kind == "scalar":
        if raw is not None and type(raw) not in (str, bool, int, float):
            fail("Invalid scalar index label.")
        if type(raw) in (int, float):
            real(raw, "index number")
    elif kind == "tuple":
        if not isinstance(raw, (list, tuple)) or len(raw) > 16:
            fail("Index tuple exceeds 16 components.")
        for cell in raw:
            label_header(cell, depth + 1)
    elif kind in ("nan", "nat", "missing"):
        if raw is not None:
            fail("Missing index marker requires null.")
    elif kind in ("timestamp", "date"):
        if not isinstance(raw, str) or len(raw) > 128:
            fail("Date index labels require bounded ISO strings.")
    elif kind == "timedelta":
        integer(raw, "timedelta", -(2**63), 2**63 - 1)
    else:
        fail("Unsupported index label encoding.")


def index_header(value, n, *, exact=True, depth=0):
    if not isinstance(value, Mapping) or depth > 16:
        fail("Index descriptor exceeds supported nesting.")
    kind = value.get("kind")
    if kind == "multi":
        keys(value, ("kind", "names", "levels", "codes", "sortorder"), "multi-index")
        levels, codes, labels = (value[k] for k in ("levels", "codes", "names"))
        if (
            not all(isinstance(v, (list, tuple)) for v in (levels, codes, labels))
            or not 1 <= len(levels) <= 16
            or len(codes) != len(levels)
            or len(labels) != len(levels)
        ):
            fail("Multi-index dimensions disagree.")
        if value["sortorder"] is not None:
            integer(value["sortorder"], "index sortorder", 0, len(levels))
        for level, code, label in zip(levels, codes, labels):
            count = index_header(level, MAX_ROWS, exact=False, depth=depth + 1)
            label_header(label)
            if not isinstance(code, (tuple, list)) or len(code) != n:
                fail("Index code length disagrees with every physical row.")
            for cell in code:
                integer(cell, "index code", -1, count - 1)
        return n
    label_header(value.get("name"))
    if kind == "range":
        keys(value, ("kind", "name", "start", "stop", "step"), "range index")
        for key in ("start", "stop", "step"):
            integer(value[key], key, -(2**63), 2**63 - 1)
        if value["step"] == 0:
            fail("Index step cannot be zero.")
        count = len(range(value["start"], value["stop"], value["step"]))
    elif kind == "categorical":
        keys(value, ("kind", "name", "categories", "ordered", "codes"), "categorical index")
        categories = index_header(value["categories"], MAX_ROWS, exact=False, depth=depth + 1)
        if type(value["ordered"]) is not bool or not isinstance(value["codes"], (list, tuple)):
            fail("Categorical index schema disagrees.")
        count = len(value["codes"])
        for cell in value["codes"]:
            integer(cell, "index code", -1, categories - 1)
    elif kind in ("index", "datetime", "timedelta"):
        required = ("kind", "name", "dtype", "values") + (("freq",) if kind != "index" else ())
        if kind == "index" and isinstance(value.get("dtype"), str) and len(value["dtype"]) <= 128:
            parsed = pd.api.types.pandas_dtype(value["dtype"])
            if isinstance(parsed, pd.StringDtype):
                required += ("string_storage", "string_na")
                if value.get("string_storage") not in ("python", "pyarrow") or value.get(
                    "string_na"
                ) not in ("nan", "missing"):
                    fail("String index requires its complete storage/missing convention.")
        keys(value, required, "index descriptor")
        if (
            not isinstance(value["dtype"], str)
            or len(value["dtype"]) > 128
            or not isinstance(value["values"], (list, tuple))
        ):
            fail("Index descriptor schema disagrees.")
        if (
            kind != "index"
            and value["freq"] is not None
            and (not isinstance(value["freq"], str) or len(value["freq"]) > 64)
        ):
            fail("Index frequency requires a bounded string/null.")
        parsed = pd.api.types.pandas_dtype(value["dtype"])
        if kind == "datetime" and not pd.api.types.is_datetime64_any_dtype(parsed):
            fail("Datetime index requires a datetime/timezone dtype.")
        if kind == "timedelta" and not pd.api.types.is_timedelta64_dtype(parsed):
            fail("Timedelta index requires a timedelta dtype.")
        if kind == "index" and (
            pd.api.types.is_complex_dtype(parsed)
            or getattr(parsed, "itemsize", 8) > 8
            and not pd.api.types.is_object_dtype(parsed)
        ):
            fail("Index numeric widths/roles exceed the lossless supported domain.")
        for label in value["values"]:
            label_header(label)
        count = len(value["values"])
    else:
        fail("Unsupported complete row-index kind.")
    if (count != n) if exact else (count > n):
        fail("Index row envelope disagrees.")
    return count


def encode_index(index):
    if isinstance(index, pd.MultiIndex):
        return dict(
            kind="multi",
            names=[_label(n) for n in index.names],
            levels=[encode_index(level) for level in index.levels],
            codes=[code.tolist() for code in index.codes],
            sortorder=index.sortorder,
        )
    if isinstance(index, pd.CategoricalIndex):
        return dict(
            kind="categorical",
            name=_label(index.name),
            categories=encode_index(index.categories),
            ordered=bool(index.ordered),
            codes=index.codes.tolist(),
        )
    out = _encode_index(index)
    if out["kind"] == "index" and isinstance(index.dtype, pd.StringDtype):
        out.update(
            string_storage=index.dtype.storage,
            string_na="missing" if index.dtype.na_value is pd.NA else "nan",
        )
    return out


def decode_index(value):
    if value["kind"] == "multi":
        return pd.MultiIndex(
            levels=[decode_index(level) for level in value["levels"]],
            codes=[list(code) for code in value["codes"]],
            names=[_unlabel(n) for n in value["names"]],
            sortorder=value["sortorder"],
            verify_integrity=True,
        )
    if value["kind"] == "categorical":
        return pd.CategoricalIndex(
            pd.Categorical.from_codes(
                list(value["codes"]),
                categories=decode_index(value["categories"]),
                ordered=value["ordered"],
            ),
            name=_unlabel(value["name"]),
        )
    if value["kind"] == "index" and "string_storage" in value:
        parsed = (
            pd.StringDtype(storage=value["string_storage"])
            if value["string_na"] == "missing"
            else pd.StringDtype(storage=value["string_storage"], na_value=float("nan"))
        )
        return pd.Index(
            [_unlabel(label) for label in value["values"]],
            dtype=parsed,
            name=_unlabel(value["name"]),
            tupleize_cols=False,
        )
    return _decode_index(value)


def raw_index(index):
    """Charge the actual compact descriptor geometry before label/code copies."""
    if len(index) > MAX_ROWS:
        fail("Index levels/categories exceed 4096 entries.", "unsupported_index")
    size = 0
    pending = [iter(index.names)]
    if isinstance(index, pd.MultiIndex):
        size += 24 * len(index) * index.nlevels
        if size > MAX_INDEX:
            fail("Multi-index code descriptor exceeds1MiB.", "state_limit")
        for level in index.levels:
            raw_index(level)
    elif isinstance(index, pd.CategoricalIndex):
        size += 24 * len(index)
        raw_index(index.categories)
    elif not isinstance(index, pd.RangeIndex):
        pending.append(iter(index))
    while pending:
        try:
            cell = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        if isinstance(cell, tuple):
            if len(cell) > 16:
                fail("Index tuple exceeds 16 components.")
            pending.append(iter(cell))
        else:
            size += 96 + 12 * len(cell) if isinstance(cell, str) else 128
        if size > MAX_INDEX:
            fail("Row-index escaped descriptor exceeds 1 MiB.", "state_limit")
    if size > MAX_INDEX:
        fail("Row-index escaped descriptor exceeds 1 MiB.", "state_limit")
    plan_workspace("interval Weibull row identity", {"escaped descriptor and copies": 8 * size})
    state = encode_index(index)
    measure(state, maximum=MAX_INDEX)
    index_header(state, len(index))
    return state


def numeric_cell(value, text, label, *, nullable=False):
    if value is None and nullable:
        return
    if type(value) is int and abs(value) > 2**53:
        fail("Integer source cell cannot be converted losslessly to float64.")
    value = real(value, label)
    if abs(value) > 1e140:
        fail("Source magnitude exceeds the admitted float64 domain; rescale explicitly.")
    parsed = dtype(text)
    if is_integer_dtype(parsed):
        signed = not text.lower().startswith("u")
        low, high = (
            (-(2 ** (8 * parsed.itemsize - 1)), 2 ** (8 * parsed.itemsize - 1) - 1)
            if signed
            else (0, 2 ** (8 * parsed.itemsize) - 1)
        )
        if value != int(value) or abs(value) > 2**53 or not low <= value <= high:
            fail("Integer source cell exceeds its lossless dtype/range.")
    elif parsed.itemsize < 8:
        code = "e" if parsed.itemsize == 2 else "f"
        try:
            if struct.unpack(code, struct.pack(code, value))[0] != value:
                fail("Source cell is not exactly representable in its declared float dtype.")
        except (OverflowError, struct.error):
            fail("Source cell exceeds its declared float dtype.")


def source_header(value, *, upper=None):
    keys(value, ("columns", "dtypes", "values", "index", "positions", "upper_encoding"), "source")
    cols = names(value["columns"], empty=upper is None)
    if not isinstance(value["values"], (list, tuple)):
        fail("Source requires a complete resident rectangular matrix.")
    n = integer(len(value["values"]), "source rows", 1, MAX_ROWS)
    if not isinstance(value["dtypes"], (list, tuple)) or len(value["dtypes"]) != len(cols):
        fail("Complete source dtype count disagrees.")
    if upper is not None and upper not in cols:
        fail("Upper endpoint column is absent.")
    uj = cols.index(upper) if upper is not None else None
    codes = value["upper_encoding"]
    if not isinstance(codes, (tuple, list)) or len(codes) != (n if upper else 0):
        fail("Original upper-endpoint coding dimensions disagree.")
    for text in value["dtypes"]:
        if text != "object":
            dtype(text)
    for j, text in enumerate(value["dtypes"]):
        if text == "object" and j != uj:
            fail("Only the upper endpoint can use object dtype for explicit right-censor nulls.")
    for i, row in enumerate(value["values"]):
        if not isinstance(row, (list, tuple)) or len(row) != len(cols):
            fail("Source row dimensions disagree.")
        for j, cell in enumerate(row):
            if j == uj:
                if codes[i] not in ("finite", "none", "positive_infinity"):
                    fail("Upper endpoint coding requires finite/none/positive_infinity.")
                if codes[i] != "finite":
                    if cell is not None or codes[i] == "none" and value["dtypes"][j] != "object":
                        fail("Right endpoint coding disagrees with original dtype/cell.")
                    continue
                if cell is None:
                    fail("Finite upper endpoint cannot be null.")
            if value["dtypes"][j] == "object":
                real(cell, "upper endpoint")
            else:
                numeric_cell(cell, value["dtypes"][j], "source cell")
    positions = value["positions"]
    if (
        not isinstance(positions, (list, tuple))
        or len(positions) != n
        or any(type(v) is not int or v != i for i, v in enumerate(positions))
    ):
        fail("Every original physical row requires its exact integer position.")
    measure(value["index"], maximum=MAX_INDEX)
    index_header(value["index"], n)
    encoded, resident = measure(value["index"], maximum=MAX_INDEX)
    plan_workspace(
        "Weibull complete index reconstruction", {"descriptor copies": 8 * (encoded + resident)}
    )
    restored = decode_index(value["index"])
    equal(encode_index(restored), value["index"], "complete index", rtol=0)
    return cols, n


def source(data, columns, *, upper=None):
    """Plan/scan resident cells and row identity before pandas/tensor allocations."""
    cols = names(columns, empty=upper is None)
    if isinstance(data, pd.DataFrame):
        if data.columns.has_duplicates or any(col not in data for col in cols):
            fail("Source columns must exist uniquely.", "invalid_argument")
        n, index = len(data), data.index
        arrays = [data[col] for col in cols]
        texts = [str(array.dtype) for array in arrays]
    elif isinstance(data, Mapping):
        if any(col not in data for col in cols):
            fail("Selected source columns are absent.", "invalid_argument")
        arrays = [data[col] for col in cols]
        if not arrays:
            fail("An intercept-only query still requires a DataFrame carrying row identity.")
        if any(not isinstance(array, (pd.Series, list, tuple)) for array in arrays):
            fail("Mappings require resident Series/list/tuple columns; no Dataset collection.")
        n = len(arrays[0])
        if any(len(array) != n for array in arrays):
            fail("Mapping columns have unequal physical lengths.")
        series = [array for array in arrays if isinstance(array, pd.Series)]
        if series:
            if len(series) != len(arrays) or any(
                not array.index.equals(series[0].index) for array in series
            ):
                fail("Mapping Series require one identical complete index.")
            index = series[0].index
        else:
            index = pd.RangeIndex(n)
        texts = [
            str(array.dtype)
            if isinstance(array, pd.Series)
            else "object"
            if col == upper and any(v is None for v in array)
            else "int64"
            if all(type(v) is int for v in array)
            else "float64"
            for col, array in zip(cols, arrays)
        ]
    else:
        fail(
            "Supply a resident DataFrame or aligned column mapping; Dataset/device collection is unsupported."
        )
    integer(n, "source rows", 1, MAX_ROWS)
    plan_workspace(
        "Weibull before source copies",
        {"source/pandas/metadata buffers": 512 * n * max(1, len(cols))},
    )
    for j, text in enumerate(texts):
        if text == "object" and cols[j] == upper:
            continue
        dtype(text)
    index_state = raw_index(index)
    rows, coding = [], []
    for i in range(n):
        row = []
        for j, array in enumerate(arrays):
            cell = array.iloc[i] if isinstance(array, pd.Series) else array[i]
            if type(cell).__module__.startswith("numpy") and hasattr(cell, "item"):
                cell = cell.item()
            if cols[j] == upper and (
                cell is None or type(cell) in (float, int) and cell == math.inf
            ):
                coding.append("none" if cell is None else "positive_infinity")
                row.append(None)
            else:
                if cols[j] == upper:
                    coding.append("finite")
                if texts[j] == "object":
                    real(cell, "upper endpoint")
                else:
                    numeric_cell(cell, texts[j], "source cell")
                row.append(cell)
        rows.append(row)
    value = dict(
        columns=cols,
        dtypes=texts,
        values=rows,
        index=index_state,
        positions=list(range(n)),
        upper_encoding=coding,
    )
    source_header(value, upper=upper)
    return value


def frame(value, *, upper=None):
    cols, n = source_header(value, upper=upper)
    plan_workspace(
        "Weibull source frame", {"complete frame and row buffers": 512 * n * max(1, len(cols))}
    )
    index = decode_index(value["index"])
    out = {}
    for j, col in enumerate(cols):
        cells = [row[j] for row in value["values"]]
        if col == upper:
            cells = [
                math.inf if code == "positive_infinity" else cell
                for cell, code in zip(cells, value["upper_encoding"])
            ]
        out[col] = pd.Series(cells, index=index, dtype=value["dtypes"][j])
    return pd.DataFrame(out, index=index)
