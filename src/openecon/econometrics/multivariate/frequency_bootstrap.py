"""Compressed literal-IID-unit bootstrap admission, counts and stable moments.

Integer frequencies are counts of independently observed units. They are not
analytic/probability weights or a claim that cloned observations are independent.
No repeated N-by-p measurement matrix is constructed by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_integer_dtype

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.summary_state import saved_summary, summary_state
from openecon.resources import plan_workspace

from . import common as c
from . import canon_uncertainty as cu
from . import pca_subspace as ps

MAX_ROWS = 10_000
MAX_UNITS = 100_000
MAX_VARIABLES = 16
MAX_PARAMETERS = 260
MAX_REPLICATIONS = 1_999
MAX_WORK = 250_000_000
MAX_EXPORT_BYTES = 32 * 1024**2
STATE_SCHEMA = "openecon.frequency-bootstrap.v1"
DRAW_PROTOCOL = "integer-copy-ranks-single-call-v1"


@dataclass(frozen=True)
class FrequencySample:
    values: torch.Tensor
    counts: torch.Tensor
    total: int
    names: list[str]
    weights: str
    replications: int
    confidence: float
    seed: int
    missing: str
    physical_rows: int
    positions: list[int]
    source_index_codes: list[str]
    attrs: dict
    accounting: pd.DataFrame


@dataclass(frozen=True)
class Moments:
    mean: torch.Tensor
    origin: torch.Tensor
    offset: torch.Tensor
    covariance: torch.Tensor
    sd: torch.Tensor
    centered: torch.Tensor
    counts: torch.Tensor
    positions: torch.Tensor
    total: int


def _frequency(value: Any) -> int | None:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.device.type != "cpu" or value.requires_grad \
                or value.layout != torch.strided:
            raise AnalysisError("invalid_weights", "Frequency cells must be plain CPU real scalars.")
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, bool) or isinstance(value, complex):
        raise AnalysisError("invalid_weights", "Boolean and complex frequencies are unsupported.")
    if isinstance(value, Integral):
        count = int(value)
    elif isinstance(value, Real):
        value = float(value)
        if math.isnan(value):
            return None
        if not math.isfinite(value) or not value.is_integer():
            raise AnalysisError("invalid_weights", "Frequencies must be finite exact nonnegative integers.")
        count = int(value)
    else:
        raise AnalysisError("invalid_weights", "Frequencies must be numeric integers, not strings or arbitrary scalar objects.")
    if count < 0 or count > 2**53:
        raise AnalysisError("invalid_weights", "Frequencies must be nonnegative integers up to exact float64 precision 2^53.")
    return count


def prepare(data: Any, names: list[str], *, weights: str, weight_type: str,
            replications: int, confidence: float, seed: int, missing: str,
            parameters: int, fit_work: int | None = None, minimum_units: int = 20,
            label: str = "frequency bootstrap") -> FrequencySample:
    """Plan before table/tensor copies, then admit a complete frequency sample.

    Invalid frequency values are refused globally, even in a row that would
    later be omitted for a missing measurement. Missing weights participate
    in the same listwise drop/raise rule as the measurements.
    """
    names = c.name_list(names, "columns")
    weights = c.check_name(weights, "weights")
    c.check_choice(weight_type, "weight_type", ("fweight",))
    if weights in names:
        raise AnalysisError("invalid_spec", "Frequency and measurement roles must be distinct.")
    replications = c.check_count(replications, "replications", minimum=19, maximum=MAX_REPLICATIONS)
    confidence = c.check_number(confidence, "confidence", minimum=0, maximum=1, exclusive=True)
    if confidence >= 1 or (replications + 1) * (1 - confidence) / 2 < 1 - 1e-12:
        raise AnalysisError("insufficient_replications", "Each percentile tail needs at least one expected bootstrap order statistic.")
    seed = c.check_count(seed, "seed", minimum=0, maximum=2**63 - 1)
    c.check_choice(missing, "missing", ("drop", "raise"))
    parameters = c.check_count(parameters, "parameters", maximum=MAX_PARAMETERS)
    minimum_units = c.check_count(minimum_units, "minimum_units", minimum=2)
    roles = [*names, weights]
    rows, p, q = cu._rows(data, roles), len(names), parameters
    if rows > MAX_ROWS or p > MAX_VARIABLES:
        raise AnalysisError("workspace_limit", "Frequency uncertainty supports at most10000 physical rows and16 variables.")
    if not rows:
        raise AnalysisError("empty_data", "The dataset contains no observations.")
    if any(len(name.encode()) > cu.MAX_LABEL_BYTES for name in roles):
        raise AnalysisError("resource_limit", "One declared column name exceeds the identity byte limit.")
    ps._measurement_admission(data, roles)
    fit_work = 64 * p**3 if fit_work is None else c.check_count(fit_work, "fit_work", minimum=0)
    # Count and tensor storage uses physical rows, with only one bounded vector
    # of virtual copy ranks at a time. Matrix/score/result buffers are planned
    # conservatively for all four families before any selected-source copy.
    plan = plan_workspace(label, {
        "source_selection_and_weighted_blocks": rows * (p + 1) * 192,
        "integer_copy_ranks_search_and_counts": MAX_UNITS * 24 + rows * 64,
        "full_count_draws": replications * rows * 16,
        "all_parameter_moment_and_diagnostic_draws": replications * (q + 8 * p * p + 8 * p + 16) * 32,
        "joint_covariance_and_quantile_buffers": q * q * 64 + replications * q * 32,
        "fixed_point_matrix_buffers": p * p * 1024,
        "bounded_identity_state": 4 * cu.MAX_IDENTITY_BYTES,
    }).record()
    cells = replications * (q + 8 * p * p + 8 * p + 16) + q * q + rows * p + 12 * p * p
    export_bytes = 32 * cells + 24 * replications * rows + 256 * rows + 4 * cu.MAX_IDENTITY_BYTES + 131072
    if export_bytes > MAX_EXPORT_BYTES:
        raise AnalysisError("resource_limit", "Complete frequency uncertainty state exceeds its32MiB portable domain.")
    selected = cu._selected(data, roles)
    raw = c.source(selected)
    c.require_numeric(raw, names)
    if any(is_bool_dtype(raw[name].dtype) for name in names):
        raise AnalysisError("invalid_data", "Boolean measurements are unsupported in frequency spectral uncertainty.")
    if is_bool_dtype(raw[weights].dtype):
        raise AnalysisError("invalid_weights", "Boolean frequencies are unsupported.")
    # Exact Python integer validation precedes int64/float64 count conversion;
    # it cannot truncate fractions or lose integers above2^53.
    frequencies = [_frequency(value) for value in raw[weights]]
    complete = [not bool(flag) and frequencies[i] is not None
                for i, flag in enumerate(raw[names].isna().any(axis=1))]
    dropped = len(complete) - sum(complete)
    if dropped and missing == "raise":
        raise AnalysisError("missing_values", "Measurement and frequency roles require one complete listwise sample.")
    positions = [i for i, flag in enumerate(complete) if flag]
    if not positions:
        raise AnalysisError("empty_sample", "No complete observations remain.")
    admitted = [frequencies[i] for i in positions]
    total = sum(admitted)
    if total > 2**53:
        raise AnalysisError("invalid_weights", "Frequency total exceeds exact float64 integer precision2^53.")
    if total > MAX_UNITS:
        raise AnalysisError("resource_limit", "Frequency bootstrap admits at most100000 literal IID units; this is a sampling/work boundary, not a point-fit statistical limit.")
    if total < minimum_units:
        raise AnalysisError("insufficient_observations", f"Frequency bootstrap needs at least{minimum_units} complete IID units.")
    sampling_work = replications * total * ((len(positions) - 1).bit_length() + 4)
    work = (replications + 1) * (rows * p * p + fit_work) + sampling_work + 2 * replications * q * q + cu.MAX_IDENTITY_NODES
    if work > MAX_WORK:
        raise AnalysisError("work_limit", "Count generation, all moment fits and joint inference exceed250million planned work units.")
    all_codes, index_names, column_codes, identity_bytes = cu._identities(raw.index, [True] * rows, roles)
    column_axis_names = [cu._identity(name, [0]) for name in raw.columns.names]
    identity_bytes += sum(len(code.encode()) for code in column_axis_names)
    if identity_bytes > cu.MAX_IDENTITY_BYTES:
        raise AnalysisError("resource_limit", "Complete source and column-axis identities exceed2MiB.")
    source_codes = [all_codes[i] for i in positions]
    sample = raw.iloc[positions]
    values = torch.tensor(sample.loc[:, names].to_numpy(dtype="float64").copy(), dtype=c.FLOAT)
    if not bool(torch.isfinite(values).all()):
        raise AnalysisError("invalid_data", "All admitted measurement values must be finite float64 numbers.")
    counts = torch.tensor(admitted, dtype=torch.int64)
    accounting = pd.DataFrame({
        "source_position": list(range(rows)), "index_code": all_codes,
        "complete": complete,
        "frequency_json": ["null" if value is None else str(value) for value in frequencies],
        "zero_complete_frequency": [bool(flag and frequencies[i] == 0) for i, flag in enumerate(complete)],
    })
    attrs = dict(weights=weights, weight_type="fweight", n_units=total, nobs=total,
                 n_physical=rows, physical_rows=rows, n_complete=len(positions),
                 n_positive_frequency=int((counts > 0).sum()), n_zero_weight=int((counts == 0).sum()),
                 n_missing=dropped, n_dropped=dropped, covariance_divisor=total - 1,
                 replications=replications, successful_replications=replications,
                 confidence=confidence, seed=seed, missing=missing,
                 index_names_json=index_names, source_columns_json=column_codes,
                 column_axis_names_json=column_axis_names, encoded_source_identity_bytes=identity_bytes,
                 moment_covariance_divisor=total - 1,
                 source_index_codec="openecon.postest.index.v1",
                 original_weights_source=weights, frequency_state_schema=STATE_SCHEMA,
                 draw_protocol=DRAW_PROTOCOL, torch_version=str(torch.__version__),
                 sampling_law="Multinomial(N,original_integer_frequencies/N)",
                 sampling_unit="literal independent replicated unit", physical_row_resampling=False,
                 expanded_measurements_materialized=False, unit_limit=MAX_UNITS,
                 physical_row_limit=MAX_ROWS, resource_plan=plan, estimated_work=work,
                 sampling_work=sampling_work, estimated_complete_export_bytes=export_bytes,
                 portable_export_limit_bytes=MAX_EXPORT_BYTES, joint_parameter_count=q,
                 fixed_dimension=True, fixed_frequency_sample=True,
                 zero_frequency_source_rows_preserved=True)
    return FrequencySample(values, counts, total, names, weights, replications,
                           confidence, seed, missing, rows, positions, source_codes, attrs, accounting)


def moments(values: torch.Tensor, counts: torch.Tensor) -> Moments:
    """Two-pass anchored weighted moments; zero counts precede subtraction."""
    if values.ndim != 2 or counts.ndim != 1 or len(values) != len(counts) \
            or counts.dtype != torch.int64 or values.dtype != c.FLOAT \
            or values.device.type != "cpu" or counts.device.type != "cpu" \
            or bool((counts < 0).any()):
        raise AnalysisError("invalid_state", "Weighted moment inputs need CPU float64 rows and nonnegative int64 counts.")
    total = int(counts.sum())
    if total < 2 or total > MAX_UNITS:
        raise AnalysisError("insufficient_observations", "Weighted moments need2..100000 literal units.")
    positions = torch.nonzero(counts > 0, as_tuple=False).flatten()
    x, positive_counts = values[positions], counts[positions]
    weights = positive_counts.to(dtype=c.FLOAT)
    origin = x[0].clone()
    shifted = x - origin
    offset = (shifted * weights[:, None]).sum(0) / total
    offset += ((shifted - offset) * weights[:, None]).sum(0) / total
    centered = shifted - offset
    covariance = centered.T @ (weights[:, None] * centered) / (total - 1)
    covariance = (covariance + covariance.T) / 2
    mean = origin + offset
    if not bool(torch.isfinite(mean).all() and torch.isfinite(covariance).all()) \
            or bool((covariance.diagonal() < 0).any()):
        raise AnalysisError("numerical_failure", "Weighted moments exceed finite float64 precision; rescale the measurements.")
    sd = covariance.diagonal().sqrt()
    return Moments(mean, origin, offset, covariance, sd, centered, positive_counts, positions, total)


def correlation(value: Moments) -> torch.Tensor:
    variance = value.covariance.diagonal()
    if bool((variance <= 0).any()):
        raise AnalysisError("zero_variance", "Every standardized variable must vary among positive-frequency units.")
    if bool((variance < torch.finfo(c.FLOAT).tiny).any()):
        raise AnalysisError("numerical_failure", "A weighted variance is subnormal before standardization; rescale measurements.")
    output = value.covariance / torch.outer(value.sd, value.sd)
    output = (output + output.T) / 2
    output.diagonal().fill_(1)
    if not bool(torch.isfinite(output).all()):
        raise AnalysisError("numerical_failure", "Weighted correlation exceeds finite float64 precision.")
    return output


def draw_counts(sample: FrequencySample, generator: torch.Generator) -> torch.Tensor:
    """Exactly sample virtual unit ranks; no rounded probability vector."""
    ranks = torch.randint(sample.total, (sample.total,), generator=generator, dtype=torch.int64, device="cpu")
    source_rows = torch.searchsorted(sample.counts.cumsum(0), ranks, right=True)
    return torch.bincount(source_rows, minlength=len(sample.counts))


def source_tables(sample: FrequencySample) -> dict[str, pd.DataFrame]:
    return {
        "sample": c.table(sample.values.numpy(), columns=sample.names),
        "source_frequencies": c.table({"frequency": sample.counts.tolist()}),
        "source_index": c.table({"source_position": sample.positions, "index_code": sample.source_index_codes}),
        "source_accounting": c.table(sample.accounting.copy()),
    }


def joint(point: torch.Tensor, draws: torch.Tensor, confidence: float):
    if draws.ndim != 2 or point.ndim != 1 or draws.shape[1] != len(point) \
            or len(draws) < 2 or not bool(torch.isfinite(draws).all() and torch.isfinite(point).all()):
        raise AnalysisError("numerical_failure", "Joint bootstrap inference needs complete finite planned vectors.")
    origin = draws[0]
    shifted = draws - origin
    centered, offset = c.centre(shifted)
    covariance = centered.T @ centered / (len(draws) - 1)
    covariance = (covariance + covariance.T) / 2
    diagonal = covariance.diagonal()
    changed = (shifted != 0).any(0)
    if not bool(torch.isfinite(covariance).all()) or bool((diagonal < 0).any()) \
            or bool((changed & (diagonal < torch.finfo(c.FLOAT).tiny)).any()):
        raise AnalysisError("numerical_failure", "A joint bootstrap variance overflows or underflows float64; rescale the functional.")
    se = diagonal.sqrt()
    bias = (origin - point) + offset
    tail = (1 - confidence) / 2
    quantiles = torch.quantile(draws, torch.tensor([tail, 1 - tail], dtype=c.FLOAT), dim=0)
    if not bool(torch.isfinite(bias).all() and torch.isfinite(quantiles).all()):
        raise AnalysisError("numerical_failure", "Joint bias or percentiles exceed finite float64 precision.")
    return covariance, se, bias, quantiles[0], quantiles[1]


def _digest(result: TableSet) -> str:
    payload = {"title": result.title,
               "attrs": {key: value for key, value in result.attrs.items()
                         if key not in ("state_content_sha256", "state_sha256")},
               "tables": {name: {"index": list(frame.index), "columns": list(frame.columns),
                                 "index_names": list(frame.index.names), "column_names": list(frame.columns.names),
                                 "data": frame.to_numpy().tolist()}
                          for name, frame in result.items()}}
    encoded = json.dumps(cu._json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_EXPORT_BYTES:
        raise AnalysisError("resource_limit", "The full frequency-bootstrap portable state exceeds32MiB.")
    return hashlib.sha256(encoded).hexdigest()


def seal(result: TableSet) -> TableSet:
    result.attrs.pop("state_content_sha256", None)
    result.attrs.pop("state_sha256", None)
    result.attrs["frequency_state_schema"] = STATE_SCHEMA
    saved_summary(result)
    digest = _digest(result)
    result.attrs["state_content_sha256"] = digest
    result.attrs["state_sha256"] = digest
    summary_state(result)  # actual final complete export, including hash attrs
    return result


def validate_saved(result: TableSet) -> None:
    """Check full integrity and count structure, without scientific refitting.

    Neither a checksum nor these structural checks certify semantic replay.
    Numerical reconstruction requires independently refitting the saved source
    and all saved count rows under the declared functional/identification.
    """
    try:
        attrs = result.attrs
        count_tables = result["replicate_counts"], result["source_frequencies"]
        n, b, r = attrs["n_units"], attrs["replications"], len(result["sample"])
        valid = isinstance(result, TableSet) and attrs["frequency_state_schema"] == STATE_SCHEMA \
            and type(n) is int and 2 <= n <= MAX_UNITS and type(b) is int and 19 <= b <= MAX_REPLICATIONS \
            and 0 < r <= MAX_ROWS and count_tables[0].shape == (b, r) \
            and count_tables[1].shape == (r, 1)
        if not valid or any(any(not is_integer_dtype(dtype) or is_bool_dtype(dtype) for dtype in frame.dtypes)
                            for frame in count_tables):
            raise ValueError("Invalid count structure")
        source = count_tables[1].iloc[:, 0]
        draws = count_tables[0]
        if bool(((source < 0) | (source > n)).any()) \
                or bool(((draws < 0) | (draws > n)).any().any()) \
                or int(source.sum()) != n \
                or not bool((draws.sum(axis=1) == n).all()) \
                or bool((draws.loc[:, source.to_numpy() == 0] != 0).any().any()):
            raise ValueError("Invalid draw law support")
        digest = _digest(result)
        if attrs.get("state_content_sha256") != digest or attrs.get("state_sha256") != digest:
            raise ValueError("Invalid integrity")
    except (KeyError, TypeError, ValueError, AttributeError, AnalysisError, OverflowError):
        raise AnalysisError("invalid_state", "Saved frequency bootstrap state fails full integrity/count-structure checks.") from None
