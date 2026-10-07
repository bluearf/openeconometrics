"""Small multivariate moments and bounded, integrity-checked score sources."""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import pandas as pd

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.stats.replay import Replay, numeric_moments
from . import common as c


def moments(data: Dataset, names: list[str], missing: str, *, minimum: int = 2):
    sample = Replay(data, names, names, missing, check_small=False)
    state = numeric_moments(sample, names)
    if state.n < minimum:
        raise AnalysisError("insufficient_observations", f"The analysis needs at least {minimum} complete observations.")
    flat = [name for name, ss, raw in zip(names, state.sscp.diagonal().tolist(),
             state.raw_ss.tolist(), strict=True) if not ss > c._CONSTANT * max(raw, 1e-300)]
    if flat:
        raise AnalysisError("constant_column", f"Column(s) {', '.join(flat)} do not vary in the sample.")
    return state, sample.attrs(), sample.dropped


def score_source(data: Dataset, names: list[str], columns: list[str],
                 callback: Callable[[pd.DataFrame], Any], *, procedure: str,
                 options: dict[str, Any] | None = None) -> Dataset:
    """Lazy score blocks preserve input row indexes, including missing rows.

    A complete-pass digest is captured on creation and checked on every replay.
    Consumers must exhaust a pass for its content-integrity check, as for any
    Dataset factory; exported/materialized results perform that complete pass.
    """
    sample = Replay(data, names, names, "drop", check_small=False)
    baseline = hashlib.sha256()
    rows = 0
    for raw in data.iter_batches(names, batch_rows=sample.rows):
        c.require_numeric(raw, names)
        complete = raw.loc[~raw.isna().any(axis=1)]
        for name in names:
            c.column(complete, name)
        baseline.update(pd.util.hash_pandas_object(raw, index=True).to_numpy(dtype="uint64").tobytes())
        rows += len(raw)
    if not rows:
        raise AnalysisError("empty_data", "The dataset contains no observations.")
    digest = baseline.hexdigest()

    def batches():
        observed = hashlib.sha256()
        for raw in data.iter_batches(names, batch_rows=sample.rows):
            observed.update(pd.util.hash_pandas_object(raw, index=True).to_numpy(dtype="uint64").tobytes())
            yield callback(raw)
        if observed.hexdigest() != digest:
            raise AnalysisError("source_changed", "The source changed while computing multivariate scores.")

    return Dataset.from_batches(batches, columns=columns, row_count=rows,
                                metadata={"procedure": procedure, "streaming": True,
                                          "source_content_sha256": digest, "precision": "float64",
                                          "batch_rows": sample.rows, "variables": names,
                                          "resource_plan": sample.plan.record(), **(options or {})})
