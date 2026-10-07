"""Measured GEE work includes full passes before short-panel re-preparation."""

import numpy as np
import pandas as pd
import pytest

from openecon.dataset import Dataset
from openecon.econometrics.mixed.gee import fit_xtgee
from openecon.econometrics.replay_sample import ReplaySample
from openecon.econometrics.streaming_gee import fit_streaming_gee
from openecon.models import ModelSpec


@pytest.mark.parametrize("corr", ["ar1", "stationary", "nonstationary"])
def test_short_panel_selection_preserves_measured_work_and_numerical_fit(corr, monkeypatch):
    rng = np.random.default_rng(77317)
    group = np.concatenate((np.arange(72), np.repeat(np.arange(72, 78), 2)))
    time = np.concatenate((np.zeros(72, dtype=int), np.tile(np.arange(2), 6)))
    x = rng.normal(size=len(group))
    frame = pd.DataFrame({"g": group, "t": time, "x": x,
                          "y": .4 + .6 * x + rng.normal(size=len(group))})
    spec = ModelSpec(estimator="xtgee", outcome="y", predictors=["x"], panel="g", time="t",
                     covariance="robust", options={"corr": corr, "corr_order": 1})
    measured = {"source_passes": 0, "numeric_peak": 0}

    def reader():
        measured["source_passes"] += 1
        for start in range(0, len(frame), 16):
            yield frame.iloc[start:start + 16].copy()

    original = ReplaySample.batches

    def batches(self, *args, **kwargs):
        for batch in original(self, *args, **kwargs):
            measured["numeric_peak"] = max(measured["numeric_peak"], len(batch.frame))
            yield batch

    monkeypatch.setattr(ReplaySample, "batches", batches)
    source = Dataset.from_batches(reader, frame.columns.tolist(), row_count=len(frame))
    result = fit_streaming_gee(spec, source, batch_rows=16)
    native = fit_xtgee(spec, frame)
    record = result.provenance["streaming"]
    assert measured["numeric_peak"] == 16
    assert record["actual_numeric_peak_rows"] == measured["numeric_peak"]
    assert record["maximum_batch_rows"] == measured["numeric_peak"]
    assert record["passes"] == measured["source_passes"]
    assert result.nobs == 12 and result.dropped_rows == 72
    np.testing.assert_allclose([c.estimate for c in result.coefficients],
                               [c.estimate for c in native.coefficients], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(result.covariance_matrix, native.covariance_matrix, rtol=1e-8, atol=1e-10)
