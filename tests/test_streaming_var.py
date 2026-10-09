"""Replay VAR preserves system inference, diagnostics, IRFs and forecasts."""
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.dataset import Dataset
from openecon.econometrics.streaming_var import fit_streaming_var


def numerical_blocks(monkeypatch, maximum):
    """Shrink the sorted snapshot reader too, not only the original source."""
    from openecon.econometrics.ordered_replay import OrderedReplay
    original = OrderedReplay.__init__
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.rows = min(self.rows, maximum)
    monkeypatch.setattr(OrderedReplay, "__init__", initialize)


@pytest.fixture(scope="module")
def data():
    gen = torch.Generator().manual_seed(932)
    noise = torch.randn((691, 2), generator=gen, dtype=torch.float64)
    levels = torch.zeros_like(noise)
    for i in range(1, len(levels)):
        levels[i] = levels[i-1]@torch.tensor([[.5, .1], [-.2, .4]], dtype=torch.float64)+noise[i]
    frame = pd.DataFrame(levels.numpy(), columns=["a", "b"])
    frame["x"] = torch.randn(691, generator=gen, dtype=torch.float64).numpy()
    frame["t"] = torch.arange(691).numpy()
    frame["cat"] = ["a", "b", "c"]*230+["a"]
    return frame


def source(frame, block):
    return Dataset.from_batches(lambda: (frame.iloc[i:i+block] for i in range(0, len(frame), block)),
                                list(frame.columns), row_count=len(frame))


def numeric_tree(a, b, tolerance=2e-8):
    if isinstance(a, dict):
        for key, value in a.items():
            if key in b:
                numeric_tree(value, b[key], tolerance)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for one, other in zip(a, b, strict=True):
            numeric_tree(one, other, tolerance)
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        assert b == pytest.approx(a, rel=tolerance, abs=tolerance)
    elif a is None:
        assert b is None


@pytest.mark.parametrize("block", [17, 501])
@pytest.mark.parametrize("options", [{}, {"covariance": "robust"}, {"small": True},
                                    {"dfk": True}, {"constant": False}, {"trend": True},
                                    {"lags": 1}, {"maxlag": 4}])
def test_full_model_contract(data, block, options, monkeypatch):
    kwargs = dict(y=["a", "b"], x=["x"], time="t", lags=2)
    kwargs.update(options)
    dense = oe.var(data=data, **kwargs)
    numerical_blocks(monkeypatch, block)
    streamed = fit_streaming_var(dense.spec, source(data, block))
    assert dense.nobs == streamed.nobs and dense.dropped_rows == streamed.dropped_rows
    assert [c.term for c in dense.coefficients] == [c.term for c in streamed.coefficients]
    numeric_tree([c.estimate for c in dense.coefficients], [c.estimate for c in streamed.coefficients])
    numeric_tree(dense.covariance_matrix, streamed.covariance_matrix)
    numeric_tree(dense.metrics, streamed.metrics)
    assert set(dense.tests) == set(streamed.tests)
    numeric_tree(dense.tests, streamed.tests)
    for name in ("sigma", "sigma_ml", "normality", "equations", "granger", "lag_exclusion", "irf", "forecast", "lag_order_selection"):
        numeric_tree(dense.extra.get(name), streamed.extra.get(name))
    forecasts = (oe.var_forecast(result, 3, exog=pd.DataFrame({"x": [0., .1, .2]})) for result in (dense, streamed))
    first, second = forecasts
    numeric_tree(first.model_dump() if hasattr(first, "model_dump") else first,
                 second.model_dump() if hasattr(second, "model_dump") else second)


def test_sorted_missing_positions_dates_and_categories(data):
    frame = data.copy()
    frame.loc[[0, 690], "a"] = float("nan")
    frame.t = pd.date_range("2020-01-01", periods=len(frame), freq="MS")
    frame = frame.sample(frac=1, random_state=42).reset_index(drop=True)
    kwargs = dict(y=["a", "b"], x=["x", "cat"], categorical=["cat"], time="t", missing="drop", lags=2)
    dense = oe.var(data=frame, **kwargs)
    streamed = fit_streaming_var(dense.spec, source(frame, 113))
    numeric_tree(dense.covariance_matrix, streamed.covariance_matrix)
    numeric_tree(dense.tests, streamed.tests)
    expected = frame.dropna().sort_values("t").index.tolist()[2:402]
    assert [row["row"] for row in streamed.predictions] == expected
    assert streamed.nobs_original == len(frame)


def test_physical_parquet_arrow_numeric_tail_survives_saved_forecast(data, tmp_path):
    from openecon.econometrics import registry
    from openecon.models import ResultBundle
    path = tmp_path / "var.parquet"
    data.to_parquet(path, index=False, row_group_size=127)
    specification = oe.var(data=data, y=["a", "b"], x=["x"], time="t", lags=2).spec
    expected = registry.load_entry(registry.get("var"))(specification, pd.read_parquet(path))
    actual = oe.fit(specification, data=oe.scan(path))
    actual = ResultBundle.model_validate_json(actual.model_dump_json())
    assert actual.inference["reference"] == expected.inference["reference"]
    future = pd.DataFrame({"x": [.1, .2, -.4]})
    one, two = oe.forecast(expected, 3, exog=future), oe.forecast(actual, 3, exog=future)
    pd.testing.assert_frame_equal(one, two, rtol=3e-7, atol=3e-7)
    numeric_tree(expected.covariance_matrix, actual.covariance_matrix, 3e-7)
    numeric_tree(expected.extra["forecast"]["last_values"], actual.extra["forecast"]["last_values"], 3e-7)


def test_strict_global_rank_and_time_guards(data):
    frame = data.assign(b=2*data.a)
    valid_spec = oe.var(data=data, y=["a", "b"], time="t").spec
    with pytest.raises(oe.AnalysisError) as caught:
        fit_streaming_var(valid_spec, source(frame, 31))
    assert caught.value.code in {"collinear_system", "singular_sigma"}
    frame = data.copy()
    frame.loc[63, "t"] = 61
    with pytest.raises(oe.AnalysisError) as caught:
        fit_streaming_var(valid_spec, source(frame, 31))
    assert caught.value.code == "duplicate_time"


def test_exogenous_omissions_and_original_source_reverified(data):
    frame = data.assign(alias=data.x)
    kwargs = dict(y=["a", "b"], x=["x", "alias"], lags=2)
    dense = oe.var(data=frame, **kwargs)
    streamed = fit_streaming_var(dense.spec, source(frame, 137))
    numeric_tree(dense.covariance_matrix, streamed.covariance_matrix)
    assert any("alias" in value for value in streamed.warnings)
    calls = 0
    def changed():
        nonlocal calls
        calls += 1
        yield frame.assign(a=frame.a+calls)
    with pytest.raises(oe.AnalysisError) as caught:
        fit_streaming_var(dense.spec, Dataset.from_batches(changed, list(frame.columns)))
    assert caught.value.code == "source_changed"
