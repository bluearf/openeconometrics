"""Complete Johansen/VEC contracts use global rows, not factor-row counts."""
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.streaming_vec import fit_streaming_vec
from test_streaming_var import numeric_tree, numerical_blocks


@pytest.fixture(scope="module")
def data():
    g = torch.Generator().manual_seed(733)
    walk = torch.randn(887, generator=g, dtype=torch.float64).cumsum(0)
    levels = torch.stack((walk+torch.randn(887, generator=g, dtype=torch.float64),
                          .7*walk+torch.randn(887, generator=g, dtype=torch.float64),
                          -.3*walk+torch.randn(887, generator=g, dtype=torch.float64)), dim=1)
    frame = pd.DataFrame(levels.numpy(), columns=["a", "b", "c"])
    frame["t"] = range(len(frame))
    return frame


def source(frame, block):
    return Dataset.from_batches(lambda: (frame.iloc[i:i+block] for i in range(0, len(frame), block)),
                                list(frame.columns), row_count=len(frame))


@pytest.mark.parametrize("trend", ["none", "rconstant", "constant", "rtrend", "trend"])
@pytest.mark.parametrize("block", [3, 719])
@pytest.mark.parametrize("rank,lags", [(1, 1), (2, 3)])
def test_global_vec_contract(data, trend, block, rank, lags, monkeypatch):
    dense = oe.vec(data=data, y=["a", "b", "c"], time="t", trend=trend, rank=rank, lags=lags)
    numerical_blocks(monkeypatch, block)
    result = fit_streaming_vec(dense.spec, source(data, block))
    assert result.nobs == dense.nobs and result.dropped_rows == dense.dropped_rows
    assert [c.term for c in result.coefficients] == [c.term for c in dense.coefficients]
    assert [c.equation for c in result.coefficients] == [c.equation for c in dense.coefficients]
    numeric_tree([c.estimate for c in dense.coefficients], [c.estimate for c in result.coefficients], 3e-8)
    numeric_tree(dense.covariance_matrix, result.covariance_matrix, 3e-8)
    numeric_tree(dense.metrics, result.metrics, 3e-8)
    assert set(dense.tests) == set(result.tests)
    numeric_tree(dense.tests, result.tests, 3e-8)
    for key in ("beta", "beta_matrix", "alpha", "eigenvalues", "rank_test", "ce_constant", "ce_trend",
                "pi", "omega", "equations", "stability", "var_representation", "normality", "forecast"):
        numeric_tree(dense.extra[key], result.extra[key], 3e-8)
    forecasts = [oe.vec_forecast(model, 5) for model in (dense, result)]
    numeric_tree(forecasts[0], forecasts[1], 3e-8)
    assert not result.sample_positions
    assert result.provenance["sample_position_count"] == result.nobs
    assert result.provenance["factor_rows_are_observations"] is False


@pytest.mark.parametrize("trend", ["rconstant", "constant", "rtrend", "trend"])
def test_large_level_units_and_sorted_dates(data, trend):
    frame = data.copy()
    frame.a = frame.a+2**29
    frame.b = frame.b+2**27
    frame.c = frame.c+2**28
    frame.t = pd.date_range("2020-01-01", periods=len(frame), freq="h", tz="Europe/Istanbul")
    frame.loc[[0, len(frame)-1], "a"] = float("nan")
    frame = frame.sample(frac=1, random_state=45).reset_index(drop=True)
    dense = oe.vec(data=frame, y=["a", "b", "c"], time="t", missing="drop", lags=2, trend=trend)
    result = fit_streaming_vec(dense.spec, source(frame, 11))
    numeric_tree([c.estimate for c in dense.coefficients], [c.estimate for c in result.coefficients], 3e-6)
    numeric_tree(dense.covariance_matrix, result.covariance_matrix, 3e-6)
    numeric_tree(dense.metrics, result.metrics, 3e-6)
    numeric_tree(dense.tests, result.tests, 3e-6)
    expected = frame.dropna().sort_values("t").index.tolist()[2:402]
    assert [p["row"] for p in result.predictions] == expected


def test_actual_file_cleanup_and_immutable_source(data, tmp_path):
    path = tmp_path/"vec.parquet"
    data.to_parquet(path, index=False, row_group_size=17)
    dense = oe.vec(data=data, y=["a", "b", "c"], time="t", lags=2)
    result = fit_streaming_vec(dense.spec, oe.scan(path))
    numeric_tree(dense.metrics, result.metrics, 3e-8)
    calls = 0
    def factory():
        nonlocal calls
        calls += 1
        changed = data.copy()
        if calls > 1:
            changed.loc[900 % len(changed), "a"] += .01
        yield changed
    with pytest.raises(AnalysisError, match="changed"):
        fit_streaming_vec(dense.spec, Dataset.from_batches(factory, list(data.columns), row_count=len(data)))
