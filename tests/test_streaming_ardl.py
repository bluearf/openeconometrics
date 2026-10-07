"""ARDL bounded lag-grid geometry and actual-period covariance contracts."""
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.dataset import Dataset
from openecon.econometrics.streaming_ardl import fit_streaming_ardl
from test_streaming_var import numeric_tree, numerical_blocks


@pytest.fixture(scope="module")
def data():
    g = torch.Generator().manual_seed(875)
    n = 707
    x = torch.randn(n, generator=g, dtype=torch.float64).cumsum(0)
    z = torch.randn(n, generator=g, dtype=torch.float64)
    noise = torch.randn(n, generator=g, dtype=torch.float64)
    y = torch.zeros(n, dtype=torch.float64)
    for i in range(2, n):
        y[i] = .5*y[i-1]-.1*y[i-2]+.25*x[i]+.3*z[i]+noise[i]
    frame = pd.DataFrame({"y": y.numpy(), "x": x.numpy(), "z": z.numpy(), "t": range(n)})
    return frame


def source(frame, block):
    return Dataset.from_batches(lambda: (frame.iloc[i:i+block] for i in range(0, len(frame), block)),
                                list(frame.columns), row_count=len(frame))


@pytest.mark.parametrize("block", [2, 603])
@pytest.mark.parametrize("options", [{"lags": [2, 1]}, {"lags": [2, 1], "ec": True},
                                    {"lags": [2, 1], "covariance": "robust"},
                                    {"lags": [2, 1], "covariance": "HC2"},
                                    {"lags": [2, 1], "covariance": "HC3"},
                                    {"lags": [0, 2], "trend": "none"},
                                    {"maxlags": [4, 3], "ic": "bic"},
                                    {"maxlags": [4, 3], "trend": "trend", "restricted": True},
                                    {"maxlags": [3, 2], "ec": True, "exog": ["z"]}])
def test_full_ardl_contract(data, block, options, monkeypatch):
    dense = oe.ardl(data=data, y="y", x=["x"], time="t", **options)
    numerical_blocks(monkeypatch, block)
    streamed = fit_streaming_ardl(dense.spec, source(data, block))
    assert dense.nobs == streamed.nobs and dense.dropped_rows == streamed.dropped_rows
    assert [c.term for c in dense.coefficients] == [c.term for c in streamed.coefficients]
    numeric_tree([c.estimate for c in dense.coefficients], [c.estimate for c in streamed.coefficients], 3e-8)
    numeric_tree(dense.covariance_matrix, streamed.covariance_matrix, 3e-8)
    numeric_tree(dense.metrics, streamed.metrics, 3e-8)
    numeric_tree(dense.extra, streamed.extra, 3e-8)
    assert set(dense.tests) == set(streamed.tests)
    numeric_tree(dense.tests, streamed.tests, 3e-8)
    for prediction in streamed.predictions:
        assert prediction["row"] >= len(data)-streamed.nobs
    assert streamed.provenance["sample_position_count"] == streamed.nobs


def test_actual_parquet_and_missing_endpoints(data, tmp_path):
    frame = data.copy()
    frame.loc[[0, len(frame)-1], "y"] = float("nan")
    frame.t = pd.date_range("2020-01-01", periods=len(frame), freq="h", tz="Europe/Istanbul")
    frame = frame.sample(frac=1, random_state=18).reset_index(drop=True)
    path = tmp_path/"ardl.parquet"
    frame.to_parquet(path, index=False, row_group_size=13)
    dense = oe.ardl(data=frame, y="y", x=["x"], time="t", missing="drop", lags=[2, 1], covariance="HC3")
    result = fit_streaming_ardl(dense.spec, oe.scan(path))
    numeric_tree(dense.metrics, result.metrics, 3e-8)
    numeric_tree(dense.covariance_matrix, result.covariance_matrix, 3e-8)
    expected = frame.dropna().sort_values("t").index.tolist()[2:402]
    assert [p["row"] for p in result.predictions] == expected
