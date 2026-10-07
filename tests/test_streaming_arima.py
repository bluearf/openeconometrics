import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.dataset import Dataset
from openecon.econometrics.streaming_arima import fit_streaming_arima
from openecon.econometrics.ordered_replay import OrderedReplay


@pytest.fixture
def data():
    rng = np.random.default_rng(519)
    x, innovation = rng.normal(size=(2, 419))
    error = np.zeros(len(x))
    for t in range(1, len(x)):
        error[t] = .39*error[t-1]+innovation[t]+.13*innovation[t-1]
    return pd.DataFrame({"y": 2.4+.8*x+error, "x": x, "t": np.arange(len(x))})


@pytest.mark.parametrize("options", [
    {"order": (0, 0, 0), "covariance": "opg"},
    {"order": (1, 0, 0), "covariance": "robust"},
    {"order": (0, 1, 1), "covariance": "nonrobust", "method": "css"},
    {"order": (1, 1, 0), "covariance": "opg", "constant": False},
    {"order": (1, 0, 0), "seasonal": (0, 1, 0), "period": 4, "covariance": "nonrobust"},
])
def test_full_fit_covariance_diagnostics_and_forecast(data, options, monkeypatch):
    expected = oe.arima(data=data, y="y", x=["x"], time="t", **options)
    original = OrderedReplay.__init__
    def bounded(self, spec, source):
        original(self, spec, source)
        self.rows = min(self.rows, 73)
    monkeypatch.setattr(OrderedReplay, "__init__", bounded)
    source = Dataset.from_batches(lambda: (data.iloc[start:start+47].copy() for start in range(0, len(data), 47)), list(data), row_count=len(data))
    actual = fit_streaming_arima(expected.spec, source)
    assert actual.nobs == expected.nobs
    assert actual.nobs_original == expected.nobs_original
    assert actual.dropped_rows == expected.dropped_rows
    np.testing.assert_allclose([row.estimate for row in actual.coefficients], [row.estimate for row in expected.coefficients], rtol=2e-6, atol=3e-7)
    np.testing.assert_allclose(actual.covariance_matrix, expected.covariance_matrix, rtol=2e-5, atol=3e-7)
    for name in ("log_likelihood", "aic", "bic", "sigma"):
        assert actual.metrics[name] == pytest.approx(expected.metrics[name], rel=2e-7, abs=3e-7)
    for name in ("model", "ljung_box", "jarque_bera", "arch_lm"):
        assert actual.tests[name]["statistic"] == pytest.approx(expected.tests[name]["statistic"], rel=3e-6, abs=3e-6)
    future = pd.DataFrame({"x": [.1, -.3, .7, .2, -.1]})
    np.testing.assert_allclose(oe.forecast(actual, 5, exog=future)[["forecast", "std_error"]], oe.forecast(expected, 5, exog=future)[["forecast", "std_error"]], rtol=3e-6, atol=5e-7)
    assert len(actual.predictions) <= 400
    assert actual.sample_positions == []
    assert actual.provenance["score_factor_rows_are_observations"] is False


def test_category_reference_before_end_missing_filter(data):
    data = data.copy()
    data["c"] = np.where(data.t % 2, "z", "y")
    data.loc[0, ["y", "c"]] = [np.nan, "a"]
    expected = oe.arima(data=data, y="y", x=["x", "c"], categorical=["c"], time="t", order=(1, 0, 0), missing="drop", covariance="nonrobust")
    actual = fit_streaming_arima(expected.spec, Dataset.from_batches(lambda: (data.iloc[start:start+71].copy() for start in range(0, len(data), 71)), list(data), row_count=len(data)))
    assert [row.term for row in actual.coefficients] == [row.term for row in expected.coefficients]
    assert actual.provenance["categorical_encoding"]["c"]["reference"] == "a"
    np.testing.assert_allclose([row.estimate for row in actual.coefficients], [row.estimate for row in expected.coefficients], rtol=3e-6, atol=5e-7)
    np.testing.assert_allclose(actual.covariance_matrix, expected.covariance_matrix, rtol=3e-5, atol=8e-7)
