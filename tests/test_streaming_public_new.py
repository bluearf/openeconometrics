"""Public dispatch and all-period helpers use verified bounded native kernels."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.dataset import Dataset
from test_econ_streaming_ucm import data as ucm_data, spec as ucm_spec
from test_econ_streaming_mswitch import data as ms_data, specification as ms_spec


def source(data):
    return Dataset.from_batches(lambda: (data.iloc[j:j+7].copy() for j in range(0, len(data), 7)), list(data), row_count=len(data))


def test_public_ucm_components_forecast_are_not_forced_back_into_ram():
    data = ucm_data(67)
    result = oe.fit(ucm_spec(), data=source(data))
    parts = oe.ucm_components(result, source(data))
    assert isinstance(parts, Dataset) and parts.row_count == len(data)
    assert parts.head(5).shape[0] == 5
    forecast = oe.ucm_forecast(result, steps=3, data=source(data), exog={"x": np.zeros(3)})
    saved = oe.ucm_forecast(result, steps=3, exog={"x": np.zeros(3)})
    assert_allclose(forecast.select_dtypes("number"), saved.select_dtypes("number"), rtol=2e-10, atol=1e-9)
    assert result.provenance["execution"]["device"] in {"cpu", "mps", "cuda"}


def test_public_mswitch_probabilities_stay_disk_backed():
    data = ms_data(105)
    result = oe.fit(ms_spec(), data=source(data))
    probabilities = oe.mswitch_probabilities(result, source(data))
    assert isinstance(probabilities, Dataset) and probabilities.row_count == len(data)
    assert probabilities.head(5).shape[0] == 5
    for block in probabilities.iter_batches(batch_rows=9):
        for kind in ("filtered", "smoothed", "predicted"):
            assert_allclose(block[[kind+"_state1", kind+"_state2"]].sum(axis=1), 1., rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("cov", ["nonrobust", "opg", "robust"])
def test_native_ucm_zero_state_gaussian_regression_matches_independent_oracle(cov):
    rng = np.random.default_rng(4509)
    x = rng.normal(size=151)
    y = .73*x+rng.normal(size=len(x))
    data = pd.DataFrame({"y": y, "x": x})
    native = oe.ucm(data=data, y="y", x=["x"], model="ntrend", covariance=cov)
    replay = oe.ucm(data=source(data), y="y", x=["x"], model="ntrend", covariance=cov)
    beta = x@y/(x@x)
    sigma2 = np.mean((y-beta*x)**2)
    coefficients = {coefficient.term: coefficient.estimate for coefficient in native.coefficients}
    assert coefficients["x"] == pytest.approx(beta, rel=2e-12)
    assert coefficients["/var(e)"] == pytest.approx(sigma2, rel=2e-7)
    assert_allclose([row.estimate for row in replay.coefficients], [row.estimate for row in native.coefficients], rtol=3e-7, atol=3e-8)
    assert_allclose(replay.covariance_matrix, native.covariance_matrix, rtol=8e-4, atol=2e-6)
    assert native.metrics["log_likelihood"] == pytest.approx(-len(y)/2*(np.log(2*np.pi*sigma2)+1), abs=2e-10)
    assert oe.ucm_components(native, data).shape[0] == len(data)
    assert len(oe.ucm_forecast(native, steps=2, exog={"x": np.zeros(2)})) == 2
