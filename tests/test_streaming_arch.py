import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_econ_arch_oracle import simulate, oracle_of

import openecon as oe
from openecon.dataset import Dataset
from openecon.econometrics.streaming_arch import fit_streaming_arch


@pytest.mark.parametrize("model,dist,seed", [("garch", "normal", 11), ("gjr", "normal", 12),
    ("garch", "t", 14), ("egarch", "normal", 13)])
def test_entire_fit_information_diagnostics_and_forecast(model, dist, seed, monkeypatch):
    monkeypatch.setenv("OPENECON_DISABLE_AUTO_STREAMING", "1")
    frame = simulate(700, seed, model=model, dist=dist)
    dense = oe.arch(data=frame, y="y", x=["x"], time="t", model=model, dist=dist, covariance="robust")
    from openecon.econometrics.ordered_replay import OrderedReplay
    original = OrderedReplay.__init__
    def small(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.rows = min(73, self.rows)
    monkeypatch.setattr(OrderedReplay, "__init__", small)
    source = Dataset.from_batches(lambda: (frame.iloc[j:j+17] for j in range(0, len(frame), 17)), frame.columns, row_count=len(frame))
    result = fit_streaming_arch(dense.spec, source)
    assert result.nobs == len(frame)
    assert len(result.predictions) == 400
    assert_allclose([row.estimate for row in result.coefficients], [row.estimate for row in dense.coefficients], rtol=3e-5, atol=3e-6)
    assert_allclose(result.covariance_matrix, dense.covariance_matrix, rtol=3e-4, atol=3e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(dense.metrics["log_likelihood"], abs=1e-7)
    assert result.metrics["aic"] == pytest.approx(dense.metrics["aic"], abs=2e-7)
    for name in ("ljung_box", "ljung_box_squared", "jarque_bera", "arch_lm_residuals"):
        assert result.tests[name]["statistic"] == pytest.approx(dense.tests[name]["statistic"], rel=3e-5, abs=3e-6)
        assert result.tests[name]["nobs"] == dense.tests[name]["nobs"] if "nobs" in result.tests[name] else True
    assert_allclose(result.extra["conditional_variance_tail"], dense.extra["conditional_variance_tail"], rtol=3e-5, atol=3e-6)
    forecast = oe.arch_forecast(result, steps=4, exog={"x": np.zeros(4)})
    expected = oe.arch_forecast(dense, steps=4, exog={"x": np.zeros(4)})
    assert_allclose(forecast.select_dtypes("number"), expected.select_dtypes("number"), rtol=3e-5, atol=3e-6)
    theta, oracle = oracle_of(result, frame)
    signs = None
    if model == "egarch":
        signs = np.where(oracle.path(theta)[0]<0., -1., 1.)
        record = result.provenance["optimizer"]
        signs[record.get("kink_observations", [])] = record.get("kink_weights", [])
    assert result.metrics["log_likelihood"] == pytest.approx(float(oracle(theta, signs).sum()), rel=3e-13)


@pytest.mark.parametrize("simulation,keywords,seed", [
    ({"model": "egarch", "ar": .5}, {"model": "egarch", "ar": 1}, 1130),
    ({"model": "egarch", "archm": "sd", "psi": .4}, {"model": "egarch", "archm": "sd"}, 822)])
def test_disk_active_kink_equations_match_complete_native_solution(simulation, keywords, seed, monkeypatch):
    from openecon.econometrics.ordered_replay import OrderedReplay
    original = OrderedReplay.__init__
    def small(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.rows = min(47, self.rows)
    monkeypatch.setattr(OrderedReplay, "__init__", small)
    data = simulate(600, seed, **simulation)
    native = oe.arch(data=data, y="y", x=["x"], time="t", covariance="nonrobust", **keywords)
    source = Dataset.from_batches(lambda: (data.iloc[j:j+19] for j in range(0, len(data), 19)), data.columns, row_count=len(data))
    replay = fit_streaming_arch(native.spec, source)
    record = replay.provenance["optimizer"]
    assert record["kink_observations"]
    assert record["kink_observations"] == native.provenance["optimizer"]["kink_observations"]
    assert_allclose(record["kink_weights"], native.provenance["optimizer"]["kink_weights"], rtol=2e-5, atol=2e-5)
    assert_allclose([row.estimate for row in replay.coefficients], [row.estimate for row in native.coefficients], rtol=2e-5, atol=2e-6)
    assert_allclose(replay.covariance_matrix, native.covariance_matrix, rtol=3e-4, atol=3e-6)
    theta, independent = oracle_of(replay, data)
    residual, variance, _ = independent.path(theta)
    active = record["kink_observations"]
    assert np.max(np.abs(residual[active]/np.sqrt(variance[active])))<1e-8
    assert all(abs(weight)<=1. for weight in record["kink_weights"])


@pytest.mark.parametrize("simulation,keywords,seed", [
    ({}, {"model": "arch", "arch": 2}, 777),
    ({"dist": "ged"}, {"dist": "ged"}, 901),
    ({"model": "parch"}, {"model": "parch"}, 523),
    ({"archm": "log", "psi": .3}, {"archm": "log"}, 818),
    ({"ar": .4}, {"ar": [1, 3], "ma": [2]}, 1125),
    ({"het": .5}, {"variance_x": ["z"]}, 514)])
def test_full_native_extended_family_options(simulation, keywords, seed, monkeypatch):
    data = simulate(600, seed, **simulation)
    native = oe.arch(data=data, y="y", x=["x"], time="t", covariance="opg", **keywords)
    source = Dataset.from_batches(lambda: (data.iloc[j:j+73] for j in range(0, len(data), 73)), data.columns, row_count=len(data))
    from openecon.econometrics.ordered_replay import OrderedReplay
    original = OrderedReplay.__init__
    def small(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.rows = min(17, self.rows)
    monkeypatch.setattr(OrderedReplay, "__init__", small)
    actual = fit_streaming_arch(native.spec, source)
    assert actual.nobs == len(data)
    assert_allclose([c.estimate for c in actual.coefficients], [c.estimate for c in native.coefficients], rtol=4e-5, atol=5e-6)
    assert_allclose(actual.covariance_matrix, native.covariance_matrix, rtol=8e-4, atol=3e-6)
    assert actual.metrics["log_likelihood"] == pytest.approx(native.metrics["log_likelihood"], abs=2e-7)
