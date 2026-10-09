"""Independent primal KKT oracle and movement-preservation identities for Denton.

The oracle optimizes reconstructed values directly with NumPy, whereas the
runtime uses Torch adjustment / innovation coordinates. No reference software
implementation is copied or needed at runtime.
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.temporal import denton


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


METHODS = [
    ("denton_additive", "additive", 1, False),
    ("denton_additive_second", "additive", 2, False),
    ("denton_proportional", "proportional", 1, False),
    ("denton_proportional_second", "proportional", 2, False),
    ("denton_cholette", "additive", 1, True),
    ("denton_cholette", "proportional", 1, True),
]


def fixture(low_frequency="Y", high_frequency="Q", aggregation="sum", m=4):
    frequency = {"Y": "Y-DEC", "Q": "Q-DEC", "M": "M"}
    start = "2017" if low_frequency == "Y" else "2017Q1"
    low_periods = pd.period_range(start, periods=m, freq=frequency[low_frequency])
    high_periods = pd.period_range(low_periods[0].asfreq(frequency[high_frequency], how="start"),
                                   low_periods[-1].asfreq(frequency[high_frequency], how="end"),
                                   freq=frequency[high_frequency])
    ratio, n = len(high_periods) // m, len(high_periods)
    C = np.zeros((m, n))
    for row in range(m):
        if aggregation in ("sum", "mean"):
            C[row, row * ratio:(row + 1) * ratio] = 1 if aggregation == "sum" else 1 / ratio
        else:
            C[row, row * ratio + (ratio - 1 if aggregation == "last" else 0)] = 1
    t = np.arange(n)
    x = 5 + .13 * t + .4 * np.sin(.72 * t)
    y = C @ x + np.linspace(2.1, -1.3, m)
    options = dict(low_periods=[str(v) for v in low_periods],
                   high_periods=[str(v) for v in high_periods],
                   low_frequency=low_frequency, high_frequency=high_frequency,
                   aggregation=aggregation)
    return y, x, C, options


def call(name, criterion, y, x, options):
    kwargs = {"criterion": criterion} if name == "denton_cholette" else {}
    return getattr(denton, name)(y, x, **options, **kwargs)


def values(result):
    return np.asarray(result["series"]["value"], dtype=float)


def oracle(y, x, C, criterion, order, cholette):
    n, m = len(x), len(y)
    # Penalize transformed reconstructed values with the original indicator
    # on the right of the first-order condition; optimize v itself, not e/w.
    derivative = np.eye(n) - np.diag(np.ones(n - 1), -1)
    if order == 2:
        derivative = derivative @ derivative
    if cholette:
        derivative = derivative[1:]
    transform = np.eye(n) if criterion == "additive" else np.diag(1 / x)
    operator = derivative @ transform
    H = operator.T @ operator
    system = np.block([[H, C.T], [C, np.zeros((m, m))]])
    return np.linalg.solve(system, np.concatenate((H @ x, y)))[:n]


@pytest.mark.parametrize("name,criterion,order,cholette", METHODS)
@pytest.mark.parametrize("aggregation", ["sum", "mean", "first", "last"])
@pytest.mark.parametrize("frequencies", [("Y", "Q"), ("Y", "M"), ("Q", "M")])
def test_independent_primal_kkt_all_calendar_aggregation_laws(name, criterion, order, cholette,
                                                             aggregation, frequencies):
    y, x, C, options = fixture(*frequencies, aggregation)
    result = call(name, criterion, y, x, options)
    expected = oracle(y, x, C, criterion, order, cholette)
    np.testing.assert_allclose(values(result), expected, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(C @ values(result), y, atol=2e-11, rtol=2e-11)
    assert result.attrs["difference_order"] == order
    assert result.attrs["criterion"] == criterion
    assert result.attrs["stationarity_relative_error"] < 2e-8
    assert result.attrs["stochastic_draws"] == 0
    assert result.attrs["complete_inputs_saved"] is True


@pytest.mark.parametrize("criterion", ["additive", "proportional"])
@pytest.mark.parametrize("aggregation", ["sum", "mean", "first", "last"])
def test_cholette_constant_adjustment_is_exact_zero_penalty(criterion, aggregation):
    _, x, C, options = fixture(aggregation=aggregation)
    target = x - 2.3 if criterion == "additive" else 1.7 * x
    y = C @ target
    result = call("denton_cholette", criterion, y, x, options)
    np.testing.assert_allclose(values(result), target, atol=3e-11, rtol=3e-11)
    assert result.attrs["scaled_penalty"] < 1e-26
    original = "denton_additive" if criterion == "additive" else "denton_proportional"
    anchored_result = call(original, criterion, y, x, options)
    anchored = values(anchored_result)
    if aggregation == "first":
        # The first benchmark directly pins the initially anchored adjustment;
        # a constant remains optimal but its original initial penalty is nonzero.
        np.testing.assert_allclose(anchored, target, rtol=3e-11, atol=3e-11)
        assert anchored_result.attrs["scaled_penalty"] > 0
    else:
        assert np.max(np.abs(anchored - target)) > .01
    assert result.attrs["initial_condition"] == "unanchored constant adjustment"


@pytest.mark.parametrize("name,criterion,order,cholette", METHODS)
def test_no_adjustment_and_scaled_data_units(name, criterion, order, cholette):
    _, x, C, options = fixture()
    result = call(name, criterion, C @ x, x, options)
    np.testing.assert_array_equal(values(result), x)
    assert result.attrs["scaled_penalty"] == 0
    y = C @ x + [1.1, -.8, 1.6, .2]
    baseline = values(call(name, criterion, y, x, options))
    for units in (1e-40, 1e40):
        rescaled = values(call(name, criterion, y * units, x * units, options)) / units
        np.testing.assert_allclose(rescaled, baseline, atol=2e-10, rtol=2e-10)


def test_cholette_proportional_indicator_scale_invariance_and_original_anchor():
    y, x, _, options = fixture()
    baseline = values(call("denton_cholette", "proportional", y, x, options))
    np.testing.assert_allclose(values(call("denton_cholette", "proportional", y, 3 * x, options)),
                               baseline, rtol=2e-11, atol=2e-11)
    original = values(call("denton_proportional", "proportional", y, x, options))
    altered = values(call("denton_proportional", "proportional", y, 3 * x, options))
    assert np.max(np.abs(original - altered)) > .1


@pytest.mark.parametrize("name,criterion,order,cholette", METHODS)
@pytest.mark.parametrize("low_frequency,high_frequency,m", [("Y", "Q", 60), ("Y", "M", 20), ("Q", "M", 80)])
def test_maximum_calendar_long_series_constraints_and_stationarity(name, criterion, order, cholette,
                                                                  low_frequency, high_frequency, m):
    y, x, C, options = fixture(low_frequency, high_frequency, m=m)
    result = call(name, criterion, y, x, options)
    assert len(values(result)) == 240
    np.testing.assert_allclose(C @ values(result), y, atol=2e-8, rtol=2e-10)
    assert result.attrs["stationarity_relative_error"] < 2e-8


@pytest.mark.parametrize("name,criterion,order,cholette", METHODS)
def test_native_cpu_contract_ignores_global_meta_default(name, criterion, order, cholette):
    y, x, _, options = fixture()
    expected = values(call(name, criterion, y, x, options))
    previous = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        actual = call(name, criterion, y, x, options)
        assert torch.get_default_device().type == "meta"
    finally:
        torch.set_default_device(previous)
    np.testing.assert_array_equal(values(actual), expected)
    assert actual.attrs["device"] == "cpu"


@pytest.mark.parametrize("name", ["denton_proportional", "denton_proportional_second", "denton_cholette"])
@pytest.mark.parametrize("bad", [0, -1])
def test_proportional_rejects_nonpositive_indicator(name, bad):
    y, x, _, options = fixture()
    x[5] = bad
    with pytest.raises(AnalysisError, match="strictly positive"):
        getattr(denton, name)(y, x, **options)


@pytest.mark.parametrize("name,criterion,order,cholette", METHODS)
def test_one_indicator_required_and_complete_json_latex_replay(name, criterion, order, cholette):
    y, x, _, options = fixture()
    with pytest.raises(AnalysisError, match="exactly one"):
        call(name, criterion, y, np.column_stack((x, x**2)), options)
    result = call(name, criterion, y, x, options)
    saved = json.loads(json.dumps(result.attrs, allow_nan=False))
    replay_options = {key: saved[key] for key in options}
    replay = call(name, saved["criterion"], saved["low"], saved["indicator"], replay_options)
    np.testing.assert_array_equal(values(replay), values(result))
    latex = result.to_latex()
    assert "series" in latex and "aggregation" in latex and "settings" in latex
    assert len(saved["indicator"]) == len(x)
    assert len(saved["low"]) == len(y)


def test_additive_accepts_signed_indicator_and_negative_output_without_clipping():
    _, x, C, options = fixture()
    x -= 8
    y = C @ (x - 10)
    result = call("denton_cholette", "additive", y, x, options)
    np.testing.assert_allclose(values(result), x - 10, atol=3e-11)
    assert result.attrs["negative_output_count"] == len(x)


def test_cholette_unknown_criterion_is_refused():
    y, x, _, options = fixture()
    with pytest.raises(AnalysisError, match="criterion"):
        call("denton_cholette", "relative", y, x, options)


@pytest.mark.parametrize("name", ["denton_proportional", "denton_proportional_second", "denton_cholette"])
def test_proportional_extreme_indicator_span_fits_each_nonzero_benchmark(name):
    _, _, C, options = fixture()
    x = np.geomspace(1e-45, 1e45, C.shape[1])
    y = 1.4 * (C @ x)
    result = getattr(denton, name)(y, x, **options)
    np.testing.assert_allclose((C @ values(result)) / y, 1, rtol=2e-11, atol=2e-11)
    assert np.isfinite(result.attrs["objective"])


@pytest.mark.parametrize("name,criterion,order,cholette", METHODS)
@pytest.mark.parametrize("aggregation", ["first", "last"])
@pytest.mark.parametrize("low", [[0., 0.], [0., 1.]])
def test_known_zero_endpoints_are_exact_observations(name, criterion, order, cholette, aggregation, low):
    _, _, C, options = fixture(aggregation=aggregation, m=2)
    x = np.arange(1., 9.)
    result = call(name, criterion, low, x, options)
    np.testing.assert_array_equal(C @ values(result), low)
    np.testing.assert_allclose(values(result), oracle(np.asarray(low), x, C, criterion, order, cholette),
                               atol=2e-10, rtol=2e-10)
    assert result.attrs["known_endpoint_cells_preserved"] is True


@pytest.mark.parametrize("aggregation", ["sum", "mean", "first", "last"])
def test_cholette_all_zero_proportional_target_has_exact_zero_penalty(aggregation):
    _, x, C, options = fixture(aggregation=aggregation)
    result = call("denton_cholette", "proportional", np.zeros(len(C)), x, options)
    np.testing.assert_array_equal(values(result), np.zeros(len(x)))
    assert result.attrs["objective"] == 0
    assert result.attrs["stationarity_relative_error"] == 0


@pytest.mark.parametrize("aggregation", ["sum", "mean", "first", "last"])
@pytest.mark.parametrize("indicator", [-7., 0., 7.])
def test_cholette_zero_additive_target_constant_indicator_has_exact_zero_penalty(aggregation, indicator):
    _, x, C, options = fixture(aggregation=aggregation)
    result = call("denton_cholette", "additive", np.zeros(len(C)), np.full_like(x, indicator), options)
    np.testing.assert_array_equal(values(result), np.zeros(len(x)))
    assert result.attrs["objective"] == 0
    assert result.attrs["stationarity_relative_error"] == 0


@pytest.mark.parametrize("name,criterion", [("denton_additive", "additive"),
                                           ("denton_additive_second", "additive"),
                                           ("denton_cholette", "additive")])
def test_catastrophic_nonzero_benchmark_cancellation_is_refused(name, criterion):
    _, _, C, options = fixture()
    x = np.geomspace(1e-45, 1e45, C.shape[1])
    y = 1.4 * (C @ x)
    with pytest.raises(AnalysisError, match="aggregation") as caught:
        call(name, criterion, y, x, options)
    assert caught.value.code == "numerical_failure"
