"""Independent QR/Schur-complement oracles for the lag-augmented Wald test."""

import json
import math

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.stats import chi2
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.var.toda_yamamoto import tycausality


def sample(n=220, order=1, seed=321):
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=(n, 3)) @ np.array([[1., .3, -.2], [0., .9, .2], [0., 0., .8]])
    values = np.empty_like(shocks)
    values[0] = shocks[0]
    for t in range(1, n):
        values[t] = .2 + values[t-1] @ np.array([[.5, .2, 0], [0, .3, .1], [.1, 0, .4]]) + shocks[t]
    for _ in range(order):
        values = np.cumsum(values, axis=0)
    return pd.DataFrame(values, columns=["a", "b", "c"])


def oracle(frame, base, dmax, constant, trend, effect, causes):
    """Direct NumPy QR with original ML divisor; never calls OpenEcon's fit."""
    values = frame[["a", "b", "c"]].to_numpy()
    n, k = values.shape
    p = base + dmax
    cols = [values[p-j:n-j, v] for v in range(k) for j in range(1, p+1)]
    if trend:
        cols.append(np.arange(p + 1, n + 1))
    if constant:
        cols.append(np.ones(n-p))
    z = np.column_stack(cols)
    q, r = np.linalg.qr(z)
    b = np.linalg.solve(r, q.T @ values[p:])
    u = values[p:] - z @ b
    sigma = u.T @ u / (n-p)
    ri = np.linalg.solve(r, np.eye(r.shape[0]))
    vi = ri @ ri.T
    v = np.kron(sigma, vi)
    names = ["a", "b", "c"]
    ix = [names.index(effect) * z.shape[1] + names.index(cause) * p + j
          for cause in causes for j in range(base)]
    estimates = b.T.reshape(-1)[ix]
    block = v[np.ix_(ix, ix)]
    w = float(estimates @ np.linalg.solve(block, estimates))
    # Original-paper projection formula removes all other design columns first.
    restricted = [names.index(cause) * p + j for cause in causes for j in range(base)]
    other = [j for j in range(z.shape[1]) if j not in restricted]
    qo, _ = np.linalg.qr(z[:, other])
    selected = z[:, restricted] - qo @ (qo.T @ z[:, restricted])
    schur = selected.T @ selected
    alternative = float(estimates @ schur @ estimates / sigma[names.index(effect), names.index(effect)])
    return b.T, v, sigma, w, alternative


@pytest.mark.parametrize("base,dmax", [(1, 0), (1, 1), (2, 1), (2, 2)])
@pytest.mark.parametrize("constant,trend", [(False, False), (True, False), (True, True)])
def test_full_system_matches_independent_qr_and_paper_projection(base, dmax, constant, trend):
    frame = sample(order=min(dmax, 2))
    result = tycausality(data=frame, y=["a", "b", "c"], lags=base, dmax=dmax,
                        constant=constant, trend=trend)
    for row in result["tests"].to_dict("records"):
        b, v, sigma, w, projection = oracle(frame, base, dmax, constant, trend,
                                           row["effect"], row["causes"].split(", "))
        assert_allclose(result["coefficients"]["estimate"], b.reshape(-1), rtol=2e-8, atol=2e-9)
        assert_allclose(result["coefficients"]["std_error"], np.sqrt(v.diagonal()), rtol=2e-8, atol=2e-9)
        assert_allclose(result.attrs["sigma_ml"], sigma, rtol=2e-9, atol=2e-10)
        assert_allclose(row["statistic"], w, rtol=4e-8, atol=2e-9)
        assert_allclose(w, projection, rtol=2e-8, atol=2e-9)
        assert_allclose(row["p_value"], chi2.sf(w, row["df"]), rtol=5e-8, atol=2e-10)
    assert result.attrs["covariance_divisor"] == len(frame) - base - dmax
    assert result.attrs["distribution"] == "chi2" and result.attrs["asymptotic"] is True
    assert len(result["tests"]) == 9
    json.dumps(result.attrs, allow_nan=False)
    assert "\\begin{table}" in result.to_latex()


def test_augmentation_coefficients_are_nuisance_not_restrictions():
    frame = sample(order=1)
    result = tycausality(data=frame, y=["a", "b", "c"], lags=1, dmax=2,
                        causes=["b", "c"], effects=["a"])
    assert result["tests"]["df"].tolist() == [2]
    assert result.attrs["tested_terms"] == [["a:L1.b", "a:L1.c"]]
    assert len(result["coefficients"].query("role == 'augmentation'")) == 18
    _, _, _, w, _ = oracle(frame, 1, 2, True, False, "a", ["b", "c"])
    assert_allclose(result["tests"]["statistic"], [w])
    # An ordinary three-lag Granger test also constrains the nuisance lags.
    _, _, _, wrong, _ = oracle(frame, 3, 0, True, False, "a", ["b", "c"])
    assert abs(w - wrong) > .5


def test_pairwise_group_selection_and_default_effect_filter():
    frame = sample()
    pairs = tycausality(data=frame, y=["a", "b", "c"], lags=2, dmax=1,
                       causes=["b", "c"], effects=["a"], joint=False)
    assert pairs["tests"]["causes"].tolist() == ["b", "c"]
    assert pairs["tests"]["df"].tolist() == [2, 2]
    default = tycausality(data=frame, y=["a", "b", "c"], lags=1, causes=["b"])
    assert default["tests"]["effect"].tolist() == ["a", "c"]


@pytest.mark.parametrize("time_values", [
    np.arange(2**53+17, 2**53+237, dtype=np.int64),
    np.arange(2**64-230, 2**64-10, dtype=np.uint64),
    pd.date_range("2000-01-01", periods=220, freq="MS"),
    pd.date_range("2000-01-01", periods=220, freq="B"),
])
def test_exact_integer_keys_and_regular_calendars_sort_without_precision_loss(time_values):
    frame = sample()
    plain = tycausality(data=frame, y=["a", "b", "c"], lags=2)
    frame["date"] = time_values
    scrambled = frame.sample(frac=1, random_state=11)
    ordered = tycausality(data=scrambled, y=["a", "b", "c"], lags=2, time="date")
    assert_allclose(ordered["tests"]["statistic"], plain["tests"]["statistic"], rtol=1e-12)
    assert_allclose(ordered["coefficients"]["estimate"], plain["coefficients"]["estimate"], rtol=1e-12)


@pytest.mark.parametrize("operation,code", [
    (lambda d: d.assign(date=np.arange(len(d)) * 2), "time_gaps"),
    (lambda d: d.assign(date=np.ones(len(d), dtype=int)), "repeated_time_values"),
    (lambda d: d.assign(date=np.arange(len(d)) + .25), "invalid_time"),
    (lambda d: d.assign(date=np.arange(len(d)) * 2. + 2**53), "invalid_time"),
    (lambda d: d.assign(date=pd.date_range("2000-01-01", periods=len(d)+1).delete(20)), "time_gaps"),
])
def test_invalid_time_is_never_silently_compressed(operation, code):
    with pytest.raises(AnalysisError) as error:
        tycausality(data=operation(sample()), y=["a", "b", "c"], lags=1, time="date")
    assert error.value.code == code


@pytest.mark.parametrize("option", [
    {"lags": True}, {"lags": 1.5}, {"lags": 0}, {"dmax": True}, {"dmax": 3},
    {"dmax": -1}, {"constant": 1}, {"trend": "yes"}, {"joint": 0},
    {"alpha": float("nan")}, {"alpha": True}, {"alpha": 1},
    {"y": "a"}, {"y": ["a", "a"]}, {"y": ["a"]}, {"time": "a"},
    {"causes": ["d"]}, {"causes": ["a"], "effects": ["a"]},
    {"causes": []}, {"effects": []},
])
def test_invalid_specifications_fail_before_numerical_work(option):
    kwargs = {"data": sample(), "y": ["a", "b", "c"], "lags": 1, **option}
    with pytest.raises(AnalysisError):
        tycausality(**kwargs)


def test_no_missing_nonfinite_collinear_or_perfect_equation_is_silently_accepted():
    frame = sample()
    for invalid in (frame.assign(a=np.nan), frame.assign(a=np.inf),
                    frame.assign(b=frame.a), frame.assign(a=np.arange(len(frame)))):
        with pytest.raises(AnalysisError):
            tycausality(data=invalid, y=["a", "b", "c"], lags=1)


def test_sample_and_budget_rejections_precede_lag_allocation(monkeypatch):
    from openecon.econometrics.var import toda_yamamoto as implementation
    monkeypatch.setattr(implementation.kernels, "lag_block", lambda *_: pytest.fail("Allocated lag matrix"))
    with pytest.raises(AnalysisError) as error:
        tycausality(data=sample(n=10), y=["a", "b", "c"], lags=4)
    assert error.value.code == "insufficient_observations"
    monkeypatch.setattr(implementation, "_MAX_WORK_BYTES", 1)
    with pytest.raises(AnalysisError) as error:
        tycausality(data=sample(), y=["a", "b", "c"], lags=1)
    assert error.value.code == "work_budget_exceeded"


def test_workspace_guard_counts_full_system_covariance_before_allocation():
    from openecon.econometrics.var.toda_yamamoto import _budget
    # Previously, the lag/QR buffers alone fitted just under 512 MiB while
    # the full (K*m)^2 coefficient covariance crossed the declared budget.
    with pytest.raises(AnalysisError) as error:
        _budget(87_309, 10, 9, 1)
    assert error.value.code == "work_budget_exceeded"


def test_float64_cpu_scope_preserves_default_device_and_input():
    frame = sample()
    original = frame.copy(deep=True)
    with torch.device("meta"):
        result = tycausality(data=frame, y=["a", "b", "c"], lags=np.int64(2))
        assert torch.empty(1).device.type == "meta"
    assert result.attrs["device"] == "cpu" and result.attrs["dtype"] == "float64"
    pd.testing.assert_frame_equal(frame, original)


def test_unit_and_level_shift_invariance_with_constant():
    frame = sample(order=1)
    baseline = tycausality(data=frame, y=["a", "b", "c"], lags=2)
    rescaled = frame.assign(a=frame.a * 1e-8 + 1e-4,
                            b=frame.b * 1e8 - 1e11, c=frame.c + 1e5)
    changed = tycausality(data=rescaled, y=["a", "b", "c"], lags=2)
    assert_allclose(changed["tests"]["statistic"], baseline["tests"]["statistic"], rtol=2e-8, atol=1e-8)
    assert all(math.isfinite(x) for x in changed["coefficients"]["std_error"])


def test_large_covariance_rank_loss_is_not_reported_as_a_different_hypothesis(monkeypatch):
    from openecon.econometrics.var import toda_yamamoto as implementation
    monkeypatch.setattr(implementation, "wald_test", lambda *_: {"df": 0, "statistic": None})
    with pytest.raises(AnalysisError) as error:
        tycausality(data=sample(), y=["a", "b", "c"], lags=1)
    assert error.value.code == "singular_test_covariance"
