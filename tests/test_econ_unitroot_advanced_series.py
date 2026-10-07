"""Original-paper SVD replication, contracts, invariances and admission."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget

spec = importlib.util.spec_from_file_location(
    "unitroot_reference", Path(__file__).resolve().parents[1] / "scripts/unitroot_reference.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def fixture(seed=726, n=180):
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=n)
    return pd.DataFrame({"y": np.cumsum(shocks + 0.4 * np.roll(shocks, 1)), "t": np.arange(n)})


@pytest.mark.parametrize("trend", ["constant", "trend"])
@pytest.mark.parametrize("lags", [0, 1, 4, None])
@pytest.mark.parametrize("seed", [726, 512])
def test_ng_original_paper_svd_refits_and_maic(trend, lags, seed):
    frame = fixture(seed)
    expected, selected, candidates = reference.ng_reference(frame.y, trend, lags, maxlag=8)
    options = {"maxlag": 8} if lags is None else {"lags": lags}
    result = oe.ngperron(frame, "y", time="t", trend=trend, **options)
    np.testing.assert_allclose(result.statistic, expected, rtol=3e-11, atol=2e-11)
    assert result.attrs["lags"] == selected
    assert result.attrs["nobs"] == len(frame) - 1 - selected
    assert result.attrs["horizon"] == len(frame) - 1
    if lags is None:
        # Normalization changes log variance by a common additive constant.
        scores = [row["maic"] for row in result.attrs["lag_selection_results"]]
        np.testing.assert_allclose(
            np.array(scores) - scores[0], np.array(candidates) - candidates[0], atol=3e-11
        )
        assert [row["lags"] for row in result.attrs["lag_selection_results"]] == list(range(9))
    json.dumps(result.attrs, allow_nan=False)


@pytest.mark.parametrize("trend", ["none", "constant", "trend"])
@pytest.mark.parametrize("lags", [0, 1, 3])
@pytest.mark.parametrize("seed", [726, 512])
def test_kss_original_cubic_regression_svd(trend, lags, seed):
    frame = fixture(seed)
    expected = reference.kss_reference(frame.y, trend, lags)
    result = oe.kss(frame, "y", time="t", trend=trend, lags=lags)
    assert result.attrs["statistic"] == pytest.approx(expected, rel=2e-11, abs=2e-11)
    assert result.attrs["nobs"] == len(frame) - 1 - lags
    assert result.attrs["df_resid"] == len(frame) - 2 - 2 * lags
    json.dumps(result.attrs, allow_nan=False)


def test_critical_values_from_original_tables_and_only_discrete_decisions():
    kss_expected = {
        "none": [-2.82, -2.22, -1.92],
        "constant": [-3.48, -2.93, -2.66],
        "trend": [-3.93, -3.40, -3.13],
    }
    for method, cases in [(oe.kss, kss_expected)]:
        for trend, cvs in cases.items():
            result = method(fixture(), "y", trend=trend, lags=0)
            expected = np.asarray(cvs).reshape(-1, 3)
            np.testing.assert_array_equal(
                result[["critical_1pct", "critical_5pct", "critical_10pct"]], expected
            )
            np.testing.assert_array_equal(
                result[["reject_1pct", "reject_5pct", "reject_10pct"]],
                result.statistic.to_numpy()[:, None] < expected,
            )
            assert result.attrs["p_value"] is None and result.p_value.isna().all()
            assert "asymptotic" in result.attrs["distribution"]
            assert "\\begin{tabular}" in result.to_latex()


@pytest.mark.parametrize("method", [oe.ngperron, oe.kss])
@pytest.mark.parametrize("trend", ["constant", "trend"])
def test_scaling_level_sorting_and_preservation(method, trend):
    frame = fixture()
    before = frame.copy(deep=True)
    base = method(frame, "y", time="t", trend=trend, lags=1)
    for scale, origin in [(1e100, 0), (-1e-100, 0), (1, 1e9)]:
        moved = frame.assign(y=frame.y * scale + origin).sample(frac=1, random_state=1)
        result = method(moved, "y", time="t", trend=trend, lags=1)
        np.testing.assert_allclose(result.statistic, base.statistic, rtol=2e-7, atol=2e-7)
    pd.testing.assert_frame_equal(frame, before)
    np.testing.assert_allclose(
        method(frame.to_dict("list"), "y", trend=trend, lags=1).statistic, base.statistic
    )


def test_raw_kss_scale_invariance_but_no_claim_of_level_invariance():
    frame = fixture()
    base = oe.kss(frame, "y", trend="none", lags=2).attrs["statistic"]
    for scale in [1e100, -1e-100]:
        assert oe.kss(frame.assign(y=frame.y * scale), "y", trend="none", lags=2).attrs[
            "statistic"
        ] == pytest.approx(base, rel=1e-10)
    assert (
        abs(
            oe.kss(frame.assign(y=frame.y + 10), "y", trend="none", lags=2).attrs["statistic"]
            - base
        )
        > 0.1
    )


@pytest.mark.parametrize("method", [oe.ngperron, oe.kss])
@pytest.mark.parametrize(
    "change,code",
    [
        (lambda f: f.assign(y=np.nan), "missing_values"),
        (lambda f: f.assign(y=f.y.where(f.index != 10, np.inf)), "non_finite_values"),
        (lambda f: f.drop(index=10), "time_gaps"),
        (lambda f: f.assign(t=0), "repeated_time_values"),
        (lambda f: f.assign(t=pd.date_range("2000-01-01", periods=len(f))), "invalid_time"),
        (lambda f: f.assign(t=f.t + 0.5), "invalid_time"),
        (lambda f: f.assign(y=1), "constant_series"),
        (lambda f: f.assign(y=f.t * 2), "perfect_fit"),
    ],
)
def test_calendar_and_undefined_samples_refuse(method, change, code):
    with pytest.raises(AnalysisError) as error:
        method(change(fixture()), "y", time="t", trend="trend", lags=0)
    assert error.value.code == code


@pytest.mark.parametrize("method", [oe.ngperron, oe.kss])
@pytest.mark.parametrize(
    "options,code",
    [
        ({"lags": True}, "invalid_lags"),
        ({"lags": -1}, "invalid_lags"),
        ({"lags": 200}, "insufficient_observations"),
        ({"trend": "drift"}, "invalid_option"),
        ({"max_work": 1}, "work_limit"),
        ({"max_memory_mb": False}, "invalid_option"),
    ],
)
def test_option_refusals(method, options, code):
    with pytest.raises(AnalysisError) as error:
        method(fixture(), "y", **options)
    assert error.value.code == code


def test_ng_ambiguous_lag_choice_refuses_and_defaults_include_zero():
    with pytest.raises(AnalysisError) as error:
        oe.ngperron(fixture(), "y", lags=1, maxlag=5)
    assert error.value.code == "invalid_lags"
    result = oe.ngperron(fixture(), "y", maxlag=0)
    assert result.attrs["lags"] == 0 and len(result.attrs["lag_selection_results"]) == 1
    assert oe.ngperron(fixture(), "y").attrs["maxlag"] == int(12 * (179 / 100) ** 0.25)


@pytest.mark.parametrize("method", [oe.ngperron, oe.kss])
def test_workspace_admission_precedes_numeric_tensor_buffers(method, monkeypatch):
    from openecon.econometrics.unitroot import advanced_series

    def forbidden(*a, **k):
        raise AssertionError("tensor conversion must not run after refusal")

    monkeypatch.setattr(advanced_series, "load_series", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        method(fixture(n=20_000), "y", lags=1)
    assert error.value.code == "workspace_limit"
    assert error.value.resource_plan["estimated_workspace_bytes"] > 1024**2


def test_ng_unverified_inference_refuses_without_fabricated_critical_values():
    result = oe.ngperron(fixture(), "y")
    assert result.attrs["critical_values"] is result.attrs["reject"] is None
    assert not any(name.startswith(("reject_", "critical_")) for name in result.columns)
    assert result.attrs["calibration_status"] == "unresolved_published_table_replication"
    with pytest.raises(AnalysisError) as error:
        oe.ngperron(fixture(), "y", inference="published")
    assert error.value.code == "critical_values_unverified"
