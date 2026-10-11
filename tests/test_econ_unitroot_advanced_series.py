"""Original-paper SVD replication, contracts, invariances and admission."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.unitroot.advanced_series import NG_CRITICAL
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
    # Ng & Perron (2001) Table I constant row: MZa, MZt, MSB, MPT.
    ng_expected = {
        "constant": [
            [-13.8, -8.1, -5.7],
            [-2.58, -1.98, -1.62],
            [0.174, 0.233, 0.275],
            [1.78, 3.17, 4.45],
        ]
    }
    for method, cases, options in [
        (oe.kss, kss_expected, {}),
        (oe.ngperron, ng_expected, {"inference": "published"}),
    ]:
        for trend, cvs in cases.items():
            result = method(fixture(), "y", trend=trend, lags=0, **options)
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


@pytest.mark.parametrize("trend", ["constant", "trend"])
def test_ng_default_inference_returns_statistics_only(trend):
    result = oe.ngperron(fixture(), "y", trend=trend)
    assert result.attrs["inference"] == "none"
    assert result.attrs["critical_values"] is result.attrs["reject"] is None
    assert result.attrs["distribution"] is None and result.attrs["p_value"] is None
    assert not any(name.startswith(("reject_", "critical_")) for name in result.columns)
    assert result.attrs["calibration_status"] == {
        "constant": "published_table_replicated",
        "trend": "published_trend_row_inconsistent_with_theorem1_limit",
    }[trend]


@pytest.mark.parametrize("lags", [0, 2, None])
def test_ng_constant_published_inference_keeps_statistics_and_lower_tail_decisions(lags):
    options = {"maxlag": 8} if lags is None else {"lags": lags}
    plain = oe.ngperron(fixture(), "y", **options)
    result = oe.ngperron(fixture(), "y", inference="published", **options)
    assert result.statistic.tolist() == plain.statistic.tolist()
    assert result.attrs["lags"] == plain.attrs["lags"]
    assert result.attrs["inference"] == "published" and result.attrs["p_value"] is None
    assert result.p_value.isna().all()
    assert "Theorem 1 asymptotic" in result.attrs["distribution"]
    for row in result.to_dict("records"):
        name, statistic = row["test"], row["statistic"]
        assert result.attrs["critical_values"][name] == dict(
            zip(["1%", "5%", "10%"], NG_CRITICAL["constant"][name], strict=True)
        )
        for percent, cv in zip((1, 5, 10), NG_CRITICAL["constant"][name], strict=True):
            assert row[f"critical_{percent}pct"] == cv
            assert row[f"reject_{percent}pct"] == (statistic < cv)
            assert result.attrs["reject"][name][f"{percent}%"] == (statistic < cv)
    json.dumps(result.attrs, allow_nan=False)
    assert "\\begin{tabular}" in result.to_latex()


def test_ng_published_decisions_follow_stationarity_evidence():
    rng = np.random.default_rng(136)
    stationary = np.zeros(400)
    for t in range(1, 400):
        stationary[t] = 0.3 * stationary[t - 1] + rng.normal()
    rejected = oe.ngperron(pd.DataFrame({"y": stationary}), "y", lags=0, inference="published")
    assert rejected.attrs["reject"]["MZa"]["1%"] and rejected.attrs["reject"]["MPT"]["1%"]
    kept = oe.ngperron(
        pd.DataFrame({"y": np.cumsum(rng.normal(size=400))}), "y", lags=0, inference="published"
    )
    assert kept.attrs["statistics"]["MZa"] > NG_CRITICAL["constant"]["MZa"][0]


def test_ng_trend_published_inference_refuses_without_fabricated_critical_values():
    with pytest.raises(AnalysisError) as error:
        oe.ngperron(fixture(), "y", trend="trend", inference="published")
    assert error.value.code == "critical_values_unverified"
    assert "Theorem-1" in str(error.value)
