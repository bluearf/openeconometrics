"""Exact ordered/patterned GEE replay, independent GLS oracles and guards."""
from __future__ import annotations

import importlib
import builtins

import numpy as np
import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.mixed.gee import fit_xtgee
from openecon.econometrics.mixed.xt import fit_xtlogit, fit_xtpoisson, fit_xtprobit
from openecon.econometrics.streaming_gee import fit_pa, fit_streaming_gee
from openecon.econometrics.streaming_gee_ordered import OrderedPanels
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget

CORRELATIONS = ["ar1", "stationary", "nonstationary", "unstructured"]


def panel_data(family="gaussian", *, seed=591983, unbalanced=True):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(4, 7, 80) if unbalanced else np.full(80, 6)
    group = np.repeat(np.arange(80), sizes)
    period = np.concatenate([np.arange(size) for size in sizes])
    x, e = rng.normal(size=(2, len(group)))
    eta = .4 + .15 * x + .03 * np.sin(group)
    if family == "gaussian":
        y = eta + e
    elif family == "binomial":
        y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    elif family in {"poisson", "nbinomial"}:
        y = rng.poisson(np.exp(eta))
    else:
        y = np.exp(eta + .25 * e)
    frame = pd.DataFrame({"g": group, "t": period, "x": x, "y": y,
                          "cat": pd.Categorical(period % 3, categories=[0, 1, 2, 3]),
                          "offset": np.full(len(group), .08), "exposure": np.full(len(group), 1.1)})
    return frame.sample(frac=1, random_state=392).reset_index(drop=True)


def specification(corr, family="gaussian", covariance="robust", **options):
    return ModelSpec(estimator="xtgee", outcome="y", predictors=["x"], panel="g", time="t",
                     covariance=covariance,
                     options={"family": family, "corr": corr,
                              "corr_order": 2 if corr in {"stationary", "nonstationary"} else 1,
                              **options})


def source_of(frame, reader_rows=11):
    def reader():
        for start in range(0, len(frame), reader_rows):
            yield frame.iloc[start:start + reader_rows].copy()
    source = Dataset.from_batches(reader, frame.columns.tolist(), row_count=len(frame))
    source.collect = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Never collect a GEE sample."))
    return source


def assert_parity(actual, expected, *, rows=17):
    assert [c.term for c in actual.coefficients] == [c.term for c in expected.coefficients]
    for a, e in zip(actual.coefficients, expected.coefficients, strict=True):
        for key in ["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"]:
            assert getattr(a, key) == pytest.approx(getattr(e, key), rel=3e-7, abs=3e-9), (a.term, key)
    np.testing.assert_allclose(actual.covariance_matrix, expected.covariance_matrix, rtol=3e-7, atol=3e-9)
    for key, value in expected.metrics.items():
        assert actual.metrics[key] == pytest.approx(value, rel=3e-7, abs=3e-9), key
    for key in ["family", "link", "corr", "nmp", "scale_method", "corr_order"]:
        if key in expected.extra:
            assert actual.extra[key] == expected.extra[key], key
    for key in ["alpha", "working_correlation"]:
        if key in expected.extra:
            np.testing.assert_allclose(actual.extra[key], expected.extra[key], rtol=3e-7, atol=3e-9)
    for key in ["statistic", "p_value", "df", "df2"]:
        assert actual.tests["model"].get(key) == pytest.approx(expected.tests["model"].get(key))
    assert actual.nobs == expected.nobs
    assert actual.nobs_original == expected.nobs_original
    assert actual.dropped_rows == expected.dropped_rows
    assert actual.sample_positions == []
    assert actual.provenance["streaming"]["maximum_batch_rows"] <= rows
    assert not actual.provenance["streaming"]["dense_observation_matrix"]
    assert ResultBundle.model_validate_json(actual.model_dump_json()).nobs == actual.nobs
    assert "\\begin{tabular}" in actual.to_latex()


@pytest.mark.parametrize("corr", CORRELATIONS)
@pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson", "gamma", "nbinomial", "igaussian"])
@pytest.mark.parametrize("covariance", ["nonrobust", "robust"])
def test_all_ordered_correlations_family_covariance_contracts(corr, family, covariance):
    frame = panel_data(family)
    options = {"link": "log"} if family in {"gamma", "igaussian"} else {}
    if family == "nbinomial":
        options["nbk"] = .6
    spec = specification(corr, family, covariance, **options)
    assert_parity(fit_streaming_gee(spec, source_of(frame), batch_rows=17), fit_xtgee(spec, frame))


@pytest.mark.parametrize("corr", CORRELATIONS)
@pytest.mark.parametrize("scale,nmp", [(None, False), ("x2", True), ("dev", True), (1.7, False)])
def test_scale_offset_exposure_categories_and_independent_numpy_gls(corr, scale, nmp):
    frame = panel_data()
    spec = specification(corr, covariance="nonrobust", scale=scale, nmp=nmp).model_copy(update={
        "predictors": ["x", "cat"], "categorical": ["cat"],
        "columns": {"offset": "offset", "exposure": "exposure"}})
    actual = fit_streaming_gee(spec, source_of(frame), batch_rows=17)
    assert_parity(actual, fit_xtgee(spec, frame))
    x = np.column_stack((np.ones(len(frame)), frame.x, frame.cat == 1, frame.cat == 2))
    y = frame.y.to_numpy() - frame.offset.to_numpy() - np.log(frame.exposure.to_numpy())
    matrix = np.asarray(actual.extra["working_correlation"])
    bread, rhs = np.zeros((4, 4)), np.zeros(4)
    for _, part in frame.groupby("g"):
        indices = part.sort_values("t").index.to_numpy()
        inv = np.linalg.inv(matrix[:len(indices), :len(indices)])
        bread += x[indices].T @ inv @ x[indices]
        rhs += x[indices].T @ inv @ y[indices]
    beta = np.linalg.solve(bread, rhs)
    np.testing.assert_allclose([c.estimate for c in actual.coefficients], beta, atol=2e-9)
    np.testing.assert_allclose(actual.covariance_matrix, np.linalg.inv(bread) * actual.metrics["scale"], atol=2e-10)


@pytest.mark.parametrize("corr", CORRELATIONS)
def test_independent_numpy_true_panel_sandwich(corr):
    frame = panel_data()
    result = fit_streaming_gee(specification(corr), source_of(frame), batch_rows=17)
    x = np.column_stack((np.ones(len(frame)), frame.x))
    y = frame.y.to_numpy()
    beta = np.array([c.estimate for c in result.coefficients])
    matrix = np.asarray(result.extra["working_correlation"])
    bread, meat = np.zeros((2, 2)), np.zeros((2, 2))
    for _, part in frame.groupby("g"):
        indices = part.sort_values("t").index.to_numpy()
        inv = np.linalg.inv(matrix[:len(indices), :len(indices)])
        bread += x[indices].T @ inv @ x[indices]
        score = x[indices].T @ inv @ (y[indices] - x[indices] @ beta)
        meat += np.outer(score, score)
    cov = np.linalg.solve(bread, meat) @ np.linalg.inv(bread) * 80 / 79
    np.testing.assert_allclose(result.covariance_matrix, cov, rtol=3e-8, atol=2e-10)


@pytest.mark.parametrize("corr", CORRELATIONS)
@pytest.mark.parametrize("estimator,family,function", [
    ("xtlogit", "binomial", fit_xtlogit), ("xtprobit", "binomial", fit_xtprobit),
    ("xtpoisson", "poisson", fit_xtpoisson)])
def test_xt_pa_wrappers_keep_exact_contract(corr, estimator, family, function):
    frame = panel_data(family)
    spec = specification(corr, family).model_copy(update={"estimator": estimator,
        "options": {"model": "pa", "corr": corr, "corr_order": 2 if corr in {"stationary", "nonstationary"} else 1}})
    assert_parity(fit_pa(spec, source_of(frame), batch_rows=17), function(spec, frame))


@pytest.mark.parametrize("corr", ["ar1", "stationary", "nonstationary"])
def test_short_panels_drop_before_global_design_rank_and_response_moments(corr):
    frame = panel_data()
    short = pd.DataFrame({"g": [101, 102, 102], "t": [0, 0, 1], "x": [1e4, -1e4, 2e4], "y": [1e5, -1e5, 2e5]})
    frame = pd.concat((frame, short), ignore_index=True)
    frame["short_only"] = (frame.g == 101).astype(float)
    spec = specification(corr).model_copy(update={"predictors": ["x", "short_only"]})
    actual = fit_streaming_gee(spec, source_of(frame), batch_rows=17)
    assert_parity(actual, fit_xtgee(spec, frame))
    assert any("too few" in value for value in actual.warnings)
    assert "short_only" not in [c.term for c in actual.coefficients]


@pytest.mark.parametrize("corr", CORRELATIONS)
def test_gaps_force_missing_datetime_and_exact_large_periods(corr):
    frame = panel_data()
    frame.loc[frame.g == 3, "t"] *= 2
    spec = specification(corr)
    for fit, data in [(fit_xtgee, frame), (fit_streaming_gee, source_of(frame))]:
        with pytest.raises(AnalysisError) as error:
            fit(spec, data)
        assert error.value.code == "unequal_spacing"
    forced = spec.model_copy(update={"options": {**spec.options, "force": True}})
    assert_parity(fit_streaming_gee(forced, source_of(frame), batch_rows=17), fit_xtgee(forced, frame))
    frame.loc[2, "y"] = np.nan
    forced = forced.model_copy(update={"missing": "drop"})
    assert_parity(fit_streaming_gee(forced, source_of(frame), batch_rows=17), fit_xtgee(forced, frame))
    frame = panel_data()
    frame.t = frame.t.astype("int64") + 2**54
    assert_parity(fit_streaming_gee(spec, source_of(frame), batch_rows=17), fit_xtgee(spec, frame))
    frame.t = pd.Timestamp("2020-01-01") + pd.to_timedelta(frame.t - 2**54, unit="D")
    assert_parity(fit_streaming_gee(spec, source_of(frame), batch_rows=17), fit_xtgee(spec, frame))


@pytest.mark.parametrize("corr", CORRELATIONS)
def test_no_time_refuses_before_read(corr, monkeypatch):
    source = source_of(panel_data())
    monkeypatch.setattr(source, "iter_batches", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Invalid timed spec must be early.")))
    with pytest.raises(AnalysisError) as error:
        fit_streaming_gee(specification(corr).model_copy(update={"time": None}), source)
    assert error.value.code == "invalid_spec"


@pytest.mark.parametrize("corr", ["stationary", "nonstationary", "unstructured"])
def test_patterned_joint_guard_before_fetch_or_matrix_allocation(corr, monkeypatch, tmp_path):
    frame = pd.concat((panel_data(), panel_data()), ignore_index=True)
    frame.g = frame.index % 3
    frame.t = frame.groupby("g").cumcount()
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    def forbidden(*args, **kwargs):
        raise AssertionError("Panel tensors must never be fetched before their structural reservation.")
    monkeypatch.setattr(OrderedPanels, "panels", forbidden)
    with use_workspace_budget(16), pytest.raises(AnalysisError) as error:
        fit_streaming_gee(specification(corr), source_of(frame), batch_rows=17)
    assert error.value.code == "workspace_limit"
    assert "patterned_correlation" in str(error.value)
    assert not list(tmp_path.iterdir())


def test_ar1_long_panels_never_fetch_whole_group_or_import_scipy(monkeypatch):
    frame = panel_data()
    frame.g = frame.index % 3
    frame.t = frame.groupby("g").cumcount()
    monkeypatch.setattr(OrderedPanels, "panels", lambda *a: (_ for _ in ()).throw(AssertionError("AR1 must use SQL row neighbours.")))
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise AssertionError("GEE must be native.")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    actual = fit_streaming_gee(specification("ar1"), source_of(frame), batch_rows=17)
    assert actual.provenance["solver_diagnostics"]["maximum_guarded_panel_rows"] == 0
    assert actual.provenance["streaming"]["actual_numeric_peak_rows"] <= 17
    assert "working_correlation" not in actual.extra
    assert_parity(actual, fit_xtgee(specification("ar1"), frame))


@pytest.mark.parametrize("failure", ["duplicate", "all_short", "source_changed", "non_pd"])
def test_ordered_failures_cleanup(failure, monkeypatch, tmp_path):
    frame = panel_data()
    spec = specification("unstructured")
    code = "repeated_time_values"
    if failure == "duplicate":
        frame = pd.concat((frame, frame.iloc[:1]), ignore_index=True)
    elif failure == "all_short":
        spec = specification("stationary", corr_order=20)
        code = "insufficient_panel_length"
    elif failure == "non_pd":
        frame = frame[frame.g < 3].copy()
        code = "working_correlation_not_pd"
    else:
        code = "source_changed"
    calls = 0
    def reader():
        nonlocal calls
        calls += 1
        current = frame.copy()
        if failure == "source_changed" and calls >= 10:
            current.loc[current.index[2], "y"] += .2
        for start in range(0, len(current), 11):
            yield current.iloc[start:start + 11]
    source = Dataset.from_batches(reader, frame.columns.tolist(), row_count=len(frame))
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    with pytest.raises(AnalysisError) as error:
        fit_streaming_gee(spec, source, batch_rows=17)
    assert error.value.code == code
    assert not list(tmp_path.iterdir())


def test_weight_covariance_and_option_rejections_before_source_read(monkeypatch):
    source = source_of(panel_data())
    monkeypatch.setattr(source, "iter_batches", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Invalid options are early.")))
    base = specification("ar1")
    for changes in [{"weights": "x", "weight_type": "pweight"}, {"covariance": "cluster"},
                    {"options": {"corr": "ar1", "corr_order": 2}}]:
        with pytest.raises(AnalysisError):
            fit_streaming_gee(base.model_copy(update=changes), source)
    assert importlib.import_module("openecon.econometrics.streaming_gee_ordered") is not None


@pytest.mark.parametrize("corr", CORRELATIONS)
@pytest.mark.parametrize("rows,intercept", [(3, True), (29, False)])
def test_reader_partition_numeric_partition_and_no_constant(corr, rows, intercept):
    frame = panel_data()
    spec = specification(corr).model_copy(update={"intercept": intercept})
    actual = fit_streaming_gee(spec, source_of(frame, reader_rows=31), batch_rows=rows)
    assert_parity(actual, fit_xtgee(spec, frame), rows=max(rows, 31))
    assert actual.provenance["streaming"]["actual_numeric_peak_rows"] <= (rows if corr == "ar1" else max(rows, 6))
    expected = frame.iloc[:min(400, len(frame))]
    assert [record["row"] for record in actual.predictions] == expected.index.tolist()
    np.testing.assert_allclose([record["observed"] for record in actual.predictions], expected.y)


@pytest.mark.parametrize("corr", ["independent", "exchangeable"])
@pytest.mark.parametrize("time_kind", ["fractional", "text"])
def test_nontimed_correlations_allow_dense_time_labels(corr, time_kind):
    frame = panel_data()
    frame.t = frame.t / 2 if time_kind == "fractional" else frame.t.map(lambda value: f"period-{value}")
    spec = specification(corr)
    assert_parity(fit_streaming_gee(spec, source_of(frame), batch_rows=17), fit_xtgee(spec, frame))


@pytest.mark.parametrize("corr", CORRELATIONS)
@pytest.mark.parametrize("family,link", [("binomial", "probit"), ("gaussian", "reciprocal"),
                                         ("gamma", "inverse_squared"), ("poisson", "identity")])
def test_noncanonical_links_stay_exact_with_ordered_correlations(corr, family, link):
    frame = panel_data(family)
    if family == "gaussian":
        frame.y = .4 + .05 * np.tanh(frame.x) + np.random.default_rng(831).normal(0, .01, len(frame))
    frame.x *= .02
    spec = specification(corr, family, link=link)
    assert_parity(fit_streaming_gee(spec, source_of(frame), batch_rows=17), fit_xtgee(spec, frame))


@pytest.mark.parametrize("order", [0, -1, True, 1.5, None])
def test_order_domain_explicit_before_source_read(order, monkeypatch):
    source = source_of(panel_data())
    monkeypatch.setattr(source, "iter_batches", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Order validation must be early.")))
    spec = specification("stationary").model_copy(update={"options": {"corr": "stationary", "corr_order": order}})
    with pytest.raises(AnalysisError) as error:
        fit_streaming_gee(spec, source)
    assert error.value.code == "invalid_spec"


@pytest.mark.parametrize("change", [{"family": "binomial", "link": "reciprocal"},
    {"scale": 0}, {"scale": -1}, {"scale": True}, {"scale": float("inf")},
    {"scale": "other"}, {"nbk": 1.2, "family": "gaussian"}])
def test_link_scale_nbk_domain_before_source_read(change, monkeypatch):
    source = source_of(panel_data())
    monkeypatch.setattr(source, "iter_batches", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Domain validation must be early.")))
    base = specification("ar1")
    spec = base.model_copy(update={"options": {**base.options, **change}})
    with pytest.raises(AnalysisError) as error:
        fit_streaming_gee(spec, source)
    assert error.value.code == "invalid_spec"
