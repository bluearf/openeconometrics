"""Actual public parity across every shipped estimator and covariance."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.dataset import Dataset


CASES = [("ols", name) for name in ("nonrobust", "HC1", "HC3", "cluster")] + [
    (name, covariance) for name in ("logit", "probit") for covariance in ("nonrobust", "cluster")
]


@pytest.fixture
def fixture():
    rng = np.random.default_rng(18431)
    n = 321
    x, z = rng.normal(size=(2, n))
    group = np.arange(n) % 37
    category = np.asarray(["Batı", "Doğu", "Kuzey"])[np.arange(n) % 3]
    eta = .2 + .45 * x - .3 * z + .12 * (category == "Doğu")
    return pd.DataFrame({"x": x, "z": z, "category": category, "group": group,
                         "y": 1.5 + .7 * x - .2 * z + .3 * (category == "Doğu") + rng.normal(size=n),
                         "binary": rng.binomial(1, 1 / (1 + np.exp(-eta))),
                         "irrelevant": ["unprojected"] * n})


def compare(actual, expected):
    assert (actual.nobs, actual.nobs_original, actual.dropped_rows) == (
        expected.nobs, expected.nobs_original, expected.dropped_rows)
    assert [c.term for c in actual.coefficients] == [c.term for c in expected.coefficients]
    assert_allclose([[c.estimate, c.std_error, c.p_value, c.ci_low, c.ci_high] for c in actual.coefficients],
                    [[c.estimate, c.std_error, c.p_value, c.ci_low, c.ci_high] for c in expected.coefficients],
                    rtol=2e-8, atol=2e-9)
    assert_allclose(actual.covariance_matrix, expected.covariance_matrix, rtol=2e-8, atol=2e-9)
    assert actual.inference == expected.inference
    for name in ("r_squared", "adjusted_r_squared", "pseudo_r_squared", "aic", "bic",
                 "log_likelihood", "rmse", "df_resid"):
        assert actual.metrics[name] == pytest.approx(expected.metrics[name], rel=2e-8, abs=2e-9)
    assert actual.provenance["categorical_encoding"] == expected.provenance["categorical_encoding"]
    assert actual.sample_positions == []
    assert len(actual.predictions) <= 400
    assert actual.provenance["streaming"]["row_limit"] is None
    assert "\\toprule" in actual.to_latex()
    assert len(actual.model_dump_json().encode()) < 150_000


@pytest.mark.parametrize(("estimator", "covariance"), CASES)
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("chunk", [7, 63])
def test_every_current_model_streams_categories_and_inference(fixture, monkeypatch, estimator, covariance, intercept, chunk):
    monkeypatch.setattr("openecon.streaming_analysis.MAX_BATCH_ROWS", chunk)
    monkeypatch.setattr("openecon.linear_ols.streaming._READER_ROWS", chunk)
    options = {"y": "y" if estimator == "ols" else "binary", "x": ["x", "z", "category"],
               "categorical": ["category"], "covariance": covariance, "intercept": intercept}
    if covariance == "cluster":
        options["cluster"] = "group"
    function = getattr(oe, estimator)
    compare(function(data=Dataset.from_frame(fixture), **options), function(data=fixture, **options))


@pytest.mark.parametrize(("estimator", "covariance"), CASES)
@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_projected_files_and_missing_cluster_rows(fixture, tmp_path, monkeypatch, estimator, covariance, format):
    monkeypatch.setattr("openecon.streaming_analysis.MAX_BATCH_ROWS", 31)
    monkeypatch.setattr("openecon.linear_ols.streaming._READER_ROWS", 31)
    fixture.loc[::23, "x"] = np.nan
    if covariance == "cluster":
        fixture.loc[::17, "group"] = np.nan
    options = {"y": "y" if estimator == "ols" else "binary", "x": ["x", "category"],
               "categorical": ["category"], "covariance": covariance, "missing": "drop"}
    if covariance == "cluster":
        options["cluster"] = "group"
    path = tmp_path / f"data.{format}"
    if format == "csv":
        fixture.to_csv(path, index=False)
    else:
        fixture.to_parquet(path, index=False, row_group_size=43)
    function = getattr(oe, estimator)
    result = function(data=oe.scan(path), **options)
    compare(result, function(data=fixture, **options))
    columns = [options["y"], "x", "category"] + (["group"] if covariance == "cluster" else [])
    positions = np.flatnonzero(fixture[columns].notna().all(axis=1)).tolist()
    assert [p["row"] for p in result.predictions] == positions[:400]


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_explicit_binary_source_bypasses_dense_design_and_full_lp(fixture, monkeypatch, estimator):
    def forbidden(*args, **kwargs):
        raise AssertionError("An explicit source must not materialize the full design or separation LP")
    monkeypatch.setattr("openecon.analysis._prepare_data", forbidden)
    monkeypatch.setattr("openecon.analysis._frame_hasher", forbidden)
    monkeypatch.setattr("openecon.analysis._check_binary_separation", forbidden)
    result = getattr(oe, estimator)(data=Dataset.from_frame(fixture), y="binary", x=["x", "z"])
    assert result.nobs == len(fixture)
    assert result.provenance["optimizer"]["converged"] is True
    assert result.provenance["streaming"]["passes"] > 3


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_large_binary_frame_automatically_streams(fixture, monkeypatch, estimator):
    function = getattr(oe, estimator)
    dense = function(data=fixture, y="binary", x=["x", "z"])
    monkeypatch.setattr("openecon.analysis._MAX_DESIGN_BYTES", 1000)
    compare(function(data=fixture, y="binary", x=["x", "z"]), dense)


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_changed_binary_source_is_rejected_even_with_same_counts(fixture, estimator):
    passes = 0
    def factory():
        nonlocal passes
        passes += 1
        frame = fixture[["binary", "x", "z"]].copy()
        if passes > 1:
            frame.loc[0, "x"] += 1
        yield frame
    source = Dataset.from_batches(factory, ["binary", "x", "z"], row_count=len(fixture))
    with pytest.raises(AnalysisError) as exc:
        getattr(oe, estimator)(data=source, y="binary", x=["x", "z"])
    assert exc.value.code == "source_changed"


@pytest.mark.parametrize("estimator,covariance", CASES)
def test_large_object_categories_route_before_dense_preparation(fixture, estimator, covariance, monkeypatch):
    import openecon.analysis as analysis
    monkeypatch.setattr(analysis, "_MAX_DESIGN_BYTES", 1000)
    def forbidden(*args, **kwargs):
        pytest.fail("Large categorical frames must route before full dense sample preparation.")
    monkeypatch.setattr(analysis, "_prepare_data", forbidden)
    options = {"y": "y" if estimator == "ols" else "binary", "x": ["x", "z", "category"],
               "categorical": ["category"], "covariance": covariance}
    if covariance == "cluster":
        options["cluster"] = "group"
    result = getattr(oe, estimator)(data=fixture, **options)
    assert result.provenance["streaming"]["passes"] >= 4


@pytest.mark.parametrize("estimator", ["ols", "logit", "probit"])
def test_cluster_scores_are_aggregated_only_for_covariance(fixture, estimator, monkeypatch):
    from openecon.engines.streaming_groups import ClusterAccumulator
    passes = 0
    encoded_passes = []
    original = ClusterAccumulator.add
    def factory():
        nonlocal passes
        passes += 1
        yield fixture
    def encode(accumulator, keys, scores):
        encoded_passes.append(passes)
        return original(accumulator, keys, scores)
    monkeypatch.setattr(ClusterAccumulator, "add", encode)
    source = Dataset.from_batches(factory, fixture.columns.tolist(), row_count=len(fixture))
    result = getattr(oe, estimator)(data=source, y="y" if estimator == "ols" else "binary",
                                   x=["x", "z"], covariance="cluster", cluster="group")
    assert encoded_passes and set(encoded_passes) == {passes}
    assert result.inference["cluster_count"] == 37
    assert passes == result.provenance["streaming"]["passes"]


@pytest.mark.parametrize("estimator", ["ols", "logit", "probit"])
def test_cluster_mutation_is_checked_on_numeric_only_passes(fixture, estimator):
    passes = 0
    def factory():
        nonlocal passes
        passes += 1
        frame = fixture.copy()
        if passes > 1:
            frame.loc[0, "group"] = 10000
        yield frame
    source = Dataset.from_batches(factory, fixture.columns.tolist(), row_count=len(fixture))
    with pytest.raises(AnalysisError) as exc:
        getattr(oe, estimator)(data=source, y="y" if estimator == "ols" else "binary",
                               x=["x", "z"], covariance="cluster", cluster="group")
    assert exc.value.code == "source_changed"


def test_capabilities_describe_real_coverage():
    from openecon.econometrics import registry
    metadata = oe.capabilities()["streaming"]
    assert metadata["estimators"][:3] == ["ols", "logit", "probit"]
    assert len(metadata["estimators"]) == 80
    assert set(metadata["estimators"]) == {entry.name for entry in registry.all_estimators()} - set(oe.capabilities()["eager_only_estimators"])
    assert metadata["row_limit"] is None
    assert "hundred_billion_rows_hardware_validated" not in metadata
    assert metadata["devices"]["likelihood"] == ["cpu"]
    assert metadata["numeric_predictors_only"] is False
    assert metadata["covariances_by_estimator"]["ols"] == [
        "nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster", "cluster_hc2", "cluster_hc3",
        "hac", "bootstrap", "jackknife",
    ]
    assert metadata["covariances_by_estimator"]["logit"] == ["nonrobust", "cluster"]
    json.dumps(metadata, allow_nan=False)
