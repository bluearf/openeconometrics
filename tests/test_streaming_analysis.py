"""Public out-of-core OLS parity, replay integrity and bounded result contracts."""
from __future__ import annotations

import json

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.data import DataError
from openecon.dataset import Dataset
from openecon.models import ModelSpec, ResultBundle


@pytest.fixture
def numerical():
    rng = np.random.default_rng(19037)
    x, z = rng.normal(size=(2, 513))
    return pd.DataFrame({"x": x, "z": z,
                         "y": 2.1 + 0.73 * x - 1.27 * z + rng.normal(size=513) * (1 + x**2),
                         "unused": ["not selected"] * 513})


@pytest.fixture
def chunks(monkeypatch):
    def set_size(size):
        monkeypatch.setattr("openecon.streaming_analysis.MAX_BATCH_ROWS", size)
        monkeypatch.setattr("openecon.linear_ols.streaming._READER_ROWS", size)
    return set_size


def _terms(result):
    return [[c.estimate, c.std_error, c.statistic, c.p_value, c.ci_low, c.ci_high]
            for c in result.coefficients]


def _check_parity(streamed, dense, *, rtol=2e-10, atol=2e-10):
    assert streamed.nobs == dense.nobs
    assert streamed.nobs_original == dense.nobs_original
    assert streamed.dropped_rows == dense.dropped_rows
    assert [c.term for c in streamed.coefficients] == [c.term for c in dense.coefficients]
    assert_allclose(_terms(streamed), _terms(dense), rtol=rtol, atol=atol)
    assert_allclose(streamed.covariance_matrix, dense.covariance_matrix, rtol=rtol, atol=atol)
    for key in ["r_squared", "adjusted_r_squared", "aic", "bic", "log_likelihood", "rmse", "df_resid"]:
        assert streamed.metrics[key] == pytest.approx(dense.metrics[key], rel=rtol, abs=atol)
    assert streamed.inference == dense.inference


@pytest.mark.parametrize("covariance", ["nonrobust", "HC1", "HC3"])
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("batch_rows", [7, 63])
def test_public_streaming_matches_dense_for_all_supported_covariances(
    numerical, chunks, covariance, intercept, batch_rows,
):
    chunks(batch_rows)
    options = {"y": "y", "x": ["x", "z"], "covariance": covariance, "intercept": intercept}
    dense = oe.ols(data=numerical, **options)
    streamed = oe.ols(data=Dataset.from_frame(numerical), **options)
    _check_parity(streamed, dense)
    assert streamed.provenance["solver"] == "torch_tsqr"
    assert streamed.provenance["streaming"]["passes"] == 3
    assert streamed.provenance["streaming"]["batch_rows"] == batch_rows


@pytest.mark.parametrize("covariance", ["nonrobust", "HC1", "HC3"])
@pytest.mark.parametrize("batch_rows", [7, 63])
def test_centering_handles_high_offsets_and_predictor_unit_scales(
    numerical, chunks, covariance, batch_rows,
):
    chunks(batch_rows)
    frame = numerical.copy()
    frame["x"] = 100_000_000 + frame.x
    frame["z"] *= 1e-9
    dense = oe.ols(data=frame, y="y", x=["x", "z"], covariance=covariance)
    streamed = oe.ols(data=Dataset.from_frame(frame), y="y", x=["x", "z"], covariance=covariance)
    # The large original-unit intercept amplifies cancellation. Predictions,
    # slope and inference parity still diagnose a material accuracy loss.
    _check_parity(streamed, dense, rtol=2e-7, atol=2e-7)
    assert_allclose([p["fitted"] for p in streamed.predictions],
                    [p["observed"] - p["residual"] for p in streamed.predictions],
                    rtol=1e-12, atol=1e-12)


def test_missing_drop_retains_original_physical_positions_and_bounded_predictions(chunks):
    chunks(7)
    rng = np.random.default_rng(1738)
    n = 913
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    frame["y"] = 3 + frame.x * 0.7 - frame.z * 0.2 + rng.normal(size=n)
    frame.loc[::13, "x"] = np.nan
    frame.loc[::97, "y"] = np.nan
    frame.index = ["duplicate label"] * n
    options = {"y": "y", "x": ["x", "z"], "missing": "drop", "covariance": "HC3"}
    dense = oe.ols(data=frame, **options)
    streamed = oe.ols(data=Dataset.from_frame(frame), **options)
    _check_parity(streamed, dense)
    retained = np.flatnonzero(frame[["y", "x", "z"]].notna().all(axis=1))
    assert [p["row"] for p in streamed.predictions] == retained[:400].tolist()
    assert streamed.sample_positions == []
    assert len(streamed.predictions) == 400
    assert streamed.nobs == len(retained)
    assert streamed.nobs_original == n
    assert streamed.dropped_rows == n - len(retained)
    assert streamed.provenance["sample_position_count"] == len(retained)
    assert streamed.provenance["sample_positions_omitted"] is True
    beta = np.array([c.estimate for c in dense.coefficients])
    expected = np.column_stack([np.ones(400), frame.iloc[retained[:400]][["x", "z"]]]) @ beta
    assert_allclose([p["fitted"] for p in streamed.predictions], expected, atol=2e-12)


def test_missing_raise_does_not_silently_drop(numerical):
    numerical.loc[5, "x"] = np.nan
    with pytest.raises(AnalysisError) as exc:
        oe.ols(data=Dataset.from_frame(numerical), y="y", x=["x", "z"], missing="raise")
    assert exc.value.code == "missing_values"


def test_result_json_and_provenance_stay_bounded_in_total_rows(numerical):
    frame = pd.concat([numerical] * 20, ignore_index=True)
    result = oe.ols(data=Dataset.from_frame(frame), y="y", x=["x", "z"])
    encoded = result.model_dump_json()
    assert len(encoded.encode("utf-8")) < 100_000
    assert result.sample_positions == []
    assert len(result.predictions) <= 400
    assert result.nobs == len(frame)
    assert isinstance(json.loads(encoded)["nobs"], int)
    ResultBundle.model_validate_json(encoded)
    assert "\\begin{tabular}" in result.to_latex()
    assert str(len(frame)) in result.summary()


def test_explicit_source_never_calls_dense_preparation_or_full_frame_hashing(numerical, monkeypatch):
    import openecon.analysis as analysis
    source = Dataset.from_frame(numerical)

    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit streamed OLS must bypass dense full-dataset operations.")

    monkeypatch.setattr(analysis, "_prepare_data", forbidden)
    monkeypatch.setattr(analysis, "_frame_hasher", forbidden)
    monkeypatch.setattr(analysis, "_hash_frame", forbidden)
    result = oe.ols(data=source, y="y", x=["x", "z"])
    assert result.nobs == len(numerical)
    assert result.provenance["solver"] == "torch_tsqr"


def test_large_inmemory_numeric_ols_routes_automatically(numerical, monkeypatch):
    import openecon.analysis as analysis
    dense = oe.ols(data=numerical, y="y", x=["x", "z"])
    monkeypatch.setattr(analysis, "_MAX_DESIGN_BYTES", 1000)
    streamed = oe.ols(data=numerical, y="y", x=["x", "z"])
    _check_parity(streamed, dense)
    assert streamed.provenance["solver"] == "torch_tsqr"


def test_large_categorical_model_routes_to_bounded_design(numerical, monkeypatch):
    import openecon.analysis as analysis
    numerical["category"] = pd.Categorical(["a", "b", "c"] * 171)
    options = {"y": "y", "x": ["x", "category"], "categorical": ["category"]}
    dense = oe.ols(data=numerical, **options)
    monkeypatch.setattr(analysis, "_MAX_DESIGN_BYTES", 1000)
    streamed = oe.ols(data=numerical, **options)
    _check_parity(streamed, dense)
    assert streamed.provenance["streaming"]["passes"] == 4


@pytest.mark.parametrize("change", ["values", "order", "missing_selection"])
def test_changed_factory_with_same_total_count_is_rejected(numerical, chunks, change):
    chunks(63)
    calls = []

    def factory():
        calls.append("pass")
        frame = numerical.loc[:, ["y", "x", "z"]].copy()
        if len(calls) > 1:
            if change == "values":
                frame.loc[0, "y"] += 1
            elif change == "order":
                frame = frame.iloc[::-1]
            else:
                frame.loc[0, "x"] = np.nan
        yield frame

    source = Dataset.from_batches(factory, ["y", "x", "z"], row_count=len(numerical))
    with pytest.raises(AnalysisError) as exc:
        oe.ols(data=source, y="y", x=["x", "z"], missing="drop")
    assert exc.value.code == "source_changed"


def test_file_changed_before_fit_uses_source_changed_error(numerical, tmp_path):
    file = tmp_path / "data.parquet"
    numerical.to_parquet(file, index=False)
    source = oe.scan(file)
    numerical.to_parquet(file, index=False)
    with pytest.raises(DataError) as exc:
        oe.ols(data=source, y="y", x=["x", "z"])
    assert exc.value.code == "SOURCE_CHANGED"


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_scan_regression_exceeds_eager_import_limit_without_full_read(tmp_path, suffix, monkeypatch):
    import openecon.data as data
    rng = np.random.default_rng(388)
    count = 100_001
    frame = pd.DataFrame({"x": rng.normal(size=count), "z": rng.normal(size=count)})
    frame["y"] = 4 + frame.x * 1.3 - frame.z * 0.4 + rng.normal(size=count)
    file = tmp_path / f"large.{suffix}"
    if suffix == "csv":
        frame.to_csv(file, index=False)
    else:
        frame.to_parquet(file, index=False, row_group_size=8192)
    automatic = oe.read(file)
    assert isinstance(automatic, oe.Dataset)
    first = next(automatic.iter_batches(columns=["x", "y"], batch_rows=257))
    assert len(first) == 257 and list(first.columns) == ["x", "y"]

    def forbid_read(*args, **kwargs):
        raise AssertionError("The scan path must not call eager data.read.")

    monkeypatch.setattr(data, "read", forbid_read)
    result = oe.ols(data=oe.scan(file), y="y", x=["x", "z"], covariance="HC3")
    dense = oe.ols(data=frame, y="y", x=["x", "z"], covariance="HC3")
    _check_parity(result, dense, rtol=1e-9, atol=1e-9)
    assert result.nobs == count
    assert result.provenance["streaming"]["row_limit"] is None


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_binary_streaming_models_validate_the_binary_outcome(numerical, estimator):
    with pytest.raises(AnalysisError) as exc:
        oe.fit(ModelSpec(estimator=estimator, outcome="y", predictors=["x"],
                         covariance="nonrobust"), data=Dataset.from_frame(numerical))
    assert exc.value.code == "invalid_binary_outcome"


def test_streaming_cluster_and_categories_validate_identifiability(numerical):
    source = Dataset.from_frame(numerical)
    with pytest.raises(AnalysisError) as exc:
        oe.ols(data=source, y="y", x=["x"], cluster="unused")
    assert exc.value.code == "insufficient_clusters"
    result = oe.ols(data=source, y="y", x=["unused"], categorical=["unused"])
    assert [coefficient.term for coefficient in result.coefficients] == ["Intercept"]
    assert result.coefficients[0].estimate == pytest.approx(numerical.y.mean(), abs=1e-12)
    assert result.nobs == len(numerical)


def test_missing_column_error_is_actionable(numerical):
    with pytest.raises(AnalysisError) as exc:
        oe.ols(data=Dataset.from_frame(numerical), y="y", x=["absent"])
    assert exc.value.code == "missing_columns"
    assert "absent" in str(exc.value)


@pytest.mark.parametrize("invalid", [np.inf, -np.inf])
def test_streaming_nonfinite_input_is_rejected(numerical, invalid):
    numerical.loc[4, "x"] = invalid
    with pytest.raises(AnalysisError) as exc:
        oe.ols(data=Dataset.from_frame(numerical), y="y", x=["x", "z"])
    assert exc.value.code == "non_finite_values"


def test_batch_width_budget_rejects_too_many_parameters():
    frame = pd.DataFrame({f"x{i}": [1., 2., 3.] for i in range(384)})
    frame["y"] = [1., 2., 3.]
    with pytest.raises(AnalysisError) as exc:
        oe.ols(data=Dataset.from_frame(frame), y="y", x=[f"x{i}" for i in range(384)])
    assert exc.value.code == "model_too_wide"


def test_source_digest_is_same_for_different_batch_boundaries(numerical, chunks):
    chunks(7)
    first = oe.ols(data=Dataset.from_frame(numerical), y="y", x=["x", "z"])
    chunks(63)
    second = oe.ols(data=Dataset.from_frame(numerical), y="y", x=["x", "z"])
    assert first.provenance["data_hash"] == second.provenance["data_hash"]
    assert first.provenance["sample_hash"] == second.provenance["sample_hash"]
    assert first.provenance["sample_positions_hash"] == second.provenance["sample_positions_hash"]
    assert_allclose(first.covariance_matrix, second.covariance_matrix, rtol=2e-10, atol=2e-10)
