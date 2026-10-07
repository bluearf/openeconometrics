"""Period-specific QR and independent coefficient/HAC native replay oracles."""
import numpy as np
import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.panel.fmb import fit_xtfmb
from openecon.econometrics.streaming_fmb import _PeriodFactors, fit_streaming_fmb
from openecon.models import ModelSpec
from openecon.resources import use_workspace_budget

from test_econ_streaming_linear import assert_parity


def fmb_data(n=37, periods=11, *, seed=69113):
    rng = np.random.default_rng(seed)
    result = pd.DataFrame({"panel": np.tile(np.arange(n), periods),
                           "time": np.repeat(np.arange(periods)*3, n),
                           "x": rng.normal(size=n*periods), "z": rng.normal(size=n*periods),
                           "cat": pd.Categorical(np.tile(np.arange(n)%3, periods), categories=[0, 1, 2, 3])})
    result["y"] = .6+(1.1+.04*result.time)*result.x-.3*result.z+.2*(result["cat"].astype(int) == 1)+rng.normal(size=len(result))
    return result.sample(frac=1, random_state=21).reset_index(drop=True)


def fmb_spec(covariance="nonrobust", lags=None, *, categorical=False, missing="raise"):
    return ModelSpec(estimator="xtfmb", outcome="y", predictors=["x", "z", *(["cat"] if categorical else [])],
                     panel="panel", time="time", categorical=["cat"] if categorical else [],
                     covariance=covariance, missing=missing,
                     options={"lags": lags} if covariance == "hac" and lags is not None else {})


@pytest.mark.parametrize("covariance,lags", [("nonrobust", None), ("hac", 0), ("hac", 4), ("hac", None)])
@pytest.mark.parametrize("categorical", [False, True])
def test_shuffled_period_global_rank_and_covariance_dense_parity(covariance, lags, categorical):
    data = fmb_data()
    spec = fmb_spec(covariance, lags, categorical=categorical)
    dense = fit_xtfmb(spec, data)
    result = fit_streaming_fmb(spec, Dataset.from_frame(data), batch_rows=17)
    assert_parity(dense, result, rtol=4e-9, atol=3e-10)
    np.testing.assert_allclose(result.extra["period_estimates_sd"], dense.extra["period_estimates_sd"], rtol=2e-9)
    assert result.inference["period_factor_spill"]["period_estimates_materialized"] is False
    assert result.provenance["streaming"]["dense_observation_matrix"] is False
    if covariance == "hac":
        assert result.inference["HAC_spill"]["hac_rows"] == 11
        assert result.inference["HAC_resource_plan"]["estimated_workspace_bytes"] <= 512*1024**2


def numpy_fmb(data, lags):
    estimates = []
    axis = sorted(data.time.unique())
    for t in axis:
        block = data[data.time == t]
        x = block[["x", "z"]].to_numpy()
        means = x.mean(0)
        design = np.column_stack([np.ones(len(block)), x-means])
        beta = np.linalg.lstsq(design, block.y.to_numpy(), rcond=None)[0]
        beta[0] -= means@beta[1:]
        estimates.append(beta)
    estimates = np.array(estimates)
    beta = estimates.mean(0)
    errors = estimates-beta
    meat = errors.T@errors
    for i, t in enumerate(axis):
        for j in range(i):
            distance = t-axis[j]
            if 0 < distance <= lags:
                cross = np.outer(errors[i], errors[j])*(1-distance/(lags+1))
                meat += cross+cross.T
    return beta, meat/(len(axis)*(len(axis)-1)), estimates.std(0, ddof=1)


@pytest.mark.parametrize("lags", [0, 1, 4, 20])
def test_independent_numpy_period_fit_and_actual_gap_hac_oracle(lags):
    data = fmb_data(n=47, periods=15)
    spec = fmb_spec("hac", lags)
    result = fit_streaming_fmb(spec, Dataset.from_frame(data), batch_rows=17)
    beta, covariance, sd = numpy_fmb(data, lags)
    np.testing.assert_allclose([c.estimate for c in result.coefficients], beta, rtol=3e-11, atol=3e-12)
    np.testing.assert_allclose(result.covariance_matrix, covariance, rtol=4e-10, atol=3e-12)
    np.testing.assert_allclose(result.extra["period_estimates_sd"], sd, rtol=3e-11, atol=3e-12)


@pytest.mark.parametrize("time_kind", ["large_integer", "unsigned", "datetime", "tz_datetime"])
def test_exact_period_keys_and_datetime_rank(time_kind):
    data = fmb_data()
    if time_kind == "large_integer":
        data["time"] += 2**53+17
    elif time_kind == "unsigned":
        data["time"] = data.time.astype("uint64")
    elif time_kind == "datetime":
        data["time"] = pd.to_datetime("2021-01-01")+pd.to_timedelta(data.time, unit="D")
    else:
        data["time"] = pd.to_datetime("2021-01-01", utc=True)+pd.to_timedelta(data.time, unit="D")
    spec = fmb_spec("hac", 4)
    result = fit_streaming_fmb(spec, Dataset.from_frame(data), batch_rows=17)
    dense = fit_xtfmb(spec, data)
    assert_parity(dense, result, rtol=5e-9, atol=3e-10)


def test_batch_invariance_missing_and_physical_prediction_positions():
    data = fmb_data(n=47, periods=15)
    data.loc[[3, 14], "x"] = np.nan
    spec = fmb_spec("hac", 4, missing="drop")
    first = fit_streaming_fmb(spec, Dataset.from_frame(data), batch_rows=1)
    other = fit_streaming_fmb(spec, Dataset.from_frame(data), batch_rows=17)
    np.testing.assert_allclose(first.covariance_matrix, other.covariance_matrix, atol=3e-11)
    assert first.provenance["sample_positions_hash"] == other.provenance["sample_positions_hash"]
    assert other.dropped_rows == 2
    assert len(other.predictions) == 400
    assert [row["row"] for row in other.predictions][:15] == [i for i in range(17) if i not in {3, 14}]


def test_period_cache_spills_and_cleans_success_and_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(_PeriodFactors, "cache_bytes", 400)
    data = fmb_data()
    result = fit_streaming_fmb(fmb_spec(), Dataset.from_frame(data), batch_rows=17)
    assert result.inference["period_factor_spill"]["peak_period_cache_bytes"] <= 400
    assert not list(tmp_path.iterdir())
    data.loc[0, ["panel", "time"]] = data.loc[1, ["panel", "time"]]
    with pytest.raises(AnalysisError) as error:
        fit_streaming_fmb(fmb_spec(), Dataset.from_frame(data), batch_rows=17)
    assert error.value.code == "repeated_time_values"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("defect,code", [("one_period", "insufficient_periods"),
                          ("few_rows", "insufficient_observations"), ("rank", "singular_design"),
                          ("constant", "constant_outcome"), ("exact", "perfect_fit"),
                          ("fractional", "invalid_time"), ("out_of_int64", "invalid_time")])
def test_structured_statistical_and_axis_guards(defect, code):
    data = fmb_data()
    if defect == "one_period":
        data = data[data.time == data.time.min()].copy()
    elif defect == "few_rows":
        first = data.time == data.time.min()
        data = pd.concat([data[~first], data[first].head(3)], ignore_index=True)
    elif defect == "rank":
        data.loc[data.time == data.time.min(), "z"] = data.loc[data.time == data.time.min(), "x"]
    elif defect == "constant":
        data["y"] = 7.
    elif defect == "exact":
        data["y"] = 2+data.x-.4*data.z
    elif defect == "fractional":
        data["time"] = data.time+.1
    else:
        data["time"] = data.time.astype("uint64")+np.uint64(2**63)
    with pytest.raises(AnalysisError) as error:
        fit_streaming_fmb(fmb_spec(), Dataset.from_frame(data), batch_rows=17)
    assert error.value.code == code


def test_lag_state_and_shared_buffers_budget_checked_before_allocation(monkeypatch):
    data = fmb_data()
    with use_workspace_budget(16):
        with pytest.raises(AnalysisError) as error:
            fit_streaming_fmb(fmb_spec("hac", 10_000_000), Dataset.from_frame(data), batch_rows=17)
    assert error.value.code == "workspace_limit"
    assert error.value.resource_plan["buffers"]["lag_weights"] == 80_000_000


@pytest.mark.parametrize("lags", [0, 4])
def test_large_offsets_independent_centered_period_oracle(lags):
    data = fmb_data(n=47, periods=15)
    data["x"] += 2**30
    data["z"] -= 2**27
    data["y"] += 2**29
    result = fit_streaming_fmb(fmb_spec("hac", lags), Dataset.from_frame(data), batch_rows=17)
    # Center in exact represented-input differences before each QR. A direct
    # raw-coordinate least-squares matrix is not an independent stable oracle.
    centered = data.copy()
    anchors = centered[["x", "z", "y"]].iloc[0].to_numpy()
    centered[["x", "z", "y"]] = centered[["x", "z", "y"]]-anchors
    beta, covariance, sd = numpy_fmb(centered, lags)
    transform = np.eye(3)
    transform[0, 1:] = -anchors[:2]
    expected_beta = transform@beta
    expected_beta[0] += anchors[2]
    np.testing.assert_allclose([c.estimate for c in result.coefficients], expected_beta, rtol=3e-11, atol=3e-12)
    np.testing.assert_allclose(result.covariance_matrix, transform@covariance@transform.T, rtol=4e-10, atol=3e-12)
    np.testing.assert_allclose(result.extra["period_estimates_sd"][1:], sd[1:], rtol=3e-11)


def test_changed_source_cleans_period_spill(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    data = fmb_data()
    calls = 0

    def reader():
        nonlocal calls
        calls += 1
        changed = data.copy()
        if calls >= 5:
            changed.loc[3, "y"] += 1
        yield changed

    source = Dataset.from_batches(reader, data.columns.tolist(), row_count=len(data))
    with pytest.raises(AnalysisError, match="changed"):
        fit_streaming_fmb(fmb_spec(), source, batch_rows=17)
    assert calls >= 5
    assert not list(tmp_path.iterdir())


def test_missing_scratch_directory_is_structured(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path/"absent"))
    with pytest.raises(AnalysisError) as error:
        fit_streaming_fmb(fmb_spec(), Dataset.from_frame(fmb_data()), batch_rows=17)
    assert error.value.code == "period_factor_spill_failed"


def test_actual_parquet_and_more_periods_than_factor_cache(tmp_path, monkeypatch):
    from openecon.dataset import scan
    data = fmb_data(n=8, periods=211)
    path = tmp_path/"periods.parquet"
    data.to_parquet(path, row_group_size=53)
    monkeypatch.setattr(_PeriodFactors, "cache_bytes", 400)
    result = fit_streaming_fmb(fmb_spec("hac", 7), scan(path), batch_rows=17)
    beta, covariance, _ = numpy_fmb(data, 7)
    np.testing.assert_allclose([c.estimate for c in result.coefficients], beta, atol=3e-12)
    np.testing.assert_allclose(result.covariance_matrix, covariance, atol=3e-12)
    assert result.inference["period_factor_spill"]["peak_period_cache_bytes"] <= 400
    assert result.nobs == 1688
