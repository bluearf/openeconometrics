"""Independent NumPy / statsmodels oracles for native panel diagnostics."""
import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy.stats import chi2, f
from statsmodels.stats.sandwich_covariance import cov_cluster

import openecon as oe
from openecon.models import ResultBundle


def panel_data(seed=91, groups=25, periods=12, *, gaps=False, hetero=False, ar=0):
    rng = np.random.default_rng(seed)
    codes = np.repeat(np.arange(groups), periods)
    times = np.tile(np.arange(periods), groups)
    effects = rng.normal(size=groups)[codes]
    x = rng.normal(size=(len(codes), 2)) + .2 * effects[:, None]
    noise = rng.normal(size=(groups, periods))
    for t in range(1, periods):
        noise[:, t] += ar * noise[:, t - 1]
    if hetero:
        noise *= np.linspace(.3, 3, groups)[:, None]
    frame = pd.DataFrame({"id": codes, "t": times, "x1": x[:, 0], "x2": x[:, 1],
                          "y": 1 + effects + x @ [.6, -.4] + noise.ravel()})
    if gaps:
        frame = frame.drop(index=rng.choice(len(frame), size=len(frame) // 8, replace=False))
    return frame.reset_index(drop=True)


def serial_oracle(frame, columns=("x1", "x2"), delta=1):
    # pandas independently identifies differences; regression and CR1 use
    # statsmodels, not the implementation's tensor helpers.
    used = frame.dropna(subset=["y", *columns, "id", "t"]).copy()
    used["position"] = used.index
    used = used.sort_values(["id", "t"])
    grouped = used.groupby("id", sort=False)
    first = grouped[["y", *columns]].diff()
    valid = grouped.t.diff().eq(delta)
    first_fit = sm.OLS(first.loc[valid, "y"], first.loc[valid, list(columns)]).fit()
    residuals = pd.Series(np.nan, index=used.index)
    residuals.loc[valid] = first_fit.resid
    used["residuals"] = residuals
    lagged = used.groupby("id").residuals.shift(1)
    pair = valid & residuals.notna() & lagged.notna()
    second = sm.OLS(residuals[pair], lagged[pair]).fit()
    variance = cov_cluster(second, used.loc[pair, "id"], use_correction=True)[0, 0]
    correlation = second.params.iloc[0]
    statistic = (correlation + .5)**2 / variance
    groups = used.loc[pair, "id"].nunique()
    return {"statistic": statistic, "correlation": correlation, "std_error": np.sqrt(variance),
            "p_value": f.sf(statistic, 1, groups - 1), "nobs": int(pair.sum()),
            "nobs_fd": int(valid.sum()), "n_groups": groups,
            "sample_positions": used.loc[pair, "position"].tolist()}


def wald_oracle(frame, columns=("x1", "x2")):
    complete = frame.dropna(subset=["y", *columns, "id"]).copy()
    # Explicit unit dummies estimate the idiosyncratic FE residuals.
    dummies = pd.get_dummies(complete.id, dtype=float).to_numpy()
    design = np.column_stack([dummies, complete[list(columns)].to_numpy()])
    y = complete.y.to_numpy()
    residuals = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    pooled = np.var(residuals, ddof=0)
    components = []
    for group in complete.id.unique():
        e = residuals[complete.id.to_numpy() == group]
        variance = np.mean(e**2)
        sampling_variance = np.sum((e**2 - variance)**2) / (len(e) * (len(e) - 1))
        components.append((variance - pooled)**2 / sampling_variance)
    statistic = np.sum(components)
    return statistic, chi2.sf(statistic, len(components))


@pytest.mark.parametrize("gaps,hetero,ar", [(False, False, 0), (True, False, 0),
                                           (False, True, .4), (True, True, -.3)])
def test_xtserial_matches_two_independent_regressions_and_panel_cluster_f(gaps, hetero, ar):
    frame = panel_data(gaps=gaps, hetero=hetero, ar=ar)
    original = frame.copy(deep=True)
    actual = oe.xtserial(data=frame, y="y", x=["x1", "x2"], panel="id", time="t")
    expected = serial_oracle(frame)
    for key in ("statistic", "correlation", "std_error", "p_value"):
        assert_allclose(actual[key], expected[key], rtol=2e-10, atol=1e-12)
    for key in ("nobs", "nobs_fd", "n_groups", "sample_positions"):
        assert actual[key] == expected[key]
    assert actual["df"] == 1 and actual["df2"] == expected["n_groups"] - 1
    assert actual["provenance"]["stata_parity_validated"] is False
    assert json.loads(json.dumps(actual, allow_nan=False)) == actual
    pd.testing.assert_frame_equal(frame, original)


def test_serial_missing_differences_do_not_bridge_deleted_rows_or_gaps():
    frame = panel_data(gaps=True)
    frame.loc[[7, 20, 21, 63], "x1"] = np.nan
    frame.loc[[10, 35], "y"] = np.nan
    actual = oe.xtserial(data=frame, y="y", x=["x1", "x2"], panel="id", time="t", missing="drop")
    expected = serial_oracle(frame)
    assert actual["sample_positions"] == expected["sample_positions"]
    assert actual["nobs_fd"] == expected["nobs_fd"]
    assert_allclose(actual["statistic"], expected["statistic"], rtol=1e-10)
    assert any("missing" in text for text in actual["warnings"])
    with pytest.raises(oe.AnalysisError) as error:
        oe.xtserial(data=frame, y="y", x=["x1"], panel="id", time="t", missing="raise")
    assert error.value.code == "missing_values"


@pytest.mark.parametrize("increment", [2, 5])
def test_numeric_time_delta_preserves_gaps_and_original_row_positions(increment):
    frame = panel_data(gaps=True)
    frame.t = 2001 + frame.t * increment
    actual = oe.xtserial(data=frame, y="y", x=["x1", "x2"], panel="id", time="t", time_delta=increment)
    expected = serial_oracle(frame, delta=increment)
    assert actual["sample_positions"] == expected["sample_positions"]
    assert_allclose(actual["statistic"], expected["statistic"], rtol=1e-10)


@pytest.mark.parametrize("timezone,unit", [(None, "ns"), (None, "us"), ("Europe/Istanbul", "ns")])
def test_datetime_grid_detects_actual_missing_days_instead_of_ranking_dates(timezone, unit):
    frame = panel_data(gaps=True)
    expected = serial_oracle(frame)
    dates = pd.Timestamp("2024-01-01", tz=timezone) + pd.to_timedelta(frame.t, unit="D")
    frame.t = dates.dt.as_unit(unit)
    actual = oe.xtserial(data=frame, y="y", x=["x1", "x2"], panel="id", time="t", time_delta="1D")
    assert actual["sample_positions"] == expected["sample_positions"]
    assert_allclose(actual["statistic"], expected["statistic"], rtol=1e-10)
    with pytest.raises(oe.AnalysisError) as error:
        oe.xtserial(data=frame, y="y", x=["x1"], panel="id", time="t")
    assert error.value.code == "invalid_time_delta"


def test_serial_collinearity_time_invariant_columns_and_categorical_design():
    frame = panel_data()
    frame["duplicate"] = frame.x1 * 2
    frame["fixed"] = frame.id % 3
    frame["sector"] = pd.Categorical(np.where(frame.x2 > 0, "a", "b"))
    actual = oe.xtserial(data=frame, y="y", x=["x1", "duplicate", "fixed", "sector"],
                         categorical=["sector"], panel="id", time="t")
    coded = frame.assign(dummy=(frame.sector == "b").astype(float))
    expected = serial_oracle(coded, columns=("x1", "dummy"))
    assert actual["design_rank"] == 2
    assert set(actual["omitted_terms"]) == {"duplicate", "fixed"}
    assert_allclose(actual["statistic"], expected["statistic"], rtol=1e-10)


def test_serial_first_regression_retains_panels_too_short_for_the_second():
    frame = panel_data()
    frame = frame[(frame.id >= 4) | (frame.t < 2)].reset_index(drop=True)
    actual = oe.xtserial(data=frame, y="y", x=["x1", "x2"], panel="id", time="t")
    expected = serial_oracle(frame)
    assert actual["n_groups_input"] == 25 and actual["n_groups"] == 21
    assert actual["df2"] == 20 and actual["nobs_fd"] == expected["nobs_fd"]
    assert_allclose(actual["statistic"], expected["statistic"], rtol=1e-10)
    assert any("4 panel(s)" in text for text in actual["warnings"])


def test_serial_all_time_invariant_regressors_are_omitted_without_fabricating_an_intercept():
    frame = panel_data().assign(fixed=lambda d: d.id)
    actual = oe.xtserial(data=frame, y="y", x=["fixed"], panel="id", time="t")
    expected = serial_oracle(frame, columns=("fixed",))
    assert actual["design_rank"] == 0 and actual["omitted_terms"] == ["fixed"]
    assert_allclose(actual["statistic"], expected["statistic"], rtol=1e-10)


@pytest.mark.parametrize("gaps,hetero", [(False, False), (True, False), (False, True), (True, True)])
def test_xttest3_matches_dummy_variable_residuals_and_corrected_group_sampling_variance(gaps, hetero):
    frame = panel_data(gaps=gaps, hetero=hetero)
    original = frame.copy(deep=True)
    result = oe.xtreg(data=frame, y="y", x=["x1", "x2"], panel="id", time="t", model="fe")
    before = result.model_dump()
    actual = oe.xttest3(result, data=frame)
    expected_stat, expected_p = wald_oracle(frame)
    assert_allclose(actual["statistic"], expected_stat, rtol=3e-10)
    assert_allclose(actual["p_value"], expected_p, rtol=3e-10, atol=1e-30)
    assert actual["df"] == frame.id.nunique() and actual["distribution"] == "chi2"
    assert actual["provenance"]["variance_denominator"] == "T_i * (T_i - 1)"
    assert actual["provenance"]["stata_parity_validated"] is False
    assert json.loads(json.dumps(actual, allow_nan=False)) == actual
    assert result.model_dump() == before
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
def test_wald_rebuilds_categorical_fe_residuals_and_ignores_original_covariance(covariance):
    frame = panel_data()
    frame["duplicate"] = frame.x1 * 2
    frame["fixed"] = frame.id
    frame["sector"] = pd.Categorical(np.where(frame.x2 > 0, "a", "b"))
    options = {"cluster": ["id"]} if covariance == "cluster" else {}
    result = oe.xtreg(data=frame, y="y", x=["x1", "duplicate", "fixed", "sector"],
                      categorical=["sector"], panel="id", time="t", model="fe",
                      covariance=covariance, **options)
    coded = frame.assign(dummy=(frame.sector == "b").astype(float))
    actual = oe.xttest3(result, data=frame)
    assert_allclose(actual["statistic"], wald_oracle(coded, columns=("x1", "dummy"))[0], rtol=3e-10)


def test_wald_saved_result_complete_case_sample_and_changed_data_rejection():
    frame = panel_data(gaps=True)
    frame.loc[[8, 21, 66], "x1"] = np.nan
    result = oe.xtreg(data=frame, y="y", x=["x1", "x2"], panel="id", time="t", model="fe", missing="drop")
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    actual = oe.xttest3(restored, data=frame)
    assert_allclose(actual["statistic"], wald_oracle(frame)[0], rtol=3e-10)
    edited = frame.copy()
    edited.loc[0, "y"] += .1
    for wrong in [edited, frame.iloc[::-1], frame.drop(columns="x1")]:
        with pytest.raises(oe.AnalysisError) as error:
            oe.xttest3(restored, data=wrong)
        assert error.value.code == "data_mismatch"


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_diagnostics_do_not_change_when_outcome_units_are_extreme(scale):
    frame = panel_data(periods=16)
    serial = oe.xtserial(data=frame, y="y", x=["x1", "x2"], panel="id", time="t")
    result = oe.xtreg(data=frame, y="y", x=["x1", "x2"], panel="id", time="t", model="fe")
    wald = oe.xttest3(result, data=frame)
    rescaled = frame.copy()
    rescaled.y *= scale
    got = oe.xtserial(data=rescaled, y="y", x=["x1", "x2"], panel="id", time="t")
    fitted = oe.xtreg(data=rescaled, y="y", x=["x1", "x2"], panel="id", time="t", model="fe")
    assert_allclose(got["statistic"], serial["statistic"], rtol=1e-9)
    assert_allclose(oe.xttest3(fitted, data=rescaled)["statistic"], wald["statistic"], rtol=1e-9)


def test_row_reordering_preserves_test_statistics_after_refitting():
    frame = panel_data(gaps=True)
    shuffled = frame.sample(frac=1, random_state=13).reset_index(drop=True)
    a = oe.xtserial(data=frame, y="y", x=["x1", "x2"], panel="id", time="t")
    b = oe.xtserial(data=shuffled, y="y", x=["x1", "x2"], panel="id", time="t")
    assert_allclose(a["statistic"], b["statistic"], rtol=1e-10)
    results = [oe.xtreg(data=d, y="y", x=["x1", "x2"], panel="id", time="t", model="fe")
               for d in [frame, shuffled]]
    assert_allclose(oe.xttest3(results[0], data=frame)["statistic"],
                    oe.xttest3(results[1], data=shuffled)["statistic"], rtol=1e-10)
