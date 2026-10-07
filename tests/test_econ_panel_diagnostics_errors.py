"""Panel diagnostic failures must be explicit and JSON-safe."""
import numpy as np
import pandas as pd
import pytest

import openecon as oe
from test_econ_panel_diagnostics import panel_data


def serial(frame, **options):
    return oe.xtserial(data=frame, y="y", x=["x1", "x2"], panel="id", time="t", **options)


@pytest.mark.parametrize("delta", [0, -1, True, 1.5, "1D"])
def test_bad_numeric_time_delta_is_refused(delta):
    with pytest.raises(oe.AnalysisError) as error:
        serial(panel_data(), time_delta=delta)
    assert error.value.code == "invalid_time_delta"


@pytest.mark.parametrize("mutation,code", [
    (lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True), "repeated_time_values"),
    (lambda d: d.drop(columns="t"), "missing_columns"),
    (lambda d: d.assign(t=d.t + .1), "invalid_time"),
    (lambda d: d.assign(t=True), "repeated_time_values"),
    (lambda d: d.assign(x1=np.inf), "non_finite_values"),
    (lambda d: d.assign(y=1.), "degenerate_residuals"),
])
def test_serial_input_validation(mutation, code):
    with pytest.raises(oe.AnalysisError) as error:
        serial(mutation(panel_data()))
    assert error.value.code == code


def test_first_difference_perfect_fit_and_too_short_panels_have_clear_errors():
    frame = panel_data()
    perfect = frame.assign(y=1 + frame.id + 2 * frame.x1 - frame.x2)
    with pytest.raises(oe.AnalysisError) as error:
        serial(perfect)
    assert error.value.code == "degenerate_residuals"
    short = frame.loc[frame.t < 2]
    with pytest.raises(oe.AnalysisError) as error:
        serial(short)
    assert error.value.code == "insufficient_panels"


def test_no_cross_panel_lags_and_one_qualifying_panel_is_insufficient():
    frame = panel_data()
    short = frame[(frame.id == 0) | (frame.t < 2)]
    with pytest.raises(oe.AnalysisError) as error:
        serial(short)
    assert error.value.code == "insufficient_panels"


def test_off_grid_time_is_not_rounded_into_adjacency():
    frame = panel_data().assign(t=lambda d: d.t * 2)
    frame.loc[8, "t"] += 1
    with pytest.raises(oe.AnalysisError) as error:
        serial(frame, time_delta=2)
    assert error.value.code == "invalid_time"


def test_large_int64_indices_preserve_exact_period_spacing():
    frame = panel_data().assign(t=lambda d: d.t.astype("int64") + 2**53)
    got = serial(frame)
    expected = serial(panel_data())
    assert got["sample_positions"] == expected["sample_positions"]
    assert got["statistic"] == pytest.approx(expected["statistic"], rel=1e-10)


@pytest.mark.parametrize("times", [lambda t: t.astype("uint64") + 2**63,
                                   lambda t: np.where(t == 0, -(2**63), t)])
def test_integer_times_cannot_wrap_during_origin_or_int64_conversion(times):
    frame = panel_data()
    frame["t"] = times(frame.t)
    with pytest.raises(oe.AnalysisError) as error:
        serial(frame)
    assert error.value.code == "invalid_time"


@pytest.mark.parametrize("delta", [0, True, "0D", "-1D", "not-a-duration"])
def test_datetime_delta_must_be_an_explicit_positive_fixed_duration(delta):
    frame = panel_data()
    frame.t = pd.Timestamp("2024-01-01") + pd.to_timedelta(frame.t, unit="D")
    with pytest.raises(oe.AnalysisError) as error:
        serial(frame, time_delta=delta)
    assert error.value.code == "invalid_time_delta"


def test_wald_refuses_non_fe_results_and_weights():
    frame = panel_data()
    pooled = oe.xtreg(data=frame, y="y", x=["x1", "x2"], panel="id", time="t", model="pooled")
    with pytest.raises(oe.AnalysisError) as error:
        oe.xttest3(pooled, data=frame)
    assert error.value.code == "unsupported_model"
    frame["w"] = frame.id + 1
    weighted = oe.xtreg(data=frame, y="y", x=["x1", "x2"], panel="id", time="t", model="fe",
                       weights="w", weight_type="aweight")
    with pytest.raises(oe.AnalysisError) as error:
        oe.xttest3(weighted, data=frame)
    assert error.value.code == "unsupported_weights"
    with pytest.raises(oe.AnalysisError) as error:
        oe.xttest3({}, data=frame)
    assert error.value.code == "invalid_result"


def test_wald_refuses_short_panels_instead_of_silently_skipping_variance_terms():
    frame = panel_data()
    frame = frame[(frame.id != 0) | (frame.t < 2)].reset_index(drop=True)
    result = oe.xtreg(data=frame, y="y", x=["x1", "x2"], panel="id", time="t", model="fe")
    with pytest.raises(oe.AnalysisError) as error:
        oe.xttest3(result, data=frame)
    assert error.value.code == "insufficient_panel_observations"


@pytest.mark.parametrize("term", ["Intercept", "x1"])
def test_wald_saved_coefficients_are_verified_not_just_data_hash(term):
    frame = panel_data()
    result = oe.xtreg(data=frame, y="y", x=["x1", "x2"], panel="id", time="t", model="fe")
    coefficient = next(c for c in result.coefficients if c.term == term)
    coefficient.estimate += .1
    with pytest.raises(oe.AnalysisError) as error:
        oe.xttest3(result, data=frame)
    assert error.value.code == "result_mismatch"


def test_wald_duplicate_stored_coefficients_are_not_silently_collapsed():
    frame = panel_data()
    result = oe.xtreg(data=frame, y="y", x=["x1", "x2"], panel="id", time="t", model="fe")
    result.coefficients.append(result.coefficients[0].model_copy(deep=True))
    with pytest.raises(oe.AnalysisError) as error:
        oe.xttest3(result, data=frame)
    assert error.value.code == "result_mismatch"


def test_wald_degenerate_squared_residuals_are_not_silently_skipped():
    # An orthogonal within design produces nonzero residuals of constant
    # magnitude in every panel, making the variance-of-variance undefined.
    frame = pd.DataFrame({"id": np.repeat(np.arange(5), 4), "t": np.tile(np.arange(4), 5),
                          "x1": np.tile([-1., 1., -1., 1.], 5),
                          "residual": np.tile([-1., -1., 1., 1.], 5)})
    frame["y"] = frame.id + .5 * frame.x1 + frame.residual
    result = oe.xtreg(data=frame, y="y", x=["x1"], panel="id", time="t", model="fe")
    with pytest.raises(oe.AnalysisError) as error:
        oe.xttest3(result, data=frame)
    assert error.value.code == "degenerate_group_variance"
