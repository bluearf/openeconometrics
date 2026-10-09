"""Independent KM covariance integration and causal RMST admission/persistence gates."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from statsmodels.duration.survfunc import SurvfuncRight

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.causal_design.common import causal_design_load, causal_design_save
from openecon.econometrics.causal_design.survival import treatment_rmst


def _data():
    return pd.DataFrame(
        {
            "time": [0, 1, 1, 2, 4, 6, 7, 0, 1, 2, 2, 3, 6, 7],
            "event": [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
            "treatment": [0] * 7 + [1] * 7,
        },
        index=["duplicate"] * 14,
    )


def _fit(data=None, **options):
    return treatment_rmst(
        _data() if data is None else data,
        "time",
        "event",
        "treatment",
        design="randomized",
        tau=options.pop("tau", 5.0),
        **options,
    )


def _km_covariance_integral(times, events, tau):
    """Independent statsmodels KM and double integral of its survival covariance.

    This integrates Cov(S(u),S(v)), rather than reproducing the implementation's
    event-tail-area variance sum. Greenwood sums come from independently computed
    statsmodels pointwise survival SEs at event times.
    """
    fit = SurvfuncRight(np.asarray(times, float), np.asarray(events, float))
    knots = np.unique(np.r_[0.0, fit.surv_times[fit.surv_times < tau], tau])
    widths = np.diff(knots)
    values, greenwood = [], []
    for left in knots[:-1]:
        at = np.flatnonzero(fit.surv_times <= left)
        if not len(at):
            values.append(1.0)
            greenwood.append(0.0)
        else:
            j = at[-1]
            value = float(fit.surv_prob[j])
            values.append(value)
            greenwood.append(float((fit.surv_prob_se[j] / value) ** 2) if value else 0.0)
    covariance = np.zeros((len(widths), len(widths)))
    for i in range(len(widths)):
        for j in range(len(widths)):
            covariance[i, j] = values[i] * values[j] * greenwood[min(i, j)]
    return float(widths @ values), float(widths @ covariance @ widths)


@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0, 3.5, 5.0, 7.0])
def test_independent_statsmodels_km_and_double_covariance_integral(tau):
    data = _data()
    result = _fit(data, tau=tau, level=0.9)
    estimates, variances = [], []
    for arm in (0, 1):
        group = data[data.treatment == arm]
        estimate, variance = _km_covariance_integral(group.time, group.event, tau)
        estimates.append(estimate)
        variances.append(variance)
    v0, v1 = variances
    expected = np.array([[v0, 0, -v0], [0, v1, v1], [-v0, v1, v0 + v1]])
    assert_allclose(result["covariance"], expected, atol=1e-13, rtol=1e-12)
    effects = result["effects"]
    assert_allclose(effects.estimate, [*estimates, estimates[1] - estimates[0]], atol=1e-13)
    assert_allclose(effects.std_error, np.sqrt(np.diag(expected)), atol=1e-13)
    nonzero = effects.std_error.to_numpy() > 0
    z = effects.estimate[nonzero] / effects.std_error[nonzero]
    assert_allclose(effects.z[nonzero], z, atol=1e-13)
    assert_allclose(effects.p_value[nonzero], 2 * stats.norm.sf(np.abs(z)), atol=1e-13)
    assert_allclose(
        effects.ci_low, effects.estimate - stats.norm.isf(0.05) * effects.std_error, atol=1e-12
    )
    assert_allclose(
        effects.ci_high, effects.estimate + stats.norm.isf(0.05) * effects.std_error, atol=1e-12
    )
    assert effects.df.isna().all()
    assert np.linalg.matrix_rank(expected) <= 2
    assert np.linalg.eigvalsh(result["covariance"].to_numpy(float)).min() >= -1e-12


def test_tied_failure_censor_and_all_complete_risk_rows_are_saved():
    data = _data()
    result = _fit(data)
    for arm, key in ((0, "risk_control"), (1, "risk_treated")):
        group = data[data.treatment == arm]
        curve = result[key]
        assert_allclose(curve.time, np.unique(group.time))
        for row in curve.itertuples():
            assert row.n_risk == (group.time >= row.time).sum()
            assert row.n_event == ((group.time == row.time) & (group.event == 1)).sum()
            assert row.n_censored == ((group.time == row.time) & (group.event == 0)).sum()
            assert row.survival == pytest.approx(
                row.survival_before * (1 - row.n_event / row.n_risk)
            )
            if row.time >= 5:
                assert row.tail_area_to_tau == row.variance_contribution == 0
    tied = result["risk_control"].set_index("time").loc[1]
    assert (tied.n_risk, tied.n_event, tied.n_censored) == (6, 1, 1)
    assert tied.survival == pytest.approx(5 / 7)
    assert result["risk_control"].iloc[-1].n_censored == 1


@pytest.mark.parametrize("tau", [2.5, 6.0])
def test_uncensored_rmst_is_capped_mean_and_mle_plugin_mean_variance(tau):
    control, treated = np.array([1, 2, 4, 6]), np.array([0, 2, 3, 6])
    data = pd.DataFrame(
        {"time": np.r_[control, treated], "event": np.ones(8), "treatment": [0] * 4 + [1] * 4}
    )
    result = _fit(data, tau=tau)
    means = [np.minimum(group, tau).mean() for group in (control, treated)]
    variances = [np.minimum(group, tau).var(ddof=0) / len(group) for group in (control, treated)]
    assert_allclose(result["effects"].estimate, [*means, means[1] - means[0]], atol=1e-13)
    assert_allclose(np.diag(result["covariance"]), [*variances, sum(variances)], atol=1e-13)
    assert result["risk_control"].iloc[-1].survival == 0
    assert result["risk_control"].iloc[-1].variance_contribution == 0
    # This is the empirical-distribution (MLE) Greenwood plug-in law, not the
    # ddof=1 sample mean variance with an implicit finite-sample multiplier.
    assert variances[0] != np.minimum(control, tau).var(ddof=1) / len(control)
    assert "no finite-sample" in result.attrs["state"]["variance_method"]


def test_all_censored_reports_zero_plugin_variance_without_artificial_epsilon():
    data = pd.DataFrame(
        {"time": [3, 4, 5, 3, 4, 5], "event": [0] * 6, "treatment": [0] * 3 + [1] * 3}
    )
    result = _fit(data, tau=3)
    effects = result["effects"]
    assert_allclose(effects.estimate, [3, 3, 0])
    assert_allclose(effects.std_error, 0)
    assert_allclose(effects.ci_low, effects.estimate)
    assert_allclose(effects.ci_high, effects.estimate)
    assert effects.z.isna().all() and effects.p_value.isna().all()
    assert effects.degenerate_variance.all()
    assert_allclose(result["covariance"], 0)
    assert len(result["risk_control"]) == 3
    restored = causal_design_load(json.loads(json.dumps(causal_design_save(result))))
    assert restored.attrs == result.attrs
    pd.testing.assert_frame_equal(restored["effects"], effects)


def test_common_horizon_at_censored_endpoint_and_terminal_failure_are_admitted():
    data = pd.DataFrame(
        {"time": [1, 2, 4, 1, 2, 4], "event": [1, 1, 0, 1, 1, 1], "treatment": [0] * 3 + [1] * 3}
    )
    result = _fit(data, tau=4)
    assert result["risk_control"].iloc[-1].n_censored == 1
    assert result["risk_treated"].iloc[-1].survival == 0
    assert result["risk_treated"].iloc[-1].variance_contribution == 0
    assert_allclose(result["effects"].estimate, [7 / 3, 7 / 3, 0])


def test_permutation_arm_swap_and_time_scale_equivariance_preserve_private_input():
    data = _data()
    original = data.copy(deep=True)
    base = _fit(data)
    shuffled = _fit(data.sample(frac=1, random_state=8))
    for key in base:
        pd.testing.assert_frame_equal(base[key], shuffled[key])
    swapped = _fit(data.assign(treatment=1 - data.treatment))
    assert swapped["effects"].estimate.iloc[0] == base["effects"].estimate.iloc[1]
    assert swapped["effects"].estimate.iloc[1] == base["effects"].estimate.iloc[0]
    assert swapped["effects"].estimate.iloc[2] == -base["effects"].estimate.iloc[2]
    scale = 12.0
    scaled = _fit(data.assign(time=data.time * scale), tau=5 * scale)
    assert_allclose(scaled["effects"].estimate, base["effects"].estimate * scale, atol=1e-12)
    assert_allclose(scaled["effects"].std_error, base["effects"].std_error * scale, atol=1e-12)
    assert_allclose(scaled["covariance"], base["covariance"] * scale**2, atol=1e-12)
    pd.testing.assert_frame_equal(data, original)


def test_full_saved_result_tables_covariance_assumptions_and_positions(tmp_path):
    data = _data()
    data.iloc[1, data.columns.get_loc("event")] = np.nan
    result = _fit(data, missing="drop")
    path = tmp_path / "rmst.json"
    artifact = causal_design_save(result, path)
    restored = causal_design_load(path)
    assert restored.attrs == result.attrs
    assert restored.title == result.title
    for name in result:
        pd.testing.assert_frame_equal(restored[name], result[name], check_dtype=False)
    state = restored.attrs["state"]
    assert state["settings"]["tau"] == 5
    assert state["sample"]["positions"] == [0, *range(2, 14)]
    assert "right censoring independent" in state["assumptions"]
    assert state["individual_effect_inference"] is False
    assert_allclose(state["covariance_matrix"], result["covariance"])
    assert result.attrs["stata_parity_validated"] is False
    tampered = copy.deepcopy(artifact)
    tampered["payload"]["tables"]["risk_control"]["data"][0][1] += 1
    with pytest.raises(AnalysisError, match="checksum"):
        causal_design_load(tampered)


@pytest.mark.parametrize("tau", [0, -1, np.nan, np.inf, True, [2, 3], "3"])
def test_invalid_or_data_selected_horizon_is_refused(tau):
    with pytest.raises(AnalysisError) as error:
        _fit(tau=tau)
    assert error.value.code == "invalid_option"


@pytest.mark.parametrize(
    "column,value,code",
    [
        ("time", -1, "invalid_time"),
        ("time", np.inf, "non_finite_values"),
        ("event", 2, "invalid_event"),
        ("event", np.inf, "non_finite_values"),
        ("treatment", 2, "invalid_treatment"),
        ("treatment", np.inf, "non_finite_values"),
    ],
)
def test_invalid_role_values_refused(column, value, code):
    data = _data().astype(float)
    data.iloc[0, data.columns.get_loc(column)] = value
    with pytest.raises(AnalysisError) as error:
        _fit(data)
    assert error.value.code == code


def test_missing_support_arm_size_and_design_errors_are_explicit():
    with pytest.raises(AnalysisError, match="missing"):
        _fit(_data().assign(event=[np.nan] + _data().event.tolist()[1:]))
    with pytest.raises(AnalysisError) as error:
        _fit(tau=7.1)
    assert error.value.code == "unsupported_horizon"
    with pytest.raises(AnalysisError) as error:
        _fit(_data().iloc[[0, 1, 7, 8, 9]], tau=1)
    assert error.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as error:
        treatment_rmst(_data(), "time", "event", "treatment", design="observational", tau=5)
    assert error.value.code == "unsupported_design"
    with pytest.raises(TypeError):
        treatment_rmst(_data(), "time", "event", "treatment", tau=5)
    with pytest.raises(TypeError):
        treatment_rmst(_data(), "time", "event", "treatment", design="randomized")


@pytest.mark.parametrize(
    "options,code",
    [
        ({"device": "cuda"}, "unsupported_device"),
        ({"weights": "event"}, "unsupported_weights"),
        ({"level": 1}, "invalid_option"),
        ({"level": 0}, "invalid_option"),
        ({"max_work": 10}, "work_budget_exceeded"),
    ],
)
def test_unsupported_routes_and_budget_refused(options, code):
    with pytest.raises(AnalysisError) as error:
        _fit(**options)
    assert error.value.code == code


def test_work_preflight_runs_before_numerical_kernel(monkeypatch):
    from openecon.econometrics.causal_design import survival

    called = False

    def unexpected(*args):
        nonlocal called
        called = True
        raise AssertionError("KM kernel must not run above the declared budget")

    monkeypatch.setattr(survival, "_arm", unexpected)
    with pytest.raises(AnalysisError) as error:
        _fit(max_work=14 * 82 - 1)
    assert error.value.code == "work_budget_exceeded"
    assert called is False


def test_non_numeric_roles_and_duplicate_role_columns_refused():
    with pytest.raises(AnalysisError) as error:
        _fit(_data().assign(event=_data().event.astype(str)))
    assert error.value.code == "non_numeric_column"
    with pytest.raises(AnalysisError) as error:
        treatment_rmst(_data(), "time", "event", "event", design="randomized", tau=5)
    assert error.value.code == "invalid_spec"
