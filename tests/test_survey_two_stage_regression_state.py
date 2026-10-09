"""Saved two-stage model execution and conditional coefficient inference."""

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
from scipy import special, stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survey.common import digest
from openecon.econometrics.survey import two_stage_regression as fitting
from openecon.econometrics.survey.two_stage_regression_state import replay_work

FAMILIES = ("regress", "logit", "probit", "poisson")


def fixture():
    rows = []
    for h in range(2):
        for p in range(3):
            for j in range(4):
                x, z = (-1.5, -0.5, 0.5, 1.5)[j], (-0.7, 0.2, 0.9)[p] + 0.15 * h
                rows.append(
                    dict(
                        h=h,
                        p=p,
                        j=j,
                        N=6 + h,
                        M=8 + 2 * p,
                        x=x,
                        z=z,
                        y=1.1 + 0.6 * x - 0.3 * z + (j * p + h) / 7 + j * j / 20,
                        b=(j + p + h) % 2,
                        c=(0, 2, 1, 3)[(j + p + h) % 4],
                    )
                )
    frame = pd.DataFrame(rows)
    design = oe.survey_two_stage_design(
        frame, psu="p", ssu="j", strata="h", population_psu="N", population_ssu="M"
    )
    return frame, design


def run(family, **options):
    frame, design = fixture()
    outcome = {"regress": "y", "logit": "b", "probit": "b", "poisson": "c"}[family]
    return getattr(oe, "survey_two_stage_" + family)(frame, design, outcome, ["x", "z"], **options)


@pytest.fixture(scope="module")
def models():
    return {family: run(family, tolerance=1e-11) for family in FAMILIES}


def rehash(payload):
    payload["integrity_sha256"] = digest(
        {k: v for k, v in payload.items() if k != "integrity_sha256"}
    )
    return payload


@pytest.mark.parametrize("family", FAMILIES)
def test_native_coefficient_solver_called_once_and_work_includes_replay(family, monkeypatch):
    original = fitting.fit_sample
    calls = []

    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fitting, "fit_sample", observed)
    state = run(family)
    assert calls == [1]
    n, used, k, g = (
        state.metadata["n_design"],
        state.metadata["n_used"],
        len(state.labels),
        state.design.validation.n_psu,
    )
    cost = 2 * replay_work(n, used, k, g, state.family)
    assert (
        state.metadata["work"]["actual_work_units"]
        == state.metadata["convergence"]["work_units"] + cost
    )
    assert state.metadata["work"]["replay_work_units"] == cost


@pytest.mark.parametrize("family", FAMILIES)
def test_prediction_retains_full_delta_covariance_and_missing_physical_positions(family, models):
    state = models[family]
    frame = pd.DataFrame({"x": [-0.8, np.nan, 0.6], "z": [0.25, 0.1, -0.3]}, index=[4, 4, -2])
    result = state.predict(frame, missing="drop", alpha=0.1)
    X = np.array([[1.0, -0.8, 0.25], [1.0, 0.6, -0.3]])
    beta, V = np.asarray(state.coefficients), np.asarray(state.covariance)
    eta = X @ beta
    if family == "regress":
        means, derivative = eta, np.ones(2)
    elif family == "logit":
        means = special.expit(eta)
        derivative = means * (1 - means)
    elif family == "probit":
        means, derivative = special.ndtr(eta), stats.norm.pdf(eta)
    else:
        means, derivative = np.exp(eta), np.exp(eta)
    J = X * derivative[:, None]
    expected = J @ V @ J.T
    np.testing.assert_allclose(result.estimate, means, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(result.attrs["covariance_matrix"], expected, rtol=1e-12, atol=0.0)
    assert result.index.tolist() == [4, -2]
    assert result.attrs["physical_positions"] == [0, 2]
    assert result.attrs["covariate_exclusions"] == [1]
    assert result.attrs["conditional_fixed_covariates"] is True
    width = stats.t.ppf(0.95, state.df) * np.sqrt(expected.diagonal())
    np.testing.assert_allclose(result.ci_high, means + width, rtol=1e-12)
    assert result.attrs["survey_two_stage_regression_state"] == state.model_dump(mode="json")


@pytest.mark.parametrize("family", FAMILIES)
def test_lincom_and_joint_wald_use_full_covariance_and_reference_df(family, models):
    state = models[family]
    beta, V = np.asarray(state.coefficients), np.asarray(state.covariance)
    vector = np.array([0.0, 0.7, -0.4])
    combination = state.lincom({"x": 0.7, "z": -0.4}, null=0.2)
    np.testing.assert_allclose(combination.estimate, [vector @ beta], rtol=1e-12)
    np.testing.assert_allclose(combination.std_error, [np.sqrt(vector @ V @ vector)], rtol=1e-12)
    R = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    null = np.array([0.1, -0.2])
    difference = R @ beta - null
    wald = difference @ np.linalg.solve(R @ V @ R.T, difference)
    denominator = state.df - 1
    expected_F = denominator / state.df * wald / 2
    result = state.test(R.tolist(), null=null.tolist())
    np.testing.assert_allclose(result.wald, [wald], rtol=1e-12)
    np.testing.assert_allclose(result.statistic, [expected_F], rtol=1e-12)
    np.testing.assert_allclose(result.p_value, [stats.f.sf(expected_F, 2, denominator)], rtol=1e-12)
    assert result.denominator_df.iloc[0] == denominator


@pytest.mark.parametrize("family", FAMILIES)
def test_unsafe_model_copy_and_mutable_metadata_cannot_bypass_postestimation_replay(family, models):
    state = models[family]
    copied = state.model_copy(update={"coefficients": tuple(v + 1 for v in state.coefficients)})
    with pytest.raises(AnalysisError, match="failed replay"):
        copied.lincom([0.0, 1.0, 0.0])
    copied = state.model_copy(deep=True)
    copied.metadata["row_scores"][0][0] += 1
    with pytest.raises(AnalysisError, match="failed replay"):
        copied.predict(pd.DataFrame({"x": [0.5], "z": [0.1]}))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("field", ["coefficients", "null", "covariance"])
def test_rehashed_boolean_inference_values_are_never_coerced(family, field, models):
    payload = models[family].model_dump(mode="json")
    if field == "covariance":
        payload[field][0][0] = True
    else:
        payload[field][0] = True
    with pytest.raises(ValidationError):
        type(models[family]).model_validate(rehash(payload))


@pytest.mark.parametrize("value", [None, 1, "rows", True, {}, []])
def test_bad_raw_response_container_is_a_typed_validation_error(value, models):
    payload = models["regress"].model_dump(mode="json")
    payload["metadata"]["primitive_y"] = value
    with pytest.raises(ValidationError):
        type(models["regress"]).model_validate(rehash(payload))


@pytest.mark.parametrize("value", [None, 1, "geometry", True, []])
def test_bad_nested_raw_design_validation_is_a_typed_validation_error(value, models):
    payload = models["regress"].model_dump(mode="json")
    payload["design"]["validation"] = value
    with pytest.raises(ValidationError):
        type(models["regress"]).model_validate(rehash(payload))


def test_saved_small_coefficient_variance_cannot_be_erased_by_rehashing():
    frame, design = fixture()
    frame["y"] *= 1e-140
    state = oe.survey_two_stage_regress(frame, design, "y", ["x", "z"])
    assert 0 < max(abs(v) for row in state.covariance for v in row) < 1e-270
    payload = state.model_dump(mode="json")
    payload["covariance"] = [[0.0] * 3 for _ in range(3)]
    with pytest.raises(ValidationError):
        type(state).model_validate(rehash(payload))


def test_complete_work_budget_refuses_before_native_fit(monkeypatch):
    frame, design = fixture()
    regressors = ["x", "z"] + [f"r{i}" for i in range(12)]
    for i, name in enumerate(regressors[2:]):
        frame[name] = np.sin(np.arange(len(frame)) * (i + 1.3))

    def forbidden(*args, **kwargs):
        raise AssertionError("native fit ran before work admission")

    monkeypatch.setattr(fitting, "fit_sample", forbidden)
    with pytest.raises(AnalysisError) as error:
        oe.survey_two_stage_logit(frame, design, "b", regressors)
    assert error.value.code == "survey_regression_budget"


def test_cached_derived_weights_are_read_once_per_replay(monkeypatch, models):
    state = models["regress"]
    validation_type = type(state.design.validation)
    original = validation_type.weights.fget
    calls = []

    def observed(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(validation_type, "weights", property(observed))
    from openecon.econometrics.survey.two_stage_regression_state import replay_fit

    replay_fit(
        state.design,
        torch.tensor(state.metadata["primitive_X"], dtype=torch.float64),
        torch.tensor(state.metadata["primitive_y"], dtype=torch.float64),
        state.metadata["sample_positions"],
        torch.tensor(state.coefficients, dtype=torch.float64),
        state.family,
    )
    assert calls == [1]


def test_df_zero_retains_second_stage_uncertainty_without_reference_tests():
    frame, _ = fixture()
    frame = frame.loc[(frame.h == 0) & (frame.p == 0)].copy()
    frame["N"] = 1
    design = oe.survey_two_stage_design(
        frame, psu="p", ssu="j", strata="h", population_psu="N", population_ssu="M"
    )
    state = oe.survey_two_stage_regress(frame, design, "y", ["x"])
    assert state.df == 0 and state.covariance[1][1] > 0
    result = state.lincom([0.0, 1.0])
    assert result.std_error.iloc[0] > 0
    assert result[["statistic", "p_value", "ci_low", "ci_high"]].isna().all().all()
    with pytest.raises(AnalysisError):
        state.test([[0.0, 1.0]])


def test_joint_prediction_covariance_respects_declared_design_memory_budget():
    frame, _ = fixture()
    design = oe.survey_two_stage_design(
        frame, psu="p", ssu="j", strata="h", population_psu="N", population_ssu="M", max_memory_mb=1
    )
    state = oe.survey_two_stage_regress(frame, design, "y", ["x", "z"])
    with pytest.raises(AnalysisError) as error:
        state.predict(pd.DataFrame({"x": np.zeros(256), "z": np.zeros(256)}))
    assert error.value.code == "workspace_limit"


@pytest.mark.parametrize("field", ["planned_work_units", "actual_work_units", "replay_work_units"])
def test_cumulative_work_record_is_replayed_after_outer_rehash(field, models):
    payload = models["logit"].model_dump(mode="json")
    payload["metadata"]["work"][field] -= 1
    with pytest.raises(ValidationError):
        type(models["logit"]).model_validate(rehash(payload))


@pytest.mark.parametrize("delta", [1e-2, 1e-3, 1e-4, 1e-5])
@pytest.mark.parametrize("scale", [1.0, 1e-5, 1e5])
def test_identified_correlated_linear_columns_restore_at_their_own_scale(delta, scale):
    frame, design = fixture()
    frame["z"] = (frame.x + delta * np.random.default_rng(8).normal(size=len(frame))) * scale
    state = oe.survey_two_stage_regress(frame, design, "y", ["x", "z"])
    restored = type(state).model_validate_json(state.model_dump_json())
    assert restored.model_dump(mode="json") == state.model_dump(mode="json")
    X = np.column_stack((np.ones(len(frame)), frame.x, frame.z))
    w = np.asarray(design.validation.weights)
    expected = np.linalg.lstsq(X * np.sqrt(w)[:, None], frame.y * np.sqrt(w), rcond=None)[0]
    np.testing.assert_allclose(state.coefficients, expected, rtol=2e-5, atol=0.0)


def _coherent_model_payload(state, X, y, beta, *, certify=False):
    """Forge every dependent numeric record, leaving admission geometry fixed."""
    from openecon.econometrics.survey.two_stage_regression_state import replay_fit

    payload = state.model_dump(mode="json")
    positions = payload["metadata"]["sample_positions"]
    replay = replay_fit(
        state.design,
        torch.tensor(X, dtype=torch.float64),
        torch.tensor(y, dtype=torch.float64),
        positions,
        torch.tensor(beta, dtype=torch.float64),
        state.family,
        certify=certify,
    )
    metadata = payload["metadata"]
    metadata["primitive_X"], metadata["primitive_y"] = (
        np.asarray(X).tolist(),
        np.asarray(y).tolist(),
    )
    for key in ("bread", "row_scores", "stage1_covariance", "stage2_covariance"):
        metadata[key] = replay[key].tolist()
    payload["coefficients"], payload["covariance"] = (
        np.asarray(beta).tolist(),
        replay["covariance"].tolist(),
    )
    n, width = metadata["n_design"], len(state.regressors) + 1
    selected = [False] * n
    values = [[0.0] * width for _ in range(n)]
    for j, i in enumerate(positions):
        selected[i] = True
        values[i] = [float(y[j]), *np.asarray(X)[j, int(state.intercept) :].tolist()]
    metadata["sample_input_sha256"] = digest(
        [
            state.design.validation.design_input_sha256,
            selected,
            values,
            [[1.0] * width for _ in range(n)],
        ]
    )
    metadata["convergence"]["score_max_abs"] = float(replay["score"].abs().max()) / len(positions)
    metadata["convergence"]["objective"] = replay["objective"]
    return rehash(payload)


@pytest.mark.parametrize("delta", [1e-3, 1e-5])
def test_coherently_rehashed_linear_slope_shift_fails_analytic_solution_binding(delta):
    frame, design = fixture()
    frame["z"] = frame.x + delta * np.random.default_rng(8).normal(size=len(frame))
    state = oe.survey_two_stage_regress(frame, design, "y", ["x", "z"])
    beta = np.array(state.coefficients)
    beta[1], beta[2] = beta[1] + 0.1, beta[2] - 0.1
    payload = _coherent_model_payload(
        state, state.metadata["primitive_X"], state.metadata["primitive_y"], beta
    )
    with pytest.raises(ValidationError, match="normal-equation coefficients"):
        type(state).model_validate(payload)


@pytest.mark.parametrize("family", ["logit", "probit", "poisson"])
def test_coherent_complete_separation_cannot_be_restored_from_finite_saturated_beta(family, models):
    state = models[family]
    X, y = np.array(state.metadata["primitive_X"]), np.array(state.metadata["primitive_y"])
    if family == "poisson":
        y = np.where(y > 0, 2.0, 0.0)
        X[:, 1] = np.where(y > 0, 0.0, -1.0)
        beta = [np.log(2.0), 35.0, 0.0]
    else:
        X[:, 1] = 2 * y - 1
        beta = [0.0, 35.0 if family == "logit" else 8.0, 0.0]
    payload = _coherent_model_payload(state, X, y, beta)
    payload["metadata"]["convergence"]["tolerance"] = 0.001
    payload = rehash(payload)
    with pytest.raises(ValidationError, match="separating direction"):
        type(state).model_validate(payload)


@pytest.mark.parametrize("family", ["logit", "probit", "poisson"])
def test_all_three_construction_separation_checks_are_charged_before_execution(family, monkeypatch):
    from openecon.econometrics.survey import regression, two_stage_regression_state as saved

    name = "_poisson_check" if family == "poisson" else "_binary_check"
    original = getattr(regression, name)
    calls = []

    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(regression, name, observed)
    monkeypatch.setattr(saved, name, observed)
    state = run(family)
    assert calls == [1, 1, 1]
    native = state.metadata["convergence"]["diagnostic_work_allowance"]
    base_cost = replay_work(
        state.metadata["n_design"],
        state.metadata["n_used"],
        len(state.labels),
        state.design.validation.n_psu,
        "linear",
    )
    assert state.metadata["work"]["replay_work_units"] == 2 * (base_cost + native)


@pytest.mark.parametrize("boundary", ["below", "above"])
def test_impossible_native_score_evaluation_count_fails_even_with_coherent_work(boundary, models):
    state = models["logit"]
    payload = state.model_dump(mode="json")
    convergence = payload["metadata"]["convergence"]
    steps = convergence["newton_steps"]
    evaluations = 2 * steps if boundary == "below" else 41 * steps + 2
    used, n, k, g = (
        state.metadata["n_used"],
        state.metadata["n_design"],
        len(state.labels),
        state.design.validation.n_psu,
    )
    convergence["score_evaluations"] = evaluations
    convergence["work_units"] = (
        n * k * 4
        + g * k * k * 4
        + convergence["diagnostic_work_allowance"]
        + evaluations * used * k * k
    )
    payload["metadata"]["work"]["actual_work_units"] = (
        convergence["work_units"] + payload["metadata"]["work"]["replay_work_units"]
    )
    with pytest.raises(ValidationError, match="iteration and work counts"):
        type(state).model_validate(rehash(payload))
