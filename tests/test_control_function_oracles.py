"""Independent outcome criteria and full generated-design CF uncertainty."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import openecon as oe


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "control_function_independent_oracle", ROOT / "scripts/verify_control_function_oracles.py",
)
oracle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oracle)


def native_fit(kind, *, cluster=None, just_identified=False, inputs=None):
    inputs = deepcopy(inputs or oracle.fixture(kind, just_identified=just_identified))
    inputs.update(covariance="cluster" if cluster else "robust", cluster=cluster)
    frame = pd.DataFrame(inputs["data"])
    frame.index = [i//2 for i in range(len(frame))]
    before = frame.copy(deep=True)
    options = {key: inputs[key] for key in (
        "y", "endogenous", "x", "instruments", "intercept", "missing", "alpha",
        "covariance", "cluster",
    )}
    result = getattr(oe, oracle.APIS[kind])(data=frame, **options)
    pd.testing.assert_frame_equal(frame, before)
    return result, inputs


@pytest.mark.parametrize("kind", oracle.KINDS)
def test_independent_scalar_score_and_observed_derivative(kind):
    data = oracle.fixture(kind)["data"]
    y = np.asarray(data["y"][:12])
    eta = np.linspace(-1.2, 1.0, len(y))
    criterion, score, derivative = oracle.outcome_quantities(kind, y, eta)
    step = 1e-5
    upper = oracle.outcome_quantities(kind, y, eta+step)
    lower = oracle.outcome_quantities(kind, y, eta-step)
    np.testing.assert_allclose((upper[0]-lower[0])/(2*step), score, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose((upper[1]-lower[1])/(2*step), derivative, atol=2e-8, rtol=2e-8)
    assert np.isfinite(criterion).all()


@pytest.mark.parametrize("kind", oracle.KINDS)
@pytest.mark.parametrize("cluster", [None, "cluster"])
def test_all_outcomes_full_nonsymmetric_generated_design_jacobian_and_sandwich(kind, cluster):
    result, inputs = native_fit(kind, cluster=cluster)
    checked = oracle.check_case(dict(kind=kind, inputs=inputs, result=result.model_dump(mode="json")))
    assert checked["numeric_bread_asymmetry"] > 1
    assert np.max(np.abs(checked["first_outcome_cross_covariance"])) > 1e-5
    state = result.extra["control_function_state"]
    bread = np.asarray(state["bread"])
    first = len(state["gamma"])
    assert np.array_equal(bread[:first, first:], np.zeros_like(bread[:first, first:]))
    assert np.max(np.abs(bread[first:, :first])) > 1
    # A second-stage-only sandwich does not carry generated-regressor uncertainty.
    rows = np.asarray(state["row_scores"])
    second = rows[:, first:]
    if cluster:
        codes = np.asarray(state["cluster_codes"])
        units = np.zeros((state["cluster_count"], second.shape[1]))
        np.add.at(units, codes, second)
        second = units
    inverse = np.linalg.inv(bread[first:, first:])
    naive = inverse @ (second.T @ second) @ inverse.T
    actual = np.asarray(result.covariance_matrix)[first:, first:]
    assert np.max(np.abs(actual-naive)) > 1e-5


@pytest.mark.parametrize("kind", oracle.KINDS)
def test_whole_singleton_cluster_cr0_is_exact_hc0(kind):
    iid, _ = native_fit(kind)
    clustered, _ = native_fit(kind, cluster="singleton")
    np.testing.assert_allclose(clustered.covariance_matrix, iid.covariance_matrix, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        clustered.extra["control_function_state"]["meat"],
        iid.extra["control_function_state"]["meat"], atol=2e-10, rtol=2e-12,
    )


@pytest.mark.parametrize("cluster", [None, "cluster"])
@pytest.mark.parametrize("just_identified", [False, True])
def test_gaussian_point_2sls_but_only_just_identified_covariance_equality(cluster, just_identified):
    result, inputs = native_fit("gaussian", cluster=cluster, just_identified=just_identified)
    independent = oracle.fit_oracle("gaussian", inputs)
    beta, covariance = oracle.two_stage_least_squares(inputs)
    first = len(independent["gamma"])
    actual_beta = np.asarray([row.estimate for row in result.coefficients])[first:-1]
    actual_covariance = np.asarray(result.covariance_matrix)[first:-1, first:-1]
    np.testing.assert_allclose(actual_beta, beta, atol=2e-9, rtol=2e-9)
    if just_identified:
        np.testing.assert_allclose(actual_covariance, covariance, atol=2e-10, rtol=2e-9)
    else:
        assert np.max(np.abs(actual_covariance-covariance)) > 1e-5


@pytest.mark.parametrize("kind", oracle.KINDS)
@pytest.mark.parametrize("cluster", [None, "cluster"])
def test_saved_conditional_mean_full_gamma_beta_delta_and_cross_row_covariance(kind, cluster):
    result, inputs = native_fit(kind, cluster=cluster)
    saved = json.loads(result.model_dump_json())
    query = oracle.query_fixture()
    frame = pd.DataFrame(query["data"])
    frame.index = [2, 2, -1, 8]
    before = frame.copy(deep=True)
    table = oe.cf_predict(result=saved, data=frame)
    pd.testing.assert_frame_equal(frame, before)
    payload = dict(kind=kind, inputs=inputs, result=saved, predictions=[dict(
        inputs=query, table=dict(columns=list(table.columns), data=table.values.tolist(), attrs=table.attrs),
    )])
    oracle.check_case(payload)
    attrs = table.attrs
    jacobian = np.asarray(attrs["mean_jacobian"])
    covariance = np.asarray(attrs["joint_parameter_covariance"])
    full = jacobian @ covariance @ jacobian.T
    np.testing.assert_allclose(np.diag(full), table.std_error.to_numpy()**2, atol=2e-12, rtol=2e-10)
    assert np.max(np.abs(full-np.diag(np.diag(full)))) > 1e-5
    first = len(result.extra["control_function_state"]["gamma"])
    assert np.max(np.abs(jacobian[:, :first])) > 1e-4
    assert attrs["conditional_mean_only"] is True


def test_probit_conditional_normalization_is_not_marginal_structural_index():
    result, inputs = native_fit("probit")
    state = result.extra["control_function_state"]
    beta, gamma = np.asarray(state["beta"]), np.asarray(state["gamma"])
    prepared = oracle.prepare_inputs(inputs)
    residual = prepared["d"]-prepared["z"]@gamma
    normalized_eta = prepared["x"]@beta[:-1]+beta[-1]*residual
    np.testing.assert_allclose(state["fitted"], stats.norm.cdf(normalized_eta), atol=2e-10)
    assert np.max(np.abs(np.asarray(state["fitted"])-stats.norm.cdf(prepared["x"]@beta[:-1]))) > 0.05
    # No automatic multiplication by an estimated latent-error SD. The public
    # coefficients remain the conditional probit normalization actually fitted.
    first = len(gamma)
    np.testing.assert_allclose([row.estimate for row in result.coefficients][first:], beta, atol=1e-12)


def test_fractional_quasi_criterion_retains_noninteger_and_boundary_outcomes():
    result, inputs = native_fit("fractional_logit")
    state = result.extra["control_function_state"]
    y = np.asarray(state["y"])
    assert y[0] == 0 and y[1] == 1 and np.any(y != np.round(y))
    eta = np.asarray(state["design"])@np.asarray(state["beta"])
    criterion = (y*eta-np.logaddexp(0, eta)).sum()
    assert state["criterion"] == pytest.approx(criterion, abs=2e-8)
    assert result.nobs == len(inputs["data"]["y"])


def test_independent_common_missing_sample_includes_instruments_and_clusters():
    inputs = oracle.fixture("gaussian")
    inputs["data"]["z1"][3] = None
    inputs["data"]["d"][7] = None
    inputs["data"]["cluster"][11] = None
    inputs.update(missing="drop", covariance="cluster", cluster="cluster")
    result, inputs = native_fit("gaussian", inputs=inputs, cluster="cluster")
    oracle.check_case(dict(kind="gaussian", inputs=inputs, result=result.model_dump(mode="json")))
    assert result.sample_positions == [i for i in range(240) if i not in {3, 7, 11}]


@pytest.mark.parametrize("kind", oracle.KINDS)
@pytest.mark.parametrize("cluster", [None, "cluster"])
def test_declared_no_intercept_both_stage_coordinate_order(kind, cluster):
    inputs = oracle.fixture(kind)
    inputs["intercept"] = False
    result, inputs = native_fit(kind, inputs=inputs, cluster=cluster)
    oracle.check_case(dict(kind=kind, inputs=inputs, result=result.model_dump(mode="json")))
    assert result.extra["control_function_state"]["z_terms"] == ["x", "z1", "z2"]
    assert result.extra["control_function_state"]["x_terms"] == ["x", "d"]


def test_no_exogenous_and_no_intercept_still_uses_full_generated_design():
    inputs = oracle.fixture("gaussian")
    inputs.update(x=[], intercept=False)
    result, inputs = native_fit("gaussian", inputs=inputs)
    oracle.check_case(dict(kind="gaussian", inputs=inputs, result=result.model_dump(mode="json")))
    assert len(result.coefficients) == 4


def test_oracle_module_has_no_production_or_tensor_imports():
    import ast
    source = (ROOT/"scripts/verify_control_function_oracles.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        names = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
        assert not any(name and name.split(".")[0] in {"openecon", "torch"} for name in names)
