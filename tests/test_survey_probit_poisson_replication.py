"""Independent probit/Poisson observed-score coefficient PSU replication.

The reference module imports only NumPy/Pandas/SciPy, constructs declared
replicate weights itself, and never calls the implementation under test.
"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError


_spec = importlib.util.spec_from_file_location(
    "probit_poisson_replication_oracle",
    Path(__file__).resolve().parents[1]/"scripts/validate_survey_probit_poisson_replication.py",
)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)
METHODS = reference.METHODS
FAMILIES = ("probit", "poisson")


def fixture(*, fpc=False):
    """Unequal weights, repeated PSU labels across strata, two score directions."""
    frame = reference.primitive_frame()
    if fpc:
        frame["N"] = np.repeat([4, 6, 8], 16)
    design = oe.survey_design(frame, weights="w", psu="p", strata="h", fpc="N" if fpc else None)
    return frame, design, reference.bootstrap_weights(frame)


def fit(frame, design, family, method, *, supplied=None, **overrides):
    options = reference.default_options(method)
    options["tolerance"] = 1e-11
    options.update(overrides)
    if method == "bootstrap" and supplied is None:
        supplied = reference.bootstrap_weights(frame)
    if supplied is not None:
        options.pop("replicates", None)
        options["replicate_weights"] = supplied
    fn = oe.survey_probit_replicate if family == "probit" else oe.survey_poisson_replicate
    return fn(frame, design, "b" if family == "probit" else "c", ["x", "z"], method=method, **options)


def assert_reference(result, expected, *, alpha=.05, null=0):
    np.testing.assert_allclose(result.coefficients, expected["coefficients"], atol=4e-9, rtol=5e-8)
    np.testing.assert_allclose(result.covariance, expected["covariance"], atol=4e-9, rtol=5e-8)
    assert result.labels == ("_cons", "x", "z")
    assert result.df == expected["df"]
    record = result.metadata["replication"]
    assert record["replicate_ids"] == expected["ids"]
    assert record["replicate_count"] == len(expected["ids"])
    assert record["failed_replicates"] == []
    np.testing.assert_allclose(record["replicate_estimates"], expected["replicas"], atol=4e-9, rtol=5e-8)
    np.testing.assert_allclose(record["variance_multipliers"], expected["multipliers"], atol=1e-13, rtol=1e-13)
    assert result.metadata["sample_positions"] == np.flatnonzero(expected["selected"]).tolist()
    table = result.to_frame()
    assert table.index.tolist() == ["_cons", "x", "z"]
    assert table.columns.tolist() == ["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"]
    np.testing.assert_allclose(table.to_numpy(), reference.inference(
        expected["coefficients"], expected["covariance"], expected["df"], alpha=alpha, null=null,
    ), atol=5e-8, rtol=5e-8)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method,centering", [
    (method, centering) for method in METHODS
    for centering in (["original", "stratum_mean"] if method == "jackknife" else ["original", "replicate_mean"])
])
def test_all_eight_weighted_coefficients_full_covariance_and_t_inference(family, method, centering):
    frame, design, supplied = fixture()
    original = frame.copy(deep=True)
    options = {**reference.default_options(method), "centering": centering}
    expected = reference.oracle(frame, family, method, **options)
    actual = fit(frame, design, family, method, centering=centering, alpha=.1, null=.15)
    assert_reference(actual, expected, alpha=.1, null=.15)
    off_diagonal = expected["covariance"]-np.diag(expected["covariance"].diagonal())
    assert np.count_nonzero(abs(off_diagonal) > 1e-9) == 6
    assert actual.metadata["replication"]["method"] == method
    assert actual.metadata["n_design"] == 48 and actual.metadata["n_used"] == 48
    assert actual.metadata["device"] == "cpu" and actual.metadata["precision"] == "float64"
    assert actual.metadata["stata_parity_validated"] is False
    assert len(actual.metadata["replication"]["replicate_convergence"]) == len(expected["ids"])
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(supplied, reference.bootstrap_weights(frame))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
def test_saved_typed_json_state_predict_margins_contrast_and_joint_adjusted_f(family, method):
    frame, design, _ = fixture()
    result = fit(frame, design, family, method)
    payload = result.model_dump(mode="json")
    restored = oe.SurveyRegressionResult.model_validate_json(json.dumps(payload, allow_nan=False))
    assert restored == result
    assert oe.SurveyRegressionResult.model_validate(payload) == result
    expected = reference.oracle(frame, family, method, **reference.default_options(method))
    beta, covariance, df = expected["coefficients"], expected["covariance"], expected["df"]
    evaluation = pd.DataFrame({"x": [-.8, .3, 1.2], "z": [.5, -.2, .9]}, index=["a", "b", "a"])
    x = np.column_stack((np.ones(3), evaluation.to_numpy()))
    eta = x@beta
    mu = reference.response(family, eta)
    gradient = reference.response_derivative(family, eta)[:, None]*x
    predicted_covariance = gradient@covariance@gradient.T
    prediction = oe.survey_predict(restored, evaluation)
    assert prediction.index.tolist() == ["a", "b", "a"]
    np.testing.assert_allclose(prediction.to_numpy(), reference.inference(mu, predicted_covariance, df), atol=4e-8, rtol=5e-8)
    np.testing.assert_allclose(prediction.attrs["covariance_matrix"], predicted_covariance, atol=4e-9, rtol=5e-8)
    xm = np.column_stack((np.ones(len(frame)), frame[["x", "z"]].to_numpy()))
    w = frame.w.to_numpy()/frame.w.sum()
    response = reference.response(family, xm@beta)
    jacobian = (w*reference.response_derivative(family, xm@beta))@xm
    margins = oe.survey_margins(restored, frame, weights="w")
    mcov = np.array([[jacobian@covariance@jacobian]])
    np.testing.assert_allclose(margins.to_numpy(), reference.inference([w@response], mcov, df), atol=4e-8, rtol=5e-8)
    np.testing.assert_allclose(margins.attrs["covariance_matrix"], mcov, atol=4e-9, rtol=5e-8)
    average_slope = w@reference.response_derivative(family, xm@beta)
    average_curvature = (w*reference.response_second(family, xm@beta))@xm
    effects = average_slope*beta[1:]
    ame_gradient = beta[1:, None]*average_curvature + average_slope*np.eye(3)[1:]
    ame_covariance = ame_gradient@covariance@ame_gradient.T
    ame = oe.survey_margins(restored, frame, variables=["x", "z"], weights="w")
    np.testing.assert_allclose(ame.to_numpy(), reference.inference(effects, ame_covariance, df), atol=4e-8, rtol=5e-8)
    np.testing.assert_allclose(ame.attrs["covariance_matrix"], ame_covariance, atol=4e-9, rtol=5e-8)
    contrast = np.array([1., .4, -.3])
    lincom = oe.survey_lincom(restored, dict(zip(restored.labels, contrast)))
    ccov = np.array([[contrast@covariance@contrast]])
    np.testing.assert_allclose(lincom.to_numpy(), reference.inference([contrast@beta], ccov, df), atol=4e-8, rtol=5e-8)
    np.testing.assert_allclose(lincom.attrs["covariance_matrix"], ccov, atol=4e-9, rtol=5e-8)
    restrictions = np.array([[0., 1., 0.], [0., 0., 1.]])
    effects = restrictions@beta
    wald = effects@np.linalg.solve(restrictions@covariance@restrictions.T, effects)
    f = (df-1)*wald/(2*df)
    joint = oe.survey_test(restored, restrictions.tolist())
    assert joint.iloc[0].df_num == 2 and joint.iloc[0].df_den == df-1
    assert joint.iloc[0].wald_chi2 == pytest.approx(wald, rel=5e-8)
    assert joint.iloc[0].statistic == pytest.approx(f, rel=5e-8)
    assert joint.iloc[0].p_value == pytest.approx(stats.f.sf(f, 2, df-1), rel=5e-8)
    for table in (prediction, margins, ame, lincom, joint):
        assert table.attrs["survey_regression_state"] == payload


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
def test_domain_missing_keeps_complete_psu_weights_and_physical_alignment(family, method):
    frame, design, supplied = fixture()
    frame.index = np.tile([90, -3, 90, 4], 12)
    frame.loc[(frame.h == 2) & (frame.p == 1), "domain"] = 0
    outcome = "b" if family == "probit" else "c"
    frame.iloc[40:, frame.columns.get_loc(outcome)] = np.nan
    frame.iloc[2, frame.columns.get_loc("x")] = np.nan
    original = frame.copy(deep=True)
    with pytest.raises(AnalysisError):
        fit(frame, design, family, method, domain="domain")
    actual = fit(frame, design, family, method, supplied=supplied if method == "bootstrap" else None,
                 domain="domain", missing="drop")
    expected = reference.oracle(frame, family, method, domain="domain", **reference.default_options(method))
    assert_reference(actual, expected)
    assert actual.metadata["outcome_exclusions"] == [2]
    assert actual.metadata["out_of_domain_positions"] == list(range(40, 48))
    assert actual.df == 3 and actual.metadata["n_used"] == 39
    assert actual.metadata["n_design"] == 48
    # IDs and geometry include the PSU with zero target members.
    assert actual.metadata["replication"]["replicate_ids"] == expected["ids"]
    for convergence, weights in zip(actual.metadata["replication"]["replicate_convergence"], expected["weights"]):
        selected = expected["selected"] & (weights > 0)
        assert convergence["sample_positions"] == np.flatnonzero(selected).tolist()
        assert convergence["n_used"] == int(selected.sum())
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
def test_global_base_and_replica_weight_scale_preserves_coefficients_and_covariance(family, method):
    frame, design, supplied = fixture()
    original = fit(frame, design, family, method)
    scaled = frame.assign(w=frame.w*37)
    sd = oe.survey_design(scaled, weights="w", psu="p", strata="h")
    actual = fit(scaled, sd, family, method, supplied=supplied*37 if method == "bootstrap" else None)
    np.testing.assert_allclose(actual.coefficients, original.coefficients, atol=3e-9, rtol=2e-8)
    np.testing.assert_allclose(actual.covariance, original.covariance, atol=3e-9, rtol=2e-8)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
def test_no_intercept_coefficients_and_complete_covariance_keep_declared_order(family, method):
    frame, design, _ = fixture()
    actual = fit(frame, design, family, method, intercept=False)
    expected = reference.oracle(frame, family, method, intercept=False, **reference.default_options(method))
    assert actual.labels == ("x", "z") and actual.intercept is False
    assert actual.metadata["parameter_order"] == ["x", "z"]
    np.testing.assert_allclose(actual.coefficients, expected["coefficients"], atol=4e-9, rtol=5e-8)
    np.testing.assert_allclose(actual.covariance, expected["covariance"], atol=4e-9, rtol=5e-8)
    np.testing.assert_allclose(actual.to_frame().to_numpy(), reference.inference(
        expected["coefficients"], expected["covariance"], expected["df"]), atol=5e-8, rtol=5e-8)
    assert oe.SurveyRegressionResult.model_validate_json(actual.model_dump_json()) == actual


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", ["brr", "fay"])
def test_supplied_balanced_weights_equal_generated_and_retain_supplied_ids(family, method):
    frame, design, _ = fixture()
    expected = reference.oracle(frame, family, method, **reference.default_options(method))
    weights = pd.DataFrame(expected["weights"].T, columns=[f"declared-{i}" for i in range(4)])
    actual = fit(frame, design, family, method, supplied=weights)
    expected["ids"] = weights.columns.tolist()
    assert_reference(actual, expected)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", ["brr", "fay"])
def test_default_hadamard_order_and_fay_rho_match_explicit_declaration(family, method):
    frame, design, _ = fixture()
    function = oe.survey_probit_replicate if family == "probit" else oe.survey_poisson_replicate
    actual = function(frame, design, "b" if family == "probit" else "c", ["x", "z"],
                      method=method, tolerance=1e-11)
    expected = reference.oracle(frame, family, method, **reference.default_options(method))
    assert_reference(actual, expected)
    assert actual.metadata["replication"]["fay_rho"] == (0. if method == "brr" else .5)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
def test_intercept_only_replicas_equal_probit_quantile_or_poisson_log_mean_closed_form(family, method):
    frame, design, supplied = fixture()
    options = reference.default_options(method)
    function = oe.survey_probit_replicate if family == "probit" else oe.survey_poisson_replicate
    actual = function(frame, design, "b" if family == "probit" else "c", [], method=method,
                      tolerance=1e-11, **options,
                      **({"replicate_weights": supplied} if method == "bootstrap" else {}))
    weights, ids, multipliers, strata, df = reference.replica_plan(frame, method, **options)
    y = frame.b.to_numpy() if family == "probit" else frame.c.to_numpy()
    mean = float(frame.w.to_numpy()@y/frame.w.sum())
    beta = stats.norm.ppf(mean) if family == "probit" else np.log(mean)
    means = weights@y/weights.sum(1)
    replicas = stats.norm.ppf(means) if family == "probit" else np.log(means)
    centering = options.get("centering", "original")
    if centering == "original":
        differences = replicas-beta
    elif centering == "replicate_mean":
        differences = replicas-replicas.mean()
    else:
        differences = np.zeros(len(replicas))
        for h in set(strata):
            selected = np.array(strata) == h
            differences[selected] = replicas[selected]-replicas[selected].mean()
    variance = float(multipliers@differences**2)
    assert actual.labels == ("_cons",) and actual.regressors == ()
    assert actual.df == df and actual.metadata["replication"]["replicate_ids"] == ids
    np.testing.assert_allclose(actual.coefficients, [beta], atol=2e-10, rtol=2e-9)
    np.testing.assert_allclose(actual.covariance, [[variance]], atol=2e-10, rtol=2e-9)
    np.testing.assert_allclose(actual.metadata["replication"]["replicate_estimates"], replicas[:, None], atol=2e-10, rtol=2e-9)
    np.testing.assert_allclose(actual.to_frame().to_numpy(), reference.inference([beta], [[variance]], df), atol=3e-8, rtol=5e-8)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
def test_saturated_two_group_coefficients_and_replica_covariance_have_closed_form(family, method):
    frame, _, _ = fixture()
    frame["x"] = np.tile([0., 0., 0., 0., 1., 1., 1., 1.], 6)
    design = oe.survey_design(frame, weights="w", psu="p", strata="h")
    weights = reference.bootstrap_weights(frame)
    options = reference.default_options(method)
    function = oe.survey_probit_replicate if family == "probit" else oe.survey_poisson_replicate
    actual = function(frame, design, "b" if family == "probit" else "c", ["x"], method=method,
                      tolerance=1e-11, **options,
                      **({"replicate_weights": weights} if method == "bootstrap" else {}))
    rw, ids, multipliers, strata, df = reference.replica_plan(frame, method, **options)
    y = frame.b.to_numpy() if family == "probit" else frame.c.to_numpy()
    def coefficients(weight):
        means = [weight[frame.x == group]@y[frame.x == group]/weight[frame.x == group].sum()
                 for group in (0, 1)]
        indices = stats.norm.ppf(means) if family == "probit" else np.log(means)
        return np.array([indices[0], indices[1]-indices[0]])
    beta = coefficients(frame.w.to_numpy())
    replicas = np.array([coefficients(weight) for weight in rw])
    center = options.get("centering", "original")
    if center == "original":
        differences = replicas-beta
    elif center == "replicate_mean":
        differences = replicas-replicas.mean(0)
    else:
        differences = np.zeros_like(replicas)
        for h in set(strata):
            selected = np.array(strata) == h
            differences[selected] = replicas[selected]-replicas[selected].mean(0)
    covariance = np.einsum("r,ri,rj->ij", multipliers, differences, differences)
    assert abs(covariance[0, 1]) > 1e-8
    assert actual.labels == ("_cons", "x") and actual.df == df
    assert actual.metadata["replication"]["replicate_ids"] == ids
    np.testing.assert_allclose(actual.coefficients, beta, atol=2e-10, rtol=2e-9)
    np.testing.assert_allclose(actual.covariance, covariance, atol=2e-10, rtol=2e-9)
    np.testing.assert_allclose(actual.metadata["replication"]["replicate_estimates"], replicas, atol=2e-10, rtol=2e-9)
    np.testing.assert_allclose(actual.to_frame().to_numpy(), reference.inference(beta, covariance, df), atol=3e-8, rtol=5e-8)


def test_unsaturated_probit_retains_observed_curvature_instead_of_fisher_bread():
    frame, design, _ = fixture()
    actual = fit(frame, design, "probit", "fay")
    beta = reference.oracle(frame, "probit", "fay", **reference.default_options("fay"))["coefficients"]
    x = np.column_stack((np.ones(len(frame)), frame[["x", "z"]].to_numpy()))
    weights = frame.w.to_numpy()/frame.w.mean()
    observed = reference.likelihood_parts(beta, x, frame.b.to_numpy(), weights, "probit")[2]
    eta = x@beta
    probability, density = stats.norm.cdf(eta), stats.norm.pdf(eta)
    fisher = x.T@((weights*density**2/(probability*(1-probability)))[:, None]*x)
    assert np.max(abs(observed-fisher)) > .01
    np.testing.assert_allclose(actual.metadata["sensitivity"], observed, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("centering", ["original", "stratum_mean"])
@pytest.mark.parametrize("census", [False, True])
def test_delete_psu_jackknife_full_covariance_fpc_and_census_stratum(family, centering, census):
    frame, _, _ = fixture(fpc=True)
    if census:
        frame.loc[frame.h == 0, "N"] = 2
    design = oe.survey_design(frame, weights="w", psu="p", strata="h", fpc="N")
    actual = fit(frame, design, family, "jackknife", centering=centering)
    expected = reference.oracle(frame, family, "jackknife", centering=centering)
    assert_reference(actual, expected)
    assert len(expected["ids"]) == (4 if census else 6)
    if census:
        assert all(not identity.startswith("stratum-0") for identity in expected["ids"])
        np.testing.assert_allclose(expected["weights"][:, :16], np.broadcast_to(frame.w[:16], (4, 16)))


@pytest.mark.parametrize("family", FAMILIES)
def test_bootstrap_explicit_df_and_caller_variance_multipliers(family):
    frame, design, weights = fixture()
    actual = fit(frame, design, family, "bootstrap", df=6, scale=.17,
                 rscales=[.9, 1.1, .8, 1.2, 1.3, .7, 1.4, .6])
    options = {**reference.default_options("bootstrap"), "df": 6, "scale": .17,
               "rscales": [.9, 1.1, .8, 1.2, 1.3, .7, 1.4, .6]}
    expected = reference.oracle(frame, family, "bootstrap", supplied=weights, **options)
    assert_reference(actual, expected)
    assert actual.df == 6


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", ["brr", "jackknife"])
def test_every_rank_failed_replica_id_is_reported_and_no_partial_result(family, method):
    frame, _, _ = fixture()
    # Each retained covariate point occurs with both responses. Thus the full
    # likelihood is finite even when only PSU 0 identifies the z coefficient.
    frame["x"] = np.tile([-1., -1., -.3, -.3, .6, .6, 1.2, 1.2], 6)
    frame["z"] = np.tile([.2, .2, -.4, -.4, .7, .7, -.8, -.8], 6)
    frame["b"] = np.tile([0, 1], 24)
    frame.loc[~((frame.h == 0) & (frame.p == 0)), "z"] = 0.
    design = oe.survey_design(frame, weights="w", psu="p", strata="h")
    with pytest.raises(AnalysisError) as exc:
        fit(frame, design, family, method)
    assert exc.value.code == "survey_replicate_failure"
    failed = ["sylvester-1", "sylvester-3"] if method == "brr" else ["stratum-0-delete-psu-0"]
    assert [item["replicate_id"] for item in exc.value.failed_replicates] == failed
    assert all(item["code"] == "survey_regression_rank" and item["n_used"] > 0
               for item in exc.value.failed_replicates)
    for identity in failed:
        assert identity in str(exc.value)
    assert "rank" in str(exc.value).lower()


@pytest.mark.parametrize("method", ["brr", "jackknife"])
def test_every_separated_replica_id_is_reported_even_with_mixed_outcomes_in_every_psu(method):
    frame = pd.DataFrame({
        "h": np.repeat(range(3), 8), "p": np.tile(np.repeat([0, 1], 4), 3),
        "w": 1., "x": np.tile([-2., -1., 1., 2.], 6),
        "z": np.tile([.4, -.3, .5, -.7], 6),
    })
    frame["b"] = (frame.x > 0).astype(int)
    frame.loc[:3, "b"] = 1-frame.loc[:3, "b"]
    design = oe.survey_design(frame, weights="w", psu="p", strata="h")
    # The complete sample is a finite nonseparated likelihood, independently fitted.
    x = np.column_stack((np.ones(len(frame)), frame[["x", "z"]]))
    reference.weighted_fit(x, frame.b.to_numpy(), frame.w.to_numpy(), "probit")
    with pytest.raises(AnalysisError) as exc:
        fit(frame, design, "probit", method)
    assert exc.value.code == "survey_replicate_failure"
    failed = ["sylvester-1", "sylvester-3"] if method == "brr" else ["stratum-0-delete-psu-0"]
    assert [item["replicate_id"] for item in exc.value.failed_replicates] == failed
    assert all(item["code"] == "survey_regression_separation" and item["n_used"] > 0
               for item in exc.value.failed_replicates)
    for identity in failed:
        assert identity in str(exc.value)
    assert "separation" in str(exc.value).lower()


@pytest.mark.parametrize("method", ["brr", "jackknife"])
def test_poisson_all_zero_replica_is_refused_with_every_failure_id_and_finite_original(method):
    frame, _, _ = fixture()
    frame["c"] = 0
    frame.loc[:7, "c"] = [1, 2, 1, 3, 2, 1, 2, 3]
    design = oe.survey_design(frame, weights="w", psu="p", strata="h")
    x = np.column_stack((np.ones(len(frame)), frame[["x", "z"]].to_numpy()))
    reference.weighted_fit(x, frame.c.to_numpy(), frame.w.to_numpy(), "poisson")
    with pytest.raises(AnalysisError) as exc:
        fit(frame, design, "poisson", method)
    expected = ["sylvester-1", "sylvester-3"] if method == "brr" else ["stratum-0-delete-psu-0"]
    assert exc.value.code == "survey_replicate_failure"
    assert [failure["replicate_id"] for failure in exc.value.failed_replicates] == expected
    assert all(failure["code"] == "survey_regression_separation" for failure in exc.value.failed_replicates)
    assert all(identity in str(exc.value) for identity in expected)


@pytest.mark.parametrize("method", METHODS)
def test_poisson_raw_exact_integer_boundary_is_checked_before_float_rounding(method):
    frame, design, _ = fixture()
    for bad_value in [-1, .25, float("inf"), float("-inf"), 2**53+1, 2**53+2, "2", True]:
        bad = frame.copy(deep=True)
        bad["c"] = bad.c.astype(object)
        bad.iloc[0, bad.columns.get_loc("c")] = bad_value
        before = bad.copy(deep=True)
        with pytest.raises(AnalysisError):
            fit(bad, design, "poisson", method)
        pd.testing.assert_frame_equal(bad, before)


@pytest.mark.parametrize("method", METHODS)
def test_probit_requires_exact_binary_member_responses(method):
    frame, design, _ = fixture()
    for value in [-1, .25, 2, float("inf"), "1", True]:
        bad = frame.copy(deep=True)
        bad["b"] = bad.b.astype(object)
        bad.iloc[0, bad.columns.get_loc("b")] = value
        with pytest.raises(AnalysisError):
            fit(bad, design, "probit", method)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
def test_response_validation_is_scoped_to_complete_target_members(family, method):
    frame, design, weights = fixture()
    outcome = "b" if family == "probit" else "c"
    frame["domain"] = 1
    frame.loc[40:, "domain"] = 0
    frame[outcome] = frame[outcome].astype(object)
    frame.iloc[40:, frame.columns.get_loc(outcome)] = ["ignored", -1, .25, float("inf"), True, 2**53+1, None, -4]
    actual = fit(frame, design, family, method, domain="domain", supplied=weights if method == "bootstrap" else None)
    clean = frame.copy(deep=True)
    clean.iloc[40:, clean.columns.get_loc(outcome)] = 0
    expected = reference.oracle(clean, family, method, domain="domain", **reference.default_options(method))
    assert_reference(actual, expected)
    assert actual.metadata["out_of_domain_positions"] == list(range(40, 48))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", ["brr", "fay"])
def test_partial_fpc_and_wrong_psu_pair_geometry_are_rejected(family, method):
    frame, design, _ = fixture(fpc=True)
    with pytest.raises(AnalysisError):
        fit(frame, design, family, method)
    frame = reference.primitive_frame()
    frame.loc[0, "p"] = 2
    design = oe.survey_design(frame, weights="w", psu="p", strata="h")
    with pytest.raises(AnalysisError):
        fit(frame, design, family, method)


@pytest.mark.parametrize("family", FAMILIES)
def test_bootstrap_rejects_row_resampling_factors_and_wrong_physical_shape(family):
    frame, design, weights = fixture()
    original = weights.copy(deep=True)
    weights.iloc[0, 0] *= 1.01
    with pytest.raises(AnalysisError) as exc:
        fit(frame, design, family, "bootstrap", supplied=weights)
    assert exc.value.code == "incompatible_survey_replicates"
    for wrong in [original.iloc[:-1], original.T, original.rename(columns={"whole-psu-1": "whole-psu-0"})]:
        with pytest.raises(AnalysisError):
            fit(frame, design, family, "bootstrap", supplied=wrong)


@pytest.mark.parametrize("method,overrides", [
    ("brr", {"rho": 0.}), ("jackknife", {"rho": .5}), ("bootstrap", {"rho": .5}),
    ("brr", {"df": 2}), ("fay", {"df": 2}), ("jackknife", {"df": 2}),
    ("fay", {"rho": 1.}), ("fay", {"rho": -.1}),
    ("brr", {"centering": "stratum_mean"}), ("bootstrap", {"centering": "stratum_mean"}),
    ("jackknife", {"centering": "replicate_mean"}),
    ("bootstrap", {"df": 0}), ("bootstrap", {"df": 8}), ("bootstrap", {"df": True}),
    ("bootstrap", {"justification": None}), ("bootstrap", {"scale": None}),
    ("bootstrap", {"rscales": [1.]*7}), ("bootstrap", {"scale": -.1}),
    ("brr", {"replicates": 3}), ("brr", {"replicates": 2}),
])
def test_explicit_method_options_refuse_unsupported_or_ambiguous_conventions(method, overrides):
    frame, design, _ = fixture()
    with pytest.raises(AnalysisError):
        fit(frame, design, "probit", method, **overrides)


def test_native_primitive_fixture_and_all_eight_saved_payloads_are_independently_verified(tmp_path):
    displayed = []
    path = Path(__file__).resolve().parents[1]/"docs/examples/survey_probit_poisson_replication_eight.py"
    namespace = {"display": displayed.append, "__name__": "__example__"}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    assert len(displayed) == 8
    states, inputs, poststates = (namespace[key] for key in ("states", "oracle_inputs", "poststates"))
    assert tuple(states) == reference.CASES
    receipt = reference.verify(states, inputs, poststates)
    assert receipt["cases"] == 8 and receipt["full_covariance_verified"] is True
    assert receipt["native_display_verified"] is False
    for case, table in zip(reference.CASES, displayed):
        pd.testing.assert_frame_equal(table, namespace["models"][case].to_frame())
    # Saved primitive inputs and every state survive the exact JSON transport.
    for name, payload in [("states", states), ("inputs", inputs), ("poststates", poststates)]:
        path = tmp_path/(name+".json")
        path.write_text(json.dumps(payload, allow_nan=False))
        assert json.loads(path.read_text()) == payload
    mutated = deepcopy(poststates)
    mutated["probit_brr"]["coefficients"]["data"][0][0] += .1
    with pytest.raises(AssertionError):
        reference.verify(states, inputs, mutated)
    execution = {"status": "ok", "error": None, "outputs": [
        {"type": "table", "latex": "saved publication table", "data": {
            "columns": poststates[case]["coefficients"]["columns"],
            "rows": poststates[case]["coefficients"]["data"],
            "index": [[label] for label in reference.LABELS], "index_names": [None],
            "total_rows": 3, "total_columns": 6,
        }} for case in reference.CASES
    ]}
    # This exercises the transport validator; source-generated output is not
    # evidence of an actual installed native execution.
    assert reference.verify(states, inputs, poststates, execution)["native_display_verified"] is True
    wrong_execution = deepcopy(execution)
    wrong_execution["outputs"][3]["data"]["rows"][1][0] += .2
    with pytest.raises(AssertionError):
        reference.verify(states, inputs, poststates, wrong_execution)
    wrong_execution = deepcopy(execution)
    wrong_execution["outputs"][0]["data"]["index"][0] = ["wrong term"]
    with pytest.raises(AssertionError):
        reference.verify(states, inputs, poststates, wrong_execution)
