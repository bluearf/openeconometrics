"""Independent fitted-logit, stacked-equation and full-refit bootstrap oracles."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.optimize._numdiff import approx_derivative
import statsmodels.api as sm
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.causal_design import common as m
from openecon.econometrics.causal_design.observational_distribution import (
    treatment_cdf_aipw,
    treatment_cdf_ipw,
    treatment_quantile_ipw,
)
from openecon.resources import use_workspace_budget


def fixture(n=120):
    rng = np.random.default_rng(193)
    x = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-0.35 * x)))
    y = 0.25 * x + 0.5 * d + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, d=d, x=x), index=[f"person-{i}" for i in range(n)])


def run(method="ipw", data=None, **kwargs):
    function = {
        "ipw": treatment_cdf_ipw,
        "quantile": treatment_quantile_ipw,
        "aipw": treatment_cdf_aipw,
    }[method]
    options = (
        dict(quantiles=[0.25, 0.5, 0.75], reps=20)
        if method == "quantile"
        else dict(thresholds=[-0.4, 0.3, 0.9])
    )
    return function(
        fixture() if data is None else data,
        "y",
        "d",
        **(dict(x=["x"], design="unconfounded") | options | kwargs),
    )


def independent_logit(raw, d):
    raw = np.asarray(raw)
    center, scale = raw.mean(0), raw.std(0)
    z = np.column_stack((np.ones(len(d)), (raw - center) / scale))
    fitted = sm.Logit(d, z).fit(method="newton", maxiter=200, tol=1e-12, disp=False)
    return fitted, z, center, scale


def ipw_oracle(data, grid):
    fitted, z, _, _ = independent_logit(data[["x"]].to_numpy(), data.d.to_numpy())
    n, p, k = len(data), z.shape[1], len(grid)
    d, indicator = data.d.to_numpy(), (data.y.to_numpy()[:, None] <= np.array(grid)).astype(float)
    e = fitted.predict(z)
    weight = np.column_stack(((1 - d) / (1 - e), d / e))
    arms = weight.T @ indicator / weight.sum(0)[:, None]
    theta = np.r_[fitted.params, arms.ravel()]

    def equations(parameters):
        prop = 1 / (1 + np.exp(-(z @ parameters[:p])))
        w = np.column_stack(((1 - d) / (1 - prop), d / prop))
        return np.column_stack(
            (
                z * (d - prop)[:, None],
                w[:, 0, None] * (indicator - parameters[p : p + k]),
                w[:, 1, None] * (indicator - parameters[p + k :]),
            )
        )

    score = equations(theta)
    jac = approx_derivative(
        lambda parameters: equations(parameters).mean(0), theta, method="3-point"
    )
    contribution = -np.linalg.solve(jac, score.T).T / n
    full_covariance = contribution.T @ contribution
    target = contribution[:, p:]
    joint_contribution = np.column_stack((target, target[:, k:] - target[:, :k]))
    return (
        fitted,
        e,
        weight,
        arms,
        score,
        jac,
        contribution,
        full_covariance,
        joint_contribution.T @ joint_contribution,
    )


def test_ipw_full_estimated_nuisance_covariance_matches_independent_equations():
    data = fixture()
    grid = [0.9, -0.4, 0.3, 100.0]
    output = run(data=data, thresholds=grid, level=0.9)
    fitted, e, weight, arms, score, jac, influence, full, joint = ipw_oracle(data, grid)
    state = output.attrs["state"]
    np.testing.assert_allclose(state["propensity_fit"]["coefficients"], fitted.params, atol=2e-10)
    np.testing.assert_allclose(state["propensity"], e, atol=2e-11)
    np.testing.assert_allclose(state["inverse_weights"], weight, atol=2e-10)
    np.testing.assert_allclose(state["arm_targets"], arms, atol=2e-11)
    np.testing.assert_allclose(state["scores"], score, atol=2e-10)
    np.testing.assert_allclose(state["jacobian"], jac, atol=2e-10)
    np.testing.assert_allclose(state["normalized_influence"], influence, atol=2e-10)
    np.testing.assert_allclose(output["nuisance_covariance"], full, atol=2e-10)
    np.testing.assert_allclose(output["joint_covariance"], joint, atol=2e-10)
    assert abs(joint[0, 4]) > 1e-8  # shared fitted propensity induces cross-arm covariance
    effect = arms[1] - arms[0]
    se = np.sqrt(joint.diagonal()[8:])
    np.testing.assert_allclose(output["effects"].std_error, se, atol=2e-10)
    np.testing.assert_allclose(
        output["effects"].ci_low, effect - stats.norm.ppf(0.95) * se, atol=2e-10
    )
    assert output["effects"].iloc[-1].inference_status == "zero_empirical_variance"
    assert pd.isna(output["effects"].iloc[-1].p_value)
    assert state["propensity_fit"]["curvature_ridge"] == 0
    assert state["propensity_fit"]["objective_penalty"] == 0


def test_intercept_only_reduction_uses_hc0_not_fake_independent_unbiased_covariance():
    data = fixture()
    grid = [-0.4, 0.3, 0.9]
    output = run(data=data, x=[], thresholds=grid)
    empirical = [
        (data.loc[data.d == arm, "y"].to_numpy()[:, None] <= grid).astype(float) for arm in (0, 1)
    ]
    covariance = sum(np.cov(v, rowvar=False, bias=True) / len(v) for v in empirical)
    np.testing.assert_allclose(output["effects"].cdf0, empirical[0].mean(0), atol=1e-15)
    np.testing.assert_allclose(output["effects"].cdf1, empirical[1].mean(0), atol=1e-15)
    np.testing.assert_allclose(output["covariance"], covariance, atol=1e-15)


def weighted_inverse(y, d, e, quantiles):
    result = []
    for arm in (0, 1):
        mask = d == arm
        outcome = y[mask]
        weight = 1 / (e[mask] if arm else 1 - e[mask])
        order = np.argsort(outcome, kind="stable")
        cumulative = np.cumsum(weight[order])
        result.append(
            outcome[
                order[
                    np.searchsorted(cumulative, np.asarray(quantiles) * cumulative[-1], side="left")
                ]
            ]
        )
    return np.array(result)


def test_quantile_bootstrap_independently_refits_every_recorded_whole_row_draw():
    data = fixture()
    output = run("quantile", data=data, level=0.9)
    state, q = output.attrs["state"], [0.25, 0.5, 0.75]
    fitted, z, _, _ = independent_logit(data[["x"]].to_numpy(), data.d.to_numpy())
    point = weighted_inverse(data.y.to_numpy(), data.d.to_numpy(), fitted.predict(z), q)
    replicates = []
    for draw, record in zip(
        state["bootstrap_sample_indices"], state["bootstrap_records"], strict=True
    ):
        rows = data.iloc[draw]
        refit, rz, _, _ = independent_logit(rows[["x"]].to_numpy(), rows.d.to_numpy())
        np.testing.assert_allclose(
            record["propensity_fit"]["coefficients"], refit.params, atol=2e-9
        )
        np.testing.assert_allclose(
            record["propensity_fit"]["probabilities"], refit.predict(rz), atol=2e-10
        )
        expected = weighted_inverse(rows.y.to_numpy(), rows.d.to_numpy(), refit.predict(rz), q)
        replicates.append(np.r_[expected.ravel(), expected[1] - expected[0]])
        assert record["original_positions"] == draw
    replicates = np.array(replicates)
    np.testing.assert_array_equal(state["joint_targets"], np.r_[point.ravel(), point[1] - point[0]])
    np.testing.assert_array_equal(state["bootstrap_joint_targets"], replicates)
    np.testing.assert_allclose(
        output["joint_covariance"], np.cov(replicates, rowvar=False, ddof=1), atol=2e-15
    )
    np.testing.assert_allclose(
        output["effects"].ci_low, np.quantile(replicates[:, 6:], 0.05, axis=0), atol=2e-15
    )
    np.testing.assert_allclose(
        output["effects"].ci_high, np.quantile(replicates[:, 6:], 0.95, axis=0), atol=2e-15
    )
    assert output["effects"].p_value.isna().all() and output["effects"].z.isna().all()
    generator = torch.Generator().manual_seed(1729)
    expected_draws = torch.randint(len(data), (20, len(data)), generator=generator).tolist()
    assert state["bootstrap_sample_indices"] == expected_draws
    assert state["rng"]["after"] == generator.get_state().tolist()


def test_aipw_all_crossfit_nuisances_and_joint_score_covariance_independent():
    data, grid = fixture(), [-0.4, 0.3, 0.9]
    output = run("aipw", data=data, level=0.9)
    state = output.attrs["state"]
    n, k = len(data), len(grid)
    predicted, propensity = np.zeros((n, 2, k)), np.zeros(n)
    for record in state["fold_records"]:
        train, test = (
            np.array(record["train_sample_indices"]),
            np.array(record["test_sample_indices"]),
        )
        assert not set(train) & set(test)
        assert len(train) + len(test) == n
        fitted, _, center, scale = independent_logit(
            data.iloc[train][["x"]].to_numpy(), data.iloc[train].d.to_numpy()
        )
        testz = np.column_stack(
            (np.ones(len(test)), (data.iloc[test][["x"]].to_numpy() - center) / scale)
        )
        propensity[test] = fitted.predict(testz)
        for arm in (0, 1):
            rows = train[data.iloc[train].d.to_numpy() == arm]
            for j, threshold in enumerate(grid):
                response = (data.iloc[rows].y.to_numpy() <= threshold).astype(float)
                fitted, _, center, scale = independent_logit(
                    data.iloc[rows][["x"]].to_numpy(), response
                )
                testz = np.column_stack(
                    (np.ones(len(test)), (data.iloc[test][["x"]].to_numpy() - center) / scale)
                )
                predicted[test, arm, j] = fitted.predict(testz)
                np.testing.assert_allclose(
                    record["outcome_fits"][arm][j]["coefficients"], fitted.params, atol=2e-9
                )
    indicator = (data.y.to_numpy()[:, None] <= grid).astype(float)
    d = data.d.to_numpy()
    mass = np.column_stack(((1 - d) / (1 - propensity), d / propensity))
    scores = predicted + mass[:, :, None] * (indicator[:, None, :] - predicted)
    arms = scores.mean(0)
    full_scores = np.column_stack((scores[:, 0], scores[:, 1], scores[:, 1] - scores[:, 0]))
    contribution = (full_scores - full_scores.mean(0)) / n
    np.testing.assert_allclose(state["propensity"], propensity, atol=2e-10)
    np.testing.assert_allclose(state["outcome_predictions"], predicted, atol=2e-10)
    np.testing.assert_allclose(state["potential_scores"], scores, atol=2e-9)
    np.testing.assert_allclose(state["arm_targets"], arms, atol=2e-10)
    np.testing.assert_allclose(
        output["joint_covariance"], contribution.T @ contribution, atol=2e-11
    )
    assert "Both propensity" in state["inference"]
    assert "One-correct-model" in state["limitations"]


@pytest.mark.parametrize("method", ["ipw", "quantile", "aipw"])
def test_full_json_tables_dtypes_roundtrip_and_tamper_rejection(method, tmp_path):
    output = run(method)
    path = tmp_path / "full.json"
    artifact = m.causal_design_save(output, path)
    loaded = m.causal_design_load(path)
    assert m.causal_design_save(loaded) == artifact
    assert json.loads(path.read_text()) == artifact
    for name in output:
        pd.testing.assert_frame_equal(output[name], loaded[name])
    tampered = copy.deepcopy(artifact)
    tampered["payload"]["attrs"]["state"]["source"]["outcome"][0] += 1
    with pytest.raises(AnalysisError):
        m.causal_design_load(tampered)


@pytest.mark.parametrize("method", ["ipw", "quantile", "aipw"])
def test_complete_case_alignment_physical_positions_typed_duplicate_labels(method):
    data = fixture()
    data.index = [1, "1", *(["same"] * 118)]
    data.loc[data.index == 1, "x"] = np.nan
    with pytest.raises(AnalysisError, match="missing"):
        run(method, data=data)
    result = run(method, data=data, missing="drop")
    assert result.attrs["positions"] == list(range(1, 120))
    assert result.attrs["unit_labels"][0] == "1"
    assert isinstance(result.attrs["unit_labels"][0], str)
    assert len(result.attrs["state"]["source"]["outcome"]) == 119
    if method == "quantile":
        for draw, record in zip(
            result.attrs["state"]["bootstrap_sample_indices"],
            result.attrs["state"]["bootstrap_records"],
            strict=True,
        ):
            assert record["original_positions"] == [position + 1 for position in draw]
    if method == "aipw":
        assert sorted(
            p
            for record in result.attrs["state"]["fold_records"]
            for p in record["test_original_positions"]
        ) == list(range(1, 120))


@pytest.mark.parametrize("method", ["ipw", "quantile", "aipw"])
def test_shift_equivariance_and_private_rng(method):
    data = fixture()
    before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    original = run(method, data=data)
    shifted = data.copy()
    shifted.y += 10
    kwargs = {} if method == "quantile" else {"thresholds": [9.6, 10.3, 10.9]}
    second = run(method, data=shifted, **kwargs)
    np.testing.assert_allclose(original["effects"].estimate, second["effects"].estimate, atol=2e-15)
    np.testing.assert_allclose(original["joint_covariance"], second["joint_covariance"], atol=2e-15)
    assert torch.equal(before, torch.random.get_rng_state())
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


def test_quantile_ties_use_left_inverse_and_no_interpolation():
    data = fixture()
    data.y = data.y.round(0)
    result = run("quantile", data=data)
    state = result.attrs["state"]
    expected = weighted_inverse(
        data.y.to_numpy(), data.d.to_numpy(), np.array(state["propensity"]), [0.25, 0.5, 0.75]
    )
    np.testing.assert_array_equal(state["joint_targets"][:6], expected.ravel())
    assert set(state["joint_targets"][:6]) <= set(data.y)


@pytest.mark.parametrize("method", ["ipw", "quantile", "aipw"])
@pytest.mark.parametrize(
    "options",
    [
        dict(device="cuda"),
        dict(weights="w"),
        dict(max_work=1),
        dict(design="randomized"),
        dict(overlap=0.5),
        dict(overlap=0),
        dict(max_iterations=True),
        dict(max_iterations=201),
    ],
)
def test_unsupported_options_refuse(method, options):
    with pytest.raises(AnalysisError):
        run(method, **options)


@pytest.mark.parametrize("method", ["ipw", "quantile", "aipw"])
def test_workspace_fails_before_nuisance_allocation(method, monkeypatch):
    import openecon.econometrics.causal_design.observational_distribution as module

    monkeypatch.setattr(
        module, "_logit", lambda *args: pytest.fail("allocated nuisance before plan")
    )
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        run(method, data=fixture(400), max_work=1_000_000_000)


@pytest.mark.parametrize("method", ["ipw", "quantile", "aipw"])
def test_dataset_and_bad_numeric_topology(method):
    with pytest.raises(AnalysisError, match="resident"):
        run(method, data=Dataset.from_frame(fixture()))
    data = fixture()
    data.d = data.d.astype(bool)
    with pytest.raises(AnalysisError):
        run(method, data=data)
    data = fixture()
    data.loc[data.index[0], "x"] = np.inf
    with pytest.raises(AnalysisError):
        run(method, data=data)
    with pytest.raises(AnalysisError):
        run(method, x=["d"])


@pytest.mark.parametrize("method", ["ipw", "quantile", "aipw"])
def test_full_rank_and_overlap_fail_without_dropping_or_clipping(method):
    data = fixture()
    data["copy"] = data.x
    with pytest.raises(AnalysisError, match="rank"):
        run(method, data=data, x=["x", "copy"])
    data["constant"] = 1.0
    with pytest.raises(AnalysisError, match="constant"):
        run(method, data=data, x=["constant"])
    with pytest.raises(AnalysisError, match="overlap"):
        run(method, overlap=0.45)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("quasi", [False, True])
def test_separating_propensity_never_passes_small_gradient_certificate(scale, quasi):
    x = np.r_[np.linspace(-2, -0.1, 30), np.linspace(0.1, 2, 30)]
    d = np.r_[np.zeros(30), np.ones(30)]
    if quasi:
        x[29:31] = 0
    data = pd.DataFrame(dict(x=x * scale, d=d, y=np.sin(np.arange(60))))
    with pytest.raises(AnalysisError, match="finite|boundary|curvature"):
        run(data=data, overlap=1e-12, max_iterations=100)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_separating_threshold_logit_refuses_entire_aipw(scale):
    data = fixture()
    data.y = (data.x > 0).astype(float)
    data.x *= scale
    with pytest.raises(AnalysisError, match="finite|boundary|curvature"):
        run("aipw", data=data, thresholds=[0.5], max_iterations=100, max_work=500_000_000)


def test_empty_threshold_training_class_is_not_silently_constant_fit():
    with pytest.raises(AnalysisError, match="both response classes"):
        run("aipw", thresholds=[100])


def test_quantile_bad_replicate_rejects_whole_calculation(monkeypatch):
    import openecon.econometrics.causal_design.observational_distribution as module

    actual = module._logit

    def failed(raw, response, iterations, label):
        if label == "Bootstrap propensity 3":
            raise AnalysisError("nonconvergence", "independent injected failed bootstrap fit")
        return actual(raw, response, iterations, label)

    monkeypatch.setattr(module, "_logit", failed)
    with pytest.raises(AnalysisError, match="failed bootstrap"):
        run("quantile")


@pytest.mark.parametrize(
    "method,target", [("ipw", "thresholds"), ("quantile", "quantiles"), ("aipw", "thresholds")]
)
@pytest.mark.parametrize("values", [[], [0.5, 0.5], [float("nan")], [True], list(range(17))])
def test_fixed_target_validation(method, target, values):
    with pytest.raises(AnalysisError):
        run(method, **{target: values})


def test_quantile_endpoints_and_rng_limits():
    for kwargs in [
        dict(quantiles=[0]),
        dict(quantiles=[1]),
        dict(reps=19),
        dict(reps=1000),
        dict(seed=-1),
        dict(seed=True),
    ]:
        with pytest.raises(AnalysisError):
            run("quantile", **kwargs)


def test_predeclared_grid_order_and_raw_aipw_no_monotone_projection():
    output = run(thresholds=[0.9, -0.4, 0.3])
    assert output["effects"].threshold.tolist() == [0.9, -0.4, 0.3]
    aipw = run("aipw")
    state = aipw.attrs["state"]
    scores = np.array(state["potential_scores"])
    np.testing.assert_allclose(state["arm_targets"], scores.mean(0), atol=1e-15)
    assert "no clipping/rearrangement" in state["limitations"]


def test_medium_n_correct_logit_dgp_against_analytic_marginal_targets():
    # A single fixed DGP diagnostic, not a claim about repeated-sample coverage.
    rng = np.random.default_rng(440)
    x = rng.uniform(-1, 1, 1000)
    d = rng.binomial(1, 1 / (1 + np.exp(-0.3 * x)))
    data = pd.DataFrame(dict(x=x, d=d, y=0.4 * d + 0.3 * x + rng.logistic(size=len(x))))
    thresholds = np.array([-0.25, 0.75])

    # Integral over X~Uniform[-1,1] of logistic(t-.4*d-.3*X).
    def true_cdf(arm):
        t = thresholds - 0.4 * arm
        return (np.logaddexp(0, t + 0.3) - np.logaddexp(0, t - 0.3)) / 0.6

    truth = true_cdf(1) - true_cdf(0)
    for method in ("ipw", "aipw"):
        result = run(method, data=data, thresholds=thresholds.tolist(), max_work=1_000_000_000)
        np.testing.assert_allclose(result["effects"].estimate, truth, atol=0.04)
        assert (
            abs(result["effects"].estimate.to_numpy() - truth) < 3 * result["effects"].std_error
        ).all()
        if method == "aipw":
            for fold in result.attrs["state"]["fold_records"]:
                for fit in [
                    fold["propensity_fit"],
                    *(item for arm in fold["outcome_fits"] for item in arm),
                ]:
                    assert fit["converged"] and fit["scaled_gradient"] <= 1e-12
                    assert fit["undamped_relative_newton_step"] <= 1e-9
                    assert np.linalg.eigvalsh(-np.array(fit["hessian"])).min() > 0
                    np.testing.assert_allclose(
                        np.sum(fit["score_rows"], axis=0), fit["score"], atol=1e-12
                    )
    quantile = run("quantile", data=data)
    np.testing.assert_allclose(quantile["effects"].estimate, 0.4, atol=0.14)
    assert (quantile["effects"].std_error > 0).all()


@pytest.mark.parametrize("misspecified", ["propensity", "outcome"])
def test_one_correct_nuisance_dgp_diagnostic_keeps_ci_assumption_explicit(misspecified):
    rng = np.random.default_rng(923)
    x = rng.uniform(-1, 1, 1200)
    propensity_index = 0.5 * x**2 if misspecified == "propensity" else 0.3 * x
    d = rng.binomial(1, 1 / (1 + np.exp(-propensity_index)))
    location = 0.6 * x**2 if misspecified == "outcome" else 0.3 * x
    data = pd.DataFrame(dict(x=x, d=d, y=0.4 * d + location + rng.logistic(size=len(x))))
    result = run("aipw", data=data, thresholds=[-0.25, 0.75], max_work=1_000_000_000)
    nodes, weights = np.polynomial.legendre.leggauss(120)
    known_location = 0.6 * nodes**2 if misspecified == "outcome" else 0.3 * nodes

    def cdf(arm):
        return (
            np.sum(
                weights[:, None]
                / (
                    1
                    + np.exp(
                        -(np.array([-0.25, 0.75])[None, :] - 0.4 * arm - known_location[:, None])
                    )
                ),
                axis=0,
            )
            / 2
        )

    np.testing.assert_allclose(result["effects"].estimate, cdf(1) - cdf(0), atol=0.065)
    # Score covariance is deliberately not asserted to give valid coverage here.
    state = result.attrs["state"]
    assert "Both propensity" in state["inference"]
    assert "does not alone validate" in state["limitations"]


def test_constant_quantile_bootstrap_exposes_exact_zero_variance_without_epsilon():
    data = fixture()
    data.y = 2.5 + data.d * 1.75
    result = run("quantile", data=data)
    assert (result["effects"].estimate == 1.75).all()
    assert (result["effects"].std_error == 0).all()
    assert (result["effects"].inference_status == "zero_empirical_bootstrap_variance").all()
    assert result.attrs["state"]["degenerate_joint_indices"] == list(range(9))


def test_real_under_supported_arm_bootstrap_sample_refuses_without_redraw():
    data = pd.DataFrame(dict(y=np.arange(30.0), d=[1] * 6 + [0] * 24))
    with pytest.raises(AnalysisError, match="six rows"):
        run("quantile", data=data, x=[])


def test_invalid_fold_arm_is_not_repaired():
    data = fixture(30)
    data.d = [1] * 6 + [0] * 24
    with pytest.raises(AnalysisError, match="training"):
        run("aipw", data=data, x=[], folds=5)


@pytest.mark.parametrize("method", ["ipw", "quantile", "aipw"])
def test_default_device_and_dtype_cannot_redirect_float64_cpu_kernel(method):
    prior_dtype = torch.get_default_dtype()
    prior_device = torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        result = run(method)
        assert all(dtype == np.dtype("float64") for dtype in result["joint_covariance"].dtypes)
        assert result.attrs["dtype"] == "float64" and result.attrs["device"] == "cpu"
    finally:
        torch.set_default_dtype(prior_dtype)
        torch.set_default_device(prior_device)


def test_near_one_level_uses_finite_normal_critical_value():
    result = run(level=np.nextafter(1.0, 0.0))
    assert np.isfinite(result["effects"].ci_low).all()
    assert np.isfinite(result["effects"].ci_high).all()


def test_nonrepresentable_positive_uncertainty_interval_is_refused():
    with pytest.raises(AnalysisError, match="confidence shift"):
        run(level=1e-300)
