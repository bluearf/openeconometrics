"""Independent references for a fixed single-target delta sensitivity model."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm, poisson
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi import chained, sensitivity
from openecon.econometrics.mi.common import _digest
from openecon.resources import use_workspace_budget


def _frame(kind="normal", n=36):
    x = np.linspace(-1.7, 1.9, n)
    if kind == "normal":
        y = 0.6 + 1.2 * x + np.sin(np.arange(n) * 1.43)
    elif kind == "logit":
        y = ((np.arange(n) * 7) % 11 < 4).astype(float)
    else:
        y = ((np.arange(n) * 5) % 7).astype(float)
    y[[2, 9, 17, n - 1]] = np.nan
    return pd.DataFrame({"y": y, "x": x, "z": np.cos(np.arange(n))},
                        index=pd.Index([f"row-{i // 2}" for i in range(n)], name="participant"))


def _run(frame, kind="normal", delta=0.0, **options):
    return sensitivity.mi_delta(frame, list(frame), target="y", kind=kind, delta=delta, **options)


def test_gaussian_saved_fit_and_predictive_likelihood_match_independent_linear_algebra():
    frame = _frame()
    result = _run(frame, delta=1.5, m=5, seed=217)
    observed = frame.y.notna().to_numpy()
    x = np.column_stack([np.ones(len(frame)), frame.x, frame.z])
    xo, xm, y = x[observed], x[~observed], frame.y.to_numpy()[observed]
    beta_hat = np.linalg.solve(xo.T @ xo, xo.T @ y)
    sse = np.square(y - xo @ beta_hat).sum()
    fit = result.metadata["observed_model"]
    np.testing.assert_allclose(fit["beta_hat"], beta_hat, atol=2e-14)
    assert fit["sse"] == pytest.approx(sse, rel=2e-14)
    assert fit["residual_df"] == len(y) - xo.shape[1]
    for diagnostic in result.metadata["imputation_diagnostics"]:
        beta = np.asarray(diagnostic["coefficient_draw"])
        sigma = np.sqrt(diagnostic["sigma2_draw"])
        expected = xm @ beta + 1.5
        np.testing.assert_allclose(diagnostic["predictive_mean"], expected, atol=1e-14)
        likelihood = norm.logpdf(y, loc=xo @ beta, scale=sigma).sum()
        assert diagnostic["observed_log_likelihood"] == pytest.approx(likelihood, abs=2e-13)


def test_gaussian_zero_delta_and_seed_coupled_location_shift_are_exact():
    frame = _frame()
    baseline = _run(frame, delta=0, m=7, seed=604)
    shifted = _run(frame, delta=2.25, m=7, seed=604)
    missing = frame.y.isna().to_numpy()
    for a, b, da, db in zip(baseline.completed_matrices, shifted.completed_matrices,
                           baseline.metadata["imputation_diagnostics"], shifted.metadata["imputation_diagnostics"]):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(b[missing, 0] - a[missing, 0], 2.25, atol=9e-16)
        np.testing.assert_array_equal(a[~missing], b[~missing])
        assert da["coefficient_draw"] == db["coefficient_draw"]
        assert da["sigma2_draw"] == db["sigma2_draw"]
        assert da["observed_log_likelihood"] == db["observed_log_likelihood"]
        assert da["predictive_mean"] == da["base_missing_linear_predictor"]


def test_gaussian_public_predictive_joint_moments_include_posterior_and_residual_uncertainty():
    observed = np.sin(np.arange(14) * 1.2) + 0.7
    frame = pd.DataFrame({"y": [*observed, np.nan, np.nan]})
    delta = 1.3
    # Independent chains supply 2,000 joint predictions; the reference uses the
    # exact inverse-chi-square Gaussian posterior predictive moments.
    draws = []
    for group in range(20):
        result = _run(frame, delta=delta, m=100, seed=30000 + group * 100)
        draws.extend(np.asarray(result.completed_matrices)[:, -2:, 0])
    actual = np.asarray(draws)
    sse = np.square(observed - observed.mean()).sum()
    sigma_mean = sse / (len(observed) - 3)
    covariance = sigma_mean * (np.eye(2) + np.ones((2, 2)) / len(observed))
    np.testing.assert_allclose(actual.mean(0), observed.mean() + delta, atol=0.05)
    np.testing.assert_allclose(np.cov(actual.T), covariance, atol=0.025, rtol=0.11)


@pytest.mark.parametrize("kind", ["logit", "poisson"])
def test_discrete_delta_changes_link_prediction_without_changing_observed_posterior(kind):
    frame = _frame(kind)
    options = {"m": 4, "seed": 814, "mh_burn": 80, "mh_steps": 90, "prior_scale": 1.7}
    baseline = _run(frame, kind, 0, **options)
    shifted = _run(frame, kind, np.log(2.0), **options)
    observed = frame.y.notna().to_numpy()
    design = np.column_stack([np.ones(len(frame)), frame.x, frame.z])
    y = frame.y.to_numpy()[observed]
    for a, b in zip(baseline.metadata["imputation_diagnostics"], shifted.metadata["imputation_diagnostics"]):
        assert a["coefficient_draw"] == b["coefficient_draw"]
        assert a["sampler"] == b["sampler"]
        assert a["observed_log_likelihood"] == b["observed_log_likelihood"]
        eta = design @ np.asarray(a["coefficient_draw"])
        if kind == "logit":
            pa, pb = np.asarray(a["predictive_probability"]), np.asarray(b["predictive_probability"])
            np.testing.assert_allclose(pa, expit(eta[~observed]), atol=2e-16)
            np.testing.assert_allclose(pb / (1 - pb), 2 * pa / (1 - pa), rtol=3e-15)
            ll = (y * eta[observed] - np.logaddexp(0, eta[observed])).sum()
        else:
            pa, pb = np.asarray(a["predictive_mean"]), np.asarray(b["predictive_mean"])
            np.testing.assert_allclose(pa, np.exp(eta[~observed]), rtol=2e-15)
            np.testing.assert_allclose(pb, 2 * pa, rtol=3e-15)
            ll = poisson.logpmf(y, np.exp(eta[observed])).sum()
        assert a["observed_log_likelihood"] == pytest.approx(ll, abs=5e-13)


@pytest.mark.parametrize("kind", ["logit", "poisson"])
def test_discrete_target_gradient_matches_independent_likelihood_and_proper_prior(kind):
    from openecon.econometrics.mi.discrete import _poisson_log_posterior
    x = np.column_stack([np.ones(7), np.linspace(-1, 1, 7)])
    y = np.array([0, 1, 1, 0, 1, 0, 0] if kind == "logit" else [0, 1, 3, 0, 2, 1, 4], float)
    beta = np.array([0.2, -0.7])
    scale = 1.6
    b = torch.tensor(beta, requires_grad=True)
    function = chained._logit_log_posterior if kind == "logit" else _poisson_log_posterior
    target = function(torch.tensor(x), torch.tensor(y), b, scale)
    gradient = torch.autograd.grad(target, b)[0].numpy()
    eta = x @ beta
    mean = expit(eta) if kind == "logit" else np.exp(eta)
    expected_gradient = x.T @ (y - mean) - beta / scale**2
    expected_target = ((y * eta - np.logaddexp(0, eta)).sum() if kind == "logit"
                       else (y * eta - np.exp(eta)).sum()) - beta @ beta / (2 * scale**2)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-14)
    assert float(target.detach()) == pytest.approx(expected_target, abs=1e-14)


@pytest.mark.parametrize("kind", ["logit", "poisson"])
def test_actual_discrete_draws_use_shifted_probabilities_not_posthoc_integer_scaling(kind, monkeypatch):
    # Hold posterior parameters fixed, then compare the actual predictive stage
    # with independently known Bernoulli/Poisson conditional moments.
    frame = pd.DataFrame({"y": [0., 1., 2. if kind == "poisson" else 0.] + [np.nan] * 6000})
    beta = torch.tensor([0.3], dtype=torch.float64)
    def fixed(x, y, generator, **kwargs):
        yy = y.numpy()
        likelihood = ((yy * 0.3 - np.logaddexp(0, 0.3)).sum() if kind == "logit"
                      else (yy * 0.3 - np.exp(0.3)).sum())
        diagnostic = {"burn_proposals": 0, "sampling_proposals": 1,
                      "accepted_burn": 0, "accepted_sampling": 1, "acceptance_rate": 1.0,
                      "final_log_posterior": float(likelihood - 0.3**2 / (2 * 2.5**2))}
        if kind == "poisson":
            diagnostic["rejected_nonfinite"] = 0
        return beta, diagnostic
    if kind == "logit":
        monkeypatch.setattr(sensitivity, "_logit_draw", fixed)
    else:
        from openecon.econometrics.mi import discrete
        monkeypatch.setattr(discrete, "_poisson_draw", fixed)
    delta = 0.4
    result = _run(frame, kind, delta, m=1, seed=9819, mh_burn=0, mh_steps=1)
    actual = np.asarray(result.completed_matrices)[0, 3:, 0]
    expected_mean = expit(0.7) if kind == "logit" else np.exp(0.7)
    expected_variance = expected_mean * (1 - expected_mean) if kind == "logit" else expected_mean
    assert actual.mean() == pytest.approx(expected_mean, abs=0.05)
    assert actual.var() == pytest.approx(expected_variance, rel=0.09)
    if kind == "logit":
        assert set(actual) <= {0, 1}
    else:
        np.testing.assert_array_equal(actual, np.floor(actual))
        assert actual.min() >= 0
        # exp(delta) is noninteger, so multiplying nonzero counts by it would
        # violate these exact support checks.
        assert actual.max() <= 1_000_000


@pytest.mark.parametrize("kind", ["normal", "logit", "poisson"])
def test_identity_seed_rng_immutability_and_json_replay(kind):
    frame = _frame(kind)
    before = frame.copy(deep=True)
    rng = torch.random.get_rng_state().clone()
    result = _run(frame, kind, 0.4, m=3, seed=319, mh_burn=12, mh_steps=15)
    assert torch.equal(rng, torch.random.get_rng_state())
    pd.testing.assert_frame_equal(frame, before)
    assert result.method == "mi_chained"
    assert result.metadata["operation"] == "fixed_single_target_pattern_mixture"
    assert result.metadata["convergence_claim"] is False
    assert result.metadata["sampler"]["stationarity_claim"] is False
    replay = _run(frame, kind, 0.4, m=3, seed=319, mh_burn=12, mh_steps=15)
    assert replay.model_dump_json() == result.model_dump_json()
    changed = _run(frame, kind, 0.4, m=3, seed=320, mh_burn=12, mh_steps=15)
    assert changed.completed_matrices != result.completed_matrices
    saved = sensitivity.MIDeltaResult.model_validate_json(result.model_dump_json())
    for i in range(1, 4):
        actual = saved.dataset(i)
        assert actual.index.equals(frame.index)
        assert np.isfinite(actual.to_numpy()).all()
        for name in frame:
            observed = frame[name].notna()
            np.testing.assert_array_equal(actual.loc[observed, name], frame.loc[observed, name])
    assert "imputations" in result.table.columns
    assert result.to_latex()
    with pytest.raises(TypeError):
        result.metadata["delta"] = 42
    with pytest.raises(TypeError):
        result.metadata["imputation_diagnostics"][0]["coefficient_draw"] = (42,)
    mutated = copy.deepcopy(result.model_dump())
    mutated["metadata"]["delta"] = 42
    with pytest.raises(ValueError, match="checksum"):
        sensitivity.MIDeltaResult.model_validate(mutated)
    with pytest.raises(ValueError):
        result.model_copy(update={"metadata": {**result.metadata, "delta": 42}})
    altered = json.loads(result.model_dump_json())
    altered["completed_matrices"][0][0][0] += 1
    with pytest.raises(ValueError, match="observed"):
        sensitivity.MIDeltaResult.model_validate(altered)


def test_explicit_predictors_and_intercept_only_keep_other_selected_columns_complete():
    frame = _frame()
    result = _run(frame, predictors=["x"], m=1)
    assert result.metadata["design_terms"] == ("intercept", "x")
    assert len(result.metadata["imputation_diagnostics"][0]["coefficient_draw"]) == 2
    result = _run(frame, predictors=[], m=1)
    assert result.metadata["design_terms"] == ("intercept",)


@pytest.mark.parametrize("kind", ["logit", "poisson"])
def test_proper_prior_permits_single_class_or_all_zero_and_rank_deficient_predictors(kind):
    frame = pd.DataFrame({"y": [0.] * 9 + [np.nan, np.nan], "x": [1.] * 11})
    result = _run(frame, kind, 0, m=2, mh_burn=10, mh_steps=20)
    assert np.isfinite(np.asarray(result.completed_matrices)).all()
    assert result.metadata["prior"]["family"] == "independent Gaussian"


@pytest.mark.parametrize("options", [
    {"delta": True}, {"delta": float("inf")}, {"delta": "1"}, {"kind": "pmm"},
    {"m": True}, {"m": 0}, {"m": 101}, {"seed": True}, {"seed": -1},
    {"prior_scale": 0}, {"prior_scale": float("nan")}, {"prior_scale": 1e-300},
    {"proposal_scale": -1}, {"mh_burn": -1}, {"mh_steps": 0}, {"mh_steps": True},
    {"max_work": True}, {"predictors": "x"}, {"predictors": ["y"]},
    {"predictors": ["x", "x"]}, {"predictors": ["absent"]},
])
def test_invalid_options_are_refused(options):
    arguments = {"target": "y", "kind": "normal", "delta": 0, **options}
    with pytest.raises(AnalysisError):
        sensitivity.mi_delta(_frame(), ["y", "x", "z"], **arguments)


@pytest.mark.parametrize("mutation", ["other_missing", "all_missing", "no_missing", "infinite", "nonnumeric"])
def test_unsupported_missing_patterns_and_data_are_refused(mutation):
    frame = _frame()
    if mutation == "other_missing":
        frame.loc[frame.index[1], "x"] = np.nan
    elif mutation == "all_missing":
        frame["y"] = np.nan
    elif mutation == "no_missing":
        frame["y"] = frame.y.fillna(0)
    elif mutation == "infinite":
        frame.iloc[1, 0] = np.inf
    else:
        frame["x"] = "a"
    with pytest.raises(AnalysisError):
        _run(frame)


@pytest.mark.parametrize("value", [-1, 0.5, 1_000_001])
def test_poisson_outcome_support_refusal(value):
    frame = _frame("poisson")
    frame.iloc[0, 0] = value
    with pytest.raises(AnalysisError, match="integers"):
        _run(frame, "poisson")


def test_binary_outcome_and_gaussian_rank_or_sse_refusal():
    frame = _frame("logit")
    frame.iloc[0, 0] = 2
    with pytest.raises(AnalysisError, match="exactly 0 or 1"):
        _run(frame, "logit")
    frame = _frame()
    frame["x"] = 1.0
    with pytest.raises(AnalysisError, match="rank deficient"):
        _run(frame)
    frame = pd.DataFrame({"y": [2., 2., 2., np.nan]})
    with pytest.raises(AnalysisError, match="residual variance"):
        _run(frame)


@pytest.mark.parametrize("delta", [100., -1000., 1000.])
def test_poisson_predictive_limits_refuse_without_clipping(delta, monkeypatch):
    from openecon.econometrics.mi import discrete
    monkeypatch.setattr(discrete, "_poisson_draw", lambda *a, **kw: (torch.zeros(3, dtype=torch.float64), {}))
    with pytest.raises(AnalysisError):
        _run(_frame("poisson"), "poisson", delta, m=1)


def test_work_and_workspace_are_refused_before_first_tensor_allocation(monkeypatch):
    frame = _frame("logit", n=100)
    def unexpected(*args, **kwargs):
        raise AssertionError("Tensor was allocated before full admission")
    monkeypatch.setattr(torch, "tensor", unexpected)
    with pytest.raises(AnalysisError, match="work"):
        _run(frame, "logit", m=100, mh_steps=100_000, max_work=1000)
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError, match="workspace"):
            _run(frame, "logit", m=100)


def test_nonresident_and_unselected_target_and_extras_are_refused():
    with pytest.raises(AnalysisError, match="resident"):
        sensitivity.mi_delta(object(), ["y"], target="y", kind="normal", delta=0)
    with pytest.raises(AnalysisError, match="target"):
        sensitivity.mi_delta(_frame(), ["x"], target="y", kind="normal", delta=0)
    for extra in ({"weights": "w"}, {"device": "cuda"}):
        with pytest.raises(TypeError):
            _run(_frame(), **extra)


def test_runtime_source_uses_no_reference_numerical_engines():
    source = Path(sensitivity.__file__).read_text()
    assert "import numpy" not in source and "import scipy" not in source and "import statsmodels" not in source


@pytest.mark.parametrize("mutation", [
    "delta", "delta_bool", "delta_identified", "kind", "prior", "beta", "eta",
    "prediction", "likelihood", "positions", "seed", "work", "plan", "stationarity",
    "proposal_count", "acceptance_rate", "posterior_target", "binary_support",
])
def test_rehashed_forged_binary_scientific_state_is_refused(mutation):
    result = _run(_frame("logit"), "logit", 0.3, m=1, seed=604, mh_burn=7, mh_steps=8)
    saved = result.model_dump()
    meta = saved["metadata"]
    diagnostic = meta["imputation_diagnostics"][0]
    if mutation == "delta":
        meta["delta"] += 1
    elif mutation == "delta_bool":
        meta["delta"] = True
    elif mutation == "delta_identified":
        meta["delta_identified_from_observed_data"] = True
    elif mutation == "kind":
        meta["kind"] = "normal"
    elif mutation == "prior":
        meta["prior"]["scale"] *= 2
    elif mutation == "beta":
        diagnostic["coefficient_draw"][0] += 1
    elif mutation == "eta":
        diagnostic["base_missing_linear_predictor"][0] += 1
    elif mutation == "prediction":
        diagnostic["predictive_probability"][0] = True
    elif mutation == "likelihood":
        diagnostic["observed_log_likelihood"] += 1
    elif mutation == "positions":
        meta["missing_positions"][0] += 1
    elif mutation == "seed":
        diagnostic["seed"] += 1
    elif mutation == "work":
        meta["projected_work"] += 1
        meta["work_estimate"] += 1
    elif mutation == "plan":
        plan = meta["chained_resource_plan"]
        plan["buffers"]["predictive vectors and local sampler state"] += 1
        plan["estimated_workspace_bytes"] += 1
    elif mutation == "stationarity":
        meta["sampler"]["stationarity_claim"] = True
    elif mutation == "proposal_count":
        diagnostic["sampler"]["sampling_proposals"] += 1
    elif mutation == "acceptance_rate":
        diagnostic["sampler"]["acceptance_rate"] = 0.12345
    elif mutation == "posterior_target":
        diagnostic["sampler"]["final_log_posterior"] += 1
    else:
        matrices = [[list(row) for row in matrix] for matrix in saved["completed_matrices"]]
        matrices[0][meta["missing_positions"][0]][0] = 0.5
        saved["completed_matrices"] = matrices
    saved["integrity_sha256"] = _digest({key: value for key, value in saved.items() if key != "integrity_sha256"})
    with pytest.raises(ValueError):
        sensitivity.MIDeltaResult.model_validate(saved)


@pytest.mark.parametrize("kind,field", [("normal", "sigma2_draw"), ("normal", "predictive_mean"),
                                        ("poisson", "predictive_mean"), ("poisson", "observed_log_likelihood")])
def test_rehashed_normal_and_count_model_forgery_is_refused(kind, field):
    result = _run(_frame(kind), kind, 0.2, m=1, mh_burn=5, mh_steps=6)
    saved = result.model_dump()
    diagnostic = saved["metadata"]["imputation_diagnostics"][0]
    if field == "sigma2_draw":
        diagnostic[field] *= 2
    elif field == "predictive_mean":
        diagnostic[field][0] += 1
    else:
        diagnostic[field] += 1
    saved["integrity_sha256"] = _digest({key: value for key, value in saved.items() if key != "integrity_sha256"})
    with pytest.raises(ValueError):
        sensitivity.MIDeltaResult.model_validate_json(json.dumps(saved))


@pytest.mark.parametrize("shape", ["empty", "rows", "columns"])
def test_shape_limits_are_refused_before_tensor_construction(shape, monkeypatch):
    if shape == "empty":
        frame = pd.DataFrame({"y": []})
    elif shape == "rows":
        frame = pd.DataFrame({"y": np.zeros(10001)})
    else:
        frame = pd.DataFrame({"y": [0.] * 8, **{f"x{i}": [1.] * 8 for i in range(16)}})
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: pytest.fail("tensor constructed before shape refusal"))
    with pytest.raises(AnalysisError):
        _run(frame)


def test_target_last_and_resident_mapping_preserve_selected_column_geometry():
    frame = _frame()[["z", "x", "y"]]
    result = _run(frame, m=1, seed=94)
    assert isinstance(result, sensitivity.MIDeltaResult)
    assert result.columns == ("z", "x", "y")
    np.testing.assert_array_equal(np.asarray(result.completed_matrices)[0, :, :2], frame[["z", "x"]])
    mapping = {name: frame[name].tolist() for name in frame}
    replay = sensitivity.mi_delta(mapping, list(frame), target="y", kind="normal", delta=0, m=1, seed=94)
    np.testing.assert_array_equal(replay.completed_matrices, result.completed_matrices)


@pytest.mark.parametrize("kind", ["normal", "logit", "poisson"])
def test_creation_and_saved_state_replay_ignore_ambient_dtype_and_device(kind):
    frame = _frame(kind)
    options = {"m": 2, "seed": 517, "mh_burn": 5, "mh_steps": 7}
    reference = _run(frame, kind, .4, **options)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            actual = _run(frame, kind, .4, **options)
            restored = sensitivity.MIDeltaResult.model_validate_json(reference.model_dump_json())
    finally:
        torch.set_default_dtype(old_dtype)
    assert actual.model_dump_json() == reference.model_dump_json()
    assert restored.model_dump_json() == reference.model_dump_json()
