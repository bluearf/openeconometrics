"""Independent deterministic complete-state tests; no Monte Carlo study outcomes."""

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import invwishart, t

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.bayesian_var import draws as draws_module
from openecon.econometrics.bayesian_var import forecast as forecast_module
from openecon.econometrics.bayesian_var.admission import admit, digest, metadata_size
from openecon.econometrics.bayesian_var.api import (
    BayesianVARContrast,
    BayesianVARPrediction,
    bayes_var,
    bayes_var_contrast,
    bayes_var_predict,
)
from openecon.econometrics.bayesian_var.design import restore_index
from openecon.econometrics.bayesian_var.draws import BayesianVARDraws, bayes_var_draws
from openecon.econometrics.bayesian_var.forecast import BayesianVARForecast, bayes_var_forecast
from openecon.econometrics.bayesian_var.impulse import BayesianVARImpulse, bayes_var_irf
from openecon.econometrics.bayesian_var.kernels import matrix_t_logpdf
from openecon.econometrics.bayesian_var.moments import availability
from openecon.econometrics.bayesian_var.posterior import BayesianVARPosterior, raw, restore
from reference import bayesian_var_mniw_oracle as oracle


def fixture(m=2, p=2, n=16, intercept=True, nu=7.0):
    time = np.arange(100, 100 + n, dtype=np.int64)
    levels = np.column_stack([np.sin(time / (3 + j)) + 0.03 * time * (j + 1) for j in range(m)])
    k = m * p + int(intercept)
    a = np.eye(k) + 0.06 * np.ones((k, k))
    s = np.eye(m) + 0.23 * np.ones((m, m))
    prior = {
        "mean": (np.arange(k * m).reshape(k, m) / 100).tolist(),
        "row_scale": (a @ a.T).tolist(),
        "innovation_scale": s.tolist(),
        "degrees_of_freedom": nu,
    }
    names = [f"y{j}" for j in range(m)]
    data = pd.DataFrame({"time": time, **dict(zip(names, levels.T, strict=True))})
    result = bayes_var(data, names, time="time", prior=prior, lags=p, intercept=intercept)
    return data, levels, prior, result


@pytest.mark.parametrize("m,p,intercept", [(2, 1, True), (2, 4, False), (3, 2, True), (4, 4, True)])
def test_augmented_qr_full_joint_oracle(m, p, intercept):
    _, levels, prior, result = fixture(m, p, 24, intercept)
    x, y, mn, vn, sn, nu = oracle.posterior(
        levels,
        p,
        intercept,
        np.array(prior["mean"]),
        np.array(prior["row_scale"]),
        np.array(prior["innovation_scale"]),
        prior["degrees_of_freedom"],
    )
    np.testing.assert_allclose(result.algebra.location, mn, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(result.algebra.row_scale, vn, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(result.algebra.innovation_scale, sn, rtol=2e-12, atol=2e-12)
    expected_cov = np.kron(sn / (nu - m - 1), vn)
    np.testing.assert_allclose(
        result.algebra.coefficient_covariance, expected_cov, rtol=3e-12, atol=3e-12
    )
    assert abs(expected_cov[0, len(vn)]) > 1e-8  # Dropping cross-equation blocks is detectable.
    np.testing.assert_allclose(result.algebra.innovation_mean, invwishart.mean(nu, sn))
    np.testing.assert_allclose(
        result.algebra.innovation_covariance,
        oracle.sigma_covariance_polarized(sn, nu),
        rtol=2e-11,
        atol=2e-12,
    )
    q = len(vn) * m
    sigma_covariance = oracle.sigma_covariance_polarized(sn, nu)
    joint = np.zeros((q + len(sigma_covariance), q + len(sigma_covariance)))
    joint[:q, :q], joint[q:, q:] = expected_cov, sigma_covariance
    np.testing.assert_allclose(result.algebra.joint_covariance, joint, rtol=2e-11, atol=2e-12)
    positions = [(i, j) for j in range(m) for i in range(j, m)]
    for j, (i, k) in enumerate(positions):
        np.testing.assert_allclose(
            result.algebra.innovation_covariance[j][j], invwishart.var(nu, sn)[i, k]
        )
    a = mn + np.arange(mn.size).reshape(mn.shape) / 200
    np.testing.assert_allclose(
        matrix_t_logpdf(a, mn, result.algebra.precision, sn, nu),
        oracle.matrix_t_logpdf(a, mn, vn, sn, nu),
        atol=3e-12,
    )
    # Matrix-t coefficient density does not equal a generic elliptical multivariate-t.
    from scipy.stats import multivariate_t

    wrong = multivariate_t.logpdf(
        a.T.reshape(-1), loc=mn.T.reshape(-1), shape=np.kron(sn, vn) / (nu - m + 1), df=nu - m + 1
    )
    assert abs(wrong - oracle.matrix_t_logpdf(a, mn, vn, sn, nu)) > 1e-4
    assert result.nobs == len(y) and result.nobs_original == len(levels)


def test_exact_rank_one_df_and_fixed_design_full_predictive_covariance():
    _, _, _, result = fixture()
    left, right = np.array([1.0, 0.3, -0.2, 0.4, 0.1]), np.array([0.7, -0.4])
    contrast = bayes_var_contrast(result, left.tolist(), right.tolist(), threshold=0.2)
    a = result.algebra
    location = left @ np.array(a.location) @ right
    df = a.degrees_of_freedom - 2 + 1
    scale = np.sqrt(
        (left @ np.array(a.row_scale) @ left) * (right @ np.array(a.innovation_scale) @ right) / df
    )
    np.testing.assert_allclose(
        contrast.interval, t.ppf([0.025, 0.975], df, loc=location, scale=scale)
    )
    np.testing.assert_allclose(
        contrast.probability_le_threshold, t.cdf(0.2, df, loc=location, scale=scale)
    )
    assert df != a.degrees_of_freedom
    x = [[1.0, 0.1, 0.2, 0.3, 0.4], [1.0, -0.5, 0.4, 0.7, -0.1]]
    prediction = bayes_var_predict(result, x, terms=result.terms)
    rows = np.array(x) @ np.array(a.row_scale) @ np.array(x).T + np.eye(2)
    np.testing.assert_allclose(
        prediction.full_covariance, np.kron(np.array(a.innovation_mean), rows)
    )
    assert abs(np.array(prediction.full_covariance)[0, 1]) > 1e-8
    assert abs(np.array(prediction.full_covariance)[0, 2]) > 1e-8
    np.testing.assert_allclose(
        prediction.row_predictive_scale[0], rows[0, 0] * np.array(a.innovation_scale) / df
    )


@pytest.mark.parametrize(
    "nu,coefficient_cov,joint_cov", [(1.1, False, False), (3.5, True, False), (5.0, True, True)]
)
def test_low_df_moments_do_not_get_finite_sample_rescue(nu, coefficient_cov, joint_cov):
    _, _, _, result = fixture(p=1, n=2, nu=nu)
    assert (result.algebra.coefficient_covariance is not None) is coefficient_cov
    assert (result.algebra.joint_covariance is not None) is joint_cov
    assert result.algebra.coefficient_mean is not None
    if not coefficient_cov:
        assert result.algebra.joint_mean is None
    contrast = bayes_var_contrast(result, [1.0, 0.0, 0.0], [1.0, 0.0])
    assert (contrast.variance is not None) is coefficient_cov


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(5, 37, 2, name="row"),
        pd.Index([f"row{i}" for i in range(16)], name="row"),
        pd.Index(np.arange(16, dtype=np.int32), name="row"),
        pd.date_range("2001-01-01", periods=16, freq="D", name="row").as_unit("ms"),
    ],
)
def test_source_calendar_permutation_and_exact_index(index):
    data, _, prior, baseline = fixture()
    data.index = index
    reordered = data.iloc[[3, 1, 7, 0, 2, 5, 4, 6, 15, 9, 8, 10, 12, 11, 14, 13]]
    result = bayes_var(reordered, ["y0", "y1"], time="time", prior=prior, lags=2)
    pd.testing.assert_index_equal(
        restore_index(raw(result.source.source_index)), reordered.index, exact=True
    )
    assert result.source.permutation != tuple(range(16))
    np.testing.assert_allclose(result.algebra.location, baseline.algebra.location)
    restored = BayesianVARPosterior.model_validate_json(result.model_dump_json())
    assert raw(restored) == raw(result)


def fixed_primitives(parent, count, seed):
    m, k = len(parent.source.series), len(parent.terms)
    factors = [
        (np.eye(m) * (2 + j / 10) + np.tril(np.ones((m, m)) * 0.12, -1)).tolist()
        for j in range(count)
    ]
    normals = [(np.arange(k * m).reshape(k, m) / 30 - 0.3 + j * 0.1).tolist() for j in range(count)]
    return factors, normals


def test_fixed_joint_primitives_forecast_irf_full_arrays_and_roundtrip(monkeypatch):
    _, levels, _, result = fixture()
    monkeypatch.setattr(draws_module, "primitive_stream", fixed_primitives)
    count, h = 3, 4
    z = (np.arange(count * h * 2).reshape(count, h, 2) / 50 - 0.2).tolist()
    monkeypatch.setattr(forecast_module, "innovation_stream", lambda *args: z)
    draws = bayes_var_draws(result, draws=count, seed=19)
    forecast = bayes_var_forecast(draws, horizon=h, innovation_seed=7)
    for j in range(count):
        b, sigma = oracle.joint_transform(
            np.array(result.algebra.location),
            np.array(result.algebra.row_scale),
            np.array(result.algebra.innovation_scale),
            np.array(draws.bartlett[j]),
            np.array(draws.standard_normals[j]),
        )
        np.testing.assert_allclose(draws.coefficients[j], b)
        np.testing.assert_allclose(draws.innovations[j], sigma)
        mean, outcome = oracle.recursive_paths(levels, b, sigma, z[j], 2, True)
        np.testing.assert_allclose(forecast.conditional_mean_paths[j], mean)
        np.testing.assert_allclose(forecast.outcome_paths[j], outcome)
        for orthogonalized in (False, True):
            irf = bayes_var_irf(draws, horizon=h, orthogonalized=orthogonalized)
            np.testing.assert_allclose(
                irf.responses[j], oracle.impulse(b, sigma, 2, 2, True, h, orthogonalized)
            )
            assert raw(BayesianVARImpulse.model_validate_json(irf.model_dump_json())) == raw(irf)
    flat = np.asarray(forecast.outcome_paths).reshape(count, -1)
    np.testing.assert_allclose(
        forecast.outcome_summary.covariance_mc_estimate, np.cov(flat, rowvar=False)
    )
    assert "Monte Carlo" in forecast.outcome_summary.covariance_label
    assert raw(BayesianVARForecast.model_validate_json(forecast.model_dump_json())) == raw(forecast)
    assert raw(BayesianVARDraws.model_validate_json(draws.model_dump_json())) == raw(draws)


def test_moment_guards_have_distinct_orthogonalized_extra_factor():
    assert not availability(7, 2, 3)["second"]
    assert availability(7.01, 2, 3)["second"]
    assert availability(7.01, 2, 3, impulse=True)["second"]
    assert not availability(9, 2, 3, impulse=True, orthogonalized=True)["second"]
    assert availability(9.01, 2, 3, impulse=True, orthogonalized=True)["second"]
    assert availability(1.01, 2, 0, impulse=True)["fourth"]
    assert not availability(3, 2, 0, impulse=True, orthogonalized=True)["second"]


def test_low_df_simulation_summary_withholds_population_covariance(monkeypatch):
    _, _, _, result = fixture(p=1, n=2, nu=1.1)
    monkeypatch.setattr(draws_module, "primitive_stream", fixed_primitives)
    draws = bayes_var_draws(result, draws=3, seed=2)
    monkeypatch.setattr(
        forecast_module,
        "innovation_stream",
        lambda count, h, m, seed: np.zeros((count, h, m)).tolist(),
    )
    forecast = bayes_var_forecast(draws, horizon=3, innovation_seed=3)
    assert forecast.outcome_summary.covariance_mc_estimate is None
    assert forecast.outcome_summary.mean_mc_standard_error is None
    assert forecast.outcome_summary.covariance_mc_standard_error is None
    assert forecast.outcome_summary.sample_mean is None
    assert forecast.outcome_summary.pointwise_quantiles


@pytest.mark.parametrize("domain", ["multi", "nullable", "categorical", "tz", "float", "object"])
def test_unsupported_index_refusal_precedes_tensor_and_rng(monkeypatch, domain):
    data, _, prior, _ = fixture()
    bad = {
        "multi": pd.MultiIndex.from_arrays([range(16), range(16)]),
        "nullable": pd.Index(pd.array(range(16), dtype="Int64")),
        "categorical": pd.CategoricalIndex([str(i) for i in range(16)]),
        "tz": pd.date_range("2001", periods=16, tz="UTC"),
        "float": pd.Index(np.arange(16, dtype=float)),
        "object": pd.Index([object() for _ in range(16)]),
    }[domain]
    data.index = bad
    monkeypatch.setattr(
        torch, "tensor", lambda *a, **k: pytest.fail("tensor allocated before index refusal")
    )
    with pytest.raises(AnalysisError, match="index|indices|Index"):
        bayes_var(data, ["y0", "y1"], time="time", prior=prior, lags=2)


@pytest.mark.parametrize(
    "kind", ["gap", "duplicate", "float_time", "nonfinite", "missing", "precision"]
)
def test_source_refusal_precedes_factorization(monkeypatch, kind):
    data, _, prior, _ = fixture()
    if kind == "gap":
        data.loc[0, "time"] -= 1
    elif kind == "duplicate":
        data.loc[0, "time"] = data.loc[1, "time"]
    elif kind == "float_time":
        data["time"] = data["time"].astype(float)
    elif kind == "nonfinite":
        data.loc[0, "y0"] = np.inf
    elif kind == "missing":
        data.loc[0, "y0"] = np.nan
    else:
        data["y0"] = pd.Series([2**53 + 1] * len(data), dtype="int64")
    monkeypatch.setattr(
        torch.linalg,
        "cholesky",
        lambda *a, **k: pytest.fail("factorization preceded source refusal"),
    )
    with pytest.raises(AnalysisError):
        bayes_var(data, ["y0", "y1"], time="time", prior=prior, lags=2)


@pytest.mark.parametrize(
    "kind", ["row_asymmetry", "sigma_indefinite", "improper", "boolean", "shape"]
)
def test_prior_no_repair_or_defaults(kind):
    data, _, prior, _ = fixture()
    if kind == "row_asymmetry":
        prior["row_scale"][0][1] += 0.01
    elif kind == "sigma_indefinite":
        prior["innovation_scale"][0][0] = -1
    elif kind == "improper":
        prior["degrees_of_freedom"] = 1
    elif kind == "boolean":
        prior["mean"][0][0] = True
    else:
        prior["mean"].pop()
    with pytest.raises(AnalysisError):
        bayes_var(data, ["y0", "y1"], time="time", prior=prior, lags=2)


@pytest.mark.parametrize(
    "field",
    ["location", "coefficient_covariance", "innovation_covariance", "coefficient_intervals"],
)
def test_tampered_algebra_cannot_be_rescued_with_new_digest(field):
    _, _, _, result = fixture()
    body = result.model_dump(mode="json")
    leaf = body["algebra"][field]
    while isinstance(leaf[0], list):
        leaf = leaf[0]
    leaf[0] += 0.03
    body["integrity_sha256"] = digest({k: v for k, v in body.items() if k != "integrity_sha256"})
    with pytest.raises(AnalysisError, match="algebra"):
        restore(body)


def test_unsupported_restored_index_preflight_before_tensor(monkeypatch):
    _, _, _, result = fixture()
    body = result.model_dump(mode="json")
    body["source"]["source_index"] = {
        "kind": "index",
        "dtype": "string",
        "name": None,
        "values": [str(i) for i in range(16)],
    }
    monkeypatch.setattr(
        torch, "tensor", lambda *a, **k: pytest.fail("tensor before canonical source replay")
    )
    with pytest.raises(AnalysisError):
        restore(body)


def test_query_and_contrast_roundtrip_construct_copy_summary_validation():
    _, _, _, result = fixture()
    q = bayes_var_predict(result, [[1.0, 0.1, 0.2, 0.3, 0.4]], terms=result.terms)
    c = bayes_var_contrast(result, [1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0])
    assert raw(BayesianVARPrediction.model_validate_json(q.model_dump_json())) == raw(q)
    assert raw(BayesianVARContrast.model_validate_json(c.model_dump_json())) == raw(c)
    assert raw(result.model_copy(deep=True)) == raw(result)
    with pytest.raises(AnalysisError):
        result.model_copy(update={"nobs": 999})
    forged = BayesianVARPosterior.model_construct(**result.__dict__ | {"nobs": 999})
    with pytest.raises(AnalysisError):
        forged.summary()
    with pytest.raises(AnalysisError):
        forged.model_dump_json()


def test_early_work_memory_shapes_and_metadata_limits(monkeypatch):
    with pytest.raises(AnalysisError, match="work"):
        admit(10000, 4, 4, True, max_work=1)
    with pytest.raises(AnalysisError, match="2,000,000"):
        admit(10, 2, 1, True, queries=10000)
    with pytest.raises(AnalysisError):
        metadata_size(2**1000000)
    assert metadata_size(2**127) == metadata_size(-(2**127))
    data, _, prior, _ = fixture()
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: pytest.fail("tensor before budget"))
    with pytest.raises(AnalysisError):
        bayes_var(data, ["y0", "y1"], time="time", prior=prior, lags=2, max_bytes=1024)


def test_no_silent_rank_drop_in_proper_prior_collinear_design():
    data, _, prior, _ = fixture()
    data["y1"] = data["y0"]
    result = bayes_var(data, ["y0", "y1"], time="time", prior=prior, lags=2)
    assert len(result.terms) == 5
    assert len(result.algebra.coefficient_covariance) == 10


def test_unstable_draws_are_retained_and_finite_horizon_is_computed(monkeypatch):
    _, _, _, result = fixture()

    def unstable(parent, count, seed):
        a, z = fixed_primitives(parent, count, seed)
        return a, (np.asarray(z) + 100).tolist()

    monkeypatch.setattr(draws_module, "primitive_stream", unstable)
    draws = bayes_var_draws(result, draws=3, seed=2)
    assert len(draws.coefficients) == 3
    assert not any(draws.stable)
    assert min(draws.spectral_radii) > 1
    monkeypatch.setattr(
        forecast_module,
        "innovation_stream",
        lambda count, h, m, seed: np.zeros((count, h, m)).tolist(),
    )
    forecast = bayes_var_forecast(draws, horizon=2, innovation_seed=3)
    assert len(forecast.outcome_paths) == 3


def test_honest_forecast_irf_overflow_no_draw_suppression(monkeypatch):
    _, _, _, result = fixture()

    def large(parent, count, seed):
        a, z = fixed_primitives(parent, count, seed)
        return a, (np.asarray(z) + 1e100).tolist()

    monkeypatch.setattr(draws_module, "primitive_stream", large)
    draws = bayes_var_draws(result, draws=3, seed=2)
    assert len(draws.coefficients) == 3 and not any(draws.stable)
    monkeypatch.setattr(
        forecast_module,
        "innovation_stream",
        lambda count, h, m, seed: np.zeros((count, h, m)).tolist(),
    )
    with pytest.raises(AnalysisError, match="overflowed") as forecast_error:
        bayes_var_forecast(draws, horizon=4, innovation_seed=3)
    assert forecast_error.value.code == "numeric_failure"
    with pytest.raises(AnalysisError, match="overflowed") as impulse_error:
        bayes_var_irf(draws, horizon=4)
    assert impulse_error.value.code == "numeric_failure"


def test_covariance_mc_error_requires_fourth_moment(monkeypatch):
    _, _, _, result = fixture(p=1, n=8, nu=5.0)
    monkeypatch.setattr(draws_module, "primitive_stream", fixed_primitives)
    draws = bayes_var_draws(result, draws=3, seed=2)
    monkeypatch.setattr(
        forecast_module,
        "innovation_stream",
        lambda count, h, m, seed: np.zeros((count, h, m)).tolist(),
    )
    forecast = bayes_var_forecast(draws, horizon=4, innovation_seed=3)
    assert forecast.outcome_summary.moment_availability.second
    assert not forecast.outcome_summary.moment_availability.fourth
    assert forecast.outcome_summary.covariance_mc_estimate is not None
    assert forecast.outcome_summary.mean_mc_standard_error is not None
    assert forecast.outcome_summary.covariance_mc_standard_error is None


def test_bool_cached_numeric_cell_is_refused_without_coercion():
    _, _, _, result = fixture()
    body = result.model_dump(mode="json")
    q = len(result.terms) * 2
    assert body["algebra"]["joint_covariance"][0][q] == 0.0
    body["algebra"]["joint_covariance"][0][q] = False
    with pytest.raises(AnalysisError):
        restore(body)


def test_draw_budget_refusal_precedes_rng(monkeypatch):
    _, _, _, result = fixture()
    monkeypatch.setattr(
        draws_module, "primitive_stream", lambda *args: pytest.fail("RNG before admission")
    )
    with pytest.raises(AnalysisError):
        bayes_var_draws(result, draws=10001, seed=2)
    with pytest.raises(AnalysisError):
        bayes_var_draws(result, draws=True, seed=2)


@pytest.mark.parametrize("field", ["coefficients", "innovations", "spectral_radii", "stable"])
def test_joint_draw_tampering_rejected_even_with_refreshed_digest(monkeypatch, field):
    _, _, _, result = fixture()
    monkeypatch.setattr(draws_module, "primitive_stream", fixed_primitives)
    draws = bayes_var_draws(result, draws=3, seed=2)
    body = draws.model_dump(mode="json")
    if field == "stable":
        body[field][0] = not body[field][0]
    elif field == "spectral_radii":
        body[field][0] += 0.1
    else:
        body[field][0][0][0] += 0.1
    body["integrity_sha256"] = digest({k: v for k, v in body.items() if k != "integrity_sha256"})
    with pytest.raises(AnalysisError):
        BayesianVARDraws.model_validate(body)


def test_query_admission_before_parent_tensor_replay(monkeypatch):
    _, _, _, result = fixture()
    monkeypatch.setattr(
        torch, "tensor", lambda *a, **k: pytest.fail("parent tensor replay before query admission")
    )
    with pytest.raises(AnalysisError):
        bayes_var_predict(result, [[1.0, 0.0, 0.0, 0.0, 0.0]] * 10000, terms=result.terms)
    with pytest.raises(AnalysisError):
        bayes_var_contrast(result, [True, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0])
    with pytest.raises(AnalysisError):
        bayes_var_draws(result, draws=10001, seed=2)


def test_horizon_admission_before_joint_parent_tensor_replay(monkeypatch):
    _, _, _, result = fixture()
    monkeypatch.setattr(draws_module, "primitive_stream", fixed_primitives)
    draws = bayes_var_draws(result, draws=3, seed=2)
    monkeypatch.setattr(
        torch, "tensor", lambda *a, **k: pytest.fail("tensor before horizon admission")
    )
    with pytest.raises(AnalysisError):
        bayes_var_forecast(draws, horizon=25, innovation_seed=2)
    with pytest.raises(AnalysisError):
        bayes_var_irf(draws, horizon=25)


def test_explicit_integer_queries_and_contrasts_are_exactly_normalized():
    _, _, _, result = fixture()
    q = bayes_var_predict(result, [[1, 0, 0, 0, 0]], terms=result.terms)
    c = bayes_var_contrast(result, [1, 0, 0, 0, 0], [1, 0])
    assert q.design == ((1.0, 0.0, 0.0, 0.0, 0.0),)
    assert c.left == (1.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(AnalysisError, match="representable"):
        bayes_var_predict(result, [[1, 2**53 + 1, 0, 0, 0]], terms=result.terms)


def test_unsupported_index_precedes_deep_memory_or_object_item_side_effect(monkeypatch):
    data, _, prior, _ = fixture()
    data.index = pd.MultiIndex.from_arrays([range(16), range(16)])
    monkeypatch.setattr(
        pd.MultiIndex,
        "memory_usage",
        lambda *a, **k: pytest.fail("index cache/copy before index refusal"),
    )
    with pytest.raises(AnalysisError):
        bayes_var(data, ["y0", "y1"], time="time", prior=prior, lags=2)

    class Arbitrary:
        def item(self):
            pytest.fail("arbitrary item method executed before refusal")

    data.index = pd.Index([Arbitrary() for _ in range(16)])
    with pytest.raises(AnalysisError):
        bayes_var(data, ["y0", "y1"], time="time", prior=prior, lags=2)
