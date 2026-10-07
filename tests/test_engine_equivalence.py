"""Independent estimator/inference oracles for the single float64 tensor core.

statsmodels is intentionally imported only by tests. Its declared covariance,
finite-sample correction and reference distribution match the public contract.
This establishes equivalence on these fixtures, not blanket Stata parity.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import special, stats

from openecon.analysis import fit
from openecon.models import ModelSpec


@pytest.fixture
def engine():
    import torch
    torch.set_num_threads(1)
    return {}


def _assert_matches(actual, expected, *, coefficient_atol=2e-9, covariance_rtol=2e-7):
    """Use absolute floors for null coefficients and tiny tail probabilities."""
    assert actual.provenance["backend"] == "openecon.torch"
    assert actual.provenance["device"] == "cpu"
    assert_allclose([c.estimate for c in actual.coefficients], expected.params,
                    rtol=2e-8, atol=coefficient_atol)
    assert_allclose(actual.covariance_matrix, expected.cov_params(),
                    rtol=covariance_rtol, atol=2e-10)
    assert_allclose([c.std_error for c in actual.coefficients], expected.bse,
                    rtol=covariance_rtol, atol=2e-10)
    assert_allclose([c.p_value for c in actual.coefficients], expected.pvalues,
                    rtol=2e-6, atol=2e-9)
    assert_allclose([[c.ci_low, c.ci_high] for c in actual.coefficients],
                    expected.conf_int(alpha=actual.spec.alpha), rtol=2e-7, atol=2e-8)
    chart_rows = [p["row"] for p in actual.predictions]
    assert_allclose([p["fitted"] for p in actual.predictions],
                    np.asarray(expected.predict())[chart_rows], rtol=2e-7, atol=2e-9)
    assert actual.nobs == int(expected.nobs)
    for metric, attribute in [("log_likelihood", "llf"), ("aic", "aic"), ("bic", "bic")]:
        assert actual.metrics[metric] == pytest.approx(getattr(expected, attribute), rel=2e-8, abs=2e-8)
    actual.model_dump_json()  # All emitted numbers, including intervals, must be finite.


@pytest.mark.parametrize("seed", [14, 90210, 271828])
@pytest.mark.parametrize("covariance", ["nonrobust", "HC1", "HC3", "cluster"])
@pytest.mark.parametrize("intercept", [True, False])
def test_random_heteroskedastic_ols_matches_independent_oracle(engine, seed, covariance, intercept):
    rng = np.random.default_rng(seed)
    n = 137
    x = rng.normal(size=(n, 4))
    groups = rng.choice(["north", "south", "west", "east", "central", "coast", "rural"], n)
    errors = rng.normal(size=n) * (0.4 + np.abs(x[:, 0]))
    frame = pd.DataFrame(x, columns=["x0", "x1", "x2", "x3"])
    frame["y"] = 0.6 + x @ np.array([0.8, -0.25, 0.0, 0.05]) + errors
    frame["group"] = groups
    spec = ModelSpec(outcome="y", predictors=list(frame.columns[:4]), covariance=covariance,
                     intercept=intercept, cluster="group" if covariance == "cluster" else None,
                     alpha=0.10, **engine)
    design = np.column_stack([np.ones(n), x]) if intercept else x
    options = {"groups": groups, "use_correction": True, "df_correction": True} if covariance == "cluster" else {}
    expected = sm.OLS(frame.y, design, hasconst=intercept).fit(cov_type=covariance, cov_kwds=options, use_t=True)
    actual = fit(spec, data=frame)
    _assert_matches(actual, expected)
    assert actual.metrics["r_squared"] == pytest.approx(expected.rsquared, abs=1e-12)
    assert actual.metrics["adjusted_r_squared"] == pytest.approx(expected.rsquared_adj, abs=1e-12)
    assert actual.inference["df_inference"] == (6 if covariance == "cluster" else n - design.shape[1])


@pytest.mark.parametrize("estimator", ["logit", "probit"])
@pytest.mark.parametrize("covariance", ["nonrobust", "cluster"])
@pytest.mark.parametrize("intercept", [True, False])
def test_binary_observed_information_and_cluster_scores_match_oracle(engine, estimator, covariance, intercept):
    rng = np.random.default_rng(711)
    n = 701
    x = rng.normal(size=(n, 3))
    eta = -0.35 + x @ np.array([0.65, -0.30, 0.10])
    probability = special.expit(eta) if estimator == "logit" else stats.norm.cdf(eta)
    frame = pd.DataFrame(x, columns=["x0", "x1", "x2"])
    frame["y"] = rng.binomial(1, probability)
    frame["g"] = np.arange(n) % 17
    design = np.column_stack([np.ones(n), x]) if intercept else x
    cls = sm.Logit if estimator == "logit" else sm.Probit
    kwargs = {"cov_kwds": {"groups": frame.g, "use_correction": True, "df_correction": True}} if covariance == "cluster" else {}
    expected = cls(frame.y, design).fit(disp=False, maxiter=200, tol=1e-11,
                                      cov_type=covariance, use_t=False, **kwargs)
    actual = fit(ModelSpec(estimator=estimator, outcome="y", predictors=["x0", "x1", "x2"],
                          intercept=intercept, covariance=covariance,
                          cluster="g" if covariance == "cluster" else None, **engine), data=frame)
    _assert_matches(actual, expected)
    # statsmodels' no-intercept llnull nevertheless fits an intercept-only model.
    assert actual.metrics["pseudo_r_squared"] == pytest.approx(expected.prsquared, abs=2e-8)
    assert actual.inference["use_t"] is False
    assert actual.inference["df_inference"] is None


@pytest.mark.parametrize("estimator,covariance", [("ols", "HC3"), ("ols", "cluster"), ("logit", "cluster"), ("probit", "nonrobust")])
def test_categories_missing_sample_and_explicit_reference_full_pipeline(engine, estimator, covariance):
    rng = np.random.default_rng(51)
    n = 333
    x = rng.normal(size=n)
    labels = np.resize(["B", "C", "A"], n)
    eta = 0.1 + 0.35 * x + 0.2 * (labels == "A") - 0.3 * (labels == "B")
    y = eta + rng.normal(size=n) if estimator == "ols" else rng.binomial(1, special.expit(eta))
    frame = pd.DataFrame({"x": x, "sector": pd.Categorical(labels, categories=["C", "A", "B"]),
                          "y": y.astype(float), "g": np.arange(n) % 31})
    frame.loc[[2, 14], "x"] = np.nan
    frame.loc[19, "y"] = np.nan
    expected_sample = frame.dropna()
    design = np.column_stack([np.ones(len(expected_sample)), expected_sample.x,
                              expected_sample.sector == "A", expected_sample.sector == "B"]).astype(float)
    kwargs = {"cov_kwds": {"groups": expected_sample.g, "use_correction": True, "df_correction": True}} if covariance == "cluster" else {}
    cls = {"ols": sm.OLS, "logit": sm.Logit, "probit": sm.Probit}[estimator]
    if estimator == "ols":
        expected = cls(expected_sample.y, design).fit(cov_type=covariance, use_t=True, **kwargs)
    else:
        expected = cls(expected_sample.y, design).fit(disp=False, maxiter=200, tol=1e-11,
                                                    cov_type=covariance, use_t=False, **kwargs)
    result = fit(ModelSpec(estimator=estimator, outcome="y", predictors=["x", "sector"],
                           categorical=["sector"], covariance=covariance, missing="drop",
                           cluster="g" if covariance == "cluster" else None, **engine), data=frame)
    assert result.sample_positions == expected_sample.index.tolist()
    assert result.dropped_rows == 3
    assert [c.term for c in result.coefficients] == ["Intercept", "x", "sector[A]", "sector[B]"]
    assert result.provenance["categorical_encoding"]["sector"]["reference"] == "C"
    assert_allclose([c.estimate for c in result.coefficients], expected.params, rtol=2e-8, atol=2e-9)
    assert_allclose(result.covariance_matrix, expected.cov_params(), rtol=2e-7, atol=2e-10)
    assert_allclose([p["fitted"] for p in result.predictions], expected.predict(), rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_binary_tail_probabilities_remain_finite(engine, estimator):
    rng = np.random.default_rng(123)
    x = np.concatenate([rng.normal(size=1000), [-14., -11., 11., 14.]])
    probability = stats.norm.cdf(0.15 + 0.6 * x) if estimator == "probit" else special.expit(0.15 + 0.6 * x)
    frame = pd.DataFrame({"x": x, "y": rng.binomial(1, probability)})
    cls = sm.Logit if estimator == "logit" else sm.Probit
    expected = cls(frame.y, sm.add_constant(frame.x)).fit(disp=False, tol=1e-11, maxiter=200)
    result = fit(ModelSpec(estimator=estimator, outcome="y", predictors=["x"], covariance="nonrobust", **engine), data=frame)
    _assert_matches(result, expected)
    assert all(0 <= point["fitted"] <= 1 for point in result.predictions)


def test_ols_small_sample_cluster_correction_is_exact(engine):
    # Three uneven clusters and only six residual degrees of freedom make both
    # CR1 and the G-1 reference distribution materially affect the intervals.
    x = np.array([-3., -1., 0., 1., 2., 2.5, 3., 5.])
    y = np.array([-1., 1., -0.5, 3., 2., 6., 4., 5.])
    g = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    design = np.column_stack([np.ones(8), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    bread = np.linalg.solve(design.T @ design, np.eye(2))
    scores = np.vstack([design[g == group].T @ residual[g == group] for group in np.unique(g)])
    correction = 3 / 2 * 7 / 6
    covariance = correction * bread @ scores.T @ scores @ bread
    result = fit(ModelSpec(outcome="y", predictors=["x"], covariance="cluster", cluster="g", **engine),
                 data=pd.DataFrame({"x": x, "y": y, "g": g}))
    assert_allclose(result.covariance_matrix, covariance, rtol=2e-11, atol=2e-12)
    assert_allclose([c.p_value for c in result.coefficients], 2 * stats.t.sf(abs(beta / np.sqrt(np.diag(covariance))), df=2), rtol=2e-11)
    assert result.inference["small_sample_correction"] == pytest.approx(correction)


def test_scaled_and_nearly_collinear_ols_retains_predictive_accuracy(engine):
    # Do not demand many accurate digits for intrinsically unstable individual
    # coefficients. Verify the well-determined fitted values and finite HC3.
    rng = np.random.default_rng(989)
    x = rng.normal(size=200)
    z = x + 1e-6 * rng.normal(size=200)
    design = np.column_stack([np.ones(200), x, z, 1e5 * rng.normal(size=200)])
    y = 0.8 + 2 * x + 1e-5 * design[:, 3] + rng.normal(size=200)
    frame = pd.DataFrame({"y": y, "x": x, "z": z, "scaled": design[:, 3]})
    result = fit(ModelSpec(outcome="y", predictors=["x", "z", "scaled"], covariance="HC3", **engine), data=frame)
    reference_beta = np.linalg.lstsq(design / np.linalg.norm(design, axis=0), y, rcond=None)[0] / np.linalg.norm(design, axis=0)
    assert_allclose([p["fitted"] for p in result.predictions], design @ reference_beta, rtol=2e-6, atol=2e-7)
    covariance = np.asarray(result.covariance_matrix)
    assert np.isfinite(covariance).all()
    assert np.diag(covariance).min() > 0
    assert result.metrics["condition_number_scaled"] > 1e5
