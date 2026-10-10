"""Independent observation-space integration and high-precision point-null evidence."""

import mpmath as mp
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import gammaln
from scipy.stats import invgamma, kstest, t

from openecon.econometrics.bayesian.commands import bayes_linear
from openecon.econometrics.bayesian.hypothesis import (
    bayes_hypothesis,
    bayes_hypothesis_draws,
    bayes_hypothesis_predict,
)


def fixture(n=19, shape=2.3):
    rng = np.random.default_rng(7121)
    predictors = rng.normal(size=(n, 2))
    x = np.column_stack((np.ones(n), predictors))
    y = x @ [0.4, -0.3, 0.7] + rng.normal(size=n) * 0.8
    prior = dict(
        mean=[0.1, -0.2, 0.15],
        scale_matrix=[[1.3, 0.2, -0.1], [0.2, 0.8, 0.12], [-0.1, 0.12, 1.6]],
        shape=shape,
        scale=1.2,
    )
    frame = pd.DataFrame(dict(y=y, a=predictors[:, 0], b=predictors[:, 1]))
    return frame, x, y, prior


def observation_oracle(x, y, prior, c, d):
    """Integrate in observation space, without the production free-coordinate QR."""
    m, v, c, d = map(np.asarray, (prior["mean"], prior["scale_matrix"], c, d))
    gram = c @ v @ c.T
    difference = d - c @ m
    center = m + v @ c.T @ np.linalg.solve(gram, difference)
    vc = v - v @ c.T @ np.linalg.solve(gram, c @ v)
    if len(c) == len(m):
        vc = np.zeros_like(v)
    ac = prior["shape"] + len(c) / 2
    bc = prior["scale"] + difference @ np.linalg.solve(gram, difference) / 2
    omega = np.eye(len(y)) + x @ vc @ x.T
    residual = y - x @ center
    solve = np.linalg.solve(omega, residual)
    mn = center + vc @ x.T @ solve
    vn = vc - vc @ x.T @ np.linalg.solve(omega, x @ vc)
    an, bn = ac + len(y) / 2, bc + residual @ solve / 2
    evidence = (
        -len(y) * np.log(2 * np.pi) / 2
        - np.linalg.slogdet(omega)[1] / 2
        + ac * np.log(bc)
        - an * np.log(bn)
        + gammaln(an)
        - gammaln(ac)
    )
    return center, vc, ac, bc, mn, vn, an, bn, evidence


@pytest.mark.parametrize("n", [1, 4, 19, 63])
@pytest.mark.parametrize(
    "c,d",
    [
        ([[0.0, 1.0, 0.0]], [0.2]),
        ([[1.0, 0.3, -0.2], [0.0, 1.0, 1.0]], [0.5, -0.1]),
        (np.eye(3).tolist(), [0.2, -0.3, 0.6]),
    ],
)
def test_full_conditional_prior_null_posterior_and_independent_evidence(n, c, d):
    data, x, y, prior = fixture(n)
    alternative = bayes_linear(data=data, y="y", x=["a", "b"], prior=prior)
    comparison = bayes_hypothesis(
        result=alternative, constraints=c, values=d, model_prior_odds=2.5, alpha=0.1
    )
    out = comparison.payload["results"]
    null = out["null_posterior"]
    center, vc, ac, bc, mn, vn, an, bn, lognull = observation_oracle(x, y, prior, c, d)
    np.testing.assert_allclose(null["prior_mean"], center, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(null["prior_conditional_scale_matrix"], vc, atol=4e-15, rtol=2e-12)
    assert null["prior_shape"] == ac
    assert null["prior_scale"] == pytest.approx(bc, rel=2e-12)
    np.testing.assert_allclose(null["mean"], mn, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(null["conditional_scale_matrix"], vn, atol=4e-15, rtol=2e-11)
    np.testing.assert_allclose(
        null["coefficient_covariance"], vn * bn / (an - 1), atol=4e-15, rtol=2e-11
    )
    np.testing.assert_allclose(
        null["coefficient_scale_matrix"], vn * bn / an, atol=4e-15, rtol=2e-11
    )
    assert null["shape"] == an
    assert null["scale"] == pytest.approx(bn, rel=2e-12)
    assert null["log_marginal_likelihood"] == pytest.approx(lognull, abs=1e-11)
    logbf = lognull - alternative.log_marginal_likelihood
    assert out["log_bayes_factor_null_alternative"] == pytest.approx(logbf, abs=1e-11)
    assert out["bayes_factor_null_alternative"] == pytest.approx(np.exp(logbf), rel=2e-11)
    probability = 1 / (1 + np.exp(-logbf) / 2.5)
    assert out["posterior_probability_null"] == pytest.approx(probability, abs=2e-12)
    np.testing.assert_allclose(np.asarray(c) @ np.asarray(null["mean"]), d, atol=3e-13)
    np.testing.assert_allclose(
        np.asarray(c) @ np.asarray(null["conditional_scale_matrix"]), 0, atol=4e-15
    )
    critical = t.isf(0.05, 2 * an)
    expected_intervals = np.column_stack(
        (
            mn - critical * np.sqrt(np.maximum(np.diag(vn * bn / an), 0)),
            mn + critical * np.sqrt(np.maximum(np.diag(vn * bn / an), 0)),
        )
    )
    np.testing.assert_allclose(
        null["credible_intervals"], expected_intervals, rtol=3e-11, atol=1e-7 if n == 1 else 1e-8
    )


@pytest.mark.parametrize(
    "row_transform",
    [
        np.diag([1e-180, 1e180]),
        np.array([[2.0, -3.0], [0.3, 0.7]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    ],
)
def test_constraint_units_and_nonsingular_row_basis_invariance(row_transform):
    data, _, _, prior = fixture()
    alternative = bayes_linear(data=data, y="y", x=["a", "b"], prior=prior)
    c, d = np.array([[1.0, 0.3, -0.2], [0.0, 1.0, 1.0]]), np.array([0.5, -0.1])
    original = bayes_hypothesis(
        result=alternative, constraints=c.tolist(), values=d.tolist()
    ).payload["results"]
    transformed = bayes_hypothesis(
        result=alternative,
        constraints=(row_transform @ c).tolist(),
        values=(row_transform @ d).tolist(),
    ).payload["results"]
    assert transformed["log_bayes_factor_null_alternative"] == pytest.approx(
        original["log_bayes_factor_null_alternative"], abs=3e-12
    )
    for name in (
        "mean",
        "conditional_scale_matrix",
        "coefficient_scale_matrix",
        "coefficient_covariance",
    ):
        np.testing.assert_allclose(
            transformed["null_posterior"][name],
            original["null_posterior"][name],
            rtol=3e-11,
            atol=2e-14,
        )


@pytest.mark.parametrize(
    "transform",
    [np.diag([1e-6, 1e6, 2.0]), np.array([[1.0, 0.3, -0.2], [0.1, 2.0, 0.4], [0.0, -0.2, 0.8]])],
)
def test_complete_coefficient_units_and_nonorthogonal_basis_invariance(transform):
    _, x, y, prior = fixture()
    inverse = np.linalg.inv(transform)
    c, d = np.array([[1.0, 0.3, -0.2], [0.0, 1.0, 1.0]]), np.array([0.5, -0.1])
    original = bayes_linear(
        data=pd.DataFrame(np.column_stack((y, x)), columns=["y", "q0", "q1", "q2"]),
        y="y",
        x=["q0", "q1", "q2"],
        intercept=False,
        prior=prior,
    )
    transformed_v = inverse @ np.asarray(prior["scale_matrix"]) @ inverse.T
    # The input contract deliberately requires an exactly symmetric declared
    # proper prior; write matching triangular cells when constructing that input.
    transformed_v = np.tril(transformed_v) + np.tril(transformed_v, -1).T
    transformed_prior = dict(
        prior, mean=(inverse @ prior["mean"]).tolist(), scale_matrix=transformed_v.tolist()
    )
    transformed = bayes_linear(
        data=pd.DataFrame(np.column_stack((y, x @ transform)), columns=["y", "q0", "q1", "q2"]),
        y="y",
        x=["q0", "q1", "q2"],
        intercept=False,
        prior=transformed_prior,
    )
    a = bayes_hypothesis(result=original, constraints=c.tolist(), values=d.tolist()).payload[
        "results"
    ]
    b = bayes_hypothesis(
        result=transformed, constraints=(c @ transform).tolist(), values=d.tolist()
    ).payload["results"]
    assert a["log_bayes_factor_null_alternative"] == pytest.approx(
        b["log_bayes_factor_null_alternative"], abs=2e-11
    )
    np.testing.assert_allclose(
        transform @ b["null_posterior"]["mean"], a["null_posterior"]["mean"], rtol=4e-11, atol=2e-13
    )
    np.testing.assert_allclose(
        transform @ np.asarray(b["null_posterior"]["coefficient_covariance"]) @ transform.T,
        a["null_posterior"]["coefficient_covariance"],
        rtol=4e-11,
        atol=2e-13,
    )


def high_precision_evidence(x, y, prior, c=None, d=None):
    with mp.workdps(90):
        xm, ym = mp.matrix(x.tolist()), mp.matrix(y.tolist())
        m, v = mp.matrix(prior["mean"]), mp.matrix(prior["scale_matrix"])
        a, b = mp.mpf(prior["shape"]), mp.mpf(prior["scale"])
        if c is not None:
            cm, dm = mp.matrix(c), mp.matrix(d)
            delta = dm - cm * m
            gram = cm * v * cm.T
            m = m + v * cm.T * (gram**-1) * delta
            v = v - v * cm.T * (gram**-1) * cm * v
            a += mp.mpf(len(c)) / 2
            b += (delta.T * (gram**-1) * delta)[0] / 2
        omega = mp.eye(len(y)) + xm * v * xm.T
        e = ym - xm * m
        penalty = (e.T * (omega**-1) * e)[0] / 2
        an = a + mp.mpf(len(y)) / 2
        return float(
            -len(y) * mp.log(2 * mp.pi) / 2
            - mp.log(mp.det(omega)) / 2
            + a * mp.log(b)
            - an * mp.log(b + penalty)
            + mp.loggamma(an)
            - mp.loggamma(a)
        )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 19, 20])
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_high_shape_odd_even_normalizers_against_90_digit_restricted_integration(n, rank):
    data, x, y, prior = fixture(n, shape=1e12)
    prior["scale"] = 1e12
    c = np.eye(3)[:rank].tolist()
    d = [0.2, -0.3, 0.6][:rank]
    alternative = bayes_linear(data=data, y="y", x=["a", "b"], prior=prior)
    output = bayes_hypothesis(result=alternative, constraints=c, values=d).payload["results"]
    null = high_precision_evidence(x, y, prior, c, d)
    full = high_precision_evidence(x, y, prior)
    assert output["null_posterior"]["log_marginal_likelihood"] == pytest.approx(null, abs=3e-11)
    assert output["log_bayes_factor_null_alternative"] == pytest.approx(null - full, abs=3e-11)


def test_joint_predictions_and_stochastic_variance_with_all_coefficients_fixed():
    data, x, y, prior = fixture()
    alternative = bayes_linear(data=data, y="y", x=["a", "b"], prior=prior)
    fixed = [0.2, -0.3, 0.6]
    comparison = bayes_hypothesis(result=alternative, constraints=np.eye(3).tolist(), values=fixed)
    null = comparison.payload["results"]["null_posterior"]
    assert (
        null["free_dimension"] == 0
        and null["free_mean"] == ()
        and null["free_conditional_scale_matrix"] == ()
    )
    assert null["coefficient_covariance"] == ((0.0, 0.0, 0.0),) * 3
    query = pd.DataFrame(
        dict(a=[0.1, -0.2, 0.4], b=[0.3, 0.7, -0.1]),
        index=pd.Index(["same", "same", "last"], name="query"),
    )
    prediction = bayes_hypothesis_predict(result=comparison, data=query)
    design = np.column_stack((np.ones(3), query.to_numpy()))
    pd.testing.assert_index_equal(prediction.index, query.index)
    np.testing.assert_allclose(prediction["mean"], design @ fixed, rtol=2e-13)
    np.testing.assert_array_equal(prediction.attrs["joint_mean_student_t_scale"], np.zeros((3, 3)))
    np.testing.assert_allclose(
        prediction.attrs["joint_outcome_covariance"], np.eye(3) * null["variance_mean"], rtol=2e-13
    )
    np.testing.assert_array_equal(
        prediction["mean_credible_lower"], prediction["mean_credible_upper"]
    )
    draws = bayes_hypothesis_draws(result=comparison, draws=10000, seed=100, data=query).payload[
        "arrays"
    ]
    np.testing.assert_allclose(draws["beta"], np.tile(fixed, (10000, 1)), atol=4e-15)
    assert (
        kstest(draws["sigma_squared"], invgamma(null["shape"], scale=null["scale"]).cdf).pvalue
        > 0.005
    )
    np.testing.assert_allclose(
        np.cov(np.asarray(draws["outcome_draws"]), rowvar=False),
        prediction.attrs["joint_outcome_covariance"],
        rtol=0.05,
        atol=0.015,
    )


def test_full_joint_draws_and_prediction_covariance_on_the_null_affine_support():
    data, _, _, prior = fixture()
    alternative = bayes_linear(data=data, y="y", x=["a", "b"], prior=prior)
    c, d = [[1.0, 0.3, -0.2]], [0.5]
    comparison = bayes_hypothesis(result=alternative, constraints=c, values=d)
    query = pd.DataFrame(dict(a=[-0.2, 0.4], b=[0.7, -0.1]))
    prediction = bayes_hypothesis_predict(result=comparison, data=query)
    null = comparison.payload["results"]["null_posterior"]
    design = np.column_stack((np.ones(2), query.to_numpy()))
    np.testing.assert_allclose(
        prediction.attrs["joint_mean_covariance"],
        design @ np.asarray(null["coefficient_covariance"]) @ design.T,
        atol=2e-14,
    )
    draws = bayes_hypothesis_draws(result=comparison, draws=10000, seed=71, data=query).payload[
        "arrays"
    ]
    beta = np.asarray(draws["beta"])
    np.testing.assert_allclose(beta @ np.asarray(c).T, 0.5, atol=2e-15)
    np.testing.assert_allclose(beta.mean(0), null["mean"], atol=0.014)
    np.testing.assert_allclose(
        np.cov(beta, rowvar=False), null["coefficient_covariance"], rtol=0.065, atol=0.0015
    )
    np.testing.assert_allclose(
        np.cov(np.asarray(draws["mean_draws"]), rowvar=False),
        prediction.attrs["joint_mean_covariance"],
        rtol=0.055,
        atol=0.0015,
    )


def test_no_prior_odds_means_no_inferred_posterior_model_probabilities():
    data, _, _, prior = fixture()
    alternative = bayes_linear(data=data, y="y", x=["a", "b"], prior=prior)
    comparison = bayes_hypothesis(result=alternative, constraints=[[0.0, 1.0, 0.0]], values=[0.0])
    assert comparison.payload["results"]["posterior_probability_null"] is None
    assert comparison.payload["results"]["log_posterior_model_odds_null_alternative"] is None
    assert all(
        "p-value" not in str(column)
        for table in comparison.summary().values()
        for column in table.columns
    )


def test_ambient_dtype_default_device_and_rng_are_preserved():
    data, _, _, prior = fixture()
    original_dtype = torch.get_default_dtype()
    original_device = torch.get_default_device()
    rng = torch.get_rng_state().clone()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        alternative = bayes_linear(data=data, y="y", x=["a", "b"], prior=prior)
        comparison = bayes_hypothesis(
            result=alternative, constraints=[[0.0, 1.0, 0.0]], values=[0.0]
        )
        prediction = bayes_hypothesis_predict(
            result=comparison, data=pd.DataFrame(dict(a=[0.1], b=[0.2]))
        )
        draws = bayes_hypothesis_draws(result=comparison, draws=21, seed=3)
        assert prediction["mean"].dtype == np.float64
        assert draws.payload["count"] == 21
        assert torch.get_default_dtype() == torch.float32
        assert str(torch.get_default_device()) == "meta"
    finally:
        torch.set_default_dtype(original_dtype)
        torch.set_default_device(original_device)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("shape", [1e-250, 1e-20, 1e-16, 0.2])
def test_tiny_proper_shape_keeps_original_increment_moment_denominator(shape):
    data = pd.DataFrame({"y": [0.3], "x": [0.2]})
    prior = dict(mean=[0.0, 0.0], scale_matrix=[[1.0, 0.0], [0.0, 1.0]], shape=shape, scale=0.7)
    alternative = bayes_linear(data=data, y="y", x=["x"], prior=prior)
    comparison = bayes_hypothesis(result=alternative, constraints=[[0.0, 1.0]], values=[0.0])
    null = comparison.payload["results"]["null_posterior"]
    assert null["variance_moment_denominator"] == shape
    with mp.workdps(300):
        expected_variance = (mp.mpf(0.7) + mp.mpf(0.3) ** 2 / 4) / mp.mpf(shape)
    assert null["variance_mean"] == pytest.approx(float(expected_variance), rel=2e-14)
    assert null["coefficient_covariance"][0][0] == pytest.approx(
        float(expected_variance) / 2, rel=2e-14
    )
    prediction = bayes_hypothesis_predict(result=comparison, data=pd.DataFrame({"x": [1.0]}))
    assert prediction.attrs["joint_outcome_covariance"][0][0] == pytest.approx(
        float(expected_variance) * 1.5, rel=2e-14
    )
    assert np.isfinite(comparison.summary()["null_coefficients"]["Posterior std. dev."]).all()


@pytest.mark.parametrize("value", [0.0, 0.2, 0.3, 1.0, -1e-12])
def test_declared_coordinate_equality_has_exactly_zero_free_loading_and_collapsed_interval(value):
    data, _, _, prior = fixture()
    alternative = bayes_linear(data=data, y="y", x=["a", "b"], prior=prior)
    comparison = bayes_hypothesis(result=alternative, constraints=[[0.0, 1.0, 0.0]], values=[value])
    null = comparison.payload["results"]["null_posterior"]
    assert null["mean"][1] == value
    assert null["basis"][1] == (0.0, 0.0)
    assert null["conditional_scale_matrix"][1] == (0.0, 0.0, 0.0)
    assert null["credible_intervals"][1] == (value, value)
    draws = bayes_hypothesis_draws(result=comparison, draws=17).payload["arrays"]["beta"]
    assert all(draw[1] == value for draw in draws)


def test_primary_conditional_nuisance_prior_example_has_ig_one_null_prior():
    # The Gaussian point-null example in Mulder/Wagenmakers/Marsman conditions
    # the full IG(1/2,1/2) prior to IG(1,1/2), rather than reusing its marginal.
    observed = 0.1
    alternative = bayes_linear(
        data=pd.DataFrame({"y": [observed]}),
        y="y",
        prior=dict(mean=[0.0], scale_matrix=[[1.0]], shape=0.5, scale=0.5),
    )
    comparison = bayes_hypothesis(result=alternative, constraints=[[1.0]], values=[0.0])
    null = comparison.payload["results"]["null_posterior"]
    assert null["prior_shape"] == 1.0 and null["prior_scale"] == 0.5
    expected = t.logpdf(observed, df=2, scale=np.sqrt(0.5)) - t.logpdf(
        observed, df=1, scale=np.sqrt(2.0)
    )
    assert comparison.payload["results"]["log_bayes_factor_null_alternative"] == pytest.approx(
        expected, abs=2e-13
    )


@pytest.mark.parametrize("case", ["overflow", "underflow"])
def test_log_bayes_factor_remains_finite_when_exponential_is_unrepresentable(case):
    if case == "overflow":
        data, shape, scale, fixed = (
            pd.DataFrame({"y": [0.0]}),
            float.fromhex("0x0.0000000000001p-1022"),
            1e-300,
            0.0,
        )
    else:
        data, shape, scale, fixed = pd.DataFrame({"y": [0.0] * 1000}), 2.0, 1.0, 50.0
    alternative = bayes_linear(
        data=data, y="y", prior=dict(mean=[0.0], scale_matrix=[[1.0]], shape=shape, scale=scale)
    )
    out = bayes_hypothesis(
        result=alternative, constraints=[[1.0]], values=[fixed], model_prior_odds=1.0
    ).payload["results"]
    assert out["bayes_factor_status"] == case
    assert out["bayes_factor_null_alternative"] is None
    assert np.isfinite(out["log_bayes_factor_null_alternative"])
    assert out["posterior_probability_null"] == (1.0 if case == "overflow" else 0.0)
    if case == "overflow":
        assert out["posterior_probability_alternative"] > 0.0


@pytest.mark.parametrize("unit", [1e-12, 1.0, 1e12])
def test_complete_response_units_preserve_conditional_variance_prior_and_evidence_ratio(unit):
    data, _, _, prior = fixture()
    original = bayes_linear(data=data, y="y", x=["a", "b"], prior=prior)
    c, d = [[1.0, 0.3, -0.2]], [0.5]
    first = bayes_hypothesis(result=original, constraints=c, values=d).payload["results"]
    changed = data.copy()
    changed["y"] *= unit
    scaled_prior = dict(
        prior, mean=(np.asarray(prior["mean"]) * unit).tolist(), scale=prior["scale"] * unit**2
    )
    rescaled = bayes_linear(data=changed, y="y", x=["a", "b"], prior=scaled_prior)
    second = bayes_hypothesis(result=rescaled, constraints=c, values=[0.5 * unit]).payload[
        "results"
    ]
    a, b = first["null_posterior"], second["null_posterior"]
    np.testing.assert_allclose(np.asarray(b["mean"]) / unit, a["mean"], atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(
        np.asarray(b["coefficient_covariance"]) / unit**2,
        a["coefficient_covariance"],
        atol=2e-13,
        rtol=2e-12,
    )
    assert b["prior_scale"] / unit**2 == pytest.approx(a["prior_scale"], rel=2e-12)
    assert b["variance_mean"] / unit**2 == pytest.approx(a["variance_mean"], rel=2e-12)
    assert b["log_marginal_likelihood"] == pytest.approx(
        a["log_marginal_likelihood"] - len(data) * np.log(unit), abs=2e-10
    )
    assert second["log_bayes_factor_null_alternative"] == pytest.approx(
        first["log_bayes_factor_null_alternative"], abs=2e-11
    )
