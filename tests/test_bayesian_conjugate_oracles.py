"""Independent NumPy/SciPy full-matrix proper-prior posterior comparisons."""
import math

import mpmath as mp
import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln
from scipy.stats import invgamma, kstest, t

from openecon.econometrics.bayesian.commands import bayes_contrast, bayes_draws, bayes_linear, bayes_predict


@pytest.mark.parametrize("n,shape", [
    (2, 1e-250), (2, 1e-16), (2, 1e-14),
    (1, math.nextafter(.5, 1.)), (1, .5), (1, .2), (3, 1e-250),
])
def test_moment_existence_uses_unrounded_prior_and_sample_increments(n, shape):
    from openecon.econometrics.bayesian.posterior import PosteriorBundle, PosteriorContrast

    posterior = bayes_linear(data=pd.DataFrame({"y": [.25] * n}), y="y",
                             prior=dict(mean=[0.], scale_matrix=[[1.]], shape=shape, scale=1.))
    with mp.workdps(300):
        size, a = mp.mpf(n), mp.mpf(shape)
        b = 1 + size * mp.mpf('.25')**2 / (2 * (size + 1))
        denominator = a + size/2 - 1
        variance_mean = None if denominator <= 0 else float(b / denominator)
        covariance = None if denominator <= 0 else float(b / (denominator * (size + 1)))
    if variance_mean is None:
        assert posterior.variance_mean is posterior.coefficient_covariance is None
    else:
        assert posterior.variance_mean == pytest.approx(variance_mean, rel=4e-15)
        assert posterior.coefficient_covariance[0][0] == pytest.approx(covariance, rel=4e-15)
        assert math.isfinite(posterior.summary()['Posterior std. dev.'].iloc[0])
    assert PosteriorBundle.model_validate_json(posterior.model_dump_json()) == posterior
    contrast = bayes_contrast(result=posterior, weights=[2.])
    if covariance is None:
        assert contrast.posterior_variance is None
    else:
        assert contrast.posterior_variance == pytest.approx(4 * covariance, rel=4e-15)
    assert PosteriorContrast.model_validate_json(contrast.model_dump_json()) == contrast
    prediction = bayes_predict(result=posterior, data=pd.DataFrame(index=pd.RangeIndex(3)))
    if covariance is None:
        assert prediction['mean_posterior_variance'].isna().all()
        assert prediction.attrs['mean_covariance_factor'] is None
    else:
        np.testing.assert_allclose(prediction['mean_posterior_variance'], covariance, rtol=4e-15)
        covariance_factor = np.array(prediction.attrs['mean_covariance_factor'])
        np.testing.assert_allclose(covariance_factor @ covariance_factor.T, covariance, rtol=4e-15)
        assert prediction.attrs['outcome_independent_variance'] == pytest.approx(variance_mean, rel=4e-15)


def test_prediction_finite_full_covariance_when_dimensionless_multiplier_overflows():
    shape, scale = 1e-320, 1e-300
    posterior = bayes_linear(data=pd.DataFrame({'y': [0., 0.]}), y='y',
                             prior=dict(mean=[0.], scale_matrix=[[1.]], shape=shape, scale=scale))
    prediction = bayes_predict(result=posterior, data=pd.DataFrame(index=pd.RangeIndex(3)))
    with mp.workdps(400):
        expected = float(mp.mpf(scale) / (3 * mp.mpf(shape)))
    np.testing.assert_allclose(prediction['mean_posterior_variance'], expected, rtol=4e-15)
    factor = np.array(prediction.attrs['mean_covariance_factor'])
    np.testing.assert_allclose(factor @ factor.T, expected, rtol=4e-15)
    assert prediction.attrs['scale_to_covariance_multiplier'] is None
    assert prediction.attrs['scale_to_covariance_status'] == 'unrepresentable_multiplier'
    assert math.isfinite(prediction.attrs['outcome_independent_variance'])


@pytest.mark.parametrize("shape", [1e-250, .2, 2.3, 15.999999, 16., 17., 1e3, 1e6, 1e12])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 75, 10_000])
def test_integer_and_half_integer_gamma_increments_against_90_digit_reference(shape, n):
    from openecon.econometrics.bayesian.core import _log_gamma_increment

    with mp.workdps(90):
        initial = mp.mpf(shape)
        expected = mp.loggamma(initial + mp.mpf(n)/2) - mp.loggamma(initial)
    assert _log_gamma_increment(shape, n) == pytest.approx(float(expected), rel=3e-14, abs=3e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 9999, 10_000])
@pytest.mark.parametrize("response", [0., 1e-4, .25])
def test_high_shape_full_evidence_preserves_normalizers_and_sub_ulp_scale_penalty(n, response):
    from openecon.econometrics.bayesian.posterior import PosteriorBundle

    shape = scale = 1e12
    prior = dict(mean=[0.], scale_matrix=[[1.]], shape=shape, scale=scale)
    fitted = bayes_linear(data=pd.DataFrame({"y": [response]*n}), y="y", prior=prior)
    # Scalar Gaussian integration in arbitrary precision, independent of the
    # production residual algebra and its gamma recurrence/expansion.
    with mp.workdps(90):
        a, b, size, value = mp.mpf(shape), mp.mpf(scale), mp.mpf(n), mp.mpf(response)
        penalty = (size*value**2 - (size*value)**2/(size+1))/2
        expected = (-size*mp.log(2*mp.pi)/2 - mp.log(size+1)/2
                    + a*mp.log(b) - (a+size/2)*mp.log(b+penalty)
                    + mp.loggamma(a+size/2) - mp.loggamma(a))
    assert fitted.log_marginal_likelihood == pytest.approx(float(expected), rel=3e-14, abs=3e-12)
    assert PosteriorBundle.model_validate_json(fitted.model_dump_json()) == fitted
    if response == 1e-4:
        assert fitted.scale == scale  # The finite posterior b necessarily rounds.
        with mp.workdps(90):
            baseline = (-size*mp.log(2*mp.pi)/2 - mp.log(size+1)/2 - size*mp.log(b)/2
                        + mp.loggamma(a+size/2) - mp.loggamma(a))
        assert fitted.log_marginal_likelihood < float(baseline)


@pytest.mark.parametrize("shape,scale,n,response", [
    (1e8, 2., 1, .3), (1e12, 1., 2, .1), (1e12, 1e-12, 3, .25),
    (1e12, 1e12, 4, 1e-6), (1e12, 1e24, 5, 1e6),
])
def test_high_shape_evidence_with_distinct_variance_units_against_high_precision(shape, scale, n, response):
    fitted = bayes_linear(data=pd.DataFrame({"y": [response]*n}), y="y",
                          prior=dict(mean=[0.], scale_matrix=[[1.]], shape=shape, scale=scale))
    with mp.workdps(90):
        a, b, size, value = mp.mpf(shape), mp.mpf(scale), mp.mpf(n), mp.mpf(response)
        penalty = (size*value**2-(size*value)**2/(size+1))/2
        expected = (-size*mp.log(2*mp.pi)/2-mp.log(size+1)/2
                    + a*mp.log(b)-(a+size/2)*mp.log(b+penalty)
                    + mp.loggamma(a+size/2)-mp.loggamma(a))
    assert fitted.log_marginal_likelihood == pytest.approx(float(expected), rel=3e-14, abs=3e-12)


def fixture(n=20, *, scale=1):
    generator = np.random.default_rng(2721)
    x = generator.normal(size=(n, 2))
    y = (0.7 + x @ np.array([0.4, -0.2]) + generator.normal(size=n)) * scale
    data = pd.DataFrame({"y": y, "a": x[:, 0], "b": x[:, 1]})
    prior = {"mean": [0.1*scale, -0.1*scale, 0.2*scale],
             "scale_matrix": [[1.3, 0.12, -0.2], [0.12, 0.7, 0.09], [-0.2, 0.09, 1.8]],
             "shape": 2.3, "scale": 1.1*scale**2}
    return data, prior


def oracle(data, prior, *, intercept=True, predictors=("a", "b")):
    x = data[list(predictors)].to_numpy()
    if intercept:
        x = np.column_stack((np.ones(len(data)), x))
    y = data.y.to_numpy()
    m0, v0 = np.array(prior["mean"]), np.array(prior["scale_matrix"])
    p0 = np.linalg.inv(v0)
    pn = p0 + x.T @ x
    vn = np.linalg.inv(pn)
    mn = np.linalg.solve(pn, p0 @ m0 + x.T @ y)
    an = prior["shape"] + len(data)/2
    # The oracle uses the direct sufficient-statistic identity; runtime uses
    # residual/prior penalty algebra to avoid cancellation.
    bn = prior["scale"] + (y @ y + m0 @ p0 @ m0 - mn @ pn @ mn)/2
    scale = vn * bn/an
    cov = None if an <= 1 else vn * bn/(an-1)
    evidence = (-len(y)*np.log(2*np.pi)/2 + np.linalg.slogdet(vn)[1]/2
                - np.linalg.slogdet(v0)[1]/2 + prior["shape"]*np.log(prior["scale"])
                - an*np.log(bn) + gammaln(an)-gammaln(prior["shape"]))
    return mn, vn, an, bn, scale, cov, evidence


@pytest.mark.parametrize("n", [1, 3, 20, 75])
@pytest.mark.parametrize("alpha", [0.001, 0.05, 0.4])
def test_complete_posterior_and_credible_limits(n, alpha):
    data, prior = fixture(n)
    fitted = bayes_linear(data=data, y="y", x=["a", "b"], prior=prior, alpha=alpha)
    mn, vn, an, bn, scale, covariance, evidence = oracle(data, prior)
    np.testing.assert_allclose(fitted.mean, mn, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(fitted.conditional_scale_matrix, vn, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(fitted.coefficient_scale_matrix, scale, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(fitted.coefficient_covariance, covariance, rtol=2e-12, atol=2e-13)
    assert fitted.shape == an
    assert fitted.scale == pytest.approx(bn, rel=2e-12)
    assert fitted.log_marginal_likelihood == pytest.approx(evidence, rel=2e-12, abs=2e-12)
    critical = t.isf(alpha/2, 2*an)
    expected = np.column_stack((mn-critical*np.sqrt(np.diag(scale)), mn+critical*np.sqrt(np.diag(scale))))
    np.testing.assert_allclose(fitted.credible_intervals, expected, rtol=2e-11, atol=2e-12)
    # Off-diagonal scale and covariance matter; no diagonal-only surrogate.
    assert np.max(np.abs(np.array(fitted.coefficient_covariance)-np.diag(np.diag(covariance)))) > 1e-4


def test_hand_computed_small_regression_and_covariance_not_t_scale():
    data = pd.DataFrame(dict(y=[1., 2., 3.], x=[0., 1., 2.]))
    prior = dict(mean=[0., 0.], scale_matrix=[[1., 0.], [0., 1.]], shape=2., scale=1.)
    fitted = bayes_linear(data=data, y="y", x=["x"], prior=prior)
    np.testing.assert_allclose(fitted.mean, [4/5, 14/15], rtol=1e-14)
    np.testing.assert_allclose(fitted.conditional_scale_matrix, [[2/5, -1/5], [-1/5, 4/15]], rtol=1e-14)
    assert fitted.shape == 3.5
    assert fitted.scale == pytest.approx(28/15)
    np.testing.assert_allclose(np.array(fitted.coefficient_covariance),
                               np.array(fitted.coefficient_scale_matrix)*3.5/2.5)


def test_full_contrast_and_prediction_cross_row_uncertainty():
    data, prior = fixture()
    fitted = bayes_linear(data=data, y="y", x=["a", "b"], prior=prior)
    mn, vn, an, bn, scale, covariance, _ = oracle(data, prior)
    weights = np.array([0.4, 1., -0.7])
    contrast = bayes_contrast(result=fitted, weights=weights.tolist(), threshold=0.2, alpha=0.1)
    target_mean, target_scale = weights @ mn, math.sqrt(weights @ scale @ weights)
    assert contrast.mean == pytest.approx(target_mean, rel=2e-12)
    assert contrast.student_t_scale == pytest.approx(target_scale, rel=2e-12)
    assert contrast.posterior_variance == pytest.approx(weights @ covariance @ weights, rel=2e-12)
    assert contrast.probability_above_threshold == pytest.approx(t.cdf((target_mean-.2)/target_scale, 2*an), rel=2e-11)
    query = pd.DataFrame(dict(a=[-1., .1, .8], b=[.2, -.3, 2.]), index=pd.Index(["q", "q", "r"], name="row"))
    design = np.column_stack((np.ones(3), query.to_numpy()))
    predictions = bayes_predict(result=fitted, data=query)
    pd.testing.assert_index_equal(predictions.index, query.index)
    mean_joint = design @ scale @ design.T
    factor = np.array(predictions.attrs["mean_scale_factor"])
    np.testing.assert_allclose(factor @ factor.T, mean_joint, rtol=2e-12, atol=1e-13)
    np.testing.assert_allclose(predictions["mean"], design @ mn, rtol=2e-12)
    np.testing.assert_allclose(predictions["outcome_student_t_scale"]**2, np.diag(mean_joint)+bn/an, rtol=2e-12)
    np.testing.assert_allclose(predictions["mean_posterior_variance"], np.diag(design @ covariance @ design.T), rtol=2e-12)
    critical = t.isf(.025, 2*an)
    np.testing.assert_allclose(predictions["outcome_predictive_lower"], design @ mn-critical*np.sqrt(np.diag(mean_joint)+bn/an), rtol=2e-11)


@pytest.mark.parametrize("scale", [1e-12, 1., 1e12])
def test_response_unit_scaling_preserves_complete_posterior(scale):
    data, prior = fixture(20, scale=scale)
    original_data, original_prior = fixture(20)
    fitted = bayes_linear(data=data, y="y", x=["a", "b"], prior=prior)
    reference = bayes_linear(data=original_data, y="y", x=["a", "b"], prior=original_prior)
    np.testing.assert_allclose(np.array(fitted.mean)/scale, reference.mean, rtol=2e-12)
    np.testing.assert_allclose(np.array(fitted.coefficient_covariance)/scale**2, reference.coefficient_covariance, rtol=2e-12)
    assert fitted.scale/scale**2 == pytest.approx(reference.scale, rel=2e-12)
    assert fitted.log_marginal_likelihood == pytest.approx(reference.log_marginal_likelihood-len(data)*np.log(scale), abs=2e-12)


def test_no_intercept_and_proper_prior_rank_deficient_design():
    data = pd.DataFrame(dict(y=[1., 2.], a=[1., 2.], b=[2., 4.]))
    prior = dict(mean=[.1, -.2], scale_matrix=[[1., .2], [.2, 2.]], shape=.3, scale=.7)
    fitted = bayes_linear(data=data, y="y", x=["a", "b"], intercept=False, prior=prior)
    mn, _, _, _, scale, covariance, _ = oracle(data, prior, intercept=False)
    assert fitted.source_rank == 1 and len(fitted.terms) == 2
    np.testing.assert_allclose(fitted.mean, mn, rtol=2e-12)
    np.testing.assert_allclose(fitted.coefficient_scale_matrix, scale, rtol=2e-12)
    np.testing.assert_allclose(fitted.coefficient_covariance, covariance, rtol=2e-12)


def test_infinite_variance_posterior_keeps_finite_credible_and_predictive_intervals():
    data = pd.DataFrame(dict(y=[2.]))
    prior = dict(mean=[0.], scale_matrix=[[1.]], shape=.2, scale=1.)
    fitted = bayes_linear(data=data, y="y", prior=prior)
    assert fitted.shape == .7
    assert fitted.coefficient_covariance is None and fitted.variance_mean is None
    assert fitted.summary()["Posterior std. dev."].isna().all()
    query = pd.DataFrame(index=["x", "y"])
    prediction = bayes_predict(result=fitted, data=query)
    assert prediction["mean_posterior_variance"].isna().all()
    assert np.isfinite(prediction["outcome_predictive_lower"]).all()
    assert bayes_contrast(result=fitted, weights=[1]).posterior_variance is None


def test_independent_draw_recovery_declared_fixed_seed_protocol():
    data, prior = fixture(75)
    fitted = bayes_linear(data=data, y="y", x=["a", "b"], prior=prior)
    draws = bayes_draws(result=fitted, draws=10_000, seed=8193)
    beta, sigma = np.array(draws.beta), np.array(draws.sigma_squared)
    # Prior to inspection: six MC standard errors for all coefficient means,
    # KS distance <= .025 for the inverse-gamma marginal, standard-normal
    # conditional residual means <= .045 and variances within .07 of one.
    covariance = np.array(fitted.coefficient_covariance)
    assert np.all(np.abs(beta.mean(0)-fitted.mean) <= 6*np.sqrt(np.diag(covariance)/len(beta)))
    assert kstest(sigma, invgamma(fitted.shape, scale=fitted.scale).cdf).statistic <= .025
    factor = np.linalg.cholesky(np.array(fitted.conditional_scale_matrix))
    whitened = np.linalg.solve(factor, ((beta-np.array(fitted.mean))/np.sqrt(sigma)[:, None]).T).T
    assert np.max(np.abs(whitened.mean(0))) <= .045
    assert np.max(np.abs(whitened.var(0)-1)) <= .07


def test_group_difference_and_zero_contrast_are_bayesian_targets():
    data = pd.DataFrame(dict(y=[1., 2., 4., 5.], group=[0., 0., 1., 1.]))
    prior = dict(mean=[0., 0.], scale_matrix=[[10., 0.], [0., 10.]], shape=2., scale=1.)
    fitted = bayes_linear(data=data, y="y", x=["group"], prior=prior)
    difference = bayes_contrast(result=fitted, weights={"group": 1.})
    assert difference.mean == fitted.mean[1]
    assert difference.probability_above_threshold > .9
    degenerate = bayes_contrast(result=fitted, weights=[0., 0.], threshold=-1.)
    assert degenerate.student_t_scale == 0 and degenerate.credible_lower == degenerate.credible_upper == 0
    assert degenerate.probability_above_threshold == 1
    assert "p_value" not in type(difference).model_fields


def test_predictor_units_and_reordered_correlated_prior_transform_full_joint_state():
    data, prior = fixture(30)
    fitted = bayes_linear(data=data, y="y", x=["a", "b"], prior=prior)
    multiplier = np.diag([1., 1/1000, 1.])
    converted = data.copy()
    converted.a *= 1000
    transformed_prior = {**prior, "mean": (multiplier @ np.array(prior["mean"])).tolist(),
                         "scale_matrix": (multiplier @ np.array(prior["scale_matrix"]) @ multiplier.T).tolist()}
    changed = bayes_linear(data=converted, y="y", x=["a", "b"], prior=transformed_prior)
    np.testing.assert_allclose(changed.mean, multiplier @ fitted.mean, rtol=2e-11)
    np.testing.assert_allclose(changed.coefficient_covariance,
                               multiplier @ np.array(fitted.coefficient_covariance) @ multiplier.T, rtol=2e-11)
    assert changed.log_marginal_likelihood == pytest.approx(fitted.log_marginal_likelihood, abs=2e-11)
    ordering = [0, 2, 1]
    reordered_prior = {**prior, "mean": np.array(prior["mean"])[ordering].tolist(),
                       "scale_matrix": np.array(prior["scale_matrix"])[np.ix_(ordering, ordering)].tolist()}
    reordered = bayes_linear(data=data, y="y", x=["b", "a"], prior=reordered_prior)
    np.testing.assert_allclose(reordered.mean, np.array(fitted.mean)[ordering], rtol=2e-12)
    np.testing.assert_allclose(reordered.coefficient_covariance,
                               np.array(fitted.coefficient_covariance)[np.ix_(ordering, ordering)], rtol=2e-12)


def test_one_sample_posterior_and_prior_sensitivity_have_analytic_limits():
    data = pd.DataFrame(dict(y=[1., 2., 3., 4.]))
    prior = dict(mean=[10.], scale_matrix=[[.01]], shape=2., scale=1.)
    fitted = bayes_linear(data=data, y="y", prior=prior)
    assert fitted.mean[0] == pytest.approx((100*10+10)/(100+4), rel=2e-14)
    diffuse = bayes_linear(data=data, y="y", prior={**prior, "scale_matrix": [[1e10]]})
    assert diffuse.mean[0] == pytest.approx(data.y.mean(), abs=2e-9)
    contrast = bayes_contrast(result=fitted, weights={"Intercept": 1}, threshold=8.)
    assert contrast.mean == fitted.mean[0]
    assert contrast.probability_above_threshold > .99


def test_standalone_independent_receipt_verifier_and_refused_corrupt_prediction():
    import copy
    import importlib.util
    import io
    from pathlib import Path
    import runpy
    from contextlib import redirect_stdout

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("bayesian_oracle", root/"scripts/verify_bayesian_conjugate_oracles.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    stream = io.StringIO()
    with redirect_stdout(stream):
        runpy.run_path(str(root/"examples/bayesian_conjugate.py"))
    import json
    receipt = json.loads(stream.getvalue().split("BAYES_CONJUGATE_RECEIPT=", 1)[1])
    report = verifier.verify(receipt)
    assert report["status"] == "passed"
    assert {"coefficient_covariance", "prediction.full_joint_scale", "contrast.posterior_variance",
            "prediction.full_joint_covariance", "prediction.full_outcome_covariance"} <= report["comparisons"].keys()
    contrast_report = verifier.verify(receipt["contrast"])
    assert contrast_report["status"] == "passed"
    assert "contrast.posterior_variance" in contrast_report["comparisons"]
    assert receipt["restored_contrast_equal"]
    altered = copy.deepcopy(receipt)
    altered["prediction"]["attrs"]["mean_scale_factor"][0][1] += .05
    with pytest.raises(AssertionError, match="full joint prediction scale"):
        verifier.verify(altered)
    altered = copy.deepcopy(receipt)
    # A row sign change preserves every marginal variance but corrupts the
    # off-diagonal dependence; independent acceptance must inspect all entries.
    altered['prediction']['attrs']['mean_covariance_factor'][0] = [
        -v for v in altered['prediction']['attrs']['mean_covariance_factor'][0]]
    with pytest.raises(AssertionError, match='full cross-query posterior covariance'):
        verifier.verify(altered)
