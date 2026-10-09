"""Independent likelihood, stacked-score and response-contrast oracles.

SciPy is used only by this development test. Nothing below imports the native
kernel or uses its designs, likelihood, derivatives, or standardization code.
"""
from __future__ import annotations

import json
from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.special import expit, gammaln, log_ndtr, ndtr
from scipy.stats import norm
import torch

import openecon as oe


LINKS = ("logit", "probit")
OUTCOMES = ("gaussian", "logit", "probit", "poisson")
MEANS = ("mu00", "mu10", "mu01", "mu11")
EFFECTS = ("PNDE", "TNDE", "PNIE", "TNIE", "TE")
CONTRAST = np.array([[-1, 1, 0, 0], [0, 0, -1, 1], [-1, 0, 1, 0],
                     [0, -1, 0, 1], [-1, 0, 0, 1]], dtype=float)


def fixture(link, outcome, *, seed=947, n=360):
    rng = np.random.default_rng(seed)
    a = rng.binomial(1, .47, n)
    c = rng.normal(size=(n, 2))
    eta_m = -.35 + .7*a + .3*c[:, 0] - .2*c[:, 1]
    m = rng.binomial(1, expit(eta_m) if link == "logit" else ndtr(eta_m))
    eta_y = -.2 + .4*a + .65*m - .35*a*m + .3*c[:, 0] - .2*c[:, 1]
    if outcome == "gaussian":
        # Nonconstant variance and skewness exercise nuisance cross-covariance.
        error = (rng.exponential(size=n)-1)*(.5+.15*np.abs(c[:, 0]))
        y = eta_y + error
    elif outcome == "poisson":
        y = rng.poisson(np.exp(eta_y))
    else:
        y = rng.binomial(1, expit(eta_y) if outcome == "logit" else ndtr(eta_y))
    return pd.DataFrame(dict(y=y, a=a, m=m, c0=c[:, 0], c1=c[:, 1]))


def designs(frame, interaction):
    a, m = frame.a.to_numpy(), frame.m.to_numpy()
    controls = [name for name in frame if name.startswith("c")]
    c = frame[controls].to_numpy()
    xm = np.column_stack((np.ones(len(a)), a, c))
    columns = [np.ones(len(a)), a, m]
    if interaction:
        columns.append(a*m)
    xy = np.column_stack((*columns, c))
    return xm, xy, c


def likelihood_parts(beta, design, y, model):
    """Observation likelihood/score and observed negative Hessian, original units."""
    eta = design@beta
    if model == "logit":
        p = expit(eta)
        ll = y*eta-np.logaddexp(0., eta)
        score = design*(y-p)[:, None]
        info = design.T@((p*(1-p))[:, None]*design)
    elif model == "probit":
        signed = 2*y-1
        z = signed*eta
        ll = log_ndtr(z)
        mills = np.exp(-.5*z*z-.5*np.log(2*np.pi)-ll)
        score = design*(signed*mills)[:, None]
        info = design.T@((mills*(z+mills))[:, None]*design)
    elif model == "poisson":
        p = np.exp(eta)
        ll = y*eta-p-gammaln(y+1)
        score = design*(y-p)[:, None]
        info = design.T@(p[:, None]*design)
    else:
        raise AssertionError(model)
    return ll, score, info


def scipy_stage(design, y, model):
    if model == "gaussian":
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y-design@beta
        sigma2 = np.mean(residual**2)
        theta = np.r_[beta, .5*np.log(sigma2)]
        ll = -.5*np.log(2*np.pi)-theta[-1]-.5*residual**2/sigma2
        score = np.column_stack((design*residual[:, None]/sigma2,
                                 residual**2/sigma2-1))
        info = np.zeros((len(theta), len(theta)))
        info[:-1, :-1] = design.T@design/sigma2
        info[:-1, -1] = info[-1, :-1] = 2*design.T@residual/sigma2
        info[-1, -1] = 2*np.sum(residual**2/sigma2)
        return theta, ll, score, info
    def objective(beta):
        ll, score, _ = likelihood_parts(beta, design, y, model)
        return -ll.sum(), -score.sum(axis=0)
    # A different optimizer from the native safeguarded Newton kernel.
    fit = minimize(objective, np.zeros(design.shape[1]), method="BFGS", jac=True,
                   options=dict(gtol=2e-9, maxiter=2000))
    ll, score, info = likelihood_parts(fit.x, design, y, model)
    assert np.max(np.abs(score.sum(axis=0))) < 2e-6, fit.message
    return fit.x, ll, score, info


def response_means(theta, c, link, outcome, interaction):
    km = c.shape[1]+2
    bm, by = theta[:km], theta[km:]
    if outcome == "gaussian":
        by = by[:-1]
    values = []
    for a, aprime in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xm = np.column_stack((np.ones(len(c)), np.full(len(c), aprime), c))
        eta = xm@bm
        p = expit(eta) if link == "logit" else ndtr(eta)
        means = []
        for m in (0, 1):
            columns = [np.ones(len(c)), np.full(len(c), a), np.full(len(c), m)]
            if interaction:
                columns.append(np.full(len(c), a*m))
            eta_y = np.column_stack((*columns, c))@by
            mean = (eta_y if outcome == "gaussian" else np.exp(eta_y)
                    if outcome == "poisson" else expit(eta_y)
                    if outcome == "logit" else ndtr(eta_y))
            means.append(mean)
        values.append(np.mean((1-p)*means[0]+p*means[1]))
    return np.array(values)


def five_point_jacobian(function, theta):
    # Numerical derivatives, independently of native automatic differentiation.
    result = np.empty((4, len(theta)))
    for j in range(len(theta)):
        h = 2e-4*max(1., abs(theta[j]))
        step = np.zeros_like(theta)
        step[j] = h
        result[:, j] = (function(theta-2*step)-8*function(theta-step)
                         +8*function(theta+step)-function(theta+2*step))/(12*h)
    return result


def counterfactual_oracle(theta, frame, link, outcome, interaction):
    _, _, c = designs(frame, interaction)
    km = c.shape[1]+2
    bm, by = theta[:km], theta[km:]
    if outcome == "gaussian":
        by = by[:-1]
    def mediator_predict(a):
        linear = np.column_stack((np.ones(len(c)), a, c))@bm
        return expit(linear) if link == "logit" else ndtr(linear)
    def outcome_predict(a, m):
        columns = [np.ones(len(c)), a, m]
        if interaction:
            columns.append(a*m)
        linear = np.column_stack((*columns, c))@by
        return (linear if outcome == "gaussian" else np.exp(linear)
                if outcome == "poisson" else expit(linear)
                if outcome == "logit" else ndtr(linear))
    zero, one = np.zeros(len(c)), np.ones(len(c))
    ps = [mediator_predict(zero), mediator_predict(one)]
    ys = [outcome_predict(a, m) for a, m in ((zero, zero), (one, zero), (zero, one), (one, one))]
    mus = [(1-ps[b])*ys[a]+ps[b]*ys[2+a] for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))]
    return np.column_stack((*ps, *ys, *mus, mediator_predict(frame.a.to_numpy()),
                            outcome_predict(frame.a.to_numpy(), frame.m.to_numpy())))


def oracle(frame, link, outcome, interaction, covariance):
    xm, xy, c = designs(frame, interaction)
    bm, lm, sm, hm = scipy_stage(xm, frame.m.to_numpy(), link)
    by, ly, sy, hy = scipy_stage(xy, frame.y.to_numpy(), outcome)
    theta = np.r_[bm, by]
    information = block_diag(hm, hy)
    bread = np.linalg.inv(information)
    scores = np.column_stack((sm, sy))
    cov = bread if covariance == "OIM" else bread@scores.T@scores@bread
    def fun(b):
        return response_means(b, c, link, outcome, interaction)
    means = fun(theta)
    jac = five_point_jacobian(fun, theta)
    means_cov = jac@cov@jac.T
    return dict(theta=theta, information=information, bread=bread, scores=scores,
                row_loglikelihood=np.column_stack((lm, ly)), covariance=cov,
                means=means, means_covariance=means_cov, effects=CONTRAST@means,
                effects_covariance=CONTRAST@means_cov@CONTRAST.T,
                delta_jacobian=np.vstack((jac, CONTRAST@jac)))


def table_matrix(result, key, labels):
    table = result[key]
    # Public square tables retain a label column and every named matrix column.
    assert list(table.columns[1:]) == list(labels)
    assert table.iloc[:, 0].tolist() == list(labels)
    return table.iloc[:, 1:].to_numpy(dtype=float)


def public_fit(frame, link, outcome, interaction=True, covariance="HC0", **options):
    controls = [name for name in frame if name.startswith("c")]
    return oe.mediation_binary(data=frame, y="y", treatment="a", mediator="m", controls=controls,
                               mediator_link=link, outcome_model=outcome,
                               interaction=interaction, covariance=covariance, **options)


@pytest.mark.parametrize("link,outcome,interaction,covariance", tuple(product(LINKS, OUTCOMES, (False, True), ("HC0", "OIM"))))
@pytest.mark.parametrize("seed", (947, 991))
def test_all_eight_models_full_likelihood_scores_and_uncertainty(link, outcome, interaction, covariance, seed):
    frame = fixture(link, outcome, seed=seed)
    expected = oracle(frame, link, outcome, interaction, covariance)
    result = public_fit(frame, link, outcome, interaction, covariance)
    assert result["inputs"].row.tolist() == list(range(len(frame)))
    np.testing.assert_allclose(result["inputs"][["y", "a", "m", "c0", "c1"]], frame[["y", "a", "m", "c0", "c1"]], atol=0, rtol=0)
    parameters = result["parameters"]
    names = parameters.parameter.tolist()
    np.testing.assert_allclose(parameters.estimate, expected["theta"], rtol=3e-6, atol=2e-7)
    for key in ("information", "bread", "covariance"):
        np.testing.assert_allclose(table_matrix(result, key, names), expected[key], rtol=8e-6, atol=5e-8)
    np.testing.assert_allclose(result["scores"][names], expected["scores"], rtol=8e-6, atol=3e-6)
    np.testing.assert_allclose(result["scores"][names].sum(), 0, atol=3e-5)
    likelihood_columns = [c for c in result["model_loglikelihood"] if c != "row"]
    np.testing.assert_allclose(result["model_loglikelihood"][likelihood_columns],
                               expected["row_loglikelihood"], rtol=8e-6, atol=2e-6)
    cf = result["counterfactual_rows"]
    assert cf.row.tolist() == list(range(len(frame)))
    np.testing.assert_allclose(cf.iloc[:, 1:], counterfactual_oracle(expected["theta"], frame, link, outcome, interaction),
                               rtol=8e-6, atol=3e-6)
    summary = {row.setting: json.loads(row.json) for row in result["fit_summary"].itertuples()}
    assert summary["log_likelihood"] == pytest.approx(expected["row_loglikelihood"].sum(), abs=2e-6)
    assert result["means"]["mean"].tolist() == list(MEANS)
    assert result["effects"].effect.tolist() == list(EFFECTS)
    np.testing.assert_allclose(result["means"].estimate, expected["means"], rtol=3e-6, atol=2e-7)
    np.testing.assert_allclose(result["effects"].estimate, expected["effects"], rtol=8e-6, atol=3e-7)
    np.testing.assert_allclose(table_matrix(result, "means_covariance", MEANS),
                               expected["means_covariance"], rtol=9e-6, atol=2e-8)
    np.testing.assert_allclose(table_matrix(result, "effects_covariance", EFFECTS),
                               expected["effects_covariance"], rtol=9e-6, atol=2e-8)
    jacobian = result["delta_jacobian"]
    assert jacobian.iloc[:, 0].tolist() == [*MEANS, *EFFECTS]
    np.testing.assert_allclose(jacobian[names], expected["delta_jacobian"], rtol=8e-6, atol=3e-7)
    state = result.attrs["binary_mediation_state"]
    complete_target_covariance = expected["delta_jacobian"]@expected["covariance"]@expected["delta_jacobian"].T
    np.testing.assert_allclose(state["derived"]["covariance"], complete_target_covariance, rtol=9e-6, atol=2e-8)
    # Saved state also retains the positive Gaussian sigma transformation,
    # including every cross-covariance with both equations' coefficients.
    natural = expected["theta"].copy()
    natural_jacobian = np.eye(len(natural))
    if outcome == "gaussian":
        natural[-1] = np.exp(natural[-1])
        natural_jacobian[-1, -1] = natural[-1]
    np.testing.assert_allclose(state["derived"]["natural_parameters"], natural, rtol=3e-6, atol=2e-7)
    np.testing.assert_allclose(state["derived"]["natural_covariance"], natural_jacobian@expected["covariance"]@natural_jacobian,
                               rtol=8e-6, atol=5e-8)
    if covariance == "HC0":
        km = designs(frame, interaction)[0].shape[1]
        # The oracle materially distinguishes a stacked sandwich from fitting
        # two robust models and pretending their estimates are independent.
        assert np.max(np.abs(expected["covariance"][:km, km:])) > 1e-5
    for key, labels in (("effects", EFFECTS), ("means", MEANS)):
        table = result[key]
        se = np.sqrt(np.diag(expected[key+"_covariance"]))
        np.testing.assert_allclose(table.std_error, se, rtol=8e-6, atol=2e-7)
        np.testing.assert_allclose(table.ci_lower, expected[key]-norm.ppf(.975)*se, rtol=8e-6, atol=3e-7)
        np.testing.assert_allclose(table.ci_upper, expected[key]+norm.ppf(.975)*se, rtol=8e-6, atol=3e-7)
        if "p_value" in table:
            np.testing.assert_allclose(table.p_value, 2*norm.sf(abs(expected[key]/se)), rtol=4e-5, atol=2e-6)
    se = np.sqrt(np.diag(expected["covariance"]))
    np.testing.assert_allclose(parameters.std_error, se, rtol=8e-6, atol=2e-7)
    np.testing.assert_allclose(parameters.ci_lower, expected["theta"]-norm.ppf(.975)*se, rtol=8e-6, atol=3e-7)
    np.testing.assert_allclose(parameters.ci_upper, expected["theta"]+norm.ppf(.975)*se, rtol=8e-6, atol=3e-7)
    tested = len(se)-int(outcome == "gaussian")
    np.testing.assert_allclose(parameters.z[:tested], expected["theta"][:tested]/se[:tested], rtol=8e-6, atol=5e-6)
    np.testing.assert_allclose(parameters.p_value[:tested], 2*norm.sf(abs(expected["theta"][:tested]/se[:tested])), rtol=4e-5, atol=2e-6)
    if outcome == "gaussian":
        assert pd.isna(parameters.z.iloc[-1]) and pd.isna(parameters.p_value.iloc[-1])
        np.testing.assert_allclose(expected["delta_jacobian"][:, -1], 0, atol=1e-12)


@pytest.mark.parametrize("link,outcome", tuple(product(LINKS, OUTCOMES)))
def test_saturated_no_control_binary_arm_closed_form(link, outcome):
    rows = []
    rates, counts = {}, {}
    for a, m in product((0, 1), repeat=2):
        count = 48+12*a+6*m
        if outcome == "gaussian":
            y = 1+.8*a+.5*m-.35*a*m+np.tile([-1., -.4, .1, .3, .4, .6], count//6)
        elif outcome == "poisson":
            y = np.resize([0, 1, 2, 2+a, 3+m, 4+a*m], count)
        else:
            y = np.r_[np.ones(18+6*a+6*m), np.zeros(count-18-6*a-6*m)]
        rates[a, m], counts[a, m] = np.mean(y), count
        rows.extend((float(v), a, m) for v in y)
    frame = pd.DataFrame(rows, columns=["y", "a", "m"])
    result = public_fit(frame, link, outcome)
    probabilities = {a: counts[a, 1]/(counts[a, 0]+counts[a, 1]) for a in (0, 1)}
    closed = np.array([(1-probabilities[b])*rates[a, 0]+probabilities[b]*rates[a, 1]
                       for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))])
    np.testing.assert_allclose(result["means"].estimate, closed, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(result["effects"].estimate, CONTRAST@closed, atol=3e-8, rtol=3e-8)
    effects = result["effects"].set_index("effect").estimate
    assert effects.TE == pytest.approx(effects.PNDE+effects.TNIE, abs=2e-14)
    assert effects.TE == pytest.approx(effects.TNDE+effects.PNIE, abs=2e-14)


def json_tables(result):
    return {name: table.astype(object).where(table.notna(), None).to_dict(orient="split")
            for name, table in result.items()}


@pytest.mark.parametrize("link,outcome", tuple(product(LINKS, OUTCOMES)))
def test_complete_state_json_restores_all_joint_uncertainty_without_refit(link, outcome, monkeypatch):
    result = public_fit(fixture(link, outcome, seed=953, n=240), link, outcome)
    payload = json.loads(json.dumps(dict(attrs=result.attrs, tables=json_tables(result), latex=result.to_latex()), sort_keys=True, allow_nan=False))
    from openecon.econometrics.decomposition import binary_kernels
    from openecon.econometrics.decomposition import binary
    def forbid(*args, **kwargs):
        raise AssertionError("Saved mediation restore must not fit a model")
    monkeypatch.setattr(binary_kernels, "fit_joint", forbid)
    if hasattr(binary, "fit_joint"):
        monkeypatch.setattr(binary, "fit_joint", forbid)
    restored = oe.mediation_binary_restore(payload["attrs"])
    assert restored.attrs == payload["attrs"]
    assert json_tables(restored) == payload["tables"]
    assert restored.to_latex() == payload["latex"]
    assert set(restored) == {"inputs", "parameters", "information", "bread", "scores", "model_loglikelihood",
                             "covariance", "counterfactual_rows", "means", "means_covariance", "effects",
                             "effects_covariance", "delta_jacobian", "fit_summary"}


def test_nonconstant_controls_fixed_empirical_target_not_mean_control_shortcut():
    frame = fixture("probit", "poisson", seed=973)
    result = public_fit(frame, "probit", "poisson")
    theta = result["parameters"].estimate.to_numpy()
    c = frame[["c0", "c1"]].to_numpy()
    correct = response_means(theta, c, "probit", "poisson", True)
    shortcut = response_means(theta, c.mean(axis=0, keepdims=True), "probit", "poisson", True)
    np.testing.assert_allclose(result["means"].estimate, correct, atol=1e-12)
    assert np.max(np.abs(shortcut-correct)) > .01


def test_hc0_gaussian_log_sigma_kept_in_joint_scores_covariance_but_null_response_derivative():
    frame = fixture("logit", "gaussian", seed=977)
    result = public_fit(frame, "logit", "gaussian")
    names = result["parameters"].parameter.tolist()
    covariance = table_matrix(result, "covariance", names)
    assert np.max(np.abs(covariance[-1, :-1])) > 1e-4
    np.testing.assert_allclose(result["delta_jacobian"][names[-1]], 0, atol=1e-14)


@pytest.mark.parametrize("outcome", OUTCOMES)
def test_affine_control_units_transform_complete_joint_state_and_preserve_target(outcome):
    frame = fixture("probit", outcome, seed=983)
    shifted = frame.copy()
    scale, offset = np.array([100., .02]), np.array([7., -3.])
    shifted[["c0", "c1"]] = frame[["c0", "c1"]]*scale+offset
    original = public_fit(frame, "probit", outcome)
    changed = public_fit(shifted, "probit", outcome)
    names = original["parameters"].parameter.tolist()
    p = len(names)
    transform = np.eye(p)
    for start, control_start in ((0, 2), (4, 4)):
        transform[start, start+control_start:start+control_start+2] = -offset/scale
        transform[start+control_start:start+control_start+2, start+control_start:start+control_start+2] = np.diag(1/scale)
    inverse = np.linalg.inv(transform)
    theta = original["parameters"].estimate.to_numpy()
    np.testing.assert_allclose(changed["parameters"].estimate, transform@theta, rtol=2e-6, atol=2e-6)
    for key in ("bread", "covariance"):
        expected = transform@table_matrix(original, key, names)@transform.T
        np.testing.assert_allclose(table_matrix(changed, key, names), expected, rtol=5e-6, atol=2e-6)
    expected = inverse.T@table_matrix(original, "information", names)@inverse
    np.testing.assert_allclose(table_matrix(changed, "information", names), expected, rtol=5e-6, atol=2e-5)
    np.testing.assert_allclose(changed["scores"][names], original["scores"][names].to_numpy()@inverse,
                               rtol=5e-6, atol=3e-6)
    for key in ("means", "effects"):
        np.testing.assert_allclose(changed[key].estimate, original[key].estimate, rtol=2e-6, atol=2e-7)
        labels = MEANS if key == "means" else EFFECTS
        np.testing.assert_allclose(table_matrix(changed, key+"_covariance", labels),
                                   table_matrix(original, key+"_covariance", labels), rtol=5e-6, atol=2e-8)


def test_fit_delta_and_complete_saved_restore_ignore_global_default_device_and_precision():
    frame = fixture("probit", "poisson", seed=997, n=240)
    expected = public_fit(frame, "probit", "poisson")
    previous_device, previous_dtype = torch.get_default_device(), torch.get_default_dtype()
    try:
        torch.set_default_device("meta")
        torch.set_default_dtype(torch.float32)
        actual = public_fit(frame, "probit", "poisson")
        restored = oe.mediation_binary_restore(json.loads(json.dumps(actual.attrs, sort_keys=True, allow_nan=False)))
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)
    assert actual.attrs == expected.attrs and restored.attrs == expected.attrs
    assert json_tables(actual) == json_tables(expected) == json_tables(restored)
