"""Independent forward-link, statsmodels and finite-difference effect oracles."""

import copy

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit, ndtr
import statsmodels.api as sm
import torch

import openecon as oe
from openecon.models import ResultBundle


def data(seed=812, n=280):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n),
                          "g": pd.Categorical(np.resize(["B", "A", "C"], n), categories=["B", "A", "C"]),
                          "w": rng.uniform(.5, 3, n), "exposure": rng.uniform(.7, 2, n)})
    index = .3 + .4 * frame.x - .2 * frame.z + .25 * (frame.g == "A") - .15 * (frame.g == "C")
    frame["binary"] = (rng.uniform(size=n) < expit(index)).astype(int)
    frame["count"] = rng.poisson(np.exp(index) * frame.exposure)
    frame["nb"] = rng.negative_binomial(.7, .7 / (.7 + np.exp(index)))
    frame["fraction"] = rng.beta(expit(index) * 15, (1 - expit(index)) * 15)
    frame["y"] = index + rng.normal(size=n)
    return frame


def design(frame, terms):
    arrays = {"Intercept": np.ones(len(frame)), "x": frame.x.to_numpy(), "z": frame.z.to_numpy(),
              "g[A]": (frame.g == "A").to_numpy(float), "g[C]": (frame.g == "C").to_numpy(float)}
    return np.column_stack([arrays.get(term, np.zeros(len(frame))) for term in terms])


def stored(model):
    return ResultBundle.model_validate_json(model.model_dump_json())


@pytest.fixture(scope="module", params=["logit", "probit", "poisson", "nbreg", "fracreg", "cloglog", "betareg", "gnbreg"])
def mean_model(request):
    df = data()
    name = request.param
    kwargs = {"data": df, "x": ["x", "z", "g"], "categorical": ["g"], "missing": "drop"}
    if name in {"logit", "probit", "cloglog"}:
        model = getattr(oe, name)(y="binary", **kwargs)
    elif name in {"betareg", "fracreg"}:
        model = getattr(oe, name)(y="fraction", **kwargs)
    elif name == "poisson":
        model = oe.poisson(y="count", exposure="exposure", **kwargs)
    else:
        model = getattr(oe, name)(y="nb", **kwargs)
    return name, df, stored(model)


def inverse(name, eta):
    if name == "probit":
        return ndtr(eta)
    if name == "cloglog":
        return -np.expm1(-np.exp(eta))
    if name in {"logit", "fracreg", "betareg"}:
        return expit(eta)
    return np.exp(eta)


def slope(name, eta):
    if name == "probit":
        return stats.norm.pdf(eta)
    if name == "cloglog":
        return np.exp(eta - np.exp(eta))
    if name in {"logit", "fracreg", "betareg"}:
        p = expit(eta)
        return p * (1 - p)
    return np.exp(eta)


def test_new_row_predictions_fixed_design_links_covariance_and_coef_order(mean_model):
    name, _, model = mean_model
    new = pd.DataFrame({"x": [-1., .2, 1.1], "z": [.3, -.4, .8], "g": ["A", "A", "C"],
                        "exposure": [1., 1.7, 2.]}, index=["duplicate", "duplicate", "third"])
    before = copy.deepcopy(model.model_dump())
    terms = [c.term for c in model.coefficients]
    beta = np.array([c.estimate for c in model.coefficients])
    x = design(new, terms)
    offset = np.log(new.exposure.to_numpy()) if name == "poisson" else 0
    eta = x @ beta + offset
    response = inverse(name, eta)
    jac = slope(name, eta)[:, None] * x
    variance = np.einsum("nk,kl,nl->n", jac, model.covariance_matrix, jac)
    prediction = oe.predict(model, new, interval="mean")
    assert prediction.index.tolist() == new.index.tolist()
    assert_allclose(prediction.response, response, rtol=2e-12)
    assert_allclose(prediction.std_error, np.sqrt(variance), rtol=2e-12)
    assert_allclose(prediction.ci_low, response - stats.norm.ppf(.975) * np.sqrt(variance), rtol=2e-10)
    assert_allclose(oe.predict(model, new, kind="xb").xb, eta, rtol=2e-12)
    assert_allclose(oe.predict(model, new, kind="stdp").stdp,
                    np.sqrt(np.einsum("nk,kl,nl->n", x, model.covariance_matrix, x)), rtol=2e-12)
    assert_allclose(oe.predict(model, new, kind="derivative", term="x")["dydx[x]"],
                    slope(name, eta) * beta[terms.index("x")], rtol=2e-12)
    permutation = list(range(len(terms)))[::-1]
    changed = model.model_copy(deep=True)
    changed.coefficients = [changed.coefficients[i] for i in permutation]
    changed.covariance_matrix = np.array(changed.covariance_matrix)[permutation][:, permutation].tolist()
    assert_allclose(oe.predict(changed, new, interval="mean"), prediction, rtol=2e-12)
    assert model.model_dump() == before


@pytest.mark.parametrize("method", ["ame", "mem"])
def test_continuous_and_categorical_margins_delta_covariance_finite_difference(mean_model, method):
    name, frame, model = mean_model
    frame = frame.drop(columns=[model.spec.outcome])
    terms = [c.term for c in model.coefficients]
    beta = np.array([c.estimate for c in model.coefficients])
    x = design(frame, terms)
    offset = np.log(frame.exposure.to_numpy()) if name == "poisson" else np.zeros(len(frame))
    if method == "mem":
        x, offset = x.mean(axis=0, keepdims=True), np.array([offset.mean()])
    alternatives = []
    for level in ("A", "C"):
        changed = frame.copy()
        changed.g = level
        base = frame.copy()
        base.g = "B"
        alternative, reference = design(changed, terms), design(base, terms)
        if method == "mem":
            alternative, reference = alternative.mean(axis=0, keepdims=True), reference.mean(axis=0, keepdims=True)
        alternatives.append((alternative, reference))

    def oracle(b):
        continuous = np.mean(slope(name, x @ b + offset) * b[terms.index("x")])
        discrete = [np.mean(inverse(name, alt @ b + offset) - inverse(name, ref @ b + offset))
                    for alt, ref in alternatives]
        return np.r_[continuous, discrete]

    expected = oracle(beta)
    gradient = np.column_stack([(oracle(beta + np.eye(len(beta))[i] * 1e-5)
                                - oracle(beta - np.eye(len(beta))[i] * 1e-5)) / 2e-5
                               for i in range(len(beta))])
    variance = np.einsum("ik,kl,il->i", gradient, model.covariance_matrix, gradient)
    actual = oe.margins(model, ["x", "g"], data=frame, method=method)
    assert actual.variable.tolist() == ["x", "g[A]", "g[C]"]
    assert_allclose(actual.estimate, expected, rtol=2e-12)
    assert_allclose(actual.std_error, np.sqrt(variance), rtol=2e-8, atol=1e-12)
    assert "gradient" not in actual.columns
    assert actual.attrs["parameter_terms"] == [coefficient.term for coefficient in model.coefficients]
    assert_allclose(actual.attrs["delta_gradients"], gradient, rtol=2e-8, atol=1e-10)


def test_glm_poisson_and_binary_predictions_match_statsmodels():
    df = data()
    x = np.column_stack([np.ones(len(df)), df.x, df.z])
    for command, outcome, family in [("logit", "binary", sm.families.Binomial()),
                                     ("probit", "binary", sm.families.Binomial(link=sm.families.links.Probit())),
                                     ("poisson", "count", sm.families.Poisson())]:
        model = getattr(oe, command)(data=df, y=outcome, x=["x", "z"])
        reference = sm.GLM(df[outcome], x, family=family).fit()
        new = pd.DataFrame({"x": [-.5, .7], "z": [.1, -.2]})
        assert_allclose(oe.predict(model, new).response, reference.predict(sm.add_constant(new)), rtol=2e-7)
        effect = oe.margins(model, ["x", "z"], data=df)
        if command in {"logit", "probit"}:
            oracle = (sm.Logit if command == "logit" else sm.Probit)(df[outcome], x).fit(disp=False)
            assert_allclose(effect.estimate, oracle.get_margeff(at="overall").margeff, rtol=2e-7)
            assert_allclose(effect.std_error, oracle.get_margeff(at="overall").margeff_se, rtol=2e-7)


def test_binomial_trials_response_is_count_mean_and_margins_scales_by_trials():
    rng = np.random.default_rng(988)
    frame = pd.DataFrame({"x": rng.normal(size=180), "n": rng.integers(3, 10, size=180)})
    frame["y"] = rng.binomial(frame.n, expit(.2 + .5 * frame.x))
    model = stored(oe.glm(data=frame, y="y", x=["x"], family="binomial", trials="n"))
    new = pd.DataFrame({"x": [-1., 0., 1.], "n": [2, 5, 10]})
    beta = np.array([c.estimate for c in model.coefficients])
    eta = np.column_stack([np.ones(3), new.x]) @ beta
    assert_allclose(oe.predict(model, new).response, new.n * expit(eta))
    assert_allclose(oe.predict(model, new, kind="derivative", term="x")["dydx[x]"],
                    new.n * expit(eta) * (1 - expit(eta)) * beta[1])
    assert oe.margins(model, "x", data=new).estimate.iloc[0] == pytest.approx(
        np.mean(new.n * expit(eta) * (1 - expit(eta)) * beta[1]))


def test_constrained_fixed_coefficients_are_used_without_imputing_uncertainty():
    frame = data()
    model = stored(oe.cnsreg(data=frame, y="y", x=["x", "z"],
                            constraints=[{"terms": {"x": 1}, "value": .4}]))
    assert "x" not in [c.term for c in model.coefficients]
    assert model.extra["constrained_terms"] == {"x": .4}
    reference = sm.OLS(frame.y - .4 * frame.x, sm.add_constant(frame[["z"]])).fit()
    new = pd.DataFrame({"x": [-3., 2.], "z": [-.2, .5]})
    actual = oe.predict(model, new, interval="mean")
    assert_allclose(actual.response, reference.predict(sm.add_constant(new[["z"]])) + .4 * new.x)
    assert_allclose(actual.std_error, reference.get_prediction(sm.add_constant(new[["z"]])).se_mean, rtol=2e-12)
    fixed = oe.margins(model, "x", data=new)
    assert fixed.estimate.iloc[0] == .4 and fixed.std_error.iloc[0] == 0
    assert fixed.p_value.iloc[0] is None
    assert oe.margins(model, "z", data=new).estimate.iloc[0] == pytest.approx(reference.params.z)


def test_iv_new_row_predictions_require_endogenous_values_without_instruments_or_outcome():
    rng = np.random.default_rng(781)
    frame = pd.DataFrame({"x": rng.normal(size=180), "instrument": rng.normal(size=180)})
    u = rng.normal(size=180)
    frame["d"] = .8 * frame.instrument + .3 * u + rng.normal(size=180)
    frame["y"] = .2 + .4 * frame.x + .7 * frame.d + u
    model = stored(oe.ivregress(data=frame, y="y", x=["x"], endog=["d"], instruments=["instrument"], small=True))
    new = pd.DataFrame({"x": [-.5, .5], "d": [.2, -.2]})
    terms = [c.term for c in model.coefficients]
    x = np.column_stack([np.ones(2) if term == "Intercept" else new[term] for term in terms])
    predicted = oe.predict(model, new, interval="mean")
    assert_allclose(predicted.response, x @ np.array([c.estimate for c in model.coefficients]))
    se = np.sqrt(np.einsum("nk,kl,nl->n", x, model.covariance_matrix, x))
    critical = stats.t.ppf(.975, model.inference["df_inference"])
    assert_allclose(predicted.ci_low, predicted.response - critical * se, rtol=2e-10)


def test_rank_omissions_and_nb_dispersion_are_not_extra_mean_slopes():
    frame = data()
    frame["duplicate"] = 2 * frame.x
    model = stored(oe.poisson(data=frame, y="count", x=["x", "duplicate"]))
    assert "duplicate" in model.provenance["omitted_terms"]
    new = pd.DataFrame({"x": [-1., 1.], "duplicate": [12., -9.]})
    terms = [c.term for c in model.coefficients]
    expected = np.exp(np.column_stack([np.ones(2), new.x]) @ [c.estimate for c in model.coefficients])
    assert terms == ["Intercept", "x"]
    assert_allclose(oe.predict(model, new).response, expected)
    nb = stored(oe.nbreg(data=frame, y="nb", x=["x", "z"]))
    assert nb.coefficients[-1].term == "/lnalpha"
    baseline = oe.predict(nb, frame[["x", "z"]], interval="mean")
    nb.coefficients[-1].estimate += 10
    assert_allclose(oe.predict(nb, frame[["x", "z"]], interval="mean"), baseline)


def test_at_grids_category_reference_and_parameter_immutability(mean_model):
    _, frame, model = mean_model
    original = copy.deepcopy(model.model_dump())
    actual = oe.margins(model, "x", data=frame, at={"x": [-1., 1.], "g": ["A", "C"]})
    assert len(actual) == 4
    for _, row in actual.iterrows():
        scenario = frame.copy()
        scenario.x, scenario.g = row["at[x]"], row["at[g]"]
        derivative = oe.predict(model, scenario, kind="derivative", term="x")["dydx[x]"]
        assert row.estimate == pytest.approx(derivative.mean())
    assert model.model_dump() == original


def test_weight_normalization_remains_finite_for_large_finite_weights():
    frame = data(n=180)
    model = stored(oe.poisson(data=frame, y="count", x=["x"], weights="w", weight_type="aweight"))
    ordinary = pd.DataFrame({"x": [-1., .4], "w": [1., 1.]})
    huge = ordinary.assign(w=[1e308, 1e308])
    assert_allclose(oe.margins(model, "x", data=ordinary).select_dtypes("number"),
                    oe.margins(model, "x", data=huge).select_dtypes("number"), rtol=1e-12)
    model.spec.weights = None
    assert_allclose(oe.margins(model, "x", data=ordinary).select_dtypes("number"),
                    oe.margins(model.model_copy(deep=True), "x", data=ordinary).select_dtypes("number"))


def test_ols_original_prediction_margins_and_weights_dispatch_unchanged():
    frame = data(n=100)
    model = oe.ols(data=frame, y="y", x=["x", "z", "g"], categorical=["g"],
                   weights="w", weight_type="aweight", covariance="HC3")
    new = frame[["x", "z", "g"]].head(4)
    assert_allclose(oe.predict(model, new, interval="mean"), model.predict(new, interval="mean"))
    assert_allclose(oe.margins(model, ["x"]).select_dtypes("number"),
                    model.margins(["x"]).select_dtypes("number"))


def test_no_refit_occurs_and_inference_mode_margins_remain_valid(mean_model, monkeypatch):
    _, frame, model = mean_model
    import openecon.analysis
    monkeypatch.setattr(openecon.analysis, "fit", lambda *args, **kwargs: pytest.fail("prediction refitted"))
    reference = oe.margins(model, "x", data=frame)
    with torch.inference_mode():
        actual = oe.margins(model, "x", data=frame)
    assert_allclose(actual.select_dtypes("number"), reference.select_dtypes("number"))
    assert len(oe.predict(model, frame)) == len(frame)


@pytest.mark.parametrize("form", ["poisson", "mean", "constant"])
def test_censored_count_latent_mean_ignores_outcomes_event_columns_and_dispersion(form):
    from test_econ_count_censored import estimate, make_data
    training = make_data(form, "mixed")
    model = stored(estimate(training, form))
    original = copy.deepcopy(model.model_dump())
    new = pd.DataFrame({"x": [-1., .2, .8], "exposure": [1., 2., .8]})
    terms = [coefficient.term for coefficient in model.coefficients]
    beta = np.array([coefficient.estimate for coefficient in model.coefficients])
    x = np.column_stack([np.ones(3), new.x, *([np.zeros(3)] if form != "poisson" else [])])
    eta = x @ beta + np.log(new.exposure)
    mu = np.exp(eta)
    prediction = oe.predict(model, new, interval="mean")
    assert prediction.attrs["response_definition"] == "latent unconditional E[Y|X]"
    assert_allclose(prediction.response, mu, rtol=1e-12)
    assert_allclose(prediction.std_error,
                    np.sqrt(np.einsum("nk,kl,nl->n", mu.to_numpy()[:, None] * x,
                                      model.covariance_matrix, mu.to_numpy()[:, None] * x)), rtol=1e-12)
    effect = oe.margins(model, "x", data=new)
    expected = np.mean(mu * beta[terms.index("x")])
    gradient = np.mean(mu.to_numpy()[:, None] * beta[terms.index("x")] * x, axis=0)
    gradient[terms.index("x")] += np.mean(mu)
    assert effect.estimate.iloc[0] == pytest.approx(expected)
    assert_allclose(effect.attrs["delta_gradients"][0], gradient, rtol=1e-12)
    assert effect.std_error.iloc[0] == pytest.approx(np.sqrt(gradient @ np.array(model.covariance_matrix) @ gradient))
    if form != "poisson":
        ancillary = "/lnalpha" if form == "mean" else "/lndelta"
        assert terms[-1] == ancillary
        changed = model.model_copy(deep=True)
        changed.coefficients[-1].estimate += 4
        assert_allclose(oe.predict(changed, new, interval="mean"), prediction)
        assert effect.attrs["delta_gradients"][0][-1] == 0
    assert model.model_dump() == original


@pytest.mark.parametrize("link", ["identity", "log", "logit", "probit", "cloglog", "loglog",
                                  "reciprocal", "inverse_squared", "power", "nbinomial"])
def test_saved_glm_link_forward_and_effect_derivatives_independent_finite_differences(link):
    """Explicit saved parameter fixtures isolate each adapter from the fitter."""
    model = stored(oe.glm(data=data(n=100), y="y", x=["x"], family="gaussian"))
    family = "nbinomial" if link == "nbinomial" else "gaussian"
    model.spec.options.update(family=family, link=link, power=.5, dispersion=1.3)
    model.extra.update(family=family, link="power(0.5)" if link == "power" else link)
    model.coefficients[0].estimate = -2. if link == "nbinomial" else 2.
    model.coefficients[1].estimate = .1
    model = stored(model)
    new = pd.DataFrame({"x": [-.8, 0., .7]})
    x = np.column_stack([np.ones(3), new.x])
    beta = np.array([coefficient.estimate for coefficient in model.coefficients])

    def forward(eta):
        return {"identity": lambda: eta, "log": lambda: np.exp(eta), "logit": lambda: expit(eta),
                "probit": lambda: ndtr(eta), "cloglog": lambda: -np.expm1(-np.exp(eta)),
                "loglog": lambda: np.exp(-np.exp(-eta)), "reciprocal": lambda: 1 / eta,
                "inverse_squared": lambda: eta ** -.5, "power": lambda: eta ** 2,
                "nbinomial": lambda: np.exp(eta) / (1.3 * (1 - np.exp(eta)))}[link]()

    def derivative(eta):
        step = 1e-3
        return (forward(eta - 2 * step) - 8 * forward(eta - step)
                + 8 * forward(eta + step) - forward(eta + 2 * step)) / (12 * step)

    def margin(b):
        return np.mean(derivative(x @ b) * b[1])

    expected = forward(x @ beta)
    jacobian = derivative(x @ beta)[:, None] * x
    predicted = oe.predict(model, new, interval="mean")
    assert_allclose(predicted.response, expected, rtol=1e-12)
    assert_allclose(predicted.std_error, np.sqrt(np.einsum("nk,kl,nl->n", jacobian,
                                                        model.covariance_matrix, jacobian)), rtol=2e-8)
    actual = oe.margins(model, "x", data=new)
    directions = np.eye(2) * 1e-3
    gradient = np.array([(margin(beta - 2 * direction) - 8 * margin(beta - direction)
                          + 8 * margin(beta + direction) - margin(beta + 2 * direction)) / .012
                         for direction in directions])
    assert_allclose(actual.estimate, margin(beta), rtol=2e-8)
    assert_allclose(actual.attrs["delta_gradients"][0], gradient, rtol=3e-6, atol=1e-8)
