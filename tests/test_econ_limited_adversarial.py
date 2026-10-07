"""Stress and invariance tests of the limited-dependent-variable family.

Units of the data and of the weights, heavy censoring, observations far in the
tails, degenerate samples and the registry contract of every estimator.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import stats
from test_econ_limited import covariance, errors, estimates, terms

import openecon as oe
from openecon.analysis import AnalysisError, fit
from openecon.econometrics import registry
from openecon.models import ModelSpec, ResultBundle

X = ["x1", "x2"]
NAMES = ("tobit", "truncreg", "intreg", "heckman", "heckprobit", "ivprobit", "ivtobit")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20264)
    n = 1500
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                          "z1": rng.normal(size=n), "z2": rng.normal(size=n),
                          "w": rng.uniform(0.5, 2.0, size=n)})
    shocks = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
    frame["latent"] = 0.3 + 0.8 * frame.x1 - 0.5 * frame.x2 + shocks[:, 0]
    frame["y"] = frame.latent.clip(lower=0.0)
    frame["lo"] = np.floor(frame.latent)
    frame["hi"] = frame.lo + 1.0
    frame["s"] = (0.2 + 0.5 * frame.x1 + 0.8 * frame.z1 + shocks[:, 1] > 0) * 1.0
    frame["ys"] = np.where(frame.s == 1, frame.latent, np.nan)
    frame["d"] = np.where(frame.s == 1, (frame.latent > 0) * 1.0, np.nan)
    frame["e"] = 0.5 * frame.x1 + 0.7 * frame.z1 - 0.4 * frame.z2 + shocks[:, 1]
    frame["b"] = (0.2 + 0.5 * frame.x1 - 0.6 * frame.e + shocks[:, 0] > 0) * 1.0
    frame["t"] = (0.2 + 0.5 * frame.x1 - 0.6 * frame.e + shocks[:, 0]).clip(lower=0.0)
    return frame


def calls(frame, **options):
    """One call of every estimator of the family."""
    iv = {"x": ["x1"], "endog": ["e"], "instruments": ["z1", "z2"]}
    selection = {"x": X, "select": "s", "select_x": ["x1", "z1"]}
    return {
        "tobit": lambda: oe.tobit(data=frame, y="y", x=X, ll=0, **options),
        "truncreg": lambda: oe.truncreg(data=frame, y="latent", x=X, ll=-1.0, **options),
        "intreg": lambda: oe.intreg(data=frame, y_low="lo", y_high="hi", x=X, **options),
        "heckman": lambda: oe.heckman(data=frame, y="ys", **selection, **options),
        "heckprobit": lambda: oe.heckprobit(data=frame, y="d", **selection, **options),
        "ivprobit": lambda: oe.ivprobit(data=frame, y="b", **iv, **options),
        "ivtobit": lambda: oe.ivtobit(data=frame, y="t", ll=0, **iv, **options),
    }


def test_registry_contract_of_the_family():
    catalogue = {info.name: info for info in registry.all_estimators()
                 if info.family == "limited"}
    assert tuple(catalogue) == NAMES
    exports = registry.public_exports()
    capabilities = oe.capabilities()["estimators"]
    for name, info in catalogue.items():
        assert info.function == name and exports[name][1] == name
        assert info.covariances == ("nonrobust", "opg", "robust", "cluster")
        assert set(info.weights) == {"fweight", "aweight", "pweight", "iweight"}
        assert info.inference == ("t" if name == "tobit" else "z")
        assert capabilities[name]["family"] == "limited" and capabilities[name]["stata"]
        assert registry.load_entry(info).__name__ == f"fit_{name}"
    assert {role.name for role in catalogue["heckman"].roles} == {"select", "select_x"}
    assert {role.name for role in catalogue["intreg"].roles} == {"upper", "offset"}
    assert {role.name for role in catalogue["ivtobit"].roles} == {"endogenous", "instruments"}
    assert catalogue["ivprobit"].option("method").choices == ("ml", "twostep")
    assert {option.name for option in catalogue["tobit"].options} \
        == {"ll", "ul", "ll_at_min", "ul_at_max"}
    assert catalogue["ivprobit"].predictors == "optional"


def test_every_estimator_runs_through_the_generic_fit_and_round_trips(data):
    specs = {
        "tobit": {"outcome": "y", "predictors": X, "options": {"ll": 0}},
        "truncreg": {"outcome": "latent", "predictors": X, "options": {"ll": -1.0}},
        "intreg": {"outcome": "lo", "predictors": X, "columns": {"upper": "hi"}},
        "heckman": {"outcome": "ys", "predictors": X,
                    "columns": {"select": "s", "select_x": ["x1", "z1"]}},
        "heckprobit": {"outcome": "d", "predictors": X,
                       "columns": {"select": "s", "select_x": ["x1", "z1"]}},
        "ivprobit": {"outcome": "b", "predictors": ["x1"],
                     "columns": {"endogenous": ["e"], "instruments": ["z1", "z2"]}},
        "ivtobit": {"outcome": "t", "predictors": ["x1"], "options": {"ll": 0},
                    "columns": {"endogenous": ["e"], "instruments": ["z1", "z2"]}},
    }
    direct = calls(data)
    for name, fields in specs.items():
        result = fit(ModelSpec(estimator=name, **fields), data=data)
        assert_allclose(estimates(result), estimates(direct[name]()), rtol=1e-12)
        restored = ResultBundle.model_validate_json(result.model_dump_json())
        assert restored == result
        assert all(np.isfinite([c.estimate, c.std_error, c.p_value, c.ci_low, c.ci_high]).all()
                   for c in result.coefficients)
        assert result.inference["covariance"] == "nonrobust"
        assert result.provenance["estimator"] == name
        assert len(terms(result)) == len(set(terms(result)))
        assert result.summary() and result.to_latex()
    for name in ("heckman", "ivprobit", "ivtobit"):
        fields = {**specs[name], "options": {**specs[name].get("options", {}),
                                             "method": "twostep"}}
        result = fit(ModelSpec(estimator=name, **fields), data=data)
        assert ResultBundle.model_validate_json(result.model_dump_json()) == result
        assert "two-step" in result.title


@pytest.mark.parametrize("factor", [1e-9, 1e9])
def test_sampling_and_importance_weights_do_not_depend_on_their_unit(data, factor):
    scaled = data.assign(w=data.w * factor)
    for name in NAMES:
        base = calls(data, weights="w", weight_type="pweight")[name]()
        other = calls(scaled, weights="w", weight_type="pweight")[name]()
        assert base.spec.covariance == "robust"
        assert_allclose(estimates(other), estimates(base), rtol=1e-8, atol=1e-10)
        assert_allclose(covariance(other), covariance(base), rtol=1e-6, atol=1e-12)
    base = calls(data, weights="w", weight_type="iweight")["tobit"]()
    other = calls(scaled, weights="w", weight_type="iweight")["tobit"]()
    assert_allclose(estimates(other), estimates(base), rtol=1e-8)
    assert_allclose(covariance(other) * factor, covariance(base), rtol=1e-6)


def test_estimates_are_equivariant_to_the_units_of_the_data(data):
    base = oe.tobit(data=data, y="y", x=X, ll=0)
    units = data.assign(y=data.y * 1e6, x1=data.x1 * 1e-4, x2=data.x2 * 1e5 + 1e7)
    other = oe.tobit(data=units, y="y", x=X, ll=0)
    assert_allclose(estimates(other)[1:] / estimates(base)[1:], [1e10, 10.0, 1e6], rtol=1e-8)
    assert_allclose(other.tests["model"]["statistic"], base.tests["model"]["statistic"],
                    rtol=1e-8)
    assert_allclose(other.metrics["pseudo_r_squared"],
                    1 - other.metrics["log_likelihood"] / other.extra["null_log_likelihood"])
    shifted = oe.intreg(data=data.assign(lo=data.lo * 1e5, hi=data.hi * 1e5), y_low="lo",
                        y_high="hi", x=X)
    reference = oe.intreg(data=data, y_low="lo", y_high="hi", x=X)
    assert_allclose(estimates(shifted)[:3], 1e5 * estimates(reference)[:3], rtol=1e-8)
    assert_allclose(estimates(shifted)[3], estimates(reference)[3] + np.log(1e5), rtol=1e-9)
    scaled = oe.heckman(data=data.assign(ys=data.ys * 1e3), y="ys", x=X, select="s",
                        select_x=["x1", "z1"])
    plain = oe.heckman(data=data, y="ys", x=X, select="s", select_x=["x1", "z1"])
    assert_allclose(estimates(scaled)[:3], 1e3 * estimates(plain)[:3], rtol=1e-7)
    assert_allclose(scaled.metrics["rho"], plain.metrics["rho"], rtol=1e-8)
    assert_allclose(scaled.metrics["lambda"], 1e3 * plain.metrics["lambda"], rtol=1e-7)


def test_heavy_censoring_tail_observations_and_narrow_intervals(data):
    heavy = data.assign(y=data.latent.clip(lower=2.0))
    result = oe.tobit(data=heavy, y="y", x=X, ll=2.0)
    assert result.metrics["n_left_censored"] > 0.85 * len(data)
    assert_allclose(estimates(result)[1:3], [0.8, -0.5], atol=0.3)
    # A censored observation 40 standard deviations beyond its limit stays finite.
    outlier = data.assign(x1=np.where(np.arange(len(data)) == 0, 60.0, data.x1))
    outlier.loc[0, "y"] = 0.0
    far = oe.tobit(data=outlier, y="y", x=X, ll=0)
    assert np.isfinite(estimates(far)).all() and np.isfinite(far.metrics["log_likelihood"])
    assert far.provenance["optimizer"]["converged"]
    # Intervals of width 2e-7 around the exact outcome reproduce the point-data regression.
    narrow = data.assign(lo=data.latent - 1e-7, hi=data.latent + 1e-7)
    interval = oe.intreg(data=narrow, y_low="lo", y_high="hi", x=X)
    exact = oe.tobit(data=data, y="latent", x=X)
    assert_allclose(estimates(interval)[:3], estimates(exact)[:3], rtol=1e-6)
    assert_allclose(np.exp(estimates(interval)[3]), estimates(exact)[3], rtol=1e-6)
    assert interval.metrics["n_interval"] == len(data)


def test_tobit_without_constant_and_confidence_level(data):
    result = oe.tobit(data=data, y="y", x=X, ll=0, intercept=False, alpha=0.1)
    assert terms(result) == ["x1", "x2", "/sigma"]
    assert result.metrics["pseudo_r_squared"] is None
    assert result.metrics["df_model"] == 2 and result.metrics["df_resid"] == len(data) - 2
    test = result.tests["model"]
    assert test["distribution"] == "F" and test["df2"] == len(data) - 2
    half = stats.t.ppf(0.95, len(data) - 2) * errors(result)
    assert_allclose([c.ci_low for c in result.coefficients], estimates(result) - half, rtol=1e-9)
    assert result.inference["confidence_level"] == pytest.approx(0.9)
    constant = oe.tobit(data=data.assign(k=1.0), y="y", x=["k"], ll=0)
    assert terms(constant) == ["Intercept", "/sigma"] and "model" not in constant.tests
    assert constant.metrics["df_model"] == 0


def test_degenerate_samples_raise_helpful_errors(data):
    def code(call):
        with pytest.raises(AnalysisError) as error:
            call()
        assert len(str(error.value)) > 20
        return error.value.code

    rng = np.random.default_rng(2)
    dummy = data.assign(dz=(data.z1 > 0) * 1.0)
    dummy["s"] = (0.2 + 0.8 * dummy.dz + rng.normal(size=len(dummy)) > 0) * 1.0
    dummy["ys"] = np.where(dummy.s == 1, 1 + 0.7 * dummy.dz + rng.normal(size=len(dummy)), np.nan)
    # With one binary selection regressor the Mills ratio is a linear function of it.
    assert code(lambda: oe.heckman(data=dummy, y="ys", x=["dz"], select="s", select_x=["dz"],
                                   method="twostep")) == "collinear_mills_ratio"
    assert code(lambda: oe.heckman(data=dummy, y="ys", x=["dz"], select="s", select_x=["dz"])) \
        in {"nonconvergence", "boundary_solution", "singular_information"}
    perfect = data.assign(ys=np.where(data.s == 1, 1 + 2 * data.x1, np.nan))
    # An exactly fitted outcome has no error variance: diagnosed before any iteration.
    for method in ("ml", "twostep"):
        assert code(lambda: oe.heckman(data=perfect, y="ys", x=["x1"], select="s",
                                       select_x=["x1", "z1"], method=method)) == "perfect_fit"
    tiny = data.head(4)
    assert code(lambda: oe.ivprobit(data=tiny, y="b", x=["x1"], endog=["e"],
                                    instruments=["z1", "z2"])) \
        in {"insufficient_observations", "constant_outcome", "separation_detected",
            "underidentified", "collinear_endogenous"}
    assert code(lambda: oe.tobit(data=data.head(0), y="y", x=X, ll=0)) == "empty_data"
    assert code(lambda: oe.tobit(data=data, y="nope", x=X, ll=0)) == "missing_columns"


def test_weak_instruments_inflate_standard_errors_and_are_reported(data):
    rng = np.random.default_rng(5)
    weak = data.assign(e=0.5 * data.x1 + 0.02 * data.z1 + rng.normal(size=len(data)))
    weak["b"] = (0.2 + 0.5 * weak.x1 - 0.6 * weak.e + rng.normal(size=len(data)) > 0) * 1.0
    strong = oe.ivprobit(data=data, y="b", x=["x1"], endog=["e"], instruments=["z1"])
    poor = oe.ivprobit(data=weak, y="b", x=["x1"], endog=["e"], instruments=["z1"])
    assert poor.extra["first_stage"]["e"]["f_statistic"] < 10 \
        < strong.extra["first_stage"]["e"]["f_statistic"]
    assert errors(poor)[2] > 5 * errors(strong)[2]


def test_two_step_and_ml_agree_on_exogenous_data():
    # Without endogeneity the probit on the regressors is consistent: every estimator agrees.
    rng = np.random.default_rng(6)
    n = 6000
    frame = pd.DataFrame({"x1": rng.normal(size=n), "z1": rng.normal(size=n)})
    frame["e"] = 0.5 * frame.x1 + 0.9 * frame.z1 + rng.normal(size=n)
    frame["b"] = (0.2 + 0.5 * frame.x1 - 0.6 * frame.e + rng.normal(size=n) > 0) * 1.0
    probit = sm.Probit(frame.b, sm.add_constant(frame[["x1", "e"]])).fit(disp=0)
    for method in ("ml", "twostep"):
        result = oe.ivprobit(data=frame, y="b", x=["x1"], endog=["e"], instruments=["z1"],
                             method=method)
        assert_allclose(estimates(result)[:3], probit.params, atol=0.08)
        assert result.tests["exogeneity"]["p_value"] > 0.01
