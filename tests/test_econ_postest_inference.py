"""Independent NumPy/statsmodels oracles for generic stored-result inference."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import Coefficient, ModelSpec, ResultBundle


def bundle(beta=(1., .6, -.3), covariance=None, *, use_t=False, df=37., terms=None,
           equations=None, alpha=.05):
    beta = np.asarray(beta, dtype=float)
    covariance = np.asarray(covariance if covariance is not None else
                            [[.2, .015, -.025], [.015, .04, .012], [-.025, .012, .09]])
    names = terms or ["Intercept", "x", "z"]
    se = np.sqrt(np.diag(covariance))
    critical = stats.t.ppf(1 - alpha / 2, df) if use_t else stats.norm.ppf(1 - alpha / 2)
    return ResultBundle(
        id="generic-oracle", created_at="2026-10-04T00:00:00Z",
        spec=ModelSpec(estimator="poisson", outcome="y", predictors=["x", "z"]),
        nobs=40, nobs_original=40, dropped_rows=0,
        coefficients=[Coefficient(term=name, equation=(equations or [None] * len(beta))[i],
                                  estimate=b, std_error=se[i], statistic=b / se[i],
                                  p_value=2 * stats.norm.sf(abs(b / se[i])),
                                  ci_low=b - critical * se[i], ci_high=b + critical * se[i])
                      for i, (name, b) in enumerate(zip(names, beta, strict=True))],
        covariance_matrix=covariance.tolist(), metrics={}, warnings=[], predictions=[],
        sample_positions=list(range(40)), provenance={"stata_parity_validated": False},
        inference={"use_t": use_t, "distribution": "t" if use_t else "normal",
                   "df_inference": df if use_t else None, "df_resid": 37., "alpha": alpha})


@pytest.mark.parametrize("use_t,df", [(False, 37.), (True, 37.), (True, 5.), (True, 1e11)])
@pytest.mark.parametrize("restriction,matrix,null", [
    ("x", [[0, 1, 0]], .2),
    (["x", "z"], [[0, 1, 0], [0, 0, 1]], 0),
    ({"x": 1, "z": -2}, [[0, 1, -2]], -.1),
    ([{"x": 1}, {"z": 2}], [[0, 1, 0], [0, 0, 2]], [.1, .3]),
    (np.array([[1, 0, 1], [0, 1, 0]]), [[1, 0, 1], [0, 1, 0]], [1, .1]),
])
def test_wald_matches_independent_numpy_and_reference_tails(use_t, df, restriction, matrix, null):
    model = bundle(use_t=use_t, df=df)
    original = copy.deepcopy(model.model_dump())
    outcome = oe.test(model, restriction, null)
    r = np.array(matrix, dtype=float)
    target = np.broadcast_to(null, len(r))
    beta = np.array([c.estimate for c in model.coefficients])
    v = np.array(model.covariance_matrix)
    difference = r @ beta - target
    wald = difference @ np.linalg.solve(r @ v @ r.T, difference)
    expected = wald / len(r) if use_t else wald
    tail = stats.f.sf(expected, len(r), df) if use_t else stats.chi2.sf(expected, len(r))
    assert outcome["statistic"] == pytest.approx(expected, rel=2e-12)
    assert outcome["p_value"] == pytest.approx(tail, rel=2e-9, abs=1e-14)
    assert outcome["distribution"] == ("F" if use_t else "chi2")
    if use_t:
        assert outcome["df_num"] == len(r) and outcome["df_denom"] == df
    else:
        assert outcome["df"] == len(r)
    if len(r) == 1:
        assert outcome["t_statistic" if use_t else "z_statistic"] == pytest.approx(
            difference[0] / np.sqrt((r @ v @ r.T)[0, 0]))
    assert model.model_dump() == original


@pytest.mark.parametrize("use_t", [False, True])
def test_lincom_and_nlcom_match_numpy_delta_method_and_recorded_alpha(use_t):
    model = bundle(use_t=use_t, alpha=.01)
    model = ResultBundle.model_validate_json(model.model_dump_json())
    beta = np.array([c.estimate for c in model.coefficients])
    covariance = np.array(model.covariance_matrix)
    linear = oe.lincom(model, {"x": 2, "z": -1}, .1)
    nonlinear = oe.nlcom(model, lambda b: b["x"] / b["z"], null=1)
    for actual, estimate, gradient, null in [
        (linear, 2 * beta[1] - beta[2] + .1, np.array([0, 2, -1]), 0),
        (nonlinear, beta[1] / beta[2], np.array([0, 1 / beta[2], -beta[1] / beta[2]**2]), 1),
    ]:
        se = np.sqrt(gradient @ covariance @ gradient)
        statistic = (estimate - null) / se
        critical = stats.t.ppf(.995, 37) if use_t else stats.norm.ppf(.995)
        tail = 2 * (stats.t.sf(abs(statistic), 37) if use_t else stats.norm.sf(abs(statistic)))
        assert actual["estimate"] == pytest.approx(estimate)
        assert actual["std_error"] == pytest.approx(se)
        assert actual["statistic"] == pytest.approx(statistic)
        assert actual["p_value"] == pytest.approx(tail, rel=2e-10)
        assert_allclose([actual["ci_low"], actual["ci_high"]],
                        [estimate - critical * se, estimate + critical * se], rtol=2e-10)
        assert actual["distribution"] == ("t" if use_t else "normal")
        assert actual["alpha"] == .01
        assert_allclose(list(actual["gradient"].values()), gradient)


@pytest.mark.parametrize("covariance", ["nonrobust", "HC3", "cluster"])
def test_public_dispatch_preserves_actual_ols_methods_and_adjustments(covariance):
    rng = np.random.default_rng(730)
    data = pd.DataFrame({"x": rng.normal(size=120), "z": rng.normal(size=120),
                         "group": np.arange(120) % 12})
    data["y"] = .4 + data.x - .2 * data.z + rng.normal(size=120)
    model = oe.ols(data=data, y="y", x=["x", "z"], covariance=covariance,
                   cluster="group" if covariance == "cluster" else None)
    assert oe.test(model, {"x": 1, "z": -1}, .2) == model.test({"x": 1, "z": -1}, .2)
    assert oe.testparm(model, ["x", "z"]) == model.testparm(["x", "z"])
    assert oe.lincom(model, {"x": 1, "z": -1}) == model.lincom({"x": 1, "z": -1})
    assert oe.nlcom(model, lambda b: b["x"] / b["z"], 1) == model.nlcom(
        lambda b: b["x"] / b["z"], 1)
    # Retained fitted contrast controls must never be replaced by generic b,V.
    model._state["contrast_inference"] = lambda gradient: {"df": 7., "scale": 1.2}
    assert oe.lincom(model, {"x": 1}) == model.lincom({"x": 1})
    assert oe.test(model, "x") == model.test("x")
    assert oe.lincom(model, {"x": 1})["df"] == 7


def test_actual_registry_poisson_matches_statsmodels_wald_and_ttest():
    rng = np.random.default_rng(793)
    data = pd.DataFrame({"x": rng.normal(size=250), "z": rng.normal(size=250)})
    data["y"] = rng.poisson(np.exp(.4 + .3 * data.x - .2 * data.z))
    model = oe.poisson(data=data, y="y", x=["x", "z"], covariance="nonrobust")
    reference = sm.GLM(data.y, sm.add_constant(data[["x", "z"]]), family=sm.families.Poisson()).fit()
    terms = [c.term for c in model.coefficients]
    r = np.zeros((2, len(terms)))
    r[0, terms.index("x")], r[1, terms.index("z")] = 1, 1
    oracle = reference.wald_test(r, scalar=True)
    actual = oe.test(model, ["x", "z"])
    assert actual["statistic"] == pytest.approx(float(oracle.statistic), rel=2e-7)
    assert actual["p_value"] == pytest.approx(float(oracle.pvalue), rel=2e-7)
    r1 = np.array([0., 1., -1.])
    reference_linear = reference.t_test(r1)
    linear = oe.lincom(model, {"x": 1, "z": -1})
    assert linear["estimate"] == pytest.approx(float(reference_linear.effect[0]), rel=2e-8)
    assert linear["std_error"] == pytest.approx(float(reference_linear.sd[0, 0]), rel=2e-7)
    assert model.provenance["stata_parity_validated"] is False


def test_equation_qualified_names_aliases_wildcards_and_ambiguous_names():
    model = bundle(terms=["a:Intercept", "a:x", "b:x"], equations=["a", "a", "b"])
    expected = oe.test(model, {"a:x": 1, "b:x": -1})
    assert oe.test(model, {"[a]x": 1, "[b]x": -1}) == expected
    assert oe.testparm(model, ["a:x", "b:*"])["terms"] == ["a:x", "b:x"]
    assert oe.testparm(model, "x")["terms"] == ["a:x", "b:x"]
    assert oe.nlcom(model, lambda b: b["[a]x"] / b["b:x"])["estimate"] == pytest.approx(-2.)
    for operation in (lambda: oe.test(model, "x"), lambda: oe.lincom(model, {"x": 1}),
                      lambda: oe.nlcom(model, lambda b: b["x"])):
        with pytest.raises(AnalysisError) as error:
            operation()
        assert error.value.code == "ambiguous_term"


@pytest.mark.parametrize("small", [False, True])
def test_actual_system_cross_equation_wald_uses_full_covariance_after_json(small):
    rng = np.random.default_rng(840)
    data = pd.DataFrame({"x": rng.normal(size=100), "z": rng.normal(size=100)})
    noise = rng.normal(size=(100, 2)) @ np.array([[1., .7], [0., .8]])
    data["y1"] = 1 + .6 * data.x + noise[:, 0]
    data["y2"] = 2 + .3 * data.x + .2 * data.z + noise[:, 1]
    model = oe.sureg(data=data, equations=[{"y": "y1", "x": ["x"], "name": "a"},
                                          {"y": "y2", "x": ["x", "z"], "name": "b"}], small=small)
    model = ResultBundle.model_validate_json(model.model_dump_json())
    names = [c.term for c in model.coefficients]
    r = np.zeros(len(names))
    r[names.index("a:x")], r[names.index("b:x")] = 1, -1
    beta, v = np.array([c.estimate for c in model.coefficients]), np.array(model.covariance_matrix)
    output = oe.test(model, {"a:x": 1, "b:x": -1})
    assert output["statistic"] == pytest.approx((r @ beta)**2 / (r @ v @ r), rel=2e-12)
    assert output["distribution"] == ("F" if model.inference["use_t"] else "chi2")
    assert oe.nlcom(model, lambda b: torch.exp(b["a:x"] - b["b:x"]))["estimate"] == pytest.approx(
        np.exp(r @ beta))


def test_testparm_factor_names_no_duplicates_and_no_matches():
    model = bundle(terms=["x", "g[B]", "g[C]"])
    selected = oe.testparm(model, ["g", "g*", "g[B]"])
    assert selected["terms"] == ["g[B]", "g[C]"]
    assert selected == {**oe.test(model, ["g[B]", "g[C]"]), "terms": ["g[B]", "g[C]"]}
    with pytest.raises(AnalysisError, match="match") as error:
        oe.testparm(model, "absent*")
    assert error.value.code == "unknown_term"


def test_equation_alias_retains_interaction_colons_and_reporting_order():
    model = bundle(terms=["Intercept", "x:g[B]", "z"], equations=[None, "outcome", "other"])
    assert oe.lincom(model, {"outcome:x:g[B]": 1}) == oe.lincom(model, {"x:g[B]": 1})
    assert oe.testparm(model, ["z", "outcome:*"])["terms"] == ["x:g[B]", "z"]
    with pytest.raises(AnalysisError) as error:
        oe.test(model, "outcome:g[B]")
    assert error.value.code == "unknown_term"


def test_legacy_method_protocol_is_unchanged():
    duck = SimpleNamespace(test=lambda *args, **kwargs: (args, kwargs))
    assert oe.test(duck, "x", value=.1) == (("x",), {"value": .1})


def test_nlcom_works_under_inference_mode_and_no_grad_without_mutation():
    model = bundle()
    before = copy.deepcopy(model.model_dump())
    ordinary = oe.nlcom(model, lambda b: torch.exp(b["x"]) + b["z"]**2)
    with torch.inference_mode():
        assert oe.nlcom(model, lambda b: torch.exp(b["x"]) + b["z"]**2) == ordinary
    with torch.no_grad():
        assert oe.nlcom(model, lambda b: torch.exp(b["x"]) + b["z"]**2) == ordinary
    assert model.model_dump() == before
