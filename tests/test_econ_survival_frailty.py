"""Shared-frailty survival models against independent references.

* Cox shared frailty (``oe.stcox_frailty``): R survival ``coxph(... frailty(g,
  distribution='gamma', sparse=FALSE))`` and R coxme (log-normal, Laplace) at fixed
  variances and at the maximizer of R's own integrated likelihood (``optimize``),
  committed by ``scripts/generate_survival_frailty_oracles.R``; the gamma EM fixed
  point written out with explicit NumPy loops (Klein 1992); analytic derivatives
  against numerical ones.
* Parametric shared frailty (``oe.streg_frailty``): the marginal likelihood integrated
  numerically over the frailty density (``scipy.integrate.quad``) and maximized with
  SciPy (``scripts/generate_survival_frailty_oracles.py numpy``).
* Survival treatment effects (``oe.stteffects``): the stacked estimating equations
  written from their definitions in NumPy (statsmodels logit, SciPy Weibull fits,
  central-difference Jacobian, ``A^-1 B A^-T``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survival.data import RiskSets
from openecon.econometrics.survival.frailty_kernels import FrailtyCox, PenalizedFrailty
from openecon.engines.optimize import check_derivatives
from openecon.models import ResultBundle

FIXTURES = Path(__file__).parent / "fixtures" / "survival_frailty"
DATA = json.loads((FIXTURES / "data_cox.json").read_text())["datasets"]
R = json.loads((FIXTURES / "r_reference.json").read_text())
CASES = {"cox_gamma": {}, "cox_entry": {"entry": "t0", "strata": "s"}}


def frame(name: str) -> pd.DataFrame:
    return pd.DataFrame(DATA[name])


def fit(name: str, ties: str, frailty: str = "gamma", **options):
    return oe.stcox_frailty(data=frame(name), time="t", failure="d", x=R[name]["terms"],
                            shared="g", ties=ties, frailty=frailty, **CASES[name], **options)


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def covariance(result):
    return np.array(result.covariance_matrix)


def frailties(result):
    return np.array([row["log_frailty"] for row in result.extra["frailty_estimates"]])


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_gamma_fixed_theta_matches_r_coxph(name, ties):
    ref = R[name][ties]["gamma_fixed_0.7"]
    result = fit(name, ties, theta=0.7)
    assert_allclose(estimates(result), ref["coef"], rtol=0, atol=1e-8)
    assert_allclose(covariance(result), ref["var"], rtol=1e-7, atol=1e-12)
    se2 = np.array(list(result.extra["se2"].values()))
    assert_allclose(se2, np.sqrt(np.diag(ref["var2"])), rtol=1e-7)
    assert_allclose(frailties(result), ref["frail"], rtol=0, atol=1e-7)
    assert result.metrics["log_likelihood"] == pytest.approx(ref["ilik"], abs=1e-8)
    assert result.metrics["partial_log_likelihood"] == pytest.approx(ref["partial"], abs=1e-8)
    penalized = ref["partial"] - ref["penalty"]
    assert result.metrics["penalized_log_likelihood"] == pytest.approx(penalized, abs=1e-8)
    assert "theta" not in result.tests                      # no test at a fixed variance
    assert result.metrics["log_likelihood_cox"] == pytest.approx(R[name][ties]["cox"]["loglik"],
                                                                 abs=1e-9)


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_gamma_estimated_theta_matches_r_profile_maximum(name, ties):
    ref = R[name][ties]["gamma_profile_max"]
    result = fit(name, ties)
    assert result.metrics["theta"] == pytest.approx(ref["theta"], rel=2e-5)
    # Our maximizer is at least as high as R's optimize() solution of the same profile.
    assert result.metrics["log_likelihood"] >= ref["ilik"] - 1e-9
    assert result.metrics["log_likelihood"] == pytest.approx(ref["ilik"], abs=1e-8)
    assert_allclose(estimates(result), ref["coef"], rtol=0, atol=2e-6)
    assert_allclose(np.sqrt(np.diag(covariance(result))), np.sqrt(np.diag(ref["var"])), rtol=2e-5)
    cox = R[name][ties]["cox"]["loglik"]
    test = result.tests["theta"]
    assert test["statistic"] == pytest.approx(2 * (ref["ilik"] - cox), abs=1e-7)
    assert test["p_value"] == pytest.approx(0.5 * stats.chi2.sf(test["statistic"], 1), rel=1e-9)
    assert test["distribution"] == "chibar2" and test["df"] == 1
    # R's own default EM search lands on the same maximum (to its looser tolerance).
    assert result.metrics["theta"] == pytest.approx(R[name][ties]["gamma_em"]["theta"], rel=2e-3)


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_lognormal_matches_coxme(name, ties):
    fixed = R[name][ties]["lognormal_fixed_0.5"]
    result = fit(name, ties, "lognormal", theta=0.5)
    assert_allclose(estimates(result), fixed["coef"], rtol=0, atol=1e-8)
    assert_allclose(covariance(result), fixed["var"], rtol=1e-7, atol=1e-12)
    assert_allclose(frailties(result), fixed["frail"], rtol=0, atol=1e-7)
    assert result.metrics["log_likelihood"] == pytest.approx(fixed["integrated"], abs=1e-8)
    assert result.metrics["penalized_log_likelihood"] == pytest.approx(fixed["penalized"], abs=1e-8)
    best = R[name][ties]["lognormal_profile_max"]
    result = fit(name, ties, "lognormal")
    assert result.metrics["theta"] == pytest.approx(best["theta"], rel=2e-5)
    assert result.metrics["log_likelihood"] >= best["integrated"] - 1e-9
    assert_allclose(estimates(result), best["coef"], rtol=0, atol=2e-6)
    cox = R[name][ties]["cox"]["loglik"]
    assert result.tests["theta"]["statistic"] == pytest.approx(2 * (best["integrated"] - cox),
                                                               abs=1e-7)
    # coxme's default BFGS search (reltol 1e-5) agrees loosely.
    assert result.metrics["theta"] == pytest.approx(R[name][ties]["lognormal_default"]["theta"],
                                                    rel=2e-2)


def test_boundary_estimate_is_the_cox_model():
    ref = R["cox_null"]
    # R's integrated likelihood lies below the Cox partial likelihood on its whole grid.
    assert max(ref["gamma_gain"]) < 0 and max(ref["lognormal_gain"]) < 0
    data = frame("cox_null")
    cox = oe.stcox(data=data, time="t", failure="d", x=["x1"])
    for frailty in ("gamma", "lognormal"):
        result = oe.stcox_frailty(data=data, time="t", failure="d", x=["x1"], shared="g",
                                  frailty=frailty)
        assert result.metrics["theta"] == 0.0
        assert result.extra["frailty"]["boundary"] is True
        assert result.tests["theta"]["statistic"] == 0.0 and result.tests["theta"]["p_value"] == 1.0
        assert_allclose(estimates(result), ref["cox"]["coef"], atol=1e-9)
        assert_allclose(covariance(result), ref["cox"]["var"], rtol=1e-8)
        assert_allclose(estimates(result), estimates(cox), atol=1e-12)
        assert result.metrics["log_likelihood"] == pytest.approx(ref["cox"]["loglik"], abs=1e-9)
        assert result.metrics["aic"] == pytest.approx(-2 * ref["cox"]["loglik"] + 2, abs=1e-8)
        assert np.all(frailties(result) == 0)


def test_gamma_breslow_solution_is_the_em_fixed_point():
    """Klein (1992) EM: w_g = ln((nu + D_g) / (nu + A_g)) with A_g from the Breslow hazard."""
    data = frame("cox_gamma")
    terms = R["cox_gamma"]["terms"]
    result = oe.stcox_frailty(data=data, time="t", failure="d", x=terms, shared="g", theta=0.7)
    beta, omega = estimates(result), frailties(result)
    labels = [row["group"] for row in result.extra["frailty_estimates"]]
    position = {label: i for i, label in enumerate(labels)}
    g = data.g.map(position).to_numpy()
    t, d = data.t.to_numpy(), data.d.to_numpy()
    risk = np.exp(data[terms].to_numpy() @ beta + omega[g])
    cumulative = np.zeros(len(t))
    for tau in np.unique(t[d == 1]):
        jump = ((t == tau) & (d == 1)).sum() / risk[t >= tau].sum()
        cumulative[t >= tau] += jump
    nu = 1 / 0.7
    events = np.bincount(g, d, len(labels))
    exposure = np.bincount(g, risk / np.exp(omega[g]) * cumulative, len(labels))
    assert_allclose(omega, np.log((nu + events) / (nu + exposure)), atol=1e-8)
    assert_allclose(np.exp(omega).sum(), len(labels), rtol=1e-10)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("distribution", ["gamma", "lognormal"])
def test_penalized_derivatives(ties, distribution):
    data = frame("cox_entry")
    t = torch.tensor(data.t.to_numpy(), dtype=torch.float64)
    t0 = torch.tensor(data.t0.to_numpy(), dtype=torch.float64)
    d = torch.tensor(data.d.to_numpy(), dtype=torch.float64)
    strata = torch.tensor(data.s.to_numpy(), dtype=torch.int64)
    groups = torch.tensor(pd.factorize(data.g)[0], dtype=torch.int64)
    x = torch.tensor(data[["x1", "x2"]].to_numpy(), dtype=torch.float64)
    x = x - x.mean(0)
    risk = RiskSets.build(t, t0, d, strata)
    q = int(groups.max()) + 1
    cox = FrailtyCox(x, groups, q, risk, None, ties)
    point = torch.cat([torch.tensor([0.3, -0.2], dtype=torch.float64),
                       0.4 * torch.sin(torch.arange(q, dtype=torch.float64))])
    for objective in (cox, PenalizedFrailty(cox, distribution, 0.6)):
        report = check_derivatives(objective, point)
        assert report["gradient_max_rel_error"] < 1e-7
        assert report["hessian_max_rel_error"] < 1e-6


def test_missing_rows_and_row_order():
    data = frame("cox_gamma")
    terms = R["cox_gamma"]["terms"]
    base = oe.stcox_frailty(data=data, time="t", failure="d", x=terms, shared="g")
    shuffled = data.sample(frac=1.0, random_state=3).reset_index(drop=True)
    again = oe.stcox_frailty(data=shuffled, time="t", failure="d", x=terms, shared="g")
    assert_allclose(estimates(again), estimates(base), atol=1e-9)
    assert again.metrics["theta"] == pytest.approx(base.metrics["theta"], rel=1e-8)
    holed = pd.concat([data, data.iloc[:1].assign(x1=np.nan)], ignore_index=True)
    with pytest.raises(AnalysisError) as error:
        oe.stcox_frailty(data=holed, time="t", failure="d", x=terms, shared="g")
    assert error.value.code == "missing_values"
    dropped = oe.stcox_frailty(data=holed, time="t", failure="d", x=terms, shared="g",
                               missing="drop")
    assert dropped.dropped_rows == 1 and dropped.nobs == len(data)
    assert_allclose(estimates(dropped), estimates(base), atol=1e-12)


def test_failure_contract():
    data = frame("cox_gamma")

    def call(**options):
        arguments = dict(data=data, time="t", failure="d", x=["x1"], shared="g")
        arguments.update(options)
        with pytest.raises(AnalysisError) as error:
            oe.stcox_frailty(**arguments)
        return error.value.code

    assert call(frailty="weibull") == "invalid_spec"
    assert call(ties="exactp") == "invalid_spec"
    assert call(theta=0.0) == "invalid_option"
    assert call(theta=-1.0) == "invalid_spec"
    assert call(data=data.assign(g=1)) == "invalid_groups"
    assert call(data=data.assign(d=0)) == "no_failures"
    assert call(data=data.assign(d=2)) == "invalid_failure_indicator"
    assert call(data=data.assign(t=-data.t)) == "invalid_survival_time"
    assert call(shared="missing_column") == "missing_columns"
    with pytest.raises(TypeError):
        oe.stcox_frailty(data=data, time="t", failure="d", x=["x1"], shared="g",
                         covariance="robust")
    with pytest.raises(TypeError):
        oe.stcox_frailty(data=data, time="t", failure="d", x=["x1"], shared="g", weights="x2")


def test_json_round_trip_and_rendering():
    result = fit("cox_entry", "efron", "lognormal")
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    assert restored.coefficients == result.coefficients
    assert restored.covariance_matrix == result.covariance_matrix
    assert restored.tests == result.tests and restored.extra == result.extra
    assert restored.metrics == result.metrics
    assert restored.provenance["stata_parity_validated"] is False
    text = result.summary()
    assert "shared lognormal frailty" in text and "chibar2" in text
    assert "\\begin{tabular" in result.summary("latex")
    assert math.isfinite(result.metrics["theta"]) and result.metrics["n_groups"] == 30


# ---- parametric shared frailty -----------------------------------------------------------

PARAMETRIC = json.loads((FIXTURES / "data_parametric.json").read_text())["datasets"]
NUMPY = json.loads((FIXTURES / "numpy_reference.json").read_text())
PARAMETRIC_CASES = [(data, frailty, distribution)
                    for data, frailty in (("param_gamma", "gamma"), ("param_invgauss", "invgaussian"))
                    for distribution in ("weibull", "exponential")]


def parametric(name, frailty, distribution, **options):
    return oe.streg_frailty(data=pd.DataFrame(PARAMETRIC[name]), time="t", failure="d",
                            x=["x1", "x2"], shared="g", frailty=frailty,
                            distribution=distribution, **options)


@pytest.mark.parametrize(("name", "frailty", "distribution"), PARAMETRIC_CASES)
def test_parametric_frailty_matches_numerical_integration(name, frailty, distribution):
    ref = NUMPY["parametric"][f"{name}_{distribution}_ph"]
    result = parametric(name, frailty, distribution)
    terms = ["Intercept", "x1", "x2"] + (["/ln_p"] if distribution == "weibull" else []) + ["/ln_the"]
    assert [c.term for c in result.coefficients] == terms
    assert_allclose(estimates(result), ref["params"], rtol=0, atol=2e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(ref["log_likelihood"], abs=1e-8)
    assert result.metrics["log_likelihood"] >= ref["log_likelihood"] - 1e-9
    assert_allclose(covariance(result), ref["covariance"], rtol=2e-4, atol=1e-7)
    assert result.metrics["log_likelihood_no_frailty"] == pytest.approx(ref["null_log_likelihood"],
                                                                        abs=1e-8)
    statistic = 2 * (ref["log_likelihood"] - ref["null_log_likelihood"])
    assert result.tests["theta"]["statistic"] == pytest.approx(statistic, abs=1e-7)
    assert result.tests["theta"]["p_value"] == pytest.approx(0.5 * stats.chi2.sf(statistic, 1),
                                                             rel=1e-8)
    theta, se = math.exp(ref["params"][-1]), math.sqrt(ref["covariance"][-1][-1])
    assert result.metrics["theta"] == pytest.approx(theta, rel=2e-6)
    low = theta * math.exp(-stats.norm.isf(0.025) * se)
    assert result.extra["theta"]["ci_low"] == pytest.approx(low, rel=1e-4)


@pytest.mark.parametrize(("name", "frailty", "distribution"), PARAMETRIC_CASES)
def test_parametric_likelihood_equals_quadrature(name, frailty, distribution):
    from openecon.econometrics.survival.parametric_frailty import FrailtyLikelihood

    data = pd.DataFrame(PARAMETRIC[name])
    t = torch.tensor(data.t.to_numpy(), dtype=torch.float64)
    d = torch.tensor(data.d.to_numpy(), dtype=torch.float64)
    g = torch.tensor(pd.factorize(data.g)[0], dtype=torch.int64)
    x = torch.tensor(np.column_stack([np.ones(len(data)), data[["x1", "x2"]].to_numpy()]),
                     dtype=torch.float64)
    q = int(g.max()) + 1
    model = FrailtyLikelihood(distribution, "ph", x, t, d, g, q, None, frailty)
    shift = float((d * torch.log(t)).sum())
    for check in NUMPY["parametric"][f"{name}_{distribution}_ph"]["evaluations"]:
        point = torch.tensor(check["params"], dtype=torch.float64)
        assert float(model.value(point)) + shift == pytest.approx(check["log_likelihood"], abs=1e-9)
        for metric in ("ph", "aft"):
            other = FrailtyLikelihood(distribution, metric, x, t, d, g, q, 0.1 * x[:, 1], frailty)
            report = check_derivatives(other, point)
            assert report["gradient_max_rel_error"] < 1e-8
            assert report["hessian_max_rel_error"] < 1e-7


def test_parametric_metrics_and_no_frailty_model():
    ph = parametric("param_gamma", "gamma", "weibull")
    aft = parametric("param_gamma", "gamma", "weibull", metric="aft")
    p = math.exp(estimates(ph)[3])
    assert aft.metrics["log_likelihood"] == pytest.approx(ph.metrics["log_likelihood"], abs=1e-9)
    assert_allclose(estimates(aft)[:3], -estimates(ph)[:3] / p, atol=1e-7)
    assert aft.metrics["theta"] == pytest.approx(ph.metrics["theta"], rel=1e-7)
    assert ph.extra["hazard_ratios"]["x1"]["ratio"] == pytest.approx(math.exp(estimates(ph)[1]))
    assert "time_ratios" in aft.extra
    streg = oe.streg(data=pd.DataFrame(PARAMETRIC["param_gamma"]), time="t", failure="d",
                     x=["x1", "x2"])
    assert ph.metrics["log_likelihood_no_frailty"] == pytest.approx(
        streg.metrics["log_likelihood"], abs=1e-9)


def test_parametric_boundary_is_the_model_without_frailty():
    data = frame("cox_null")
    for frailty in ("gamma", "invgaussian"):
        for distribution in ("exponential", "weibull"):
            result = oe.streg_frailty(data=data, time="t", failure="d", x=["x1"], shared="g",
                                      frailty=frailty, distribution=distribution)
            streg = oe.streg(data=data, time="t", failure="d", x=["x1"],
                             distribution=distribution)
            assert result.extra["frailty"]["score_at_zero"] < 0
            assert result.metrics["theta"] == 0.0 and result.extra["frailty"]["boundary"]
            assert result.tests["theta"]["statistic"] == 0 and result.tests["theta"]["p_value"] == 1
            assert "/ln_the" not in [c.term for c in result.coefficients]
            assert_allclose(estimates(result), estimates(streg), atol=1e-8)
            assert_allclose(covariance(result), covariance(streg), rtol=1e-7, atol=1e-12)
            assert result.metrics["log_likelihood"] == pytest.approx(
                streg.metrics["log_likelihood"], abs=1e-9)


def test_parametric_failure_contract_and_round_trip():
    from openecon.econometrics.core import make_spec

    data = pd.DataFrame(PARAMETRIC["param_invgauss"])

    def code(**options):
        arguments = dict(data=data, time="t", failure="d", x=["x1"], shared="g")
        arguments.update(options)
        with pytest.raises(AnalysisError) as error:
            oe.streg_frailty(**arguments)
        return error.value.code

    assert code(distribution="gompertz") == "invalid_spec"
    assert code(frailty="lognormal") == "invalid_spec"
    assert code(data=data.assign(g=0)) == "invalid_groups"
    assert code(data=data.assign(d=0)) == "no_failures"
    assert code(data=data.assign(d=3)) == "invalid_failure_indicator"
    with pytest.raises(AnalysisError) as error:          # no delayed entry in this model
        make_spec("streg_frailty", outcome="t", predictors=["x1"], covariance="nonrobust",
                  columns={"failure": "d", "shared": "g", "entry": "x2"})
    assert error.value.code == "invalid_spec"
    result = parametric("param_invgauss", "invgaussian", "weibull")
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    assert restored.coefficients == result.coefficients and restored.extra == result.extra
    assert "invgaussian frailty" in result.summary() and "chibar2" in result.summary()
    assert "\\begin{tabular" in result.summary("latex")


# ---- survival treatment effects ---------------------------------------------------------

EFFECTS = pd.DataFrame(json.loads((FIXTURES / "data_stteffects.json").read_text())
                       ["datasets"]["stteffects"]).assign(cl=lambda df: np.arange(len(df)) // 15)
EFFECT_CASES = {f"{m}_{e}": (m, e, {}) for m in ("ra", "ipw", "ipwra") for e in ("ate", "atet")}
EFFECT_CASES["ipwra_ate_cluster15"] = ("ipwra", "ate", {"covariance": "cluster", "cluster": "cl"})
EFFECT_CASES["ipw_ate_cx_x2"] = ("ipw", "ate", {"cx": ["x2"]})


def effects(method, estimand, data=EFFECTS, **options):
    return oe.stteffects(data=data, time="t", failure="d", treatment="treat", x=["x1", "x2"],
                         method=method, estimand=estimand, **options)


@pytest.mark.parametrize("case", list(EFFECT_CASES))
def test_stteffects_matches_explicit_gmm(case):
    method, estimand, options = EFFECT_CASES[case]
    ref = NUMPY["stteffects"][case]
    result = effects(method, estimand, **options)
    pom = np.array(ref["pom"])
    pom_cov = np.array(ref["pom_covariance"])
    stored = np.array(list(result.extra["potential_outcome_means"].values()))
    assert_allclose(stored, pom, rtol=0, atol=1e-8)
    assert_allclose(result.extra["potential_outcome_covariance"], pom_cov, rtol=1e-7)
    transform = np.array([[-1.0, 1.0], [1.0, 0.0]])
    assert_allclose(estimates(result), transform @ pom, atol=1e-8)
    assert_allclose(covariance(result), transform @ pom_cov @ transform.T, rtol=1e-7)
    name = "ATET" if estimand == "atet" else "ATE"
    assert [c.term for c in result.coefficients] == [f"{name}:r1vs0.treat", "POmean:0.treat"]
    assert result.inference["small_sample_correction"] == 1.0
    if "cluster" in options:
        assert result.inference["cluster_count"] == 40


def test_stteffects_pomeans_control_and_auxiliary_equations():
    ate = effects("ipwra", "ate")
    pomeans = effects("ipwra", "pomeans")
    assert [c.term for c in pomeans.coefficients] == ["POmeans:0.treat", "POmeans:1.treat"]
    assert_allclose(estimates(pomeans), list(ate.extra["potential_outcome_means"].values()),
                    atol=1e-12)
    assert_allclose(covariance(pomeans), ate.extra["potential_outcome_covariance"], rtol=1e-12)
    flipped = effects("ipwra", "ate", control=1)
    assert estimates(flipped)[0] == pytest.approx(-estimates(ate)[0], abs=1e-10)
    assert flipped.coefficients[0].std_error == pytest.approx(ate.coefficients[0].std_error,
                                                              rel=1e-9)
    tables = ate.extra["auxiliary_equations"]
    assert set(tables) == {"TME1", "OME0", "OME1"}
    assert [row["term"] for row in tables["OME1"]] == ["Intercept", "x1", "x2", "/ln_p"]
    ipw = effects("ipw", "ate")
    assert set(ipw.extra["auxiliary_equations"]) == {"TME1", "CME0", "CME1"}
    ra = effects("ra", "ate")
    # The RA outcome equations are the per-arm Weibull ML fits of oe.streg.
    for level in (0, 1):
        streg = oe.streg(data=EFFECTS[EFFECTS.treat == level], time="t", failure="d",
                         x=["x1", "x2"])
        table = ra.extra["auxiliary_equations"][f"OME{level}"]
        assert_allclose([row["estimate"] for row in table], estimates(streg), atol=1e-7)


def test_stteffects_failure_contract_and_round_trip():
    def code(data=EFFECTS, **options):
        arguments = dict(data=data, time="t", failure="d", treatment="treat", x=["x1", "x2"])
        arguments.update(options)
        with pytest.raises(AnalysisError) as error:
            oe.stteffects(**arguments)
        return error.value.code

    assert code(data=EFFECTS.assign(treat=EFFECTS.index % 3)) == "invalid_treatment"
    assert code(tx=["x1"]) == "invalid_spec"
    assert code(cx=["x1"], method="ipwra") == "invalid_spec"
    assert code(x=["x1", "treat"]) == "invalid_spec"
    assert code(method="ipw", pstolerance=0.2) == "overlap_violation"
    assert code(method="ipw", data=EFFECTS.assign(d=1)) == "insufficient_censoring"
    assert code(method="aipw") == "invalid_spec"
    assert code(covariance="nonrobust") == "invalid_spec"
    assert code(data=EFFECTS.assign(d=0)) == "no_failures"
    with pytest.raises(TypeError):
        oe.stteffects(data=EFFECTS, time="t", failure="d", treatment="treat", weights="x1")
    result = effects("ipw", "atet")
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    assert restored.coefficients == result.coefficients
    assert restored.covariance_matrix == result.covariance_matrix
    assert restored.extra == result.extra
    assert "ATET:r1vs0.treat" in result.summary()
    assert "\\begin{tabular" in result.summary("latex")
