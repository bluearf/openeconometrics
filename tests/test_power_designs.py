"""Independent probability laws, published fixtures and design inversion gates.

SciPy is a development oracle, never a runtime dependency. The design tests
derive df/noncentrality/variance independently from the declared experiment.
"""
import json
import math

import numpy as np
import pytest
from scipy import integrate, stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.stats.power_distributions import chi_power, f_power, t_power


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def plan(result):
    return result["plan"].iloc[0]


def t_oracle(delta, df, alpha, alternative):
    cut = stats.t.isf(alpha / (2 if alternative == "two-sided" else 1), df)
    if alternative == "two-sided":
        return stats.nct.sf(cut, df, delta) + stats.nct.cdf(-cut, df, delta)
    return stats.nct.cdf(-cut, df, delta) if alternative == "lower" else stats.nct.sf(cut, df, delta)


@pytest.mark.parametrize("df", [1, 2, 5, 30, 1000, 19999])
@pytest.mark.parametrize("alpha,delta", [(1e-8, 2.), (.001, -20.), (.05, 0.), (.49, 64.)])
def test_t_probability_law_grid(df, alpha, delta):
    for alternative in ("upper", "lower", "two-sided"):
        assert t_power(delta, df, alpha, alternative) == pytest.approx(
            t_oracle(delta, df, alpha, alternative), abs=2e-10)


def test_t_independent_conditional_density_integral():
    df, delta, alpha = 7, 2.8, .03
    cut = stats.t.isf(alpha / 2, df)
    def integrand(v):
        boundary = cut * np.sqrt(v / df)
        return (stats.norm.sf(boundary - delta) + stats.norm.cdf(-boundary - delta)) * stats.chi2.pdf(v, df)
    expected = integrate.quad(integrand, 0, np.inf, epsabs=2e-12)[0]
    assert t_power(delta, df, alpha, "two-sided") == pytest.approx(expected, abs=3e-11)


@pytest.mark.parametrize("nc", [0., .001, 1., 40., 1000., 4096.])
@pytest.mark.parametrize("dfs", [(1, 1), (4, 15), (29, 100), (100, 10000)])
def test_noncentral_f_chi_laws(nc, dfs):
    df1, df2 = dfs
    cut = stats.f.isf(.01, df1, df2)
    expected = stats.f.sf(cut, df1, df2) if nc == 0 else stats.ncf.sf(cut, df1, df2, nc)
    assert f_power(cut, df1, df2, nc) == pytest.approx(expected, abs=2e-10)
    cut = stats.chi2.isf(.01, df1)
    assert chi_power(cut, df1, nc) == pytest.approx(stats.ncx2.sf(cut, df1, nc), abs=2e-10)


@pytest.mark.parametrize("alternative", ["two-sided", "upper", "lower"])
@pytest.mark.parametrize("name,kwargs", [
    ("power_tmean", dict(sd=1.3)),
    ("power_ttwomeans", dict(sd=1.3, ratio=.37)),
    ("power_ttwomeans", dict(sd=1.3, ratio=2.3)),
    ("power_tpaired", dict(sd1=1.2, sd2=1.8, correlation=.7)),
])
def test_t_design_law_minimum_size_and_signed_mde(alternative, name, kwargs):
    fn = getattr(oe, name)
    effect = -.7 if alternative == "lower" else .7
    n, alpha = 31, .013
    output = plan(fn(effect, n=n, alpha=alpha, alternative=alternative, **kwargs))
    if name == "power_tpaired":
        variance = kwargs["sd1"]**2 + kwargs["sd2"]**2 - 2 * kwargs["correlation"] * kwargs["sd1"] * kwargs["sd2"]
        se, df = math.sqrt(variance / n), n - 1
    elif name == "power_ttwomeans":
        n2 = math.ceil(kwargs["ratio"] * n)
        se, df = kwargs["sd"] * math.sqrt(1 / n + 1 / n2), n + n2 - 2
        assert output.n2 == n2
    else:
        se, df = kwargs["sd"] / math.sqrt(n), n - 1
    assert output.df == df
    assert output.design_standard_error == pytest.approx(se)
    assert output.power == pytest.approx(t_oracle(effect / se, df, alpha, alternative), abs=2e-10)
    sized = plan(fn(effect, power=.81, alpha=alpha, alternative=alternative, **kwargs))
    assert sized.power >= .81
    assert plan(fn(effect, n=int(sized.n)-1, alpha=alpha, alternative=alternative, **kwargs)).power < .81
    detectable = plan(fn(n=n, power=.81, alpha=alpha, alternative=alternative, **kwargs))
    assert detectable.power == pytest.approx(.81, abs=1e-9)
    assert detectable.effect * effect > 0


def test_published_primary_examples():
    # Stata 15 manuals: onemean examples 1/4; oneway example 1; rsquared example 1.
    assert plan(oe.power_tmean(25, sd=40, power=.8)).n == 23
    assert plan(oe.power_tmean(25, sd=40, n=30)).power == pytest.approx(.9112, abs=5e-5)
    means = np.array([260., 289., 295.])
    effect = np.sqrt(np.mean((means - means.mean())**2) / 4900)
    assert plan(oe.power_anova(effect, groups=3, power=.8)).n == 69
    assert plan(oe.power_regression(.1/.9, predictors=5, tested=5, power=.8)).n == 122


@pytest.mark.parametrize("name,kwargs,effect", [
    ("power_anova", dict(groups=3), .25),
    ("power_anova", dict(groups=8), .4),
    ("power_regression", dict(predictors=5, tested=2), .15),
    ("power_regression", dict(predictors=10, tested=10), .3),
])
def test_f_design_independent_df_nc_and_inversions(name, kwargs, effect):
    fn = getattr(oe, name)
    output = plan(fn(effect, n=43, **kwargs))
    if name == "power_anova":
        k = kwargs["groups"]
        df1, df2, nc = k - 1, k * 42, k * 43 * effect**2
    else:
        df1, df2, nc = kwargs["tested"], 43-kwargs["predictors"]-1, 43*effect
    expected = stats.ncf.sf(stats.f.isf(.05, df1, df2), df1, df2, nc)
    assert output.power == pytest.approx(expected, abs=2e-10)
    assert (output.df1, output.df2, output.noncentrality) == pytest.approx((df1, df2, nc))
    size = plan(fn(effect, power=.85, **kwargs))
    assert size.power >= .85
    assert plan(fn(effect, n=int(size.n)-1, **kwargs)).power < .85
    mde = plan(fn(n=43, power=.85, **kwargs))
    assert mde.power == pytest.approx(.85, abs=1e-9)
    assert plan(fn(n=200, power=.8, **kwargs)).power == pytest.approx(.8, abs=1e-9)


@pytest.mark.parametrize("icc,ratio", [(0., 1.), (.1, .37), (.9, 2.3), (1., 1.)])
def test_cluster_variance_from_full_exchangeable_covariance(icc, ratio):
    m, k, sd = 17, 31, 1.4
    covariance = sd**2 * (np.eye(m) * (1 - icc) + np.ones((m, m)) * icc)
    cluster_mean_variance = covariance.sum() / m**2
    se = np.sqrt(cluster_mean_variance * (1/k + 1/math.ceil(ratio*k)))
    result = oe.power_cluster_mean(.6, sd=sd, cluster_size=m, icc=icc, ratio=ratio, n=k)
    output = plan(result)
    cut = stats.norm.isf(.025)
    expected = stats.norm.sf(cut, loc=.6/se) + stats.norm.cdf(-cut, loc=.6/se)
    assert output.design_standard_error == pytest.approx(se)
    assert output.power == pytest.approx(expected, abs=2e-13)
    assert output.total_n == m*(k+math.ceil(ratio*k))
    sized = plan(oe.power_cluster_mean(.6, sd=sd, cluster_size=m, icc=icc, ratio=ratio, power=.8))
    assert sized.power >= .8
    if int(sized.n)>max(2, math.floor(1/ratio)+1):
        assert plan(oe.power_cluster_mean(.6, sd=sd, cluster_size=m, icc=icc, ratio=ratio, n=int(sized.n)-1)).power < .8
    mde = plan(oe.power_cluster_mean(sd=sd, cluster_size=m, icc=icc, ratio=ratio, n=k, power=.8, direction="lower"))
    assert mde.effect < 0 and mde.power == pytest.approx(.8, abs=1e-9)
    assert "approximation" in result.attrs["inference"]


@pytest.mark.parametrize("name,kwargs", [
    ("power_gof", dict(null_probabilities=[.25]*4, proposed_probabilities=[.4,.3,.2,.1])),
    ("power_independence", dict(joint_probabilities=[[.35,.15],[.15,.35]])),
    ("power_independence", dict(joint_probabilities=[[.2,.1,.1],[.1,.3,.2]])),
])
def test_pearson_probability_geometry_and_strength_inversion(name, kwargs):
    fn = getattr(oe,name)
    result = fn(n=200, strength=.7, **kwargs)
    if name == "power_gof":
        null, proposed = np.array(kwargs["null_probabilities"]), np.array(kwargs["proposed_probabilities"])
        df = len(null)-1
    else:
        joint = np.array(kwargs["joint_probabilities"])
        null = np.outer(joint.sum(axis=1), joint.sum(axis=0))
        proposed, df = joint, (joint.shape[0]-1)*(joint.shape[1]-1)
        planned = np.array(result.attrs["planned_probabilities"]).reshape(joint.shape)
        np.testing.assert_allclose(planned.sum(axis=0), joint.sum(axis=0), atol=1e-14)
        np.testing.assert_allclose(planned.sum(axis=1), joint.sum(axis=1), atol=1e-14)
    nc = 200 * np.sum((.7*(proposed-null))**2 / null)
    assert plan(result).power == pytest.approx(stats.ncx2.sf(stats.chi2.isf(.05,df),df,nc), abs=2e-10)
    size = plan(fn(power=.8, **kwargs))
    assert size.power >= .8
    minimum = math.ceil(5/null.min())
    if size.n > minimum:
        assert plan(fn(n=int(size.n)-1, **kwargs)).power < .8
    mde = plan(fn(n=200, strength=None, power=.8, **kwargs))
    assert 0 < mde.strength <= 1 and mde.power == pytest.approx(.8, abs=1e-9)
    assert "asymptotic" in result.attrs["inference"]


def test_null_size_not_observed_power_and_complete_settings():
    examples = [oe.power_tmean(0, sd=1., n=20), oe.power_anova(0,groups=3,n=20),
                oe.power_regression(0,predictors=3,tested=2,n=20),
                oe.power_gof([.5,.5],[.5,.5],n=20)]
    for result in examples:
        assert plan(result).power == pytest.approx(.05, abs=2e-10)
        settings = {k: json.loads(v) for k,v in result["settings"].itertuples(index=False,name=None)}
        assert settings == result.attrs
        assert settings["prospective"] and not settings["observations_used"]
        assert "\\begin{tabular}" in result.to_latex()


@pytest.mark.parametrize("fn,kwargs", [
    (oe.power_tmean,dict(effect=.5,sd=1,n=1)),
    (oe.power_tmean,dict(effect=.5,sd=1,n=True)),
    (oe.power_tmean,dict(effect=.5,sd=float("nan"),n=30)),
    (oe.power_ttwomeans,dict(effect=.5,sd=1,n=10,ratio=0)),
    (oe.power_tpaired,dict(effect=.5,sd1=1,sd2=1,correlation=1,n=30)),
    (oe.power_tpaired,dict(effect=.5,sd1=1,sd2=1,correlation=1.1,n=30)),
    (oe.power_anova,dict(effect=-1,groups=3,n=20)),
    (oe.power_anova,dict(effect=.5,groups=1,n=20)),
    (oe.power_regression,dict(effect=.15,predictors=3,tested=4,n=20)),
    (oe.power_regression,dict(effect=.15,predictors=3,tested=2,n=4)),
    (oe.power_cluster_mean,dict(effect=.5,sd=1,cluster_size=20,icc=-.1,n=20)),
    (oe.power_gof,dict(null_probabilities=[0,1],proposed_probabilities=[.1,.9],n=30)),
    (oe.power_gof,dict(null_probabilities=[.5,.5],proposed_probabilities=[.4,.7],n=30)),
    (oe.power_gof,dict(null_probabilities=[.5,.5],proposed_probabilities=[.4,.6],n=2)),
    (oe.power_independence,dict(joint_probabilities=[[.4,.1],[.5]],n=200)),
    (oe.power_independence,dict(joint_probabilities=[[.5,0],[0,.5]],n=200)),
])
def test_invalid_missing_and_design_refusals(fn,kwargs):
    with pytest.raises(AnalysisError):
        fn(**kwargs)


def test_limits_no_fallback_and_scenario_boundary_reported():
    with pytest.raises(AnalysisError,match="Noncentrality"):
        oe.power_tmean(100,sd=1,n=30)
    with pytest.raises(AnalysisError,match="limit"):
        oe.power_tmean(.5,sd=1,n=30000)
    with pytest.raises(AnalysisError,match="equal the null"):
        oe.power_gof([.5,.5],[.5,.5],power=.8)
    with pytest.raises(AnalysisError,match="unattainable"):
        oe.power_gof([.5,.5],[.49,.51],n=20,strength=None,power=.8)
    output = oe.power_regression(40.96,predictors=5,tested=2,n=100)
    assert len(output.attrs["omitted_adjacent_scenarios"]) == 1
    assert output.attrs["omitted_adjacent_scenarios"][0]["n"] == 101
    assert len(output["scenarios"]) == 2
    with torch.device("meta"):
        output = oe.power_tmean(.5,sd=1.,n=30)
        assert output.attrs["device"] == "cpu"


def test_unsupported_observation_device_and_welch_options_not_accepted():
    with pytest.raises(TypeError):
        oe.power_ttwomeans(.5,sd=1.,sd2=2.,n=30)
    with pytest.raises(TypeError):
        oe.power_tmean(.5,sd=1.,n=30,device="cuda")
    with pytest.raises(AnalysisError):
        oe.power_tmean([1,2],sd=1.,n=30)
