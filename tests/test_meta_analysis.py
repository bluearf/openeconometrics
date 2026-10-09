"""Author BCG fixture and independent NumPy QR/SciPy development oracles."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, special, stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.meta.common import checksum
from openecon.econometrics.meta import kernels

FIXTURE = Path(__file__).parent / "fixtures/meta/bcg.csv"


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def bcg():
    d = pd.read_csv(FIXTURE)
    e = oe.meta_effectsize(data=d, study="trial", measure="RR", columns=dict(a="tpos", b="tneg", c="cpos", d="cneg"))
    for name in ("ablat", "year", "alloc"):
        e[name] = d[name]
    e["sequence"] = d.trial
    return e


def oracle(data, names=(), method="REML", inference="z", intercept=True):
    y, v = np.asarray(data.yi), np.asarray(data.vi)
    x = np.column_stack([*([np.ones(len(y))] if intercept else []), *[np.asarray(data[n]) for n in names]])
    k, p = x.shape
    df = k-p

    def at(tau):
        w = 1/(v+tau)
        q, r = np.linalg.qr(x*np.sqrt(w[:, None]))
        beta = np.linalg.solve(r, q.T@(y*np.sqrt(w)))
        residual = y-x@beta
        ri = np.linalg.solve(r, np.eye(p))
        cv = ri@ri.T
        rss = np.dot(w, residual**2)
        objective = np.log(v+tau).sum()+rss
        if method == "REML":
            objective += 2*np.log(np.abs(np.diag(r))).sum()
        return objective, beta, cv, w, rss

    fixed = at(0)
    projection_trace = fixed[3].sum()-np.trace(fixed[2]@(x.T@(fixed[3][:, None]**2*x)))
    dl = max(0, (fixed[4]-df)/projection_trace)
    if method == "common":
        tau = 0.
    elif method == "DL":
        tau = dl
    else:
        best = optimize.minimize_scalar(lambda t: at(t)[0], bounds=(0, max(1, np.var(y)*100)), method="bounded", options={"xatol": 1e-12})
        tau = best.x if at(best.x)[0] < fixed[0] else 0.
    _, beta, cv, w, rss = at(tau)
    scale = 1 if inference == "z" else max(1, rss/df) if inference == "modified_hksj" else rss/df
    cv = cv*scale
    se = np.sqrt(cv.diagonal())
    c = stats.norm.isf(.025) if inference == "z" else stats.t.isf(.025, df)
    prob = stats.norm.sf(abs(beta/se))*2 if inference == "z" else stats.t.sf(abs(beta/se), df)*2
    return dict(beta=beta, covariance=cv, weights=w, tau2=tau, q=fixed[4], se=se, p=prob, c=c, df=df, x=x)


@pytest.mark.parametrize("measure", ["MD", "SMD"])
def test_continuous_independent_formula(measure):
    d = pd.DataFrame(dict(m1=[3., 8.], sd1=[2., 1.], n1=[2, 48], m2=[1., 7.], sd2=[1., 3.], n2=[2, 26]))
    original = d.copy(deep=True)
    result = oe.meta_effectsize(data=d, measure=measure, columns={n: n for n in d})
    effect = d.m1-d.m2
    variance = d.sd1**2/d.n1+d.sd2**2/d.n2
    if measure == "SMD":
        df = d.n1+d.n2-2
        j = special.gamma(df/2)/(np.sqrt(df/2)*special.gamma((df-1)/2))
        pooled = ((d.n1-1)*d.sd1**2+(d.n2-1)*d.sd2**2)/df
        effect = j*effect/np.sqrt(pooled)
        variance = 1/d.n1+1/d.n2+effect**2/(2*(d.n1+d.n2))
    np.testing.assert_allclose(result.yi, effect, rtol=1e-12)
    np.testing.assert_allclose(result.vi, variance, rtol=1e-12)
    np.testing.assert_allclose(result.ci_low, effect-stats.norm.isf(.025)*np.sqrt(variance))
    pd.testing.assert_frame_equal(d, original)
    assert result.attrs["n_studies"] == 2 and not result.corrected.any()


@pytest.mark.parametrize("measure", ["OR", "RR"])
def test_binary_zero_policy_and_back_transform(measure):
    d = dict(a=[0, 10], b=[20, 11], c=[4, 5], d=[16, 14])
    with pytest.raises(AnalysisError, match="Zero cells"):
        oe.meta_effectsize(data=d, measure=measure, columns={n: n for n in d})
    result = oe.meta_effectsize(data=d, measure=measure, columns={n: n for n in d}, zero="add_half")
    a, b, c, e = np.array([d[n] for n in d], dtype=float)+np.array([.5, 0])
    yi = np.log(a*e/(b*c)) if measure == "OR" else np.log(a/(a+b))-np.log(c/(c+e))
    vi = 1/a+1/b+1/c+1/e if measure == "OR" else 1/a-1/(a+b)+1/c-1/(c+e)
    np.testing.assert_allclose(result.yi, yi)
    np.testing.assert_allclose(result.vi, vi)
    np.testing.assert_allclose(result.estimate_original, np.exp(yi))
    assert result.corrected.tolist() == [True, False]
    for cells in (dict(a=[0], b=[2], c=[0], d=[3]), dict(a=[2], b=[0], c=[3], d=[0])):
        with pytest.raises(AnalysisError) as error:
            oe.meta_effectsize(data=cells, measure=measure, columns={n: n for n in cells}, zero="add_half")
        assert error.value.code == "uninformative_study"


def test_fisher_and_prep_guards():
    result = oe.meta_effectsize(data=dict(r=[.2, -.5], n=[10, 50]), measure="ZCOR", columns=dict(r="r", n="n"))
    np.testing.assert_allclose(result.yi, np.arctanh([.2, -.5]))
    np.testing.assert_allclose(result.vi, 1/(np.array([10, 50])-3))
    np.testing.assert_allclose(result.estimate_original, [.2, -.5])
    for d in (dict(r=[1.], n=[10]), dict(r=[.2], n=[3]), dict(r=[.2], n=[10.5]), dict(r=[np.nan], n=[10]), dict(r=[True], n=[10])):
        with pytest.raises(AnalysisError):
            oe.meta_effectsize(data=d, measure="ZCOR", columns=dict(r="r", n="n"))
    with pytest.raises(AnalysisError) as error:
        oe.meta_effectsize(data=dict(r=[.2], n=[10]), measure="ZCOR", columns=dict(r="r", n="n"), dependence="paired")
    assert error.value.code == "unsupported_dependence"


def test_author_published_bcg():
    """Published 4-decimal displays use author optimizer's default tolerance.

    5e-4 envelope admits that displayed solver tolerance; the independent
    high-precision QR likelihood oracle below has much tighter gates.
    """
    data = bcg()
    f = oe.meta_pool(data=data, study="study")
    c, h = f["coefficients"].iloc[0], f["heterogeneity"].iloc[0]
    np.testing.assert_allclose([c.estimate, c.std_error, c.ci_low, c.ci_high, h.tau2], [-.7145, .1798, -1.0669, -.3622, .3132], atol=5e-5)
    assert abs(h.q0-152.2330) < 5e-5
    assert abs(h.I2_percent-92.22) < .005
    r = oe.meta_regress(data=data, study="study", moderators=["ablat", "year"])
    np.testing.assert_allclose(r["coefficients"].estimate, [-3.5455, -.0280, .0019], atol=5e-4)
    np.testing.assert_allclose(r["coefficients"].std_error, [29.0959, .0102, .0147], atol=5e-4)
    assert abs(r["heterogeneity"].iloc[0].tau2-.1108) < 5e-5
    assert abs(r["tests"].iloc[1].statistic-12.2043) < 5e-4


@pytest.mark.parametrize("method", ["common", "DL", "ML", "REML"])
@pytest.mark.parametrize("inference", ["z", "hksj", "modified_hksj"])
@pytest.mark.parametrize("names", [[], ["ablat", "year"]])
def test_qr_likelihood_full_covariance_inference_prediction(method, inference, names):
    data = bcg()
    before = data.copy(deep=True)
    actual = oe.meta_regress(data=data, study="study", moderators=names, method=method, inference=inference)
    expected = oracle(data, names, method, inference)
    np.testing.assert_allclose(actual["coefficients"].estimate, expected["beta"], rtol=1e-7, atol=2e-7)
    np.testing.assert_allclose(actual["covariance"], expected["covariance"], rtol=2e-7, atol=1e-9)
    np.testing.assert_allclose(actual["coefficients"].std_error, expected["se"], rtol=2e-7)
    np.testing.assert_allclose(actual["coefficients"].p_value, expected["p"], rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(actual["studies"].weight, expected["weights"], rtol=2e-7)
    assert float(actual["heterogeneity"].iloc[0].tau2) == pytest.approx(expected["tau2"], rel=2e-7, abs=2e-9)
    state = json.loads(json.dumps(actual.attrs["prediction_state"], allow_nan=False))
    copied = copy.deepcopy(state)
    new = {"ablat": [30., 50.], "year": [1970., 2000.]} if names else None
    prediction = oe.meta_predict(state, data=new)
    x = np.column_stack([np.ones(2), [30., 50.], [1970., 2000.]]) if names else np.ones((1, 1))
    mean = x@expected["beta"]
    variance = np.einsum("ip,pq,iq->i", x, expected["covariance"], x)
    np.testing.assert_allclose(prediction.estimate, mean, atol=2e-7)
    np.testing.assert_allclose(prediction.std_error, np.sqrt(variance), rtol=2e-6)
    np.testing.assert_allclose(prediction.pi_low, mean-expected["c"]*np.sqrt(variance+expected["tau2"]), atol=3e-7)
    assert state == copied
    pd.testing.assert_frame_equal(data, before)
    assert actual.attrs["variance_solver"]["status"] == "converged"
    assert len(actual.to_latex()) > 1000


@pytest.mark.parametrize("method", ["ML", "REML", "DL", "common"])
def test_boundary_small_variance_scaling_and_no_intercept(method):
    data = pd.DataFrame(dict(yi=[1, 1.001, .999, 1.002, 1.001], vi=[1, .5, .2, 2, .8], x=[1., 2., 3., 5., 7.]))
    f = oe.meta_pool(data=data, method=method)
    assert f["heterogeneity"].iloc[0].tau2 == 0
    scaled = data.copy()
    scaled.yi *= 1e-8
    scaled.vi *= 1e-16
    s = oe.meta_pool(data=scaled, method=method)
    np.testing.assert_allclose(s["coefficients"].estimate, f["coefficients"].estimate*1e-8)
    for intercept in (True, False):
        r = oe.meta_regress(data=data, moderators=["x"], intercept=intercept, method=method)
        expected = oracle(data, ["x"], method, intercept=intercept)
        np.testing.assert_allclose(r["covariance"], expected["covariance"], rtol=1e-7)


def test_input_rank_state_resource_and_failed_convergence():
    for d in (dict(yi=[1, 2], vi=[.1, 0]), dict(yi=[1, 2], vi=[.1, -1]), dict(yi=[1, 2], vi=[.1, np.inf]), dict(yi=[1, np.nan], vi=[.1, .2]), dict(yi=[1, 2], vi=[1e-20, 1e20])):
        with pytest.raises(AnalysisError):
            oe.meta_pool(data=d)
    d = bcg()
    with pytest.raises(AnalysisError):
        oe.meta_pool(data=d, dependence="clustered")
    with pytest.raises(AnalysisError):
        oe.meta_regress(data=d, moderators=["ablat", "ablat"])
    d["duplicate"] = d.ablat*2
    with pytest.raises(AnalysisError) as error:
        oe.meta_regress(data=d, moderators=["ablat", "duplicate"])
    assert error.value.code == "singular_design"
    d["near_duplicate"] = d.ablat+np.linspace(-1, 1, len(d))*1e-7
    with pytest.raises(AnalysisError) as error:
        oe.meta_regress(data=d, moderators=["ablat", "near_duplicate"])
    assert error.value.code == "singular_design"
    with pytest.raises(AnalysisError):
        oe.meta_pool(data=pd.DataFrame(dict(yi=np.ones(2001), vi=np.ones(2001))))
    state = oe.meta_pool(data=d).attrs["prediction_state"]
    for mutation in ("checksum", "covariance", "residual_df", "tau2"):
        bad = copy.deepcopy(state)
        bad[mutation] = "broken"
        with pytest.raises(AnalysisError) as error:
            oe.meta_predict(bad)
        assert error.value.code == "invalid_prediction_state"
    bad = copy.deepcopy(state)
    bad["covariance"] = [[-1.]]
    bad.pop("checksum")
    bad["checksum"] = checksum(bad)
    with pytest.raises(AnalysisError):
        oe.meta_predict(bad)
    original = kernels.fit
    with patch("openecon.econometrics.meta.models.kernels.fit", side_effect=lambda y, v, x, method: original(y, v, x, method, max_iterations=0)):
        with pytest.raises(AnalysisError) as error:
            oe.meta_pool(data=d)
        assert error.value.code == "variance_not_converged"


def test_all_sensitivity_refits_egger_and_saved_plots():
    d = bcg()
    diagnostics = oe.meta_diagnostics(data=d, study="study", group="alloc", order="sequence")
    assert diagnostics.attrs["counts"] == {"leave_one_out": dict(planned=13, completed=13, failed=0), "cumulative": dict(planned=13, completed=12, failed=1), "subgroups": dict(planned=3, completed=3, failed=0)}
    assert diagnostics["cumulative"].iloc[0].status == "failed"
    assert diagnostics["cumulative"].iloc[0].estimate is None
    # Metafor author's printed leave1out BCG estimates, with display tolerance.
    np.testing.assert_allclose(diagnostics["leave_one_out"].estimate.astype(float),
                               [-.7071, -.6540, -.6856, -.6284, -.7642, -.7109, -.6552, -.7948, -.7412, -.6530, -.7579, -.7598, -.7775], atol=1e-4)
    for i, row in enumerate(diagnostics["leave_one_out"].itertuples()):
        expected = oracle(d.drop(index=i))
        assert row.estimate == pytest.approx(expected["beta"][0], abs=2e-8)
        assert row.tau2 == pytest.approx(expected["tau2"], rel=2e-7)
    for j in range(2, 14):
        expected = oracle(d.iloc[:j])
        assert diagnostics["cumulative"].iloc[j-1].estimate == pytest.approx(expected["beta"][0], abs=2e-8)
    for row in diagnostics["subgroups"].itertuples():
        expected = oracle(d.loc[d.alloc == row.study_or_group])
        assert row.estimate == pytest.approx(expected["beta"][0], abs=2e-8)
    # Classical Egger in standardized-effect coordinates: intercept is tested.
    se = np.sqrt(d.vi)
    x = np.column_stack([np.ones(len(d)), 1/se])
    q, r = np.linalg.qr(x)
    beta = np.linalg.solve(r, q.T@(d.yi/se))
    resid = d.yi/se-x@beta
    ri = np.linalg.solve(r, np.eye(2))
    covariance = ri@ri.T*np.dot(resid, resid)/(len(d)-2)
    statistic = beta[0]/np.sqrt(covariance[0, 0])
    egger = diagnostics["egger"].iloc[0]
    assert egger.statistic == pytest.approx(statistic, rel=1e-10)
    assert egger.p_value == pytest.approx(stats.t.sf(abs(statistic), 11)*2, rel=1e-10)
    for kind, count in (("forest", 14), ("funnel", 13)):
        plot = oe.meta_plot(diagnostics, kind=kind)
        assert plot.sample_n == plot.total_n == 13 and len(plot.data) == count
        json.dumps(plot.model_dump(), allow_nan=False)
        assert "tikzpicture" in plot.to_latex()
        assert "OpenEconCharts" in plot.to_html()
        if kind == "funnel":
            assert len(plot.config["options"]["annotations"]) == 2
            assert "segment" in plot.to_latex()
    for frame in diagnostics.values():
        json.dumps(frame.to_dict(orient="split"), allow_nan=False)


def test_diagnostic_failure_denominators_hash_and_order():
    d = bcg().iloc[:4].copy()
    d["g"] = ["one", "two", "two", "three"]
    f = oe.meta_diagnostics(data=d, group="g")
    assert f.attrs["counts"]["subgroups"] == dict(planned=3, completed=1, failed=2)
    assert f["egger"].iloc[0].status == "ineligible"
    changed = d.copy()
    changed.g = ["one", "two", "three", "three"]
    assert oe.meta_diagnostics(data=changed, group="g").attrs["sample_hash"] != f.attrs["sample_hash"]
    tied = d.copy()
    tied.year = 2000
    with pytest.raises(AnalysisError):
        oe.meta_diagnostics(data=tied, order="year")
    d["order"] = [4, 3, 2, 1]
    f = oe.meta_diagnostics(data=d, study="study", order="order")
    assert f.attrs["ordered_studies"] == [4, 3, 2, 1]
    with pytest.raises(AnalysisError):
        oe.meta_diagnostics(data=d.iloc[:2])


def test_segment_validation():
    from openecon_charts import scatter
    p = scatter(data=dict(x=[0, 2], y=[0, 2]), x="x", y="y")
    for line in (dict(type="segment", x0=0, y0=0, x1=np.inf, y1=1), dict(type="segment", x0=0, y0=0, x1=1), dict(type="segment", x0=0, y0=0, x1=1, y1=1, coords="axes")):
        with pytest.raises(ValueError):
            p.with_options(annotations=[line])
    s = p.with_options(annotations=[dict(type="segment", x0=0, y0=0, x1=2, y1=2)])
    assert "segment" in s.to_latex()
    assert not p.config
