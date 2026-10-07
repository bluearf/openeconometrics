"""Independent oracles for oe.xtlogit, oe.xtprobit and oe.xtpoisson (re / fe / pa).

Random-effects logit/probit are the melogit/meprobit likelihood (verified in
test_econ_mixed_glmm.py) with Stata's xt reporting; the gamma random-effects
Poisson model is checked against a brute-force SciPy maximization of the
negative binomial panel likelihood written independently in NumPy; the
conditional fixed-effects models against statsmodels ConditionalLogit /
ConditionalPoisson and Poisson regression with panel dummies; pa against
oe.xtgee.
"""

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln
from statsmodels.discrete.conditional_models import ConditionalLogit, ConditionalPoisson
from statsmodels.tools.numdiff import approx_fprime, approx_hess

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def make_panel(seed=0, panels=120, periods=6):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(panels), periods)
    n = panels * periods
    frame = pd.DataFrame({"id": ids, "t": np.tile(np.arange(periods), panels),
                          "x": rng.normal(size=n), "z": rng.normal(size=panels)[ids]})
    u = rng.normal(size=panels)[ids]
    eta = 0.3 + 0.7 * frame.x + 0.3 * frame.z + u
    frame["y"] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    frame["e"] = rng.uniform(0.5, 2.0, size=n)
    frame["c"] = rng.poisson(frame.e * np.exp(0.2 + 0.5 * frame.x + 0.2 * frame.z)
                             * rng.gamma(2, 0.5, panels)[ids])
    frame["cl"] = ids // 6
    return frame


@pytest.fixture(scope="module")
def panel():
    return make_panel()


def table(result):
    return {c.term: c for c in result.coefficients}


# ---- random effects ---------------------------------------------------------------------


@pytest.mark.parametrize("family", ["logit", "probit"])
def test_random_effects_binary_reporting(panel, family):
    xt = {"logit": oe.xtlogit, "probit": oe.xtprobit}[family]
    me = {"logit": oe.melogit, "probit": oe.meprobit}[family]
    result = xt(data=panel, y="y", x=["x", "z"], panel="id")
    reference = me(data=panel, y="y", x=["x", "z"], group="id", intpoints=12)
    assert_allclose(result.metrics["log_likelihood"], reference.metrics["log_likelihood"],
                    rtol=1e-12)
    terms, ref = table(result), table(reference)
    for name in ("Intercept", "x", "z"):
        assert_allclose(terms[name].estimate, ref[name].estimate, rtol=1e-9)
        assert_allclose(terms[name].std_error, ref[name].std_error, rtol=1e-8)
    variance = ref["/var(_cons[id])"]
    lnsig2u = terms["/lnsig2u"]
    assert_allclose(lnsig2u.estimate, math.log(variance.estimate), rtol=1e-10)
    assert_allclose(lnsig2u.std_error, variance.std_error / variance.estimate, rtol=1e-8)
    c = math.pi ** 2 / 3 if family == "logit" else 1.0
    sigma2 = math.exp(lnsig2u.estimate)
    assert_allclose(result.metrics["sigma_u"], math.sqrt(sigma2))
    assert_allclose(result.metrics["rho"], sigma2 / (sigma2 + c))
    rho = result.extra["rho"]
    r = sigma2 / (sigma2 + c)
    assert_allclose(rho["std_error"], r * (1 - r) * lnsig2u.std_error, rtol=1e-10)
    z = stats.norm.ppf(0.975)
    low = math.exp(lnsig2u.estimate - z * lnsig2u.std_error)
    assert_allclose(rho["ci_low"], low / (low + c), rtol=1e-10)
    assert_allclose(result.extra["sigma_u"]["ci_low"], math.sqrt(low), rtol=1e-10)
    pooled = (sm.Logit if family == "logit" else sm.Probit)(
        panel.y, sm.add_constant(panel[["x", "z"]])).fit(disp=0)
    lr = result.tests["rho"]
    assert_allclose(lr["statistic"], 2 * (result.metrics["log_likelihood"] - pooled.llf),
                    rtol=1e-9)
    assert lr["distribution"] == "chibar2"


def negative_binomial_panel(params, frame, y="c", offset=None):
    X = np.column_stack([np.ones(len(frame)), frame[["x", "z"]].to_numpy()])
    eta = X @ params[:3] + (0 if offset is None else offset)
    lam = np.exp(eta)
    a = math.exp(-params[3])
    data = pd.DataFrame({"id": frame.id, "lam": lam, "y": frame[y], "ye": frame[y] * eta,
                         "lf": gammaln(frame[y] + 1)})
    sums = data.groupby("id").sum()
    return (sums.ye - sums.lf + gammaln(sums.y + a) - gammaln(a) + a * math.log(a)
            - (sums.y + a) * np.log(sums.lam + a))


def test_gamma_random_effects_poisson_brute_force(panel):
    result = oe.xtpoisson(data=panel, y="c", x=["x", "z"], panel="id", exposure="e")
    offset = np.log(panel.e.to_numpy())

    def negative(params):
        return -negative_binomial_panel(params, panel, offset=offset).sum()

    best = minimize(negative, np.array([0.0, 0.0, 0.0, 0.0]), method="BFGS",
                    options={"gtol": 1e-9})
    terms = table(result)
    assert_allclose(result.metrics["log_likelihood"], -best.fun, rtol=1e-10)
    got = [terms[n].estimate for n in ("Intercept", "x", "z", "/lnalpha")]
    assert_allclose(got, best.x, rtol=1e-5, atol=1e-6)
    hessian = approx_hess(np.array(got), lambda p: -negative(p))
    se = np.sqrt(np.diag(np.linalg.inv(-hessian)))
    assert_allclose([terms[n].std_error for n in ("Intercept", "x", "z", "/lnalpha")], se,
                    rtol=1e-5)
    assert_allclose(result.metrics["alpha"], math.exp(terms["/lnalpha"].estimate))
    pooled = sm.Poisson(panel.c, sm.add_constant(panel[["x", "z"]]), offset=offset).fit(disp=0)
    assert_allclose(result.tests["alpha"]["statistic"],
                    2 * (result.metrics["log_likelihood"] - pooled.llf), rtol=1e-9)
    robust = oe.xtpoisson(data=panel, y="c", x=["x", "z"], panel="id", exposure="e",
                          covariance="robust")
    scores = approx_fprime(np.array(got), lambda p: negative_binomial_panel(
        p, panel, offset=offset).to_numpy(), centered=True)
    bread = np.linalg.inv(-hessian)
    groups = scores.shape[0]
    expected = groups / (groups - 1) * bread @ scores.T @ scores @ bread
    assert_allclose(np.array(robust.covariance_matrix), expected, rtol=1e-4)


def test_normal_random_effects_poisson(panel):
    result = oe.xtpoisson(data=panel, y="c", x=["x", "z"], panel="id", normal=True,
                          intpoints=20)
    reference = oe.mepoisson(data=panel, y="c", x=["x", "z"], group="id", intpoints=20)
    assert_allclose(result.metrics["log_likelihood"], reference.metrics["log_likelihood"],
                    rtol=1e-12)
    assert "/lnsig2u" in table(result) and "sigma_u" in result.tests


# ---- conditional fixed effects ------------------------------------------------------------


def test_conditional_logit_matches_statsmodels(panel):
    result = oe.xtlogit(data=panel, y="y", x=["x", "z"], panel="id", model="fe")
    assert list(table(result)) == ["x"]                 # z is constant within panels
    totals = panel.groupby("id").y.agg(["sum", "size"])
    informative = totals[(totals["sum"] > 0) & (totals["sum"] < totals["size"])].index
    used = panel[panel.id.isin(informative)]
    ref = ConditionalLogit(used.y, used[["x"]], groups=used.id).fit(disp=0, method="newton",
                                                                     maxiter=100)
    assert_allclose(table(result)["x"].estimate, ref.params["x"], rtol=1e-6)
    assert_allclose(table(result)["x"].std_error, ref.bse["x"], rtol=1e-6)
    assert result.metrics["n_groups_dropped"] == panel.id.nunique() - len(informative)
    assert result.nobs == len(used)
    assert_allclose(result.metrics["log_likelihood"], ref.llf, rtol=1e-9)


def test_conditional_poisson_matches_statsmodels_and_dummies(panel):
    frame = panel.copy()
    frame.loc[frame.id < 5, "c"] = 0                       # all-zero panels are dropped
    result = oe.xtpoisson(data=frame, y="c", x=["x", "z"], panel="id", model="fe")
    zero = frame.groupby("id").c.sum() == 0
    assert result.metrics["n_groups_dropped"] == int(zero.sum()) >= 5
    used = frame[~frame.id.isin(zero[zero].index)]
    ref = ConditionalPoisson(used.c, used[["x"]], groups=used.id).fit(disp=0, method="newton")
    assert_allclose(table(result)["x"].estimate, ref.params["x"], rtol=1e-7)
    assert_allclose(table(result)["x"].std_error, ref.bse["x"], rtol=1e-6)
    dummies = sm.GLM(used.c, np.column_stack([used.x, pd.get_dummies(used.id).to_numpy(float)]),
                     family=sm.families.Poisson()).fit(tol=1e-12)
    assert_allclose(table(result)["x"].estimate, dummies.params.iloc[0], rtol=1e-9)
    # Conditional log likelihood written out: ln Y! - sum ln y! + sum y ln p.
    b = table(result)["x"].estimate
    work = used.assign(lam=np.exp(b * used.x))
    work["p"] = work.lam / work.groupby("id").lam.transform("sum")
    totals = work.groupby("id").c.sum()
    ll = gammaln(totals + 1).sum() - gammaln(work.c + 1).sum() + (work.c * np.log(work.p)).sum()
    assert_allclose(result.metrics["log_likelihood"], ll, rtol=1e-10)
    robust = oe.xtpoisson(data=frame, y="c", x=["x"], panel="id", model="fe",
                          covariance="robust")
    work["s"] = work.c * work.x - work.groupby("id").c.transform("sum") * work.p * work.x
    scores = work.groupby("id").s.sum().to_numpy()
    groups = len(scores)
    information = 1 / table(result)["x"].std_error ** 2
    expected = groups / (groups - 1) * (scores ** 2).sum() / information ** 2
    assert_allclose(robust.coefficients[0].std_error ** 2, expected, rtol=1e-6)


# ---- population averaged -------------------------------------------------------------------


def test_population_averaged_is_xtgee(panel):
    for command, family, link, y in ((oe.xtlogit, "binomial", "logit", "y"),
                                     (oe.xtprobit, "binomial", "probit", "y"),
                                     (oe.xtpoisson, "poisson", "log", "c")):
        result = command(data=panel, y=y, x=["x", "z"], panel="id", time="t", model="pa",
                         corr="ar1", covariance="robust")
        reference = oe.xtgee(data=panel, y=y, x=["x", "z"], panel="id", time="t",
                             family=family, link=link, corr="ar1", covariance="robust")
        assert_allclose([c.estimate for c in result.coefficients],
                        [c.estimate for c in reference.coefficients], rtol=1e-12)
        assert_allclose(result.covariance_matrix, reference.covariance_matrix, rtol=1e-12)
        assert result.spec.estimator == command.__name__


# ---- errors and rendering -----------------------------------------------------------------


def test_error_contract(panel):
    def code(command=oe.xtlogit, **kwargs):
        options = {"data": panel, "y": "y", "x": ["x"], "panel": "id"}
        options.update(kwargs)
        with pytest.raises(AnalysisError) as error:
            command(**options)
        return error.value.code

    assert code(command=oe.xtprobit, model="fe") == "invalid_spec"
    assert code(model="fe", covariance="robust") == "unsupported_covariance"
    assert code(corr="ar1") == "invalid_spec"
    assert code(model="fe", intpoints=7) == "invalid_spec"
    assert code(model="pa", covariance="cluster", cluster="cl") == "unsupported_covariance"
    assert code(model="gee") == "invalid_spec"
    rng = np.random.default_rng(5)
    plain = panel.assign(c=rng.poisson(np.exp(0.2 + 0.3 * panel.x)))
    assert code(command=oe.xtpoisson, data=plain, y="c") == "boundary_solution"
    constant = panel.assign(y=(panel.id % 2).astype(float))
    assert code(data=constant, model="fe") == "no_outcome_variation"
    assert code(x=["z"], model="fe") == "no_within_group_variation"
    assert code(x=["id"]) == "invalid_spec"


def test_cluster_and_round_trip(panel):
    result = oe.xtlogit(data=panel, y="y", x=["x", "z"], panel="id", cluster="cl")
    assert result.inference["cluster_count"] == 20
    assert type(result).model_validate_json(result.model_dump_json()) == result
    text = result.summary()
    # No LR comparison test with a robust (pseudo) likelihood, as Stata's mixed.
    assert "/lnsig2u" in text and "rho" not in result.tests
    plain = oe.xtlogit(data=panel, y="y", x=["x", "z"], panel="id")
    assert "LR test of rho = 0" in plain.summary()
    assert "tabular" in result.to_latex()
    fe = oe.xtpoisson(data=panel, y="c", x=["x"], panel="id", model="fe")
    assert type(fe).model_validate_json(fe.model_dump_json()) == fe
    assert oe.capabilities()["estimators"]["xtpoisson"]["options"]["model"]["default"] == "re"


def test_count_kernel_derivatives(panel):
    import torch

    from openecon.econometrics.mixed.xt_kernels import (
        ConditionalPoissonObjective, GammaPoissonObjective,
    )
    from openecon.engines.optimize import check_derivatives

    x = torch.tensor(np.column_stack([np.ones(len(panel)), panel.x, panel.z]))
    y = torch.tensor(panel.c.to_numpy(float))
    codes = torch.tensor(panel.id.to_numpy())
    offset = torch.tensor(np.log(panel.e.to_numpy()))
    gamma = GammaPoissonObjective(x, y, codes, 120, offset)
    report = check_derivatives(gamma, torch.tensor([0.1, 0.4, 0.2, -0.6], dtype=torch.float64))
    assert report["gradient_max_rel_error"] < 1e-7 and report["hessian_max_rel_error"] < 1e-7
    conditional = ConditionalPoissonObjective(x[:, 1:2], y, codes, 120, offset)
    report = check_derivatives(conditional, torch.tensor([0.3], dtype=torch.float64))
    assert report["gradient_max_rel_error"] < 1e-7 and report["hessian_max_rel_error"] < 1e-7
