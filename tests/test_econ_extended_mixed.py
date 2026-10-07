"""Independent dense, special-function and quadrature oracles for new likelihoods."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import gammaln, log_ndtr, logsumexp, roots_hermitenorm
from scipy.stats import norm
from statsmodels.tools.numdiff import approx_hess, approx_fprime

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle

FIXTURE = Path(__file__).parent / "fixtures" / "extended_mixed"


def coefficients(result):
    return np.array([c.estimate for c in result.coefficients])


def full_covariance(value, theta):
    return np.linalg.inv(-approx_hess(np.asarray(theta), value, epsilon=2e-4))


@pytest.mark.parametrize("model", ["re", "fe"])
def test_hhg_published_and_independent_full_information(model):
    d = pd.read_csv(FIXTURE / "airacc.csv")
    r = oe.xtnbreg(
        data=d,
        y="i_cnt",
        x=["inprog"],
        panel="airline",
        time="time",
        exposure="pmiles",
        model=model,
    )
    x = np.column_stack((np.ones(len(d)), d.inprog))
    y = d.i_cnt.to_numpy()
    offset = np.log(d.pmiles)
    g = pd.factorize(d.airline)[0]

    def likelihood(b):
        lam = np.exp(x @ b[:2] + offset)
        total = np.bincount(g, weights=lam)
        yy = np.bincount(g, weights=y)
        row = (gammaln(lam + y) - gammaln(lam) - gammaln(y + 1)).sum()
        if model == "fe":
            return row + (gammaln(total) + gammaln(yy + 1) - gammaln(total + yy)).sum()
        a, s = np.exp(b[2:])
        return (
            row
            + (
                gammaln(a + s)
                - gammaln(a)
                - gammaln(s)
                + gammaln(a + total)
                + gammaln(s + yy)
                - gammaln(a + s + total + yy)
            ).sum()
        )

    b = coefficients(r)
    assert_allclose(r.metrics["log_likelihood"], likelihood(b), atol=2e-9)
    assert_allclose(r.covariance_matrix, full_covariance(likelihood, b), rtol=2e-3, atol=3e-5)
    irr = np.exp(b[:2])
    irr_se = irr * np.sqrt(np.diag(r.covariance_matrix)[:2])
    assert_allclose(
        irr, [0.0367524, 0.911673] if model == "re" else [0.0329025, 0.9062669], atol=1e-6
    )
    assert_allclose(
        irr_se, [0.0407032, 0.0590277] if model == "re" else [0.0331262, 0.0613917], atol=2e-6
    )
    assert_allclose(
        r.metrics["log_likelihood"], -265.38202 if model == "re" else -174.25143, atol=5e-6
    )
    assert (
        ResultBundle.model_validate_json(r.model_dump_json()).covariance_matrix
        == r.covariance_matrix
    )


def frontier_oracle(d, theta, varying=False, half=False, cost=False):
    x = np.column_stack((np.ones(len(d)), d[["lnmachines", "lnworkers"]]))
    g = pd.factorize(d.id)[0]
    y = d.lnwidgets.to_numpy()
    residual = y - x @ theta[:3]
    mu = 0 if half else theta[3]
    start = 3 + int(not half)
    su, sv = np.exp(theta[start : start + 2])
    w = (
        np.exp(-theta[-1] * (d.t - d.groupby("id").t.transform("max")).to_numpy())
        if varying
        else np.ones(len(d))
    )
    n = np.bincount(g)
    a = su**-2 + np.bincount(g, weights=w * w) / sv**2
    b = mu / su**2 + (1 if cost else -1) * np.bincount(g, weights=w * residual) / sv**2
    ll = (
        -n / 2 * np.log(2 * np.pi)
        - n * np.log(sv)
        - np.log(su)
        - 0.5 * np.log(a)
        - 0.5 * np.bincount(g, weights=residual**2) / sv**2
        - 0.5 * mu**2 / su**2
        + 0.5 * b * b / a
        + log_ndtr(b / np.sqrt(a))
        - log_ndtr(mu / su)
    ).sum()
    m, s = b / a, 1 / np.sqrt(a)
    efficiency = np.exp(
        -w * m[g] + 0.5 * w * w * s[g] ** 2 + log_ndtr((m / s)[g] - w * s[g]) - log_ndtr((m / s)[g])
    )
    return ll, efficiency


@pytest.mark.parametrize("varying", [False, True])
def test_frontier_published_joint_likelihood_and_efficiency_uncertainty(varying):
    d = pd.read_csv(FIXTURE / "xtfrontier1.csv")
    r = oe.xtfrontier(
        data=d,
        y="lnwidgets",
        x=["lnmachines", "lnworkers"],
        panel="id",
        time="t",
        time_varying=varying,
    )
    b = np.array(r.extra["likelihood_theta"])
    v = np.array(r.extra["likelihood_covariance"])

    def fun(theta):
        return frontier_oracle(d, theta, varying)[0]

    assert_allclose(r.metrics["log_likelihood"], fun(b), atol=2e-9)
    assert_allclose(v, full_covariance(fun, b), rtol=3e-3, atol=4e-5)
    published = (
        [3.028939, 0.2907555, 0.2942412, 1.110831, 1.410723, 1.123982, 0.0016764]
        if varying
        else [3.030983, 0.2904551, 0.2943333, 1.125667, 1.421979, 1.138685]
    )
    assert_allclose(coefficients(r), published, atol=2e-4)
    assert_allclose(r.metrics["log_likelihood"], -1472.5289 if varying else -1472.6069, atol=5e-5)
    ll, eff = frontier_oracle(d, b, varying)
    jac = approx_fprime(
        b, lambda theta: frontier_oracle(d, theta, varying)[1], epsilon=1e-6, centered=True
    )
    assert_allclose([x["mean"] for x in r.extra["efficiency"]], eff, atol=2e-12)
    assert_allclose(
        [x["std_error"] for x in r.extra["efficiency"]],
        np.sqrt(np.einsum("ij,jk,ik->i", jac, v, jac)),
        rtol=2e-5,
        atol=1e-7,
    )
    # Direct numerical integration of the joint conditional density for three firms.
    x = np.column_stack((np.ones(len(d)), d[["lnmachines", "lnworkers"]]))
    mu = b[3]
    su, sv = np.exp(b[4:6])
    for _, block in list(d.groupby("id"))[:3]:
        rows = block.index.to_numpy()
        res = d.lnwidgets.to_numpy()[rows] - x[rows] @ b[:3]
        w = np.exp(-b[-1] * (block.t - block.t.max())) if varying else np.ones(len(rows))

        def integrand(u):
            return np.exp(
                norm.logpdf(u, mu, su)
                - norm.logcdf(mu / su)
                + norm.logpdf(res + w * u, 0, sv).sum()
            )

        integrated = quad(integrand, 0, np.inf, epsabs=1e-18, epsrel=1e-9)[0]
        subset = d.loc[rows].reset_index(drop=True)
        assert_allclose(np.log(integrated), frontier_oracle(subset, b, varying)[0], atol=1e-8)


def dense_lmm(d, theta, groups, nested=False):
    x = np.column_stack((np.ones(len(d)), d[["x"]] if "x" in d else np.empty((len(d), 0))))
    p = x.shape[1]
    v = np.eye(len(d)) * np.exp(2 * theta[-1])
    names = groups
    for j, name in enumerate(names):
        keys = list(zip(*[d[c].tolist() for c in groups[: j + 1]])) if nested else d[name].tolist()
        codes = pd.factorize(pd.Series(keys))[0]
        # Native reporting parameter order: lowest level, upper levels, residual.
        index = p if j == len(groups) - 1 else p + 1 + j
        v += (codes[:, None] == codes[None, :]) * np.exp(2 * theta[index])
    residual = d["y"].to_numpy() - x @ theta[:p]
    return -0.5 * (
        len(d) * np.log(2 * np.pi)
        + np.linalg.slogdet(v)[1]
        + residual @ np.linalg.solve(v, residual)
    )


def test_crossed_penicillin_ml_independent_dense_optimizer_full_covariance():
    d = pd.read_csv(FIXTURE / "penicillin.csv").rename(columns={"diameter": "y"})
    r = oe.mixedflex(data=d, y="y", group=["plate", "sample"])
    b = np.array(r.extra["likelihood_theta"])
    v = np.array(r.extra["likelihood_covariance"])

    def fun(theta):
        return dense_lmm(d, theta, ["plate", "sample"])

    ref = minimize(
        lambda theta: -fun(theta), [23.0, 0.4, -0.3, -0.5], method="BFGS", options={"gtol": 1e-6}
    )
    assert_allclose(b, ref.x, atol=2e-5)
    assert_allclose(r.metrics["log_likelihood"], fun(ref.x), atol=1e-9)
    assert_allclose(v, full_covariance(fun, b), rtol=2e-4, atol=1e-6)
    assert_allclose(coefficients(r)[0], 22.9722222222222, atol=1e-10)


def test_three_nested_levels_reused_child_labels_full_information():
    rng = np.random.default_rng(18)
    top = np.repeat(np.arange(6), 16)
    mid = np.tile(np.repeat(np.arange(2), 8), 6)
    low = np.tile(np.repeat(np.arange(2), 4), 12)
    x = rng.normal(size=len(top))
    y = (
        2
        + 0.4 * x
        + rng.normal(size=6)[top]
        + rng.normal(size=12)[top * 2 + mid]
        + rng.normal(size=24)[top * 4 + mid * 2 + low]
        + rng.normal(size=len(top)) * 0.4
    )
    d = pd.DataFrame({"y": y, "x": x, "top": top, "mid": mid, "low": low})
    r = oe.mixedflex(data=d, y="y", x=["x"], group=["top", "mid", "low"], grouping="nested")
    b = np.array(r.extra["likelihood_theta"])

    def fun(theta):
        return dense_lmm(d, theta, ["top", "mid", "low"], True)

    assert_allclose(r.metrics["log_likelihood"], fun(b), atol=1e-10)
    assert_allclose(r.extra["likelihood_covariance"], full_covariance(fun, b), rtol=2e-3, atol=1e-6)
    assert [v["n_groups"] for v in r.extra["levels"]] == [6, 12, 24]


def transport():
    d = pd.read_csv(FIXTURE / "transport.csv")
    columns = ["trcost", "trtime"]
    for a in ["Public", "Bicycle", "Walk"]:
        for c in ["age", "income"]:
            name = a + "_" + c
            d[name] = (d.alt == a) * d[c]
            columns.append(name)
        name = a + "_const"
        d[name] = (d.alt == a).astype(float)
        columns.append(name)
    return d, columns


def test_choice_public_fixture_independent_high_order_full_information_probabilities():
    d, cols = transport()
    r = oe.mixedlogit(
        data=d,
        y="choice",
        x=cols,
        group="id",
        case="t",
        alternative="alt",
        random=["trtime"],
        max_work=10_000_000_000,
    )
    b = np.array(r.extra["likelihood_theta"])
    v = np.array(r.extra["likelihood_covariance"])
    nodes, weights = roots_hermitenorm(128)
    weights /= np.sqrt(2 * np.pi)
    x = d[cols].to_numpy()
    z = d.trtime.to_numpy()
    y = d.choice.to_numpy()

    def probabilities(theta):
        eta = (x @ theta[:-1])[:, None] + z[:, None] * np.exp(theta[-1]) * nodes
        shaped = eta.reshape(500, 3, 4, -1)
        return (shaped - logsumexp(shaped, axis=2, keepdims=True)).reshape(len(d), -1)

    def likelihood(theta):
        logp = probabilities(theta)
        panel = (logp * y[:, None]).reshape(500, 12, -1).sum(1)
        return logsumexp(panel + np.log(weights), axis=1).sum()

    assert_allclose(r.metrics["log_likelihood"], likelihood(b), atol=2e-6)
    assert_allclose(v, full_covariance(likelihood, b), rtol=5e-3, atol=4e-5)
    mean = (np.exp(probabilities(b)) * weights).sum(1)
    assert_allclose([x["mean"] for x in r.extra["response"]], mean, atol=1e-6)
    assert_allclose(mean.reshape(-1, 4).sum(1), 1, atol=1e-12)
    # Official example uses a simulated Hammersley integral: compare printed
    # coefficients within simulation resolution, not its different objective.
    assert_allclose(coefficients(r)[:2], [-0.8388216, -1.508756], atol=3e-4)
    assert_allclose(np.sqrt(coefficients(r)[-1]), 1.945596, atol=1e-3)
    assert r.extra["quadrature_checks"][-1]["normalized_covariance_change"] < 0.001


def test_menbreg_published_nb2_fixture_population_full_delta():
    d = pd.read_csv(FIXTURE / "drvisits.csv")
    cols = ["reform", "age", "educ", "married", "badh", "loginc"]
    r = oe.menbreg(data=d, y="numvisit", x=cols, group="id")
    b = np.array(r.extra["likelihood_theta"])
    v = np.array(r.extra["likelihood_covariance"])
    x = np.column_stack((np.ones(len(d)), d[cols]))
    y = d.numvisit.to_numpy()
    g = pd.factorize(d.id)[0]
    nodes, weights = roots_hermitenorm(96)
    weights /= np.sqrt(2 * np.pi)

    def likelihood(theta):
        eta = (x @ theta[:7])[:, None] + np.exp(theta[7]) * nodes
        size = np.exp(-theta[-1])
        denom = np.logaddexp(np.log(size), eta)
        row = (
            gammaln(y[:, None] + size)
            - gammaln(size)
            - gammaln(y[:, None] + 1)
            + size * (np.log(size) - denom)
            + y[:, None] * (eta - denom)
        )
        panel = np.zeros((g.max() + 1, len(nodes)))
        np.add.at(panel, g, row)
        return logsumexp(panel + np.log(weights), axis=1).sum()

    assert_allclose(r.metrics["log_likelihood"], likelihood(b), atol=1e-8)
    assert_allclose(v, full_covariance(likelihood, b), rtol=3e-3, atol=2e-5)
    assert_allclose(
        np.exp(coefficients(r)[:7]),
        [0.5017199, 0.9008536, 1.003593, 1.007026, 1.089597, 3.043562, 1.136342],
        atol=2e-5,
    )
    assert_allclose(coefficients(r)[-2:], [0.4740088, -0.7962692], atol=1e-5)
    mean = np.exp(x @ b[:7] + 0.5 * np.exp(2 * b[7]))
    jac = np.column_stack((x * mean[:, None], mean * np.exp(2 * b[7]), np.zeros(len(d))))
    assert_allclose([row["mean"] for row in r.extra["response"]], mean, rtol=1e-12)
    assert_allclose(
        [row["std_error"] for row in r.extra["response"]],
        np.sqrt(np.einsum("ij,jk,ik->i", jac, v, jac)),
        rtol=1e-12,
    )


def test_extended_domains_budget_missing_and_choice_availability():
    d, cols = transport()
    small = d[d.id <= 10].copy()
    with pytest.raises(AnalysisError, match="exactly one"):
        oe.mixedlogit(
            data=small.assign(choice=0),
            y="choice",
            x=cols,
            group="id",
            case="t",
            alternative="alt",
            random=["trtime"],
        )
    small["available"] = 1
    small.loc[small.choice == 1, "available"] = 0
    with pytest.raises(AnalysisError, match="unavailable"):
        oe.mixedlogit(
            data=small,
            y="choice",
            x=cols,
            group="id",
            case="t",
            alternative="alt",
            random=["trtime"],
            availability="available",
        )
    nb = pd.read_csv(FIXTURE / "airacc.csv")
    with pytest.raises(AnalysisError, match="work"):
        oe.xtnbreg(data=nb, y="i_cnt", x=["inprog"], panel="airline", max_work=1)
    with pytest.raises(AnalysisError):
        oe.xtnbreg(data=nb.assign(i_cnt=0.1), y="i_cnt", x=["inprog"], panel="airline")
    frontier = pd.read_csv(FIXTURE / "xtfrontier1.csv")
    with pytest.raises(AnalysisError, match="calendar"):
        oe.xtfrontier(data=frontier, y="lnwidgets", x=["lnmachines"], panel="id", time_varying=True)


def test_correlated_nb2_random_slope_full_information_and_population_delta():
    import torch
    from openecon.dataset import Dataset

    rng = np.random.default_rng(952)
    g = np.repeat(np.arange(40), 6)
    x = rng.normal(size=len(g)) * 0.6
    u = rng.multivariate_normal([0, 0], [[0.36, 0.15], [0.15, 0.64]], 40)
    mu = np.exp(0.4 + 0.3 * x + u[g, 0] * x + u[g, 1])
    y = rng.negative_binomial(2, 2 / (2 + mu))
    d = pd.DataFrame(dict(g=g, x=x, y=y))
    before = torch.random.get_rng_state().clone()
    r = oe.menbreg(
        data=d,
        y="y",
        x=["x"],
        group="g",
        random=["x"],
        covstructure="unstructured",
        max_work=15_000_000_000,
    )
    assert torch.equal(before, torch.random.get_rng_state())
    b = np.array(r.extra["likelihood_theta"])
    nodes, weights = roots_hermitenorm(80)
    weights /= np.sqrt(2 * np.pi)
    grid = np.array(np.meshgrid(nodes, nodes, indexing="ij")).reshape(2, -1)
    logweights = np.log(np.outer(weights, weights).ravel())
    fixed = np.column_stack((np.ones(len(d)), x))
    z = np.column_stack((x, np.ones(len(d))))

    def likelihood(theta):
        lower = np.array([[np.exp(theta[2]), 0], [theta[3], np.exp(theta[4])]])
        eta = (fixed @ theta[:2])[:, None] + z @ lower @ grid
        size = np.exp(-theta[-1])
        denom = np.logaddexp(np.log(size), eta)
        row = (
            gammaln(y[:, None] + size)
            - gammaln(size)
            - gammaln(y[:, None] + 1)
            + size * (np.log(size) - denom)
            + y[:, None] * (eta - denom)
        )
        panel = row.reshape(40, 6, -1).sum(1)
        return logsumexp(panel + logweights, axis=1).sum()

    assert_allclose(r.metrics["log_likelihood"], likelihood(b), atol=2e-6)
    assert_allclose(
        r.extra["likelihood_covariance"], full_covariance(likelihood, b), rtol=0.005, atol=6e-4
    )
    reported = coefficients(r)

    def population(point):
        return np.exp(fixed @ point[:2] + 0.5 * (point[2] * x * x + point[3] + 2 * point[4] * x))

    jac = approx_fprime(reported, population, epsilon=1e-6, centered=True)
    actual = oe.predict(r, Dataset.from_frame(d), interval="mean", batch_rows=7)
    actual = pd.concat(list(actual.iter_batches(batch_rows=5)))
    assert_allclose(actual.response, population(reported), atol=1e-12)
    assert_allclose(
        actual.std_error,
        np.sqrt(np.einsum("ij,jk,ik->i", jac, r.covariance_matrix, jac)),
        rtol=2e-8,
    )

    def effect(point):
        return np.mean(population(point) * (point[1] + point[2] * x + point[4]))

    gradient = approx_fprime(reported, effect, epsilon=1e-6, centered=True)
    actual = oe.margins(r, "x", data=Dataset.from_frame(d), batch_rows=5)
    assert_allclose(actual.estimate, effect(reported), atol=2e-12)
    assert_allclose(actual.std_error, np.sqrt(gradient @ r.covariance_matrix @ gradient), rtol=2e-8)


def test_half_normal_cost_frontier_is_joint_signed_likelihood():
    d = pd.read_csv(FIXTURE / "xtfrontier1.csv")
    prod = oe.xtfrontier(
        data=d,
        y="lnwidgets",
        x=["lnmachines", "lnworkers"],
        panel="id",
        time="t",
        distribution="half_normal",
    )
    cost = oe.xtfrontier(
        data=d.assign(lnwidgets=-d.lnwidgets),
        y="lnwidgets",
        x=["lnmachines", "lnworkers"],
        panel="id",
        time="t",
        distribution="half_normal",
        cost=True,
    )
    b = np.array(cost.extra["likelihood_theta"])
    def fun(point):
        return frontier_oracle(
            d.assign(lnwidgets=-d.lnwidgets), point, half=True, cost=True
        )[0]
    assert_allclose(cost.metrics["log_likelihood"], prod.metrics["log_likelihood"], atol=2e-10)
    assert_allclose(coefficients(cost)[:3], -coefficients(prod)[:3], atol=1e-9)
    assert_allclose(
        cost.extra["likelihood_covariance"], full_covariance(fun, b), rtol=2e-3, atol=3e-5
    )
