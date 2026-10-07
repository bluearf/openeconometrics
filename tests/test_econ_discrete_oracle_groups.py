"""Independent oracles for clogit, hetprobit and biprobit (verification stage).

The conditional-logit denominator is ENUMERATED (every outcome vector of a
group with the observed number of positives), or, for large groups, read off
the coefficients of the polynomial ``prod_i (1 + exp(e_i) z)``. The bivariate
normal distribution function is written with Owen's T function, a
representation unrelated to the Genz quadrature of the implementation. The
calculus helpers (Richardson derivatives, brute-force maximization, Stata's
covariance formulas) come from ``test_econ_discrete_oracle``.
"""

import itertools
import math
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import integrate, special, stats
from statsmodels.discrete.conditional_models import ConditionalLogit
from test_econ_discrete_oracle import (
    KINDS, check_chi2, check_table, design, jacobian, maximize, same_fit, stata_covariance,
    table, terms, wald,
)

import openecon as oe
from openecon.analysis import AnalysisError

# ---- conditional logit -----------------------------------------------------------------


@pytest.fixture(scope="module")
def matched():
    """Groups of 2..7 rows with 0..T positives, nested in uneven clusters."""
    rng = np.random.default_rng(4471)
    sizes = rng.integers(2, 8, size=150)
    group = np.repeat(np.arange(len(sizes)), sizes)
    n = len(group)
    frame = pd.DataFrame({
        "set": np.array([f"s{g:03d}" for g in group]), "x1": rng.normal(size=n),
        "x2": (rng.random(n) < 0.4).astype(float), "x3": rng.uniform(-1, 1, size=n),
        "dose": rng.normal(scale=0.4, size=n),
    })
    effect = rng.normal(scale=1.2, size=len(sizes))[group]
    index = effect + 0.8 * frame.x1 - 0.6 * frame.x2 + 0.5 * frame.x3
    frame["y"] = (rng.random(n) < 1 / (1 + np.exp(-index))).astype(float)
    frame["clinic"] = (group // 4).astype(int)                 # groups nested in clinics
    frame["ward"] = rng.integers(0, 25, size=n)               # crossed with the groups
    frame["fw"] = rng.integers(1, 4, size=len(sizes)).astype(float)[group]
    frame["pw"] = rng.uniform(0.5, 9.0, size=len(sizes))[group]
    return frame.sample(frac=1.0, random_state=2).reset_index(drop=True)    # unsorted groups


class Enumerated:
    """The conditional likelihood by enumeration of every admissible outcome vector."""

    def __init__(self, frame, names, group="set", outcome="y", offset=None):
        y = frame[outcome].to_numpy()
        labels, codes = np.unique(frame[group].to_numpy(), return_inverse=True)
        members = [np.flatnonzero(codes == g) for g in range(len(labels))]
        self.dropped = [rows for rows in members if y[rows].sum() in (0, len(rows))]
        self.members = [rows for rows in members if 0 < y[rows].sum() < len(rows)]
        self.kept = np.sort(np.concatenate(self.members))
        self.x = frame[names].to_numpy(dtype=float)
        self.y = y
        self.shift = 0.0 if offset is None else frame[offset].to_numpy()
        self.sets = [np.array(list(itertools.combinations(rows, int(y[rows].sum()))))
                     for rows in self.members]
        # Groups with the same (size, positives) share array shapes: evaluate them together.
        classes = {}
        for position, subsets in enumerate(self.sets):
            classes.setdefault(subsets.shape, []).append(position)
        self.blocks = [(np.array(ids), np.stack([self.sets[g] for g in ids]),
                        np.stack([self.members[g][y[self.members[g]] == 1] for g in ids]))
                       for ids in classes.values()]

    def group_loglik(self, theta):
        index = self.x @ theta + self.shift
        out = np.empty(len(self.members))
        for ids, subsets, positives in self.blocks:
            out[ids] = index[positives].sum(axis=1) - special.logsumexp(
                index[subsets].sum(axis=2), axis=1)
        return out

    def null(self):
        return -sum(math.log(math.comb(len(rows), int(self.y[rows].sum())))
                    for rows in self.members)

    def row_scores(self, theta):
        """(y_i - Pr(y_i = 1 | m_g)) (x_i - xbar_g) with the probability enumerated.

        d ln L_g / d e_i = y_i - Pr(y_i = 1 | m_g) sums to zero within a group, so any
        group-constant shift of x leaves the group score unchanged; centring on the group
        mean is the split that does not depend on the origin of the regressors.
        """
        index = self.x @ theta + self.shift
        scores = np.zeros_like(self.x)
        for rows, subsets in zip(self.members, self.sets, strict=True):
            share = special.softmax(index[subsets].sum(axis=1))
            centre = self.x[rows].mean(axis=0)
            for row in rows:
                chance = share[(subsets == row).any(axis=1)].sum()
                scores[row] = (self.y[row] - chance) * (self.x[row] - centre)
        return scores


def clogit_oracle(frame, names, *, weights=None, offset=None):
    model = Enumerated(frame, names, offset=offset)
    group_weights = None if weights is None else np.array(
        [frame[weights].to_numpy()[rows[0]] for rows in model.members])
    theta, value, hessian = maximize(model.group_loglik, np.zeros(len(names)), group_weights)
    return model, theta, value, hessian, jacobian(model.group_loglik, theta), group_weights


def test_clogit_against_the_enumerated_likelihood(matched):
    names = ["x1", "x2", "x3"]
    model, theta, value, hessian, scores, _ = clogit_oracle(matched, names)
    groups, rows = len(model.members), len(model.kept)
    assert model.dropped                                            # the design has such groups
    clinic = np.array([matched.clinic.to_numpy()[r[0]] for r in model.members])
    for kind in KINDS:
        result = oe.clogit(data=matched, y="y", x=names, group="set",
                           covariance=None if kind == "cluster" else kind,
                           cluster="clinic" if kind == "cluster" else None)
        # The unit of the OPG and robust estimators is the group; robust = cluster(group).
        covariance = stata_covariance("cluster" if kind == "robust" else kind, hessian, scores,
                                      nobs=groups,
                                      clusters=clinic if kind == "cluster" else np.arange(groups))
        check_table(result, theta, covariance)
        assert terms(result) == names
        assert result.nobs == rows and result.sample_positions == model.kept.tolist()
        metrics = result.metrics
        assert metrics["n_groups"] == groups and metrics["n_groups_dropped"] == len(model.dropped)
        assert result.extra["observations_dropped"] == sum(len(r) for r in model.dropped)
        assert any("all positive or all negative" in message for message in result.warnings)
        assert_allclose(metrics["log_likelihood"], value, rtol=1e-11)
        assert_allclose(result.extra["null_log_likelihood"], model.null(), rtol=1e-12)
        assert_allclose(metrics["pseudo_r_squared"], 1 - value / model.null(), rtol=1e-9)
        assert_allclose(metrics["aic"], -2 * value + 2 * 3, rtol=1e-11)
        assert_allclose(metrics["bic"], -2 * value + 3 * math.log(rows), rtol=1e-11)
        if kind == "nonrobust":
            check_chi2(result.tests["model"], 2 * (value - model.null()), 3)
        else:
            check_chi2(result.tests["model"], wald(theta, covariance, range(3)), 3, rtol=1e-5)
        if kind == "robust":
            assert result.inference["cluster_count"] == groups
        if kind == "cluster":
            assert result.inference["cluster_count"] == len(np.unique(clinic))
    # The enumerated per-row scores add up to the group scores (oracle self-check) ...
    row_scores = model.row_scores(theta)
    assert_allclose(np.array([row_scores[r].sum(axis=0) for r in model.members]), scores,
                    atol=1e-8)
    # ... and give the covariance when the clusters cut across the groups.
    crossed = oe.clogit(data=matched, y="y", x=names, group="set", cluster="ward")
    kept = model.kept
    covariance = stata_covariance("cluster", hessian, row_scores[kept], nobs=rows,
                                  clusters=matched.ward.to_numpy()[kept])
    check_table(crossed, theta, covariance)
    assert any("split across the clusters" in message for message in crossed.warnings)
    assert not any("split across" in message for message in result.warnings)   # nested clinics


def test_clogit_matches_statsmodels(matched):
    names = ["x1", "x2", "x3"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ConditionalLogit(matched.y, matched[names], groups=matched["set"])
        reference = model.fit(disp=False)
    result = oe.clogit(data=matched, y="y", x=names, group="set")
    # statsmodels stops BFGS at its default tolerance: its estimate is close, ours is the
    # stationary point of ITS likelihood and score.
    assert_allclose(table(result)["estimate"], reference.params, rtol=2e-3)
    assert_allclose(table(result)["std_error"], reference.bse, rtol=2e-3)
    theta = table(result)["estimate"]
    assert_allclose(model.loglike(theta), result.metrics["log_likelihood"], rtol=1e-11)
    assert model.loglike(theta) >= reference.llf
    assert_allclose(model.score(theta), 0.0, atol=1e-7)
    assert_allclose(np.linalg.inv(-jacobian(model.score, theta)),
                    np.array(result.covariance_matrix), rtol=1e-6)


@pytest.mark.parametrize("weight_type,column", [("fweight", "fw"), ("iweight", "pw"),
                                               ("pweight", "pw")])
def test_clogit_group_weights_against_enumeration(matched, weight_type, column):
    names = ["x1", "x2"]
    model, theta, value, hessian, scores, weights = clogit_oracle(matched, names, weights=column)
    frequency = weights if weight_type == "fweight" else None
    weighted = scores if weight_type == "fweight" else scores * weights[:, None]
    groups = int(weights.sum()) if weight_type == "fweight" else len(model.members)
    kinds = ("robust", "cluster") if weight_type == "pweight" else KINDS
    clinic = np.array([matched.clinic.to_numpy()[r[0]] for r in model.members])
    rows = matched[column].to_numpy()[model.kept]
    for kind in kinds:
        result = oe.clogit(data=matched, y="y", x=names, group="set", weights=column,
                           weight_type=weight_type, covariance=None if kind == "cluster" else kind,
                           cluster="clinic" if kind == "cluster" else None)
        if kind == "robust" and weight_type == "fweight":
            # Duplicated groups are separate clusters: sum_g f_g s_g s_g' with G = sum f_g.
            bread = np.linalg.inv(-hessian)
            covariance = groups / (groups - 1) * bread @ (
                (scores * weights[:, None]).T @ scores) @ bread
        else:
            covariance = stata_covariance(
                "cluster" if kind == "robust" else kind, hessian, weighted, nobs=groups,
                frequency=frequency,
                clusters=clinic if kind == "cluster" else np.arange(len(model.members)))
        check_table(result, theta, covariance)
        assert result.metrics["n_groups"] == groups
        assert result.nobs == (int(rows.sum()) if weight_type == "fweight" else len(model.kept))
        assert_allclose(result.metrics["log_likelihood"], value, rtol=1e-11)
    if weight_type == "fweight":
        # The same data with every group physically replicated under a new identifier.
        copies = []
        for rows_ in [*model.members, *model.dropped]:
            block = matched.iloc[rows_]
            for copy in range(int(block.fw.iloc[0])):
                copies.append(block.assign(set=block["set"] + f"_{copy}"))
        replicated = pd.concat(copies, ignore_index=True)
        for kind in KINDS:
            options = {"covariance": None if kind == "cluster" else kind,
                       "cluster": "clinic" if kind == "cluster" else None}
            same_fit(oe.clogit(data=matched, y="y", x=names, group="set", weights="fw",
                               weight_type="fweight", **options),
                     oe.clogit(data=replicated, y="y", x=names, group="set", **options))
    # Weights that vary inside a group are refused (Stata: weights must be constant in group).
    uneven = matched.assign(pw=np.arange(len(matched)) + 1.0)
    with pytest.raises(AnalysisError) as caught:
        oe.clogit(data=uneven, y="y", x=names, group="set", weights="pw",
                  weight_type=weight_type)
    assert caught.value.code == "weights_not_constant_within_group"


def test_clogit_offset_one_positive_per_group_and_large_groups(matched):
    names = ["x1", "x3"]
    model, theta, value, hessian, _, _ = clogit_oracle(matched, names, offset="dose")
    result = oe.clogit(data=matched, y="y", x=names, group="set", offset="dose")
    check_table(result, theta, np.linalg.inv(-hessian))
    assert_allclose(result.metrics["log_likelihood"], value, rtol=1e-11)
    # One positive per group is McFadden's choice model: ln L_g = e_chosen - logsumexp(e).
    rng = np.random.default_rng(77)
    sets, alternatives = 220, 4
    choice = pd.DataFrame({"trip": np.repeat(np.arange(sets), alternatives),
                           "price": rng.normal(size=sets * alternatives),
                           "time": rng.exponential(size=sets * alternatives)})
    utility = (-0.9 * choice.price - 0.4 * choice.time
               + rng.gumbel(size=len(choice))).to_numpy().reshape(sets, alternatives)
    choice["chosen"] = (utility == utility.max(axis=1, keepdims=True)).ravel().astype(float)
    x = choice[["price", "time"]].to_numpy().reshape(sets, alternatives, 2)
    picked = choice.chosen.to_numpy().reshape(sets, alternatives).argmax(axis=1)

    def choice_loglik(point):
        index = x @ point
        return index[np.arange(sets), picked] - special.logsumexp(index, axis=1)

    theta, value, hessian = maximize(choice_loglik, np.zeros(2))
    result = oe.clogit(data=choice, y="chosen", x=["price", "time"], group="trip")
    check_table(result, theta, np.linalg.inv(-hessian))
    assert_allclose(result.metrics["log_likelihood"], value, rtol=1e-11)
    assert_allclose(result.extra["null_log_likelihood"], -sets * math.log(alternatives),
                    rtol=1e-12)
    assert result.extra["multiple_positive_outcomes"] is False
    # Large groups: the denominator is the m-th coefficient of prod_i (1 + exp(e_i) z).
    big = pd.DataFrame({"id": np.repeat(np.arange(30), 40), "x": rng.normal(size=1200),
                        "z": rng.normal(size=1200)})
    big["y"] = (rng.random(1200) < 1 / (1 + np.exp(-(0.3 + 0.7 * big.x - 0.5 * big.z)))) * 1.0
    xb, yb = big[["x", "z"]].to_numpy().reshape(30, 40, 2), big.y.to_numpy().reshape(30, 40)

    def polynomial_loglik(point):
        index = xb @ point
        out = []
        for g in range(30):
            coefficients = np.array([1.0])
            for value_ in np.exp(index[g]):
                coefficients = np.convolve(coefficients, [1.0, value_])
            out.append(index[g][yb[g] == 1].sum() - math.log(coefficients[int(yb[g].sum())]))
        return np.array(out)

    theta, value, hessian = maximize(polynomial_loglik, np.zeros(2))
    result = oe.clogit(data=big, y="y", x=["x", "z"], group="id")
    check_table(result, theta, np.linalg.inv(-hessian), cov_rtol=1e-5)
    assert_allclose(result.metrics["log_likelihood"], value, rtol=1e-10)
    assert result.extra["multiple_positive_outcomes"] is True


def test_clogit_invariances_missing_values_and_within_group_constants(matched):
    names = ["x1", "x2", "x3"]
    base = oe.clogit(data=matched, y="y", x=names, group="set", cluster="clinic")
    ordered = matched.sort_values(["set", "x1"]).reset_index(drop=True)
    same_fit(base, oe.clogit(data=ordered, y="y", x=names, group="set", cluster="clinic"))
    # Group labels of any type; a regressor constant within groups is not identified.
    numbered = matched.assign(set=matched["set"].str[1:].astype(int) * 7 - 3,
                              level=matched["set"].str[1:].astype(float) ** 2)
    other = oe.clogit(data=numbered, y="y", x=["x1", "level", "x2", "x3"], group="set",
                      cluster="clinic")
    same_fit(base, other)
    assert other.provenance["omitted_terms"] == ["level"]
    # Reversing the outcome flips the sign of every coefficient (the likelihood is symmetric).
    mirrored = oe.clogit(data=matched.assign(y=1 - matched.y), y="y", x=names, group="set",
                         cluster="clinic")
    assert_allclose(table(mirrored)["estimate"], -table(base)["estimate"], rtol=1e-8)
    assert_allclose(np.array(mirrored.covariance_matrix), np.array(base.covariance_matrix),
                    rtol=1e-7)
    assert_allclose(mirrored.metrics["log_likelihood"], base.metrics["log_likelihood"],
                    rtol=1e-11)
    scaled = oe.clogit(data=matched.assign(x1=matched.x1 * 1e8, x3=matched.x3 * 1e-8),
                       y="y", x=names, group="set", cluster="clinic")
    factor = np.array([1e-8, 1.0, 1e8])
    assert_allclose(table(scaled)["estimate"], table(base)["estimate"] * factor, rtol=1e-6)
    assert_allclose(table(scaled)["std_error"], table(base)["std_error"] * factor, rtol=1e-6)
    # Only within-group differences matter: a shift of a regressor changes nothing.
    # Only within-group differences matter: the level of a regressor changes nothing, for
    # the likelihood, its Hessian and the per-observation scores of crossed clusters alike.
    for cluster in ("clinic", "ward"):
        reference = oe.clogit(data=matched, y="y", x=names, group="set", cluster=cluster)
        moved = oe.clogit(data=matched.assign(x3=matched.x3 + 1e6, x1=matched.x1 - 4e5),
                          y="y", x=names, group="set", cluster=cluster)
        same_fit(reference, moved, rtol=1e-8)
    # Missing values: the complete rows define groups, outcome variation and the sample.
    rng = np.random.default_rng(31)
    holes = matched.copy()
    holes.loc[rng.choice(len(holes), 40, replace=False), "x2"] = np.nan
    holes.loc[rng.choice(len(holes), 30, replace=False), "y"] = np.nan
    complete = holes.dropna(subset=["x1", "x2", "x3", "y"]).reset_index(drop=True)
    dropped = oe.clogit(data=holes, y="y", x=names, group="set", missing="drop")
    same_fit(dropped, oe.clogit(data=complete, y="y", x=names, group="set"))
    model = Enumerated(complete, names)
    assert dropped.metrics["n_groups"] == len(model.members)
    assert dropped.metrics["n_groups_dropped"] == len(model.dropped)


# ---- heteroskedastic probit -------------------------------------------------------------


@pytest.fixture(scope="module")
def workers():
    rng = np.random.default_rng(60221)
    n = 700
    sizes = rng.integers(2, 30, size=120)
    firm = np.repeat(np.arange(len(sizes)), sizes)[:n]
    frame = pd.DataFrame({
        "age": rng.normal(size=n), "kids": rng.poisson(1.0, size=n).astype(float),
        "educ": rng.uniform(-1, 1, size=n), "urban": (rng.random(n) < 0.6).astype(float),
        "tenure": rng.normal(size=n), "sector": rng.choice(["a", "b", "c"], size=n),
        "firm": firm, "fw": rng.integers(1, 4, size=n).astype(float),
        "aw": rng.uniform(0.3, 2.5, size=n), "pw": rng.uniform(20, 300, size=n),
    })
    shock = rng.multivariate_normal([0, 0], [[1, 0.55], [0.55, 1]], size=n)
    noise = np.exp(0.5 * frame.urban - 0.35 * frame.tenure)
    frame["work"] = (0.4 + 0.9 * frame.age - 0.5 * frame.kids + noise * shock[:, 0] > 0) * 1.0
    frame["union"] = (0.2 + 0.7 * frame.age - 0.3 * frame.kids + shock[:, 0] > 0) * 1.0
    frame["insured"] = (-0.3 + 0.5 * frame.age + 0.8 * frame.educ + shock[:, 1] > 0) * 1.0
    return frame


def weight_setup(frame, weight_type):
    n = len(frame)
    if weight_type is None:
        return None, np.ones(n), None, n
    if weight_type == "fweight":
        return frame.fw.to_numpy(), np.ones(n), frame.fw.to_numpy(), int(frame.fw.sum())
    if weight_type == "aweight":
        scaled = frame.aw.to_numpy() * n / frame.aw.sum()
        return scaled, scaled, None, n
    raw = frame[{"iweight": "aw", "pweight": "pw"}[weight_type]].to_numpy()
    return raw, raw, None, n


WEIGHT_COLUMN = {"fweight": "fw", "aweight": "aw", "iweight": "aw", "pweight": "pw"}


def probit_loglik(theta, x, y):
    return stats.norm.logcdf((2 * y - 1) * (x @ theta))


def hetprobit_loglik(theta, x, z, y):
    """ln Phi(q x'b / exp(z'g)) with scipy's normal distribution."""
    k = x.shape[1]
    return stats.norm.logcdf((2 * y - 1) * (x @ theta[:k]) / np.exp(z @ theta[k:]))


def hetprobit_oracle(frame, x, z, *, weight_type=None, outcome="work"):
    y = frame[outcome].to_numpy()
    weights, score_weights, frequency, nobs = weight_setup(frame, weight_type)
    probit, probit_ll, _ = maximize(lambda t: probit_loglik(t, x, y), np.zeros(x.shape[1]),
                                    weights)
    start = np.concatenate([probit, np.zeros(z.shape[1])])
    theta, value, hessian = maximize(lambda t: hetprobit_loglik(t, x, z, y), start, weights)
    scores = jacobian(lambda t: hetprobit_loglik(t, x, z, y), theta) * score_weights[:, None]
    return {"theta": theta, "ll": value, "hessian": hessian, "scores": scores, "nobs": nobs,
            "frequency": frequency, "score_weights": score_weights, "probit_ll": probit_ll}


def check_hetprobit(result, oracle, covariance, kind, k, q, *, slopes):
    check_table(result, oracle["theta"], covariance)
    assert_allclose(result.metrics["log_likelihood"], oracle["ll"], rtol=1e-11)
    size = k + q
    assert_allclose(result.metrics["aic"], -2 * oracle["ll"] + 2 * size, rtol=1e-11)
    assert_allclose(result.metrics["bic"], -2 * oracle["ll"] + size * math.log(oracle["nobs"]),
                    rtol=1e-11)
    assert "pseudo_r_squared" not in result.metrics           # Stata's hetprobit reports none
    # Stata's header: Wald chi2 of the mean-equation slopes under every VCE.
    check_chi2(result.tests["model"], wald(oracle["theta"], covariance, slopes), len(slopes),
               rtol=1e-5)
    if kind in ("nonrobust", "opg"):
        check_chi2(result.tests["lnsigma"], 2 * (oracle["ll"] - oracle["probit_ll"]), q)
        assert "LR" in result.tests["lnsigma"]["label"]
    else:
        check_chi2(result.tests["lnsigma"], wald(oracle["theta"], covariance, range(k, k + q)),
                   q, rtol=1e-5)
        assert "Wald" in result.tests["lnsigma"]["label"]
    assert_allclose(result.extra["probit_log_likelihood"], oracle["probit_ll"], rtol=1e-10)


def test_hetprobit_against_brute_force_likelihood(workers):
    names, het = ["age", "kids", "educ"], ["urban", "tenure"]
    x, z = design(workers, names, constant=True), design(workers, het)
    oracle = hetprobit_oracle(workers, x, z)
    for kind in KINDS:
        result = oe.hetprobit(data=workers, y="work", x=names, het=het,
                              covariance=None if kind == "cluster" else kind,
                              cluster="firm" if kind == "cluster" else None)
        covariance = stata_covariance(kind, oracle["hessian"], oracle["scores"],
                                      nobs=len(workers), clusters=workers.firm)
        check_hetprobit(result, oracle, covariance, kind, 4, 2, slopes=[1, 2, 3])
        assert terms(result) == ["Intercept", "age", "kids", "educ", "lnsigma:urban",
                                 "lnsigma:tenure"]
        assert [row.equation for row in result.coefficients] == ["work"] * 4 + ["lnsigma"] * 2
        assert result.nobs == len(workers)
        assert result.extra["zero_outcomes"] == (workers.work == 0).sum()
        assert result.extra["nonzero_outcomes"] == (workers.work == 1).sum()
    # The chart sample is the fitted Pr(y = 1) = Phi(x'b / exp(z'g)).
    theta = oracle["theta"]
    chance = stats.norm.cdf(x @ theta[:4] / np.exp(z @ theta[4:]))
    for row in result.predictions[:25]:
        assert_allclose(row["fitted"], chance[row["row"]], rtol=1e-7)
        assert row["observed"] == workers.work[row["row"]]


@pytest.mark.parametrize("weight_type", ["fweight", "aweight", "iweight", "pweight"])
def test_hetprobit_weights_against_brute_force(workers, weight_type):
    names, het = ["age", "kids"], ["urban"]
    x, z = design(workers, names, constant=True), design(workers, het)
    oracle = hetprobit_oracle(workers, x, z, weight_type=weight_type)
    kinds = ("robust", "cluster") if weight_type == "pweight" else KINDS
    for kind in kinds:
        result = oe.hetprobit(data=workers, y="work", x=names, het=het,
                              weights=WEIGHT_COLUMN[weight_type], weight_type=weight_type,
                              covariance=None if kind == "cluster" else kind,
                              cluster="firm" if kind == "cluster" else None)
        covariance = stata_covariance(kind, oracle["hessian"], oracle["scores"],
                                      nobs=oracle["nobs"], frequency=oracle["frequency"],
                                      opg_weights=oracle["score_weights"],
                                      clusters=workers.firm)
        check_hetprobit(result, oracle, covariance, kind, 3, 1, slopes=[1, 2])
        assert result.nobs == oracle["nobs"]
    if weight_type == "fweight":
        duplicated = workers.loc[workers.index.repeat(workers.fw.astype(int))].reset_index(
            drop=True)
        for kind in KINDS:
            options = {"covariance": None if kind == "cluster" else kind,
                       "cluster": "firm" if kind == "cluster" else None}
            same_fit(oe.hetprobit(data=workers, y="work", x=names, het=het, weights="fw",
                                  weight_type="fweight", **options),
                     oe.hetprobit(data=duplicated, y="work", x=names, het=het, **options),
                     rtol=1e-8)
    if weight_type == "pweight":
        default = oe.hetprobit(data=workers, y="work", x=names, het=het, weights="pw",
                               weight_type="pweight")
        assert default.spec.covariance == "robust"
        other = oe.hetprobit(data=workers.assign(pw=workers.pw * 0.01), y="work", x=names,
                             het=het, weights="pw", weight_type="pweight")
        same_fit(default, other, rtol=1e-8, metrics=False)


def test_hetprobit_overlapping_equations_categoricals_and_no_constant(workers):
    # A variable in both equations, a categorical in each, no constant in the mean equation.
    dummies = {"sector": ["a", "b", "c"]}
    x = design(workers, ["age", "kids"], dummies=dummies)
    z = design(workers, ["age"], dummies=dummies)
    oracle = hetprobit_oracle(workers, x, z)
    result = oe.hetprobit(data=workers, y="work", x=["age", "kids", "sector"],
                          het=["age", "sector"], categorical=["sector"], intercept=False)
    assert terms(result) == ["age", "kids", "sector[b]", "sector[c]", "lnsigma:age",
                             "lnsigma:sector[b]", "lnsigma:sector[c]"]
    covariance = stata_covariance("nonrobust", oracle["hessian"], oracle["scores"],
                                  nobs=len(workers))
    check_hetprobit(result, oracle, covariance, "nonrobust", 4, 3, slopes=[0, 1, 2, 3])


def test_hetprobit_invariances_and_probit_limit(workers):
    names, het = ["age", "kids", "educ"], ["urban", "tenure"]
    base = oe.hetprobit(data=workers, y="work", x=names, het=het, cluster="firm")
    shuffled = workers.sample(frac=1.0, random_state=3).reset_index(drop=True)
    same_fit(base, oe.hetprobit(data=shuffled, y="work", x=names, het=het, cluster="firm"),
             rtol=1e-8)
    # Rescaling a mean regressor and a variance regressor.
    scaled = workers.assign(age=workers.age * 1e6, tenure=workers.tenure * 1e-5)
    other = oe.hetprobit(data=scaled, y="work", x=names, het=het, cluster="firm")
    factor = np.array([1.0, 1e-6, 1.0, 1.0, 1.0, 1e5])
    assert_allclose(table(other)["estimate"], table(base)["estimate"] * factor, rtol=1e-6)
    assert_allclose(table(other)["std_error"], table(base)["std_error"] * factor, rtol=1e-6)
    assert_allclose(other.metrics["log_likelihood"], base.metrics["log_likelihood"], rtol=1e-10)
    # Shifting a variance regressor by c multiplies the mean equation by exp(c g): the
    # model is the same, so the likelihood and the variance coefficients do not move.
    moved = oe.hetprobit(data=workers.assign(tenure=workers.tenure + 4.0), y="work", x=names,
                         het=het, cluster="firm")
    gamma = table(base)["estimate"][5]
    assert_allclose(table(moved)["estimate"][:4], table(base)["estimate"][:4]
                    * math.exp(4.0 * gamma), rtol=1e-7)
    assert_allclose(table(moved)["estimate"][4:], table(base)["estimate"][4:], rtol=1e-7)
    assert_allclose(moved.metrics["log_likelihood"], base.metrics["log_likelihood"], rtol=1e-10)
    assert_allclose(moved.tests["lnsigma"]["statistic"], base.tests["lnsigma"]["statistic"],
                    rtol=1e-6)
    # Reversing the outcome flips the mean equation and leaves the variance equation alone.
    mirrored = oe.hetprobit(data=workers.assign(work=1 - workers.work), y="work", x=names,
                            het=het, cluster="firm")
    sign = np.array([-1.0, -1, -1, -1, 1, 1])
    assert_allclose(table(mirrored)["estimate"], table(base)["estimate"] * sign, rtol=1e-7)
    assert_allclose(table(mirrored)["std_error"], table(base)["std_error"], rtol=1e-6)
    # The comparison model of the LR test is statsmodels' probit.
    probit = sm.Probit(workers.work, sm.add_constant(workers[names])).fit(disp=False, tol=1e-13)
    plain = oe.hetprobit(data=workers, y="work", x=names, het=het)
    assert_allclose(plain.extra["probit_log_likelihood"], probit.llf, rtol=1e-11)
    assert_allclose(plain.tests["lnsigma"]["statistic"],
                    2 * (plain.metrics["log_likelihood"] - probit.llf), rtol=1e-8)


# ---- bivariate probit -------------------------------------------------------------------


def binormal(h, k, r):
    """Phi2(h, k; r) from Owen's T function (Owen 1956), no bivariate quadrature."""
    h, k = np.asarray(h, dtype=float), np.asarray(k, dtype=float)
    root = np.sqrt((1 - r) * (1 + r))
    with np.errstate(divide="ignore", invalid="ignore"):
        slope_h = np.where(h == 0, np.sign(k) * np.inf, (k - r * h) / (h * root))
        slope_k = np.where(k == 0, np.sign(h) * np.inf, (h - r * k) / (k * root))
        slope_h, slope_k = np.nan_to_num(slope_h, nan=0.0), np.nan_to_num(slope_k, nan=0.0)
    same_side = (h * k > 0) | ((h * k == 0) & (h + k >= 0))
    value = (0.5 * (special.ndtr(h) + special.ndtr(k)) - special.owens_t(h, slope_h)
             - special.owens_t(k, slope_k) - np.where(same_side, 0.0, 0.5))
    # At the origin both slopes are 0/0; the orthant probability is known in closed form.
    return np.where((h == 0) & (k == 0), 0.25 + np.arcsin(r) / (2 * np.pi), value)


def test_owen_oracle_and_the_bivariate_normal_kernel_agree_with_quadrature():
    import mpmath
    import torch

    from openecon.econometrics.discrete.bivariate import bvn_cdf, log_bvn_cdf

    def tensor(value):
        return torch.tensor([value], dtype=torch.float64)

    points = [(0.3, -0.8, 0.5), (-1.2, -0.4, -0.7), (2.0, 1.5, 0.93), (0.0, 0.0, 0.3),
              (1.1, -3.0, 0.999), (0.4, 0.4, -0.999), (4.0, -4.0, 0.2), (-0.3, 2.2, 0.0)]
    for h, k, r in points:
        # Pr(X <= h, Y <= k) = int_{-inf}^{h} phi(x) Phi((k - r x) / sqrt(1 - r^2)) dx.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exact = integrate.quad(
                lambda x, k=k, r=r: stats.norm.pdf(x) * stats.norm.cdf(
                    (k - r * x) / math.sqrt(1 - r * r)),
                -np.inf, h, epsabs=1e-15, epsrel=1e-14, limit=400)[0]
        assert_allclose(binormal(h, k, r), exact, rtol=0, atol=5e-14)
        assert_allclose(float(bvn_cdf(tensor(h), tensor(k), r)), exact, rtol=0, atol=5e-14)
        assert_allclose(float(log_bvn_cdf(tensor(h), tensor(k), r)), math.log(exact),
                        rtol=0, atol=1e-11)
        library = stats.multivariate_normal([0, 0], [[1, r], [r, 1]]).cdf([h, k])
        assert_allclose(exact, library, rtol=0, atol=1e-6)
    # Vectorized comparison over the whole range of correlations.
    rng = np.random.default_rng(5)
    h, k = rng.normal(scale=1.5, size=4000), rng.normal(scale=1.5, size=4000)
    r = np.tanh(rng.normal(scale=1.5, size=4000))
    kernel = bvn_cdf(torch.from_numpy(h), torch.from_numpy(k), torch.from_numpy(r)).numpy()
    assert_allclose(kernel, binormal(h, k, r), rtol=0, atol=5e-14)
    # Tails: ln Phi2 against 40-digit quadrature (Owen's formula cancels there).
    mpmath.mp.dps = 40
    for h, k, r in [(-2.5, 0.7, -0.95), (-6.0, -7.0, 0.5), (-0.6, -0.6, -0.99),
                    (-2.0, -2.0, -0.9), (-8.0, -8.5, 0.9999), (-9.0, -4.0, -0.2)]:
        low, high = min(h, k), max(h, k)
        scale = mpmath.sqrt(1 - mpmath.mpf(r) ** 2)
        cuts = [low - width for width in (30, 10, 4, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01,
                                          0.005, 0.002, 0.001, 0)]
        value, error = mpmath.quad(
            lambda x, high=high, r=r, scale=scale: mpmath.npdf(x) * mpmath.ncdf(
                (high - r * x) / scale), cuts, error=True)
        assert error / value < 1e-13
        assert_allclose(float(log_bvn_cdf(tensor(h), tensor(k), r)), float(mpmath.log(value)),
                        rtol=0, atol=1e-10)


def biprobit_loglik(theta, x1, x2, y1, y2):
    k1 = x1.shape[1]
    q1, q2 = 2 * y1 - 1, 2 * y2 - 1
    probability = binormal(q1 * (x1 @ theta[:k1]), q2 * (x2 @ theta[k1:-1]),
                           q1 * q2 * math.tanh(theta[-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(probability)


def biprobit_oracle(frame, x1, x2, *, y1="union", y2="insured", weight_type=None):
    first, second = frame[y1].to_numpy(), frame[y2].to_numpy()
    weights, score_weights, frequency, nobs = weight_setup(frame, weight_type)
    one, ll_one, _ = maximize(lambda t: probit_loglik(t, x1, first), np.zeros(x1.shape[1]),
                              weights)
    two, ll_two, _ = maximize(lambda t: probit_loglik(t, x2, second), np.zeros(x2.shape[1]),
                              weights)

    def loglik(theta):
        return biprobit_loglik(theta, x1, x2, first, second)

    theta, value, hessian = maximize(loglik, np.concatenate([one, two, [0.0]]), weights)
    scores = jacobian(loglik, theta) * score_weights[:, None]
    return {"theta": theta, "ll": value, "hessian": hessian, "scores": scores, "nobs": nobs,
            "frequency": frequency, "score_weights": score_weights, "comparison": ll_one + ll_two}


def check_biprobit(result, oracle, covariance, kind, slopes, *, alpha=0.05):
    theta = oracle["theta"]
    size = len(theta)
    check_table(result, theta, covariance, alpha=alpha, cov_rtol=5e-6)
    assert terms(result)[-1] == "/athrho" and result.coefficients[-1].equation is None
    metrics = result.metrics
    assert_allclose(metrics["log_likelihood"], oracle["ll"], rtol=1e-11)
    assert_allclose(metrics["aic"], -2 * oracle["ll"] + 2 * size, rtol=1e-11)
    assert_allclose(metrics["bic"], -2 * oracle["ll"] + size * math.log(oracle["nobs"]),
                    rtol=1e-11)
    assert "pseudo_r_squared" not in metrics                  # Stata's biprobit reports none
    # rho as Stata displays it: tanh(athrho), delta-method SE, tanh of the athrho interval.
    athrho, se = theta[-1], math.sqrt(covariance[-1, -1])
    z = stats.norm.ppf(1 - alpha / 2)
    record = result.extra["rho"]
    assert_allclose(metrics["rho"], math.tanh(athrho), rtol=1e-7)
    assert_allclose(record["estimate"], math.tanh(athrho), rtol=1e-7)
    assert_allclose(record["std_error"], (1 - math.tanh(athrho) ** 2) * se, rtol=5e-6)
    assert_allclose([record["ci_low"], record["ci_high"]],
                    [math.tanh(athrho - z * se), math.tanh(athrho + z * se)], rtol=5e-6)
    check_chi2(result.tests["model"], wald(theta, covariance, slopes), len(slopes), rtol=1e-5)
    if kind in ("nonrobust", "opg"):
        check_chi2(result.tests["rho"], 2 * (oracle["ll"] - oracle["comparison"]), 1)
        assert "LR" in result.tests["rho"]["label"]
    else:
        check_chi2(result.tests["rho"], (athrho / se) ** 2, 1, rtol=1e-5)
        assert "Wald" in result.tests["rho"]["label"]
    assert_allclose(result.extra["comparison_log_likelihood"], oracle["comparison"], rtol=1e-10)


def test_biprobit_against_brute_force_likelihood(workers):
    names = ["age", "kids"]
    x = design(workers, names, constant=True)
    oracle = biprobit_oracle(workers, x, x)
    for kind in KINDS:
        result = oe.biprobit(data=workers, y1="union", y2="insured", x=names,
                             covariance=None if kind == "cluster" else kind,
                             cluster="firm" if kind == "cluster" else None, alpha=0.1)
        covariance = stata_covariance(kind, oracle["hessian"], oracle["scores"],
                                      nobs=len(workers), clusters=workers.firm)
        check_biprobit(result, oracle, covariance, kind, [1, 2, 4, 5], alpha=0.1)
        assert terms(result) == ["union:Intercept", "union:age", "union:kids",
                                 "insured:Intercept", "insured:age", "insured:kids", "/athrho"]
        assert [row.equation for row in result.coefficients] == ["union"] * 3 + ["insured"] * 3 \
            + [None]
        assert result.title == "Bivariate probit regression" and result.nobs == len(workers)
    counts = result.extra["outcome_counts"]
    for a in (0, 1):
        for b in (0, 1):
            assert counts[f"{a}{b}"] == ((workers.union == a) & (workers.insured == b)).sum()


def test_seemingly_unrelated_biprobit_with_categoricals(workers):
    dummies = {"sector": ["a", "b", "c"]}
    x1 = design(workers, ["age", "kids"], constant=True)
    x2 = design(workers, ["age", "educ"], constant=True, dummies=dummies)
    oracle = biprobit_oracle(workers, x1, x2)
    for kind in ("nonrobust", "robust"):
        result = oe.biprobit(data=workers, y1="union", y2="insured", x=["age", "kids"],
                             x2=["age", "educ", "sector"], categorical=["sector"],
                             covariance=kind)
        covariance = stata_covariance(kind, oracle["hessian"], oracle["scores"],
                                      nobs=len(workers))
        check_biprobit(result, oracle, covariance, kind, [1, 2, 4, 5, 6, 7])
        assert terms(result) == ["union:Intercept", "union:age", "union:kids",
                                 "insured:Intercept", "insured:age", "insured:educ",
                                 "insured:sector[b]", "insured:sector[c]", "/athrho"]
        assert result.title == "Seemingly unrelated bivariate probit"


@pytest.mark.parametrize("weight_type", ["fweight", "aweight", "iweight", "pweight"])
def test_biprobit_weights_against_brute_force(workers, weight_type):
    x1 = design(workers, ["age"], constant=True)
    x2 = design(workers, ["educ"], constant=True)
    oracle = biprobit_oracle(workers, x1, x2, weight_type=weight_type)
    kinds = ("robust", "cluster") if weight_type == "pweight" else KINDS
    for kind in kinds:
        result = oe.biprobit(data=workers, y1="union", y2="insured", x=["age"], x2=["educ"],
                             weights=WEIGHT_COLUMN[weight_type], weight_type=weight_type,
                             covariance=None if kind == "cluster" else kind,
                             cluster="firm" if kind == "cluster" else None)
        covariance = stata_covariance(kind, oracle["hessian"], oracle["scores"],
                                      nobs=oracle["nobs"], frequency=oracle["frequency"],
                                      opg_weights=oracle["score_weights"],
                                      clusters=workers.firm)
        check_biprobit(result, oracle, covariance, kind, [1, 3])
        assert result.nobs == oracle["nobs"]
    if weight_type == "fweight":
        duplicated = workers.loc[workers.index.repeat(workers.fw.astype(int))].reset_index(
            drop=True)
        for kind in KINDS:
            options = {"covariance": None if kind == "cluster" else kind,
                       "cluster": "firm" if kind == "cluster" else None}
            same_fit(oe.biprobit(data=workers, y1="union", y2="insured", x=["age"], x2=["educ"],
                                 weights="fw", weight_type="fweight", **options),
                     oe.biprobit(data=duplicated, y1="union", y2="insured", x=["age"],
                                 x2=["educ"], **options), rtol=1e-8)
    if weight_type == "pweight":
        default = oe.biprobit(data=workers, y1="union", y2="insured", x=["age"], x2=["educ"],
                              weights="pw", weight_type="pweight")
        assert default.spec.covariance == "robust"


def test_biprobit_invariances_and_equivalences(workers):
    names = ["age", "kids"]
    base = oe.biprobit(data=workers, y1="union", y2="insured", x=names, x2=["age", "educ"],
                       cluster="firm")
    shuffled = workers.sample(frac=1.0, random_state=9).reset_index(drop=True)
    same_fit(base, oe.biprobit(data=shuffled, y1="union", y2="insured", x=names,
                               x2=["age", "educ"], cluster="firm"), rtol=1e-8)
    # Swapping the equations permutes the parameters; rho is unchanged.
    swapped = oe.biprobit(data=workers, y1="insured", y2="union", x=["age", "educ"], x2=names,
                          cluster="firm")
    order = [3, 4, 5, 0, 1, 2, 6]
    assert_allclose(table(swapped)["estimate"][order], table(base)["estimate"], rtol=1e-7)
    assert_allclose(np.array(swapped.covariance_matrix)[np.ix_(order, order)],
                    np.array(base.covariance_matrix), rtol=1e-6)
    assert_allclose(swapped.metrics["log_likelihood"], base.metrics["log_likelihood"],
                    rtol=1e-11)
    # Reversing one outcome flips its equation and the sign of rho.
    mirrored = oe.biprobit(data=workers.assign(insured=1 - workers.insured), y1="union",
                           y2="insured", x=names, x2=["age", "educ"], cluster="firm")
    sign = np.array([1.0, 1, 1, -1, -1, -1, -1])
    assert_allclose(table(mirrored)["estimate"], table(base)["estimate"] * sign, rtol=1e-7)
    assert_allclose(table(mirrored)["std_error"], table(base)["std_error"], rtol=1e-6)
    assert_allclose(mirrored.metrics["rho"], -base.metrics["rho"], rtol=1e-7)
    assert_allclose(mirrored.extra["rho"]["ci_low"], -base.extra["rho"]["ci_high"], rtol=1e-6)
    # Rescaled regressors.
    scaled = oe.biprobit(data=workers.assign(age=workers.age * 1e7, educ=workers.educ * 1e-6),
                         y1="union", y2="insured", x=names, x2=["age", "educ"], cluster="firm")
    factor = np.array([1.0, 1e-7, 1.0, 1.0, 1e-7, 1e6, 1.0])
    assert_allclose(table(scaled)["estimate"], table(base)["estimate"] * factor, rtol=1e-6)
    assert_allclose(table(scaled)["std_error"], table(base)["std_error"] * factor, rtol=1e-6)
    # The comparison model: two statsmodels probits.
    plain = oe.biprobit(data=workers, y1="union", y2="insured", x=names, x2=["age", "educ"])
    one = sm.Probit(workers.union, sm.add_constant(workers[names])).fit(disp=False, tol=1e-13)
    two = sm.Probit(workers.insured, sm.add_constant(workers[["age", "educ"]])).fit(
        disp=False, tol=1e-13)
    assert_allclose(plain.extra["probit_log_likelihoods"], [one.llf, two.llf], rtol=1e-11)
    assert_allclose(plain.tests["rho"]["statistic"],
                    2 * (plain.metrics["log_likelihood"] - one.llf - two.llf), rtol=1e-8)
    # Independent errors by construction: rho is near zero and the equations are the probits.
    rng = np.random.default_rng(1)
    independent = workers.assign(coin=(rng.random(len(workers)) < 0.45) * 1.0)
    result = oe.biprobit(data=independent, y1="union", y2="coin", x=names, x2=["educ"])
    assert abs(result.metrics["rho"]) < 0.15
    assert_allclose(table(result)["estimate"][:3], one.params, atol=0.02)


# ---- analytic derivatives away from the maximum -----------------------------------------


def test_kernel_derivatives_against_independent_likelihoods(matched, workers):
    import torch

    from openecon.econometrics.discrete.bivariate import BiprobitObjective
    from openecon.econometrics.discrete.conditional import ConditionalLogitObjective
    from openecon.econometrics.discrete.kernels import HetprobitObjective
    from test_econ_discrete_oracle import hessian_of

    def compare(objective, point, loglik, weights, rows=True, slack=1.0):
        def total(p):
            return float((weights * loglik(p)).sum())

        value, gradient, hessian = objective(torch.from_numpy(point))
        assert_allclose(float(value), total(point), rtol=1e-11 * slack)
        assert_allclose(float(objective.value(torch.from_numpy(point))), total(point),
                        rtol=1e-11 * slack)
        assert_allclose(gradient.numpy(), jacobian(total, point), rtol=1e-7 * slack,
                        atol=1e-7 * slack)
        assert_allclose(hessian.numpy(), hessian_of(total, point), rtol=2e-6 * slack,
                        atol=2e-6 * slack)
        if rows:
            assert_allclose(objective.score_rows(torch.from_numpy(point)).numpy(),
                            jacobian(loglik, point), rtol=1e-6 * slack, atol=1e-8 * slack)

    # Conditional logit with group weights and an offset, groups in arbitrary row order.
    names = ["x1", "x2", "x3"]
    model = Enumerated(matched, names, offset="dose")
    kept = matched.iloc[model.kept].reset_index(drop=True)
    inner = Enumerated(kept, names, offset="dose")
    labels, codes = np.unique(kept["set"].to_numpy(), return_inverse=True)
    group_weights = np.array([kept.pw.to_numpy()[rows[0]] for rows in inner.members])
    objective = ConditionalLogitObjective(
        torch.from_numpy(inner.x), torch.from_numpy(kept.y.to_numpy()), torch.from_numpy(codes),
        len(labels), torch.from_numpy(group_weights), torch.from_numpy(kept.dose.to_numpy()))
    point = np.array([0.5, -0.4, 0.3])
    compare(objective, point, inner.group_loglik, group_weights, rows=False)
    assert_allclose(objective.group_scores(torch.from_numpy(point)).numpy(),
                    jacobian(inner.group_loglik, point), rtol=1e-7, atol=1e-9)
    assert_allclose(objective.group_log_likelihood(torch.from_numpy(point)).numpy(),
                    inner.group_loglik(point), rtol=1e-11)
    # Per-row scores (y_i - Pr(y_i = 1 | m_g)) x_i: here x is passed uncentred.
    raw = inner.row_scores(point) + (kept.y.to_numpy() - _chance(inner, point))[:, None] \
        * kept[names].groupby(kept["set"]).transform("mean").to_numpy()
    assert_allclose(objective.score_rows(torch.from_numpy(point)).numpy(), raw, atol=1e-10)

    # Heteroskedastic probit and bivariate probit with weights.
    y = workers.work.to_numpy()
    x, z = design(workers, ["age", "kids"], constant=True), design(workers, ["urban", "tenure"])
    weights = workers.aw.to_numpy()
    tensors = [torch.from_numpy(a) for a in (x, z, y, weights)]
    compare(HetprobitObjective(*tensors), np.array([0.2, 0.6, -0.3, 0.4, -0.2]),
            lambda p: hetprobit_loglik(p, x, z, y), weights)
    y1, y2 = workers.union.to_numpy(), workers.insured.to_numpy()
    x2 = design(workers, ["educ"], constant=True)
    tensors = [torch.from_numpy(a) for a in (x, x2, y1, y2, weights)]
    # Owen's formula (the oracle) cancels for joint probabilities near 1e-9, where the
    # kernel keeps its relative accuracy (checked against 40-digit quadrature above), so
    # the comparison of this sum of logarithms is looser by that margin.
    for athrho in (0.4, -1.1, 2.2):
        compare(BiprobitObjective(*tensors), np.array([0.1, 0.5, -0.2, -0.2, 0.6, athrho]),
                lambda p: biprobit_loglik(p, x, x2, y1, y2), weights, slack=200.0)


def _chance(model, theta):
    """Pr(y_i = 1 | m_g) by enumeration, for every row of the informative groups."""
    index = model.x @ theta + model.shift
    chance = np.zeros(len(model.x))
    for rows, subsets in zip(model.members, model.sets, strict=True):
        share = special.softmax(index[subsets].sum(axis=1))
        for row in rows:
            chance[row] = share[(subsets == row).any(axis=1)].sum()
    return chance
