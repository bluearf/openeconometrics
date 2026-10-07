"""Conditional (fixed-effects) logit against independent oracles.

The oracle likelihood enumerates every placement of the positives within a
group (itertools.combinations), which shares nothing with the recursive
algorithm under test. statsmodels' ConditionalLogit supplies a second
reference for estimates and conventional standard errors; scores for the
robust estimators are complex-step derivatives of the enumerated likelihood.
"""

import itertools
import json
import time
import warnings

import mpmath
import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import optimize as sopt
from scipy import stats
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.tools.numdiff import approx_fprime, approx_fprime_cs

import openecon as oe
from openecon.analysis import AnalysisError, fit
from openecon.econometrics.discrete.conditional import ConditionalLogitObjective
from openecon.engines.optimize import check_derivatives
from openecon.models import ModelSpec, ResultBundle

X = ["x1", "x2"]


def make_groups(seed=20263, groups=90, low=2, high=7):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(low, high + 1, size=groups)
    ids = np.repeat(np.arange(groups), sizes)
    n = len(ids)
    frame = pd.DataFrame({"id": ids, "x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n)})
    effect = rng.normal(size=groups)[ids]
    index = effect + 0.9 * frame.x1 - 0.6 * frame.x2
    frame["y"] = (rng.random(n) < 1 / (1 + np.exp(-index))) * 1.0
    frame["level"] = rng.normal(size=groups)[ids]           # constant within group
    frame["town"] = ids % 15                                 # nests the groups
    frame["wave"] = rng.integers(0, 12, size=n)              # does not nest the groups
    frame["fw"] = rng.integers(1, 4, size=groups)[ids].astype(float)
    frame["pw"] = rng.uniform(0.5, 2.5, size=groups)[ids]
    frame["off"] = rng.normal(scale=0.3, size=n)
    return frame.sample(frac=1.0, random_state=7).reset_index(drop=True)     # unsorted rows


@pytest.fixture(scope="module")
def data():
    return make_groups()


def informative(frame, y="y", group="id"):
    share = frame.groupby(group)[y].transform("mean")
    return frame[(share > 0) & (share < 1)].reset_index(drop=True)


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def covariance(result):
    return np.array(result.covariance_matrix)


def enumerate_groups(theta, x, y, codes, offset=None):
    """Conditional log likelihood per group and Pr(y_i = 1 | group total), by enumeration."""
    labels = np.unique(codes)
    log_likelihood = np.zeros(len(labels), dtype=np.result_type(theta, float))
    chance = np.zeros(len(y), dtype=log_likelihood.dtype)
    for position, label in enumerate(labels):
        rows = np.flatnonzero(codes == label)
        index = x[rows] @ theta + (0 if offset is None else offset[rows])
        positives = int(y[rows].sum())
        subsets = np.array(list(itertools.combinations(range(len(rows)), positives)))
        terms = np.exp(index[subsets].sum(axis=1))
        total = terms.sum()
        log_likelihood[position] = (y[rows] * index).sum() - np.log(total)
        member = np.zeros((len(subsets), len(rows)))
        member[np.arange(len(subsets))[:, None], subsets] = 1
        chance[rows] = member.T @ terms / total
    return log_likelihood, chance


def oracle(frame, theta, columns=X, weights=None, offset=None):
    """Enumerated log likelihood, Hessian, group scores and per-observation scores."""
    x, y, codes = frame[columns].to_numpy(), frame.y.to_numpy(), frame.id.to_numpy()
    off = None if offset is None else frame[offset].to_numpy()
    labels = np.unique(codes)
    w = np.ones(len(labels)) if weights is None else frame.groupby("id")[weights].first().loc[labels].to_numpy()

    def group_ll(t):
        return enumerate_groups(t, x, y, codes, off)[0]

    hessian = approx_fprime(theta, lambda t: approx_fprime_cs(t, lambda u: (w * group_ll(u)).sum()),
                            centered=True)
    group_scores = approx_fprime_cs(theta, group_ll)
    log_likelihood, chance = enumerate_groups(theta, x, y, codes, off)
    # A group's score is divided among its rows on within-group centred regressors, the
    # split that does not depend on the origin of x (identical sums when clusters nest groups).
    centred = x - (frame[columns].groupby(frame.id).transform("mean")).to_numpy()
    row_scores = centred * (y - chance)[:, None]
    return {"ll": float((w * log_likelihood).sum()), "hessian": (hessian + hessian.T) / 2,
            "group_scores": group_scores, "row_scores": row_scores, "weights": w, "labels": labels}


def test_clogit_matches_statsmodels_and_the_enumerated_likelihood(data):
    result = oe.clogit(data=data, y="y", x=X, group="id")
    used = informative(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = ConditionalLogit(data.y, data[X], groups=data.id).fit(disp=0, method="newton", tol=1e-13)
    assert [c.term for c in result.coefficients] == X
    assert_allclose(estimates(result), ref.params.values, rtol=1e-8)
    assert_allclose(errors(result), ref.bse.values, rtol=1e-6)
    assert_allclose(result.metrics["log_likelihood"], ref.llf, rtol=1e-11)
    truth = oracle(used, estimates(result))
    assert_allclose(result.metrics["log_likelihood"], truth["ll"], rtol=1e-12)
    assert_allclose(covariance(result), np.linalg.inv(-truth["hessian"]), rtol=1e-6)
    brute = sopt.minimize(lambda t: -enumerate_groups(t, used[X].to_numpy(), used.y.to_numpy(),
                                                      used.id.to_numpy())[0].sum(),
                          np.zeros(2), method="BFGS", options={"gtol": 1e-9})
    assert_allclose(estimates(result), brute.x, rtol=2e-5, atol=2e-6)
    # Sample accounting, as Stata reports it.
    groups_all, groups_used = data.id.nunique(), used.id.nunique()
    assert result.nobs == len(used) and result.nobs_original == len(data)
    assert result.metrics["n_groups"] == groups_used
    assert result.metrics["n_groups_dropped"] == groups_all - groups_used > 0
    assert result.extra["observations_dropped"] == len(data) - len(used) == result.dropped_rows
    note = f"note: {groups_all - groups_used} group(s) ({len(data) - len(used)} obs) dropped"
    assert any(note in w and "all positive or all negative outcomes" in w for w in result.warnings)
    assert sorted(result.sample_positions) == sorted(
        np.flatnonzero(data.id.isin(used.id.unique()).to_numpy()).tolist())
    # Null model b = 0: every placement of the positives is equally likely.
    sizes = used.groupby("id").y.agg(["size", "sum"])
    null = -sum(np.log(float(mpmath.binomial(int(t), int(m)))) for t, m in sizes.to_numpy())
    assert_allclose(result.extra["null_log_likelihood"], null, rtol=1e-12)
    assert_allclose(result.metrics["pseudo_r_squared"], 1 - truth["ll"] / null, rtol=1e-10)
    assert_allclose(result.metrics["aic"], -2 * truth["ll"] + 4, rtol=1e-12)
    assert_allclose(result.metrics["bic"], -2 * truth["ll"] + 2 * np.log(len(used)), rtol=1e-12)
    model = result.tests["model"]
    assert model["df"] == 2 and "LR" in model["label"]
    assert_allclose(model["statistic"], 2 * (truth["ll"] - null), rtol=1e-9)
    assert_allclose(model["p_value"], stats.chi2.sf(model["statistic"], 2), rtol=1e-8)
    assert result.extra["multiple_positive_outcomes"] is True and result.extra["group"] == "id"
    assert result.extra["group_sizes"]["max"] == sizes["size"].max()
    assert result.inference["use_t"] is False and result.predictions == []
    # Row order and group labels are irrelevant.
    relabelled = data.sort_values(["id", "x1"]).assign(id=lambda f: "g" + f.id.astype(str))
    assert_allclose(estimates(oe.clogit(data=relabelled, y="y", x=X, group="id")), estimates(result),
                    rtol=1e-10)


def test_clogit_covariances_treat_the_group_as_the_unit(data):
    used = informative(data)
    theta = estimates(oe.clogit(data=data, y="y", x=X, group="id"))
    truth = oracle(used, theta)
    bread = np.linalg.inv(-truth["hessian"])
    scores = truth["group_scores"]
    g = len(scores)
    opg = oe.clogit(data=data, y="y", x=X, group="id", covariance="opg")
    assert_allclose(covariance(opg), np.linalg.inv(scores.T @ scores), rtol=1e-6)
    assert opg.inference["correction"] == "outer product of the group scores"
    robust = oe.clogit(data=data, y="y", x=X, group="id", covariance="robust")
    assert_allclose(covariance(robust), g / (g - 1) * bread @ scores.T @ scores @ bread, rtol=1e-6)
    assert robust.inference["cluster_count"] == g and "G/(G-1)" in robust.inference["correction"]
    assert_allclose(robust.inference["small_sample_correction"], g / (g - 1))
    wald = theta @ np.linalg.solve(covariance(robust), theta)
    assert "Wald" in robust.tests["model"]["label"]
    assert_allclose(robust.tests["model"]["statistic"], wald, rtol=1e-9)
    # vce(robust) of clogit is vce(cluster group): the per-observation scores of the
    # backward pass summed within groups reproduce the group scores.
    by_group = oe.clogit(data=data, y="y", x=X, group="id", cluster="id")
    assert_allclose(covariance(by_group), covariance(robust), rtol=1e-10)
    rows = truth["row_scores"]
    for column in ("town", "wave"):             # nesting and non-nesting clusters
        labels, index = np.unique(used[column].to_numpy(), return_inverse=True)
        sums = np.zeros((len(labels), 2))
        np.add.at(sums, index, rows)
        expected = len(labels) / (len(labels) - 1) * bread @ sums.T @ sums @ bread
        result = oe.clogit(data=data, y="y", x=X, group="id", cluster=column)
        assert result.spec.covariance == "cluster"
        assert_allclose(covariance(result), expected, rtol=1e-6)
        assert result.inference["cluster_count"] == len(labels)
    twoway = oe.clogit(data=data, y="y", x=X, group="id", cluster=["town", "wave"])
    assert twoway.inference["cluster_columns"] == ["town", "wave"] and np.isfinite(errors(twoway)).all()


def test_clogit_weights_apply_to_whole_groups(data):
    used = informative(data)
    pieces = []
    for copy in range(3):
        part = data[data.fw > copy].copy()
        part["id"] = part.id * 10 + copy                 # every replicate is its own group
        pieces.append(part)
    repeated = pd.concat(pieces, ignore_index=True)
    for kind in ("nonrobust", "opg", "robust", "cluster"):
        extra = {"covariance": kind, "cluster": "town" if kind == "cluster" else None}
        weighted = oe.clogit(data=data, y="y", x=X, group="id", weights="fw", weight_type="fweight",
                             **extra)
        duplicated = oe.clogit(data=repeated, y="y", x=X, group="id", **extra)
        assert weighted.nobs == duplicated.nobs == int(used.fw.sum())
        assert weighted.metrics["n_groups"] == duplicated.metrics["n_groups"]
        assert_allclose(estimates(weighted), estimates(duplicated), rtol=1e-9)
        assert_allclose(covariance(weighted), covariance(duplicated), rtol=1e-8)
        for name in ("log_likelihood", "pseudo_r_squared", "aic", "bic"):
            assert_allclose(weighted.metrics[name], duplicated.metrics[name], rtol=1e-10)
    # iweights scale each group's contribution; pweights add the group-clustered sandwich.
    importance = oe.clogit(data=data, y="y", x=X, group="id", weights="pw", weight_type="iweight")
    truth = oracle(used, estimates(importance), weights="pw")
    assert_allclose(importance.metrics["log_likelihood"], truth["ll"], rtol=1e-12)
    bread = np.linalg.inv(-truth["hessian"])
    assert_allclose(covariance(importance), bread, rtol=1e-6)
    gradient = (truth["group_scores"] * truth["weights"][:, None]).sum(axis=0)
    assert_allclose(gradient, 0, atol=1e-7)
    sampling = oe.clogit(data=data, y="y", x=X, group="id", weights="pw", weight_type="pweight")
    assert sampling.spec.covariance == "robust"
    assert_allclose(estimates(sampling), estimates(importance), rtol=1e-12)
    weighted_scores = truth["group_scores"] * truth["weights"][:, None]
    g = len(weighted_scores)
    assert_allclose(covariance(sampling), g / (g - 1) * bread @ weighted_scores.T @ weighted_scores @ bread,
                    rtol=1e-6)
    uneven = data.assign(pw=data.pw + (data.index % 2) * 0.1)
    with pytest.raises(AnalysisError) as caught:
        oe.clogit(data=uneven, y="y", x=X, group="id", weights="pw", weight_type="iweight")
    assert caught.value.code == "weights_not_constant_within_group"
    with pytest.raises(AnalysisError) as caught:
        oe.clogit(data=data, y="y", x=X, group="id", weights="pw", weight_type="aweight")
    assert caught.value.code == "invalid_spec" and "aweight" in str(caught.value)
    with pytest.raises(AnalysisError) as caught:
        fit(ModelSpec(estimator="clogit", outcome="y", predictors=X, intercept=False, weights="pw",
                      weight_type="pweight", covariance="opg", columns={"group": "id"}), data=data)
    assert caught.value.code == "unsupported_covariance"


def test_clogit_offset_omitted_regressors_and_missing_values(data):
    used = informative(data)
    result = oe.clogit(data=data, y="y", x=X, group="id", offset="off")
    truth = oracle(used, estimates(result), offset="off")
    assert_allclose(result.metrics["log_likelihood"], truth["ll"], rtol=1e-12)
    assert_allclose(truth["group_scores"].sum(axis=0), 0, atol=1e-7)
    assert_allclose(covariance(result), np.linalg.inv(-truth["hessian"]), rtol=1e-6)
    zero = enumerate_groups(np.zeros(2), used[X].to_numpy(), used.y.to_numpy(), used.id.to_numpy(),
                            used.off.to_numpy())[0].sum()
    assert_allclose(result.extra["null_log_likelihood"], zero, rtol=1e-12)
    # No within-group variation, a constant and an exact combination are not identified.
    frame = data.assign(one=1.0, twin=data.x1 + data.x2)
    omitted = oe.clogit(data=frame, y="y", x=["x1", "level", "one", "x2", "twin"], group="id")
    assert [c.term for c in omitted.coefficients] == X
    assert set(omitted.provenance["omitted_terms"]) == {"level", "one", "twin"}
    assert_allclose(estimates(omitted), estimates(oe.clogit(data=data, y="y", x=X, group="id")), rtol=1e-10)
    with pytest.raises(AnalysisError) as caught:
        oe.clogit(data=data, y="y", x=["level"], group="id")
    assert caught.value.code == "no_within_group_variation"
    expanded = oe.clogit(data=data.assign(band=np.where(data.x2 > 0, "hi", "lo")), y="y",
                         x=["x1", "band"], categorical=["band"], group="id")
    assert [c.term for c in expanded.coefficients] == ["x1", "band[lo]"]
    holes = data.copy()
    holes.loc[[2, 11], "x1"] = np.nan
    holes.loc[5, "id"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.clogit(data=holes, y="y", x=X, group="id")
    assert caught.value.code == "missing_values"
    dropped = oe.clogit(data=holes, y="y", x=X, group="id", missing="drop")
    complete = oe.clogit(data=holes.dropna().reset_index(drop=True), y="y", x=X, group="id")
    assert_allclose(estimates(dropped), estimates(complete), rtol=1e-12)
    assert dropped.nobs == complete.nobs and 2 not in dropped.sample_positions


def test_clogit_one_positive_per_group_is_the_choice_model():
    rng = np.random.default_rng(5)
    sets, alternatives = 150, 4
    frame = pd.DataFrame({"id": np.repeat(np.arange(sets), alternatives),
                          "x1": rng.normal(size=sets * alternatives),
                          "x2": rng.normal(size=sets * alternatives)})
    utility = (0.7 * frame.x1 - 0.4 * frame.x2 + rng.gumbel(size=len(frame))).to_numpy()
    chosen = utility.reshape(sets, alternatives).argmax(axis=1)
    frame["y"] = 0.0
    frame.loc[np.arange(sets) * alternatives + chosen, "y"] = 1.0
    result = oe.clogit(data=frame, y="y", x=X, group="id")
    x = frame[X].to_numpy().reshape(sets, alternatives, 2)

    def negative(theta):
        index = x @ theta
        return -(index[np.arange(sets), chosen] - special_logsumexp(index)).sum()

    def special_logsumexp(index):
        top = index.max(axis=1)
        return top + np.log(np.exp(index - top[:, None]).sum(axis=1))

    brute = sopt.minimize(negative, np.zeros(2), method="BFGS", options={"gtol": 1e-10})
    assert_allclose(estimates(result), brute.x, rtol=1e-5, atol=1e-6)
    assert_allclose(result.metrics["log_likelihood"], -brute.fun, rtol=1e-10)
    assert_allclose(result.extra["null_log_likelihood"], -sets * np.log(alternatives), rtol=1e-12)
    assert result.extra["multiple_positive_outcomes"] is False and result.metrics["n_groups_dropped"] == 0
    # Reflecting the outcome (one negative per group) flips the sign of b.
    mirrored = oe.clogit(data=frame.assign(y=1 - frame.y), y="y", x=X, group="id")
    assert_allclose(estimates(mirrored), -estimates(result), rtol=1e-9)
    assert_allclose(errors(mirrored), errors(result), rtol=1e-9)
    assert_allclose(mirrored.metrics["log_likelihood"], result.metrics["log_likelihood"], rtol=1e-12)


def test_conditional_likelihood_derivatives_scores_and_large_groups(data):
    used = informative(data)
    labels, codes = np.unique(used.id.to_numpy(), return_inverse=True)
    weights = torch.tensor(used.groupby("id").pw.first().loc[labels].to_numpy())
    objective = ConditionalLogitObjective(
        torch.tensor(used[X].to_numpy()), torch.tensor(used.y.to_numpy()), torch.tensor(codes),
        len(labels), weights, torch.tensor(used.off.to_numpy()))
    assert objective.blocks and len(objective.single_groups)        # both code paths are exercised
    theta = torch.tensor([0.4, -0.7], dtype=torch.float64)
    report = check_derivatives(objective, theta)
    assert report["gradient_max_rel_error"] < 1e-8 and report["hessian_max_rel_error"] < 1e-8
    value, gradient, hessian = objective(theta)
    log_likelihood, chance = enumerate_groups(theta.numpy(), used[X].to_numpy(), used.y.to_numpy(),
                                              used.id.to_numpy(), used.off.to_numpy())
    assert_allclose(float(value), (weights.numpy() * log_likelihood).sum(), rtol=1e-12)
    assert_allclose(float(objective.value(theta)), float(value), rtol=1e-14)
    # codes follow np.unique order here, so groups line up with the enumeration.
    assert_allclose(objective.group_log_likelihood(theta).numpy(), log_likelihood, rtol=1e-10, atol=1e-12)
    rows = objective.score_rows(theta).numpy()
    assert_allclose(rows, used[X].to_numpy() * (used.y.to_numpy() - chance)[:, None], atol=1e-12)
    sums = np.zeros((len(labels), 2))
    np.add.at(sums, codes, rows)
    assert_allclose(objective.group_scores(theta).numpy(), sums, atol=1e-12)
    assert_allclose((sums * weights.numpy()[:, None]).sum(axis=0), gradient.numpy(), atol=1e-10)
    assert float(torch.linalg.eigvalsh(hessian)[-1]) < 0
    # A group of 400 rows with 180 positives and a wide index range: the elementary
    # symmetric function overflows float64 by hundreds of orders of magnitude, the
    # normalized recursion does not. Reference: exact recursion in 60-digit arithmetic.
    rng = np.random.default_rng(9)
    size, positives = 400, 180
    x = rng.normal(scale=4.0, size=(size, 1))
    y = np.zeros(size)
    y[rng.choice(size, positives, replace=False)] = 1
    big = ConditionalLogitObjective(torch.tensor(x), torch.tensor(y), torch.zeros(size, dtype=torch.int64),
                                    1, torch.ones(1, dtype=torch.float64))
    beta = 3.0
    mpmath.mp.dps = 60
    poly = [mpmath.mpf(1)] + [mpmath.mpf(0)] * positives
    for value_ in (x[:, 0] * beta).tolist():
        u = mpmath.exp(value_)
        for j in range(positives, 0, -1):
            poly[j] += u * poly[j - 1]
    exact = float(mpmath.mpf(float((y * x[:, 0] * beta).sum())) - mpmath.log(poly[positives]))
    value, gradient, hessian = big(torch.tensor([beta], dtype=torch.float64))
    assert_allclose(float(value), exact, rtol=1e-10)
    assert torch.isfinite(gradient).all() and float(hessian[0, 0]) < 0
    report = check_derivatives(big, torch.tensor([0.2], dtype=torch.float64))
    assert report["gradient_max_rel_error"] < 1e-7 and report["hessian_max_rel_error"] < 1e-7
    with pytest.raises(Exception) as caught:
        ConditionalLogitObjective(torch.tensor(x), torch.ones(size, dtype=torch.float64),
                                  torch.zeros(size, dtype=torch.int64), 1, torch.ones(1, dtype=torch.float64))
    assert getattr(caught.value, "code", None) == "no_outcome_variation"


def test_clogit_separation_and_error_codes(data):
    separated = data.assign(flag=data.y)
    with pytest.raises(AnalysisError) as caught:
        oe.clogit(data=separated, y="y", x=["x1", "flag"], group="id")
    assert caught.value.code == "separation_detected" and "group(s)" in str(caught.value)
    with pytest.raises(AnalysisError) as caught:
        oe.clogit(data=data.assign(y=1.0), y="y", x=X, group="id")
    assert caught.value.code == "no_outcome_variation"
    with pytest.raises(AnalysisError) as caught:
        oe.clogit(data=data.assign(y=data.y * 2), y="y", x=X, group="id")
    assert caught.value.code == "invalid_binary_outcome"
    with pytest.raises(AnalysisError) as caught:
        oe.clogit(data=data, y="y", x=X, group="x1")
    assert caught.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as caught:
        oe.clogit(data=data, y="y", x=X, group="nope")
    assert caught.value.code == "missing_columns"
    with pytest.raises(TypeError):
        oe.clogit(data=data, y="y", x=X)                              # group is required
    for arguments in [{"intercept": True, "columns": {"group": "id"}}, {"intercept": False},
                      {"intercept": False, "columns": {"group": ["id", "town"]}},
                      {"intercept": False, "columns": {"group": "id"}, "panel": "id"}]:
        with pytest.raises(ValidationError):
            ModelSpec(estimator="clogit", outcome="y", predictors=X, **arguments)


def test_clogit_round_trip_summary_and_latex(data):
    result = oe.clogit(data=data, y="y", x=X, group="id", covariance="robust")
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result and isinstance(restored, ResultBundle)
    assert fit(result.spec, data=data).spec == result.spec
    payload = json.loads(result.model_dump_json())
    assert payload["spec"]["columns"] == {"group": "id"} and payload["extra"]["group"] == "id"
    text = result.summary()
    assert "Conditional (fixed-effects) logistic regression — y" in text and "Covariance: robust" in text
    assert "n_groups:" in text and "n_groups_dropped:" in text and "dropped because of all positive" in text
    assert "Wald chi2 test of the coefficients: chi2(2)" in text and "  z  " in text
    assert "x1" in str(result.to_latex())
    assert result.provenance["stata_equivalent"] == ["clogit", "xtlogit, fe"]
    assert result.provenance["optimizer"]["converged"] is True
    assert callable(oe.clogit) and "clogit" in dir(oe)


def test_clogit_scales_to_many_groups():
    rng = np.random.default_rng(13)
    groups, periods, k = 40_000, 8, 6
    ids = np.repeat(np.arange(groups), periods)
    x = rng.normal(size=(groups * periods, k))
    beta = np.linspace(-0.5, 0.5, k)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(k)])
    frame["id"] = ids
    frame["y"] = (rng.random(len(ids)) < 1 / (1 + np.exp(-(rng.normal(size=groups)[ids] + x @ beta)))) * 1.0
    start = time.perf_counter()
    result = oe.clogit(data=frame, y="y", x=list(frame.columns[:k]), group="id", covariance="robust")
    elapsed = time.perf_counter() - start
    assert elapsed < 30 and result.metrics["n_groups"] > 30_000
    assert_allclose(estimates(result), beta, atol=0.03)
