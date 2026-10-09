"""Independent development oracles for strict full/top-prefix ranking models.

SciPy/NumPy are test-only. Designs, reduced risk sets, likelihood derivatives,
Gumbel samples and permutation sums below do not use the native estimator.
Primary mathematical references:
https://www.stata.com/manuals15/rrologit.pdf
https://hturner.github.io/PlackettLuce/articles/Overview.html
https://eml.berkeley.edu/books/choice2nd/Ch03_p34-75.pdf
"""
from __future__ import annotations

from itertools import permutations
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError


ALTERNATIVES = ("A", "B", "C", "D")
PARAMETERS = ("x1", "x2", "asc_B", "asc_C", "asc_D")
TRUE = np.array([.55, -.35, -.4, .25, .1])


@pytest.fixture(scope="module", autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def ranking_fixture(*, seed=548, cases=180, prefix=False, availability=False,
                    repeated=False, common_shocks=False):
    """Sort iid Gumbel random utilities, without drawing sequential logit choices."""
    rng = np.random.default_rng(seed)
    rows, shocks = [], {}
    for case in range(cases):
        respondent = case // 3 if repeated else case
        x = rng.normal(size=(4, 2))
        x[:, 0] += np.arange(4)*.12
        available = np.ones(4, dtype=bool)
        if availability and case % 4 == 0:
            available[(case//4) % 4] = False
        design = np.column_stack((x, np.eye(4)[:, 1:]))
        if common_shocks:
            error = shocks.setdefault(respondent, rng.gumbel(size=4))
        else:
            error = rng.gumbel(size=4)
        ordering = np.flatnonzero(available)[np.argsort(-(design@TRUE+error)[available])]
        retained = 1 + case % len(ordering) if prefix else len(ordering)
        ranks = np.zeros(4)
        ranks[ordering[:retained]] = np.arange(1, retained+1)
        for j, alt in enumerate(ALTERNATIVES):
            rows.append({"case": case, "respondent": respondent, "alternative": alt,
                         "rank": ranks[j], "available": int(available[j]), "x1": x[j, 0], "x2": x[j, 1],
                         **{"asc_"+other: int(alt == other) for other in ALTERNATIVES[1:]}})
    return pd.DataFrame(rows)


def case_designs(frame):
    result = []
    for case, group in frame.groupby("case", sort=True):
        group = group.loc[group.available == 1].copy()
        x = np.column_stack((group[["x1", "x2"]].to_numpy(),
                             *[(group.alternative == alt).to_numpy() for alt in ALTERNATIVES[1:]]))
        observed = np.flatnonzero(group["rank"].to_numpy() > 0)
        order = observed[np.argsort(group["rank"].to_numpy()[observed])]
        result.append((case, group.respondent.iloc[0], x, order))
    return result


def independent_likelihood(beta, cases):
    """Return complete case log likelihood/scores and observed information."""
    scores = np.zeros((len(cases), len(beta)))
    likelihood = np.zeros(len(cases))
    information = np.zeros((len(beta), len(beta)))
    for case_index, (_, _, x, order) in enumerate(cases):
        remaining = list(range(len(x)))
        for chosen in order:
            risk = x[remaining]
            eta = risk@beta
            logden = logsumexp(eta)
            probability = np.exp(eta-logden)
            average = probability@risk
            centered = risk-average
            likelihood[case_index] += x[chosen]@beta-logden
            scores[case_index] += x[chosen]-average
            information += centered.T@(probability[:, None]*centered)
            remaining.remove(chosen)
    return likelihood, scores, information


def scipy_oracle(frame, covariance):
    cases = case_designs(frame)
    def objective(beta):
        likelihood, scores, _ = independent_likelihood(beta, cases)
        return -likelihood.sum(), -scores.sum(axis=0)
    fit = minimize(objective, np.zeros(5), jac=True, method="BFGS",
                   options={"gtol": 5e-9, "maxiter": 2000})
    likelihood, scores, information = independent_likelihood(fit.x, cases)
    assert np.max(np.abs(scores.sum(axis=0))) < 3e-6, fit.message
    bread = np.linalg.inv(information)
    cluster_scores = np.empty((0, len(fit.x)))
    aggregated = scores
    if covariance == "CR0":
        respondents = np.array([row[1] for row in cases])
        cluster_scores = np.array([scores[respondents == person].sum(axis=0) for person in np.unique(respondents)])
        aggregated = cluster_scores
    meat = aggregated.T@aggregated
    if covariance == "OIM":
        cov = bread
    else:
        cov = bread@meat@bread
    return dict(beta=fit.x, likelihood=likelihood, scores=scores,
                cluster_scores=cluster_scores, information=information, bread=bread,
                meat=meat, covariance=cov)


def stage_oracle(frame, beta):
    scores, likelihood, probabilities, log_probabilities, jacobians, metadata = [], [], [], [], [], []
    X = np.column_stack((frame[["x1", "x2"]].to_numpy(),
                         *[(frame.alternative == alt).to_numpy() for alt in ALTERNATIVES[1:]]))
    cases, available, ranks = frame.case.to_numpy(), frame.available.to_numpy(), frame["rank"].to_numpy()
    labels = frame.alternative.tolist()
    for case in np.unique(cases):
        remaining = np.flatnonzero((cases == case) & (available == 1)).tolist()
        observed = [row for row in remaining if ranks[row] > 0]
        order = sorted(observed, key=lambda row: ranks[row])
        for stage, chosen in enumerate(order, 1):
            x = X[remaining]
            eta = x@beta
            lp = eta-logsumexp(eta)
            p = np.exp(lp)
            j = remaining.index(chosen)
            scores.append(x[j]-p@x)
            likelihood.append(lp[j])
            probabilities.extend(p)
            log_probabilities.extend(lp)
            jacobians.extend(p[:, None]*(x-p@x))
            metadata.extend((case, stage, row, labels[row], int(row == chosen)) for row in remaining)
            remaining.remove(chosen)
    return dict(scores=np.array(scores), likelihood=np.array(likelihood), probabilities=np.array(probabilities),
                log_probabilities=np.array(log_probabilities), jacobian=np.array(jacobians), metadata=metadata)


def prefix_probability(beta, x, prefix):
    remaining = list(range(len(x)))
    log_probability = 0.
    for chosen in prefix:
        log_probability += x[chosen]@beta-logsumexp(x[remaining]@beta)
        remaining.remove(chosen)
    return np.exp(log_probability)


def prefix_jacobian(beta, x, prefix):
    remaining = list(range(len(x)))
    derivative = np.zeros_like(beta)
    for chosen in prefix:
        eta = x[remaining]@beta
        p = np.exp(eta-logsumexp(eta))
        derivative += x[chosen]-p@x[remaining]
        remaining.remove(chosen)
    return prefix_probability(beta, x, prefix)*derivative


def five_point_jacobian(function, beta):
    columns = []
    for j in range(len(beta)):
        h = 2e-4*max(1., abs(beta[j]))
        step = np.zeros_like(beta)
        step[j] = h
        columns.append((-np.asarray(function(beta+2*step))+8*np.asarray(function(beta+step))
                        -8*np.asarray(function(beta-step))+np.asarray(function(beta-2*step)))/(12*h))
    return np.stack(columns, axis=-1)


def fit_public(frame, covariance="oim"):
    return oe.rologit(frame, "rank", list(PARAMETERS), case="case", alternative="alternative",
                      available="available", vce=covariance,
                      cluster="respondent" if covariance == "cr0" else None)


def full_matrix(result, key):
    return result[key].set_index("parameter").loc[list(PARAMETERS), list(PARAMETERS)].to_numpy(float)


def canonical_result(result):
    return {"attrs": result.attrs, "latex": result.to_latex(),
            "tables": {key: result[key].to_dict(orient="split") for key in result.keys()}}


@pytest.mark.parametrize("seed", [545, 546, 547])
@pytest.mark.parametrize("prefix,availability", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("vce", ["oim", "hc0", "cr0"])
def test_full_likelihood_information_and_all_covariance_entries(seed, prefix, availability, vce):
    frame = ranking_fixture(seed=seed, cases=150, prefix=prefix, availability=availability,
                            repeated=True, common_shocks=vce == "cr0")
    reference = scipy_oracle(frame, {"oim": "OIM", "hc0": "HC0", "cr0": "CR0"}[vce])
    result = fit_public(frame, vce)
    parameters = result["parameters"].set_index("parameter").loc[list(PARAMETERS)]
    np.testing.assert_allclose(parameters["estimate"], reference["beta"], rtol=2e-6, atol=3e-7)
    for key in ("information", "bread", "meat", "covariance"):
        np.testing.assert_allclose(full_matrix(result, key), reference[key], rtol=5e-6, atol=5e-8)
    np.testing.assert_allclose(result["case_scores"][list(PARAMETERS)].to_numpy(float), reference["scores"], rtol=4e-6, atol=3e-6)
    np.testing.assert_allclose(result["cluster_scores"][list(PARAMETERS)].to_numpy(float), reference["cluster_scores"], rtol=4e-6, atol=3e-6)
    np.testing.assert_allclose(result["case_likelihood"].log_likelihood, reference["likelihood"], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result["case_likelihood"].ranking_probability, np.exp(reference["likelihood"]), rtol=2e-6, atol=2e-7)
    stage = stage_oracle(frame, reference["beta"])
    np.testing.assert_allclose(result["stage_scores"][list(PARAMETERS)], stage["scores"], rtol=4e-6, atol=3e-6)
    np.testing.assert_allclose(result["stages"].log_likelihood, stage["likelihood"], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result["probabilities"].probability, stage["probabilities"], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(result["probabilities"].log_probability, stage["log_probabilities"], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result["probability_jacobian"][list(PARAMETERS)], stage["jacobian"], rtol=4e-6, atol=3e-6)
    assert list(result["probabilities"][["case", "stage", "row", "alternative", "selected"]].itertuples(index=False, name=None)) == stage["metadata"]
    assert len(result["inputs"]) == len(frame)
    np.testing.assert_array_equal(result["inputs"][list(PARAMETERS)], frame[list(PARAMETERS)])
    np.testing.assert_allclose(parameters["std_error"], np.sqrt(reference["covariance"].diagonal()), rtol=3e-6, atol=3e-8)
    z = reference["beta"]/np.sqrt(reference["covariance"].diagonal())
    np.testing.assert_allclose(parameters["z"], z, rtol=3e-6, atol=2e-7)
    np.testing.assert_allclose(parameters["p_value"], 2*norm.sf(np.abs(z)), rtol=1e-5, atol=3e-8)
    np.testing.assert_allclose(parameters["ci_lower"], reference["beta"]-norm.ppf(.975)*np.sqrt(reference["covariance"].diagonal()), atol=4e-7)
    np.testing.assert_allclose(parameters["ci_upper"], reference["beta"]+norm.ppf(.975)*np.sqrt(reference["covariance"].diagonal()), atol=4e-7)


def test_lower_rank_information_changes_fit_beyond_top_one():
    frame = ranking_fixture(seed=917, cases=120)
    top1 = frame.copy()
    top1.loc[top1["rank"] > 1, "rank"] = 0
    full_ref, top_ref = scipy_oracle(frame, "OIM"), scipy_oracle(top1, "OIM")
    assert np.linalg.norm(full_ref["beta"]-top_ref["beta"]) > .05
    full, top = fit_public(frame), fit_public(top1)
    np.testing.assert_allclose(full["parameters"]["estimate"], full_ref["beta"], atol=3e-7)
    np.testing.assert_allclose(top["parameters"]["estimate"], top_ref["beta"], atol=3e-7)
    assert np.linalg.norm(full_matrix(full, "information")-full_matrix(top, "information")) > 10


def test_reordering_case_shifts_and_unavailable_values_do_not_change_fit():
    frame = ranking_fixture(seed=913, cases=120, prefix=True, availability=True, repeated=True)
    original = fit_public(frame, "cr0")
    changed = frame.sample(frac=1, random_state=76).copy()
    # Add arbitrary case-level utility constants to every risk set, then alter
    # unavailable attributes that must never enter a denominator/score.
    changed["x1"] += np.sin(changed.case)*11
    changed["x2"] += np.cos(changed.case)*7
    changed.loc[changed.available == 0, "x1"] = 87
    changed.loc[changed.available == 0, "x2"] = -63
    shifted = fit_public(changed, "cr0")
    np.testing.assert_allclose(shifted["parameters"]["estimate"], original["parameters"]["estimate"], rtol=2e-6, atol=2e-7)
    for key in ("information", "bread", "covariance"):
        np.testing.assert_allclose(full_matrix(shifted, key), full_matrix(original, key), rtol=3e-6, atol=2e-7)


def test_canonical_json_restore_rebuilds_all_tables_and_intervals_without_fit(monkeypatch):
    result = fit_public(ranking_fixture(seed=940, cases=100, prefix=True), "hc0")
    original = canonical_result(result)
    attrs = json.loads(json.dumps(result.attrs, sort_keys=True, allow_nan=False))
    # Restore must evaluate saved sufficient state rather than optimize again.
    import openecon.econometrics.discrete.rank_ordered as native
    def forbidden(*args, **kwargs):
        raise AssertionError("Restore unexpectedly refitted")
    monkeypatch.setattr(native, "rologit", forbidden)
    monkeypatch.setattr(native, "_fit", forbidden)
    restored = oe.rologit_restore(attrs)
    assert canonical_result(restored) == original
    lower_level = oe.rologit_restore(attrs, level=.8)
    np.testing.assert_array_equal(lower_level["parameters"]["estimate"], result["parameters"]["estimate"])
    np.testing.assert_array_equal(full_matrix(lower_level, "covariance"), full_matrix(result, "covariance"))
    p = lower_level["parameters"]
    np.testing.assert_allclose(p.ci_upper-p.estimate, norm.ppf(.9)*p.std_error, rtol=2e-12, atol=2e-12)


def test_cpu_float64_is_independent_of_torch_global_defaults():
    frame = ranking_fixture(seed=939, cases=90, prefix=True)
    expected = fit_public(frame, "hc0")
    expected_prediction = oe.rologit_predict(expected, targets=[(0, "A"), (0, "B")])
    expected_margin = oe.rologit_margins(expected, targets=[(1, "A", "B", "x1")])
    dtype = torch.get_default_dtype()
    device = torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        result = fit_public(frame, "hc0")
        prediction = oe.rologit_predict(result, targets=[(0, "A"), (0, "B")])
        margin = oe.rologit_margins(result, targets=[(1, "A", "B", "x1")])
    finally:
        torch.set_default_dtype(dtype)
        torch.set_default_device(device)
    for key in ("information", "bread", "covariance"):
        np.testing.assert_array_equal(full_matrix(result, key), full_matrix(expected, key))
    np.testing.assert_array_equal(result["parameters"]["estimate"], expected["parameters"]["estimate"])
    for observed, reference in ((prediction, expected_prediction), (margin, expected_margin)):
        for key in reference.keys():
            pd.testing.assert_frame_equal(observed[key], reference[key], check_exact=True)


def test_mutated_and_resealed_nonstationary_fit_is_refused():
    fitted = fit_public(ranking_fixture(seed=930, cases=100))
    attrs = json.loads(json.dumps(fitted.attrs, allow_nan=False))
    state = attrs["rank_ordered_state"]
    state["fit"]["beta"][0] += .3
    with pytest.raises(AnalysisError):
        oe.rologit_restore(attrs)
    # A checksum is an integrity marker, not a certificate of actual maximum
    # likelihood stationarity. Recomputing it must not admit fabricated fits.
    state["checksum"] = hashlib.sha256(json.dumps(
        {key: value for key, value in state.items() if key != "checksum"}, sort_keys=True,
        ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(AnalysisError):
        oe.rologit_restore(attrs)


def query_permutations(*, prefix_length=4):
    x = np.random.default_rng(1848).normal(size=(4, 2))
    design = np.column_stack((x, np.eye(4)[:, 1:]))
    rows, orders = [], tuple(permutations(range(4)))
    for case, order in enumerate(orders):
        rank = np.zeros(4)
        rank[list(order[:prefix_length])] = np.arange(1, prefix_length+1)
        for j, alt in enumerate(ALTERNATIVES):
            rows.append({"case": case, "alternative": alt, "rank": rank[j], "available": 1,
                         **dict(zip(PARAMETERS, design[j]))})
    return pd.DataFrame(rows), design, orders


def target_matrix(result, key, names):
    return result[key].set_index("target").loc[names, names].to_numpy(float)


def test_selected_first_choice_targets_keep_full_denominator_and_transformed_intervals():
    fitted = fit_public(ranking_fixture(seed=929, cases=120), "hc0")
    beta, covariance = fitted["parameters"]["estimate"].to_numpy(), full_matrix(fitted, "covariance")
    frame, x, _ = query_permutations()
    frame = frame.loc[frame.case == 0].drop(columns="rank")
    result = oe.rologit_predict(fitted, data=frame, targets=[(0, "A"), (0, "D")])
    p = np.exp(x@beta-logsumexp(x@beta))
    selected = [0, 3]
    J = p[selected, None]*(x[selected]-p@x)
    LJ = x[selected]-p@x
    OJ = LJ/(1-p[selected, None])
    predictions = result["predictions"]
    names = predictions.target.tolist()
    np.testing.assert_allclose(predictions.probability, p[selected], rtol=2e-12)
    assert predictions.riskset_size.tolist() == [4, 4]
    np.testing.assert_allclose(result["jacobian"][list(PARAMETERS)], J, rtol=3e-11, atol=2e-12)
    np.testing.assert_allclose(result["log_probability_jacobian"][list(PARAMETERS)], LJ, rtol=3e-11, atol=2e-12)
    np.testing.assert_allclose(result["log_odds_jacobian"][list(PARAMETERS)], OJ, rtol=3e-11, atol=2e-12)
    odds = np.log(p[selected]/(1-p[selected]))
    odds_se = np.sqrt(np.diag(OJ@covariance@OJ.T))
    lower = 1/(1+np.exp(-(odds-norm.ppf(.975)*odds_se)))
    upper = 1/(1+np.exp(-(odds+norm.ppf(.975)*odds_se)))
    np.testing.assert_allclose(predictions.ci_lower, lower, rtol=2e-10)
    np.testing.assert_allclose(predictions.ci_upper, upper, rtol=2e-10)
    joint_names = names + ["log_"+name for name in names] + ["log_odds_"+name for name in names]
    stacked = np.vstack((J, LJ, OJ))
    np.testing.assert_allclose(target_matrix(result, "joint_covariance", joint_names), stacked@covariance@stacked.T,
                               rtol=2e-9, atol=3e-12)


@pytest.mark.parametrize("prefix_length", [1, 2, 3, 4])
def test_public_complete_rank_permutation_mass_partial_prefix_and_joint_delta(prefix_length):
    fitted = fit_public(ranking_fixture(seed=931, cases=150, prefix=True), "hc0")
    beta = fitted["parameters"]["estimate"].to_numpy()
    covariance = full_matrix(fitted, "covariance")
    data, x, orders = query_permutations(prefix_length=prefix_length)
    targets = [{"case": case, "kind": "ranking"} for case in range(24)]
    result = oe.rologit_predict(fitted, data=data, mode="stages", targets=targets)
    expected = np.array([prefix_probability(beta, x, order[:prefix_length]) for order in orders])
    J = np.array([prefix_jacobian(beta, x, order[:prefix_length]) for order in orders])
    numeric = five_point_jacobian(lambda b: [prefix_probability(b, x, order[:prefix_length]) for order in orders], beta)
    np.testing.assert_allclose(J, numeric, rtol=3e-8, atol=2e-10)
    predictions = result["predictions"]
    names = predictions.target.tolist()
    np.testing.assert_allclose(predictions.probability, expected, rtol=3e-12, atol=3e-14)
    np.testing.assert_allclose(predictions.log_probability, np.log(expected), atol=3e-12)
    np.testing.assert_allclose(result["jacobian"][list(PARAMETERS)], J, rtol=3e-10, atol=3e-12)
    LJ = J/expected[:, None]
    np.testing.assert_allclose(result["log_probability_jacobian"][list(PARAMETERS)], LJ, rtol=3e-10, atol=3e-12)
    np.testing.assert_allclose(target_matrix(result, "covariance", names), J@covariance@J.T, rtol=3e-9, atol=3e-12)
    np.testing.assert_allclose(target_matrix(result, "log_probability_covariance", names), LJ@covariance@LJ.T, rtol=3e-9, atol=3e-12)
    # Both full cross-target matrices and their probability/log-probability
    # cross-block are recoverable from the persisted common beta covariance.
    public_J = np.vstack((result["jacobian"][list(PARAMETERS)], result["log_probability_jacobian"][list(PARAMETERS)]))
    full = public_J@full_matrix(result, "coefficient_covariance")@public_J.T
    expected_full = np.vstack((J, LJ))@covariance@np.vstack((J, LJ)).T
    np.testing.assert_allclose(full, expected_full, rtol=3e-9, atol=3e-12)
    joint_names = names + ["log_"+name for name in names]
    np.testing.assert_allclose(target_matrix(result, "joint_covariance", joint_names), expected_full,
                               rtol=3e-9, atol=3e-12)
    np.testing.assert_allclose(result["joint_jacobian"][list(PARAMETERS)], np.vstack((J, LJ)),
                               rtol=3e-10, atol=3e-12)
    np.testing.assert_allclose(predictions.standard_error, np.sqrt((J@covariance@J.T).diagonal()), rtol=2e-9)
    log_se = np.sqrt((LJ@covariance@LJ.T).diagonal())
    np.testing.assert_allclose(predictions.log_standard_error, log_se, rtol=2e-9)
    np.testing.assert_allclose(predictions.ci_lower, np.exp(np.log(expected)-norm.ppf(.975)*log_se), rtol=3e-9)
    np.testing.assert_allclose(predictions.ci_upper, np.exp(np.minimum(0., np.log(expected)+norm.ppf(.975)*log_se)), rtol=3e-9)
    if prefix_length == 4:
        np.testing.assert_allclose(expected.sum(), 1., atol=5e-14)
        np.testing.assert_allclose(J.sum(axis=0), 0., atol=5e-14)
        np.testing.assert_allclose(target_matrix(result, "covariance", names).sum(axis=0), 0., atol=1e-13)
    else:
        all_probabilities = np.array([prefix_probability(beta, x, order) for order in orders])
        for j, order in enumerate(orders):
            compatible = [k for k, complete in enumerate(orders) if complete[:prefix_length] == order[:prefix_length]]
            np.testing.assert_allclose(expected[j], all_probabilities[compatible].sum(), atol=4e-15)


def margin_risksets(frame):
    result = {}
    for case, group in frame.groupby("case", sort=True):
        group = group.loc[group.available == 1]
        x, labels = group[list(PARAMETERS)].to_numpy(), group.alternative.tolist()
        rank = group["rank"].to_numpy()
        observed = np.flatnonzero(rank > 0)
        order = observed[np.argsort(rank[observed])]
        remaining = list(range(len(group)))
        for stage, chosen in enumerate(order, 1):
            result[(case, stage)] = (x[remaining], [labels[j] for j in remaining])
            remaining.remove(chosen)
    return result


def margin_oracle(beta, frame, targets, *, kind, weights, risksets=None):
    risksets = margin_risksets(frame) if risksets is None else risksets
    values, per_case, support = [], [], []
    for stage, outcome, changed, variable in targets:
        members, member_weights = [], []
        for case in sorted(frame.case.unique()):
            if (case, stage) not in risksets:
                continue
            x, labels = risksets[(case, stage)]
            if outcome not in labels or changed not in labels:
                continue
            p = np.exp(x@beta-logsumexp(x@beta))
            j, k, r = labels.index(outcome), labels.index(changed), PARAMETERS.index(variable)
            contrast = float(j == k)-p[k]
            value = beta[r]*contrast*(p[j] if kind == "effect" else x[k, r])
            members.append(value)
            member_weights.append(weights[case])
            per_case.append(value)
            support.append(case)
        assert members
        values.append(np.average(members, weights=member_weights))
    assert risksets
    return np.array(values), np.array(per_case), support


@pytest.mark.parametrize("kind", ["effect", "elasticity"])
@pytest.mark.parametrize("mode", ["first", "stages"])
def test_fixed_support_weighted_own_cross_margins_full_covariance_and_case_factorization(kind, mode):
    frame = ranking_fixture(seed=948, cases=110, prefix=True, availability=True, repeated=True)
    fitted = fit_public(frame, "cr0")
    beta, V = fitted["parameters"]["estimate"].to_numpy(), full_matrix(fitted, "covariance")
    stage = 1 if mode == "first" else 2
    targets = [(stage, j, k, v) for j, k in (("A", "A"), ("A", "B"), ("C", "D"), ("D", "D")) for v in ("x1", "x2")]
    weights = {int(case): 1.+case % 5 for case in frame.case.unique()}
    query = frame.copy()
    if kind == "elasticity":
        # Log-attribute elasticity has a declared positive-attribute domain.
        # A common shift preserves every risk-set probability while making
        # this a valid new-data derivative target without refitting.
        for variable in ("x1", "x2"):
            query[variable] += 1-query[variable].min()
    result = oe.rologit_margins(fitted, data=query, mode=mode, targets=targets, kind=kind, case_weights=weights)
    risksets = margin_risksets(query)
    values, individual, support = margin_oracle(beta, query, targets, kind=kind, weights=weights, risksets=risksets)
    J = five_point_jacobian(lambda b: margin_oracle(b, query, targets, kind=kind, weights=weights, risksets=risksets)[0], beta)
    CJ = five_point_jacobian(lambda b: margin_oracle(b, query, targets, kind=kind, weights=weights, risksets=risksets)[1], beta)
    margins = result["margins"]
    np.testing.assert_allclose(margins.estimate, values, rtol=3e-11, atol=3e-12)
    np.testing.assert_allclose(result["per_case"].estimate, individual, rtol=3e-11, atol=3e-12)
    assert result["per_case"]["case"].tolist() == support
    np.testing.assert_allclose(result["jacobian"][list(PARAMETERS)], J, rtol=3e-8, atol=3e-10)
    np.testing.assert_allclose(result["per_case_jacobian"][list(PARAMETERS)], CJ, rtol=3e-8, atol=3e-10)
    names = margins.target.tolist()
    np.testing.assert_allclose(target_matrix(result, "covariance", names), J@V@J.T, rtol=2e-7, atol=3e-11)
    public_CJ = result["per_case_jacobian"][list(PARAMETERS)].to_numpy()
    full_case_covariance = public_CJ@full_matrix(result, "coefficient_covariance")@public_CJ.T
    np.testing.assert_allclose(full_case_covariance, CJ@V@CJ.T, rtol=2e-7, atol=3e-11)
    np.testing.assert_allclose(result["per_case"].standard_error, np.sqrt(np.diag(full_case_covariance)), rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(margins.standard_error, np.sqrt(np.diag(J@V@J.T)), rtol=2e-7)


def test_independent_permutation_and_prefix_probability_identity():
    x = np.random.default_rng(548).normal(size=(4, 5))
    full = tuple(permutations(range(4)))
    probabilities = np.array([prefix_probability(TRUE, x, order) for order in full])
    np.testing.assert_allclose(probabilities.sum(), 1., atol=5e-15)
    for prefix in ((0,), (1, 3), (2, 0, 3)):
        compatible = [j for j, order in enumerate(full) if order[:len(prefix)] == prefix]
        np.testing.assert_allclose(probabilities[compatible].sum(), prefix_probability(TRUE, x, prefix), atol=5e-15)
        np.testing.assert_allclose(prefix_jacobian(TRUE, x, prefix),
                                   five_point_jacobian(lambda b: prefix_probability(b, x, prefix), TRUE),
                                   rtol=3e-8, atol=2e-10)


def test_independent_gumbel_utilities_match_complete_rank_probability_law():
    _, x, orders = query_permutations()
    simulations = 120_000
    utility = x@TRUE + np.random.default_rng(5_480_000).gumbel(size=(simulations, 4))
    order = np.argsort(-utility, axis=1)
    codes = order@np.array([64, 16, 4, 1])
    observed = np.array([(codes == np.dot(permutation, [64, 16, 4, 1])).mean() for permutation in orders])
    expected = np.array([prefix_probability(TRUE, x, permutation) for permutation in orders])
    allowance = 5*np.sqrt(expected*(1-expected)/simulations)+1/simulations
    assert np.all(np.abs(observed-expected) <= allowance)
    assert np.abs(observed.sum()-1) < 1e-12


def test_independent_likelihood_score_information_finite_differences():
    cases = case_designs(ranking_fixture(cases=18, prefix=True, availability=True))
    likelihood, score, information = independent_likelihood(TRUE, cases)
    numeric_score = five_point_jacobian(lambda b: independent_likelihood(b, cases)[0], TRUE)
    np.testing.assert_allclose(score, numeric_score, rtol=2e-8, atol=2e-9)
    numeric_hessian = -five_point_jacobian(lambda b: independent_likelihood(b, cases)[1].sum(axis=0), TRUE)
    np.testing.assert_allclose(information, numeric_hessian, rtol=2e-8, atol=2e-8)
    assert np.isfinite(likelihood).all() and np.linalg.eigvalsh(information).min() > 0
