"""Independent raw least-squares, complete bootstrap-law and replay oracles."""

import copy
import itertools
import math
import random

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget


@pytest.fixture
def study():
    # Deliberately nonalphabetic attribute order survives canonical JSON sorting.
    attributes = {"zbrand": ["a", "b"], "price": [10, 20, 30, 40], "aquality": ["basic", "premium"]}
    full = oe.conjoint_plan(attributes)["profiles"]
    plan = pd.concat([full]*3, ignore_index=True)
    plan.profile_id = [f"card-{i}" for i in range(len(plan))]
    x = np.column_stack([np.ones(len(plan)), np.where(plan.zbrand == "a", 1., -1.),
                         plan.price.to_numpy(float), plan.price.to_numpy(float)**2,
                         np.where(plan.aquality == "basic", 1., -1.)])
    rng = np.random.default_rng(805812)
    data = []
    subjects = [17, "17", "z", 5.5]
    for i, subject in enumerate(subjects):
        beta = [16+i*.2, .7+i*.5, -.02+i*.003, -.001+i*.0001, .3-i*.15]
        y = x@beta+rng.normal(size=len(x))*(.1+.005*plan.price.to_numpy(float))
        data.extend([subject, card, float(value), j//3] for j, (card, value) in enumerate(zip(plan.profile_id, y)))
    responses = pd.DataFrame(data, columns=["subject", "profile_id", "score", "session"],
                             index=[f"source-{i}" for i in range(len(data))])
    fit = oe.conjoint_fit(plan, responses, attributes, factors={"price":"ideal"})
    raw_map = np.eye(5).tolist()
    raw_map += [[0, 1, 0, 0, 0], [0, -1, 0, 0, 0]]
    raw_map += [[0, 0, v, v*v, 0] for v in attributes["price"]]
    raw_map += [[0, 0, 0, 0, 1], [0, 0, 0, 0, -1]]
    return fit, plan, responses, x, np.array(raw_map), attributes


def assert_moments(rows, covariance, values, point, level=.9):
    np.testing.assert_allclose(rows.estimate, point, rtol=2e-8, atol=2e-10)
    expected = np.cov(values, rowvar=False, ddof=1)
    np.testing.assert_allclose(covariance, expected, rtol=3e-8, atol=2e-10)
    np.testing.assert_allclose(rows.std_error, np.sqrt(np.diag(expected)), rtol=3e-8, atol=2e-10)
    np.testing.assert_allclose(rows.bootstrap_mean, values.mean(0), rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(rows.bias, values.mean(0)-point, rtol=3e-7, atol=2e-10)
    limits = np.quantile(values, [(1-level)/2, (1+level)/2], axis=0, method="linear")
    np.testing.assert_allclose(rows.ci_low, limits[0], rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(rows.ci_high, limits[1], rtol=2e-8, atol=2e-10)
    assert "p_value" not in rows and "df" not in rows


@pytest.mark.parametrize("method", ["residual", "rademacher", "mammen", "pairs"])
def test_every_refit_full_joint_target_covariance_and_percentile_matches_raw_numpy(study, method):
    fitted, _, responses, x, mapping, _ = study
    restored = oe.conjoint_load(oe.conjoint_save(fitted))
    output = oe.conjoint_bootstrap_fit(restored, method=method, replications=39, seed=812, level=.9)
    n, p = x.shape
    h = np.einsum("ij,jk,ik->i", x, np.linalg.inv(x.T@x), x)
    rng = random.Random(812)
    for person, record in enumerate(output.attrs["state"]["subjects"]):
        original = responses.iloc[person*n:(person+1)*n]
        y = original.score.to_numpy(float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        prediction, residual = x@beta, y-x@beta
        if method == "residual":
            errors = (residual-residual.mean())*np.sqrt(n/(n-p))
        elif method != "pairs":
            errors = residual/np.sqrt(1-h)
        expected = []
        for draw in record["draws"]:
            if method in ("pairs", "residual"):
                assert draw == [rng.randrange(n) for _ in range(n)]
            elif method == "rademacher":
                assert draw == [1. if rng.random() < .5 else -1. for _ in range(n)]
            else:
                root = np.sqrt(5)
                np.testing.assert_array_equal(draw, [(1-root)/2 if rng.random() < (root+1)/(2*root) else (1+root)/2
                                                     for _ in range(n)])
            if method == "pairs":
                coefficient = np.linalg.lstsq(x[draw], y[draw], rcond=None)[0]
            else:
                response = prediction+(errors[draw] if method == "residual" else errors*np.array(draw))
                coefficient = np.linalg.lstsq(x, response, rcond=None)[0]
            expected.append(mapping@coefficient)
        expected = np.array(expected)
        start, end = person*len(mapping), (person+1)*len(mapping)
        rows = output["estimates"].iloc[start:end]
        cov = output["covariance"].iloc[start:end, 2:].to_numpy(float)
        replicas = output["replicates"].iloc[person*39:(person+1)*39, 2:].to_numpy(float)
        np.testing.assert_allclose(replicas, expected, rtol=2e-8, atol=2e-10)
        assert_moments(rows, cov, expected, mapping@beta)
        assert record["subject"] == original.subject.iloc[0]
        assert record["source_labels"] == original.index.tolist()
        assert record["source_positions"] == list(range(person*n, (person+1)*n))
        np.testing.assert_allclose(record["observed"], y)
    assert len(output["replicates"]) == 4*39
    assert oe.conjoint_save(oe.conjoint_load(oe.conjoint_save(output))) == oe.conjoint_save(output)
    assert oe.conjoint_load(oe.conjoint_save(output)).to_latex() == output.to_latex()


def individual_targets(study, target, temperature=1., tolerance=0.):
    _, plan, responses, x, mapping, _ = study
    parameters = np.stack([np.linalg.lstsq(x, g.score.to_numpy(float), rcond=None)[0]
                           for _, g in responses.groupby("subject", sort=False)])
    if target == "importance":
        utilities = parameters@mapping[5:].T
        spans = np.column_stack([np.ptp(utilities[:,:2], axis=1), np.ptp(utilities[:,2:6], axis=1),
                                  np.ptp(utilities[:,6:], axis=1)])
        return 100*spans/spans.sum(1)[:, None]
    scores = parameters@x.T
    if target == "first_choice":
        winners = scores.max(1)[:, None]-scores <= tolerance
        return winners/winners.sum(1)[:, None]
    if target == "btl":
        return scores/scores.sum(1)[:, None]
    exponential = np.exp((scores-scores.max(1)[:, None])/temperature)
    return exponential/exponential.sum(1)[:, None]


@pytest.mark.parametrize("target", ["importance", "first_choice", "btl", "logit"])
def test_whole_respondent_targets_all_draws_full_covariance_percentiles_and_typed_identity(study, target):
    fitted, plan, _, _, _, _ = study
    options = {"temperature":.7} if target == "logit" else {}
    output = (oe.conjoint_bootstrap_importance(fitted, replications=39, seed=812, level=.9)
              if target == "importance" else oe.conjoint_bootstrap_shares(
                  fitted, plan, method=target, replications=39, seed=812, level=.9, **options))
    expected_individual = individual_targets(study, target, temperature=.7 if target == "logit" else 1)
    np.testing.assert_allclose(output["individual"].iloc[:,1:].to_numpy(float), expected_individual, rtol=2e-8, atol=1e-11)
    draws = output.attrs["state"]["draw_subject_positions"]
    rng = random.Random(812)
    assert draws == [[rng.randrange(4) for _ in range(4)] for _ in range(39)]
    assert any(len(set(d)) < 4 for d in draws)
    expected = np.array([expected_individual[draw].mean(0) for draw in draws])
    np.testing.assert_allclose(output["replicates"].iloc[:,1:].to_numpy(float), expected, rtol=2e-8, atol=1e-11)
    assert_moments(output["estimates"], output["covariance"].iloc[:,1:].to_numpy(float),
                   expected, expected_individual.mean(0))
    assert output.attrs["state"]["subjects"] == [17, "17", "z", 5.5]
    total = 100 if target == "importance" else 1
    np.testing.assert_allclose(expected.sum(1), total, atol=1e-12)
    np.testing.assert_allclose(output["covariance"].iloc[:,1:].to_numpy(float).sum(0), 0, atol=2e-10)
    restored = oe.conjoint_load(oe.conjoint_save(output))
    assert restored.to_latex() == output.to_latex()
    assert oe.conjoint_save(restored) == oe.conjoint_save(output)


def test_full_random_law_moments_and_exact_enumerated_respondent_bootstrap_variance(study):
    root = math.sqrt(5)
    values, probs = np.array([(1-root)/2, (1+root)/2]), np.array([(root+1)/(2*root), (root-1)/(2*root)])
    np.testing.assert_allclose([np.sum(probs*values**k) for k in (1,2,3)], [0,1,1], atol=1e-15)
    individual = individual_targets(study, "importance")[:3]
    exact = np.array([individual[list(draw)].mean(0) for draw in itertools.product(range(3), repeat=3)])
    centered = individual-individual.mean(0)
    # Conditional nonparametric variance uses divisor m², not the unbiased sampling divisor m(m-1).
    np.testing.assert_allclose(np.cov(exact, rowvar=False, ddof=0), centered.T@centered/9, atol=1e-12)


@pytest.mark.parametrize("method", ["residual", "rademacher", "mammen", "pairs"])
def test_clustered_fits_are_not_silently_given_an_independent_profile_law(study, method):
    _, plan, responses, _, _, attributes = study
    fitted = oe.conjoint_fit(plan, responses, attributes, factors={"price":"ideal"}, covariance="cr1", cluster="session")
    with pytest.raises(AnalysisError) as error:
        oe.conjoint_bootstrap_fit(fitted, method=method, replications=20)
    assert error.value.code == "unsupported_dependence"
    assert len(oe.conjoint_bootstrap_importance(fitted, replications=20)["replicates"]) == 20


@pytest.mark.parametrize("method", ["rademacher", "mammen"])
def test_unit_leverage_is_refused_without_changing_the_profile_sample(method):
    plan = pd.DataFrame([["u",0,0],["v",0,0],["w",1,0],["z",0,1]], columns=["profile_id", "a", "b"])
    response = pd.DataFrame({"subject":[1]*4, "profile_id":plan.profile_id, "score":[0.,1.,2.,3.]})
    fitted = oe.conjoint_fit(plan, response, {"a":[0,1], "b":[0,1]})
    with pytest.raises(AnalysisError) as error:
        oe.conjoint_bootstrap_fit(fitted, method=method, replications=20)
    assert error.value.code == "unit_leverage"


def test_unidentified_pairs_replica_fails_the_whole_operation_without_redraw():
    plan = oe.conjoint_plan({"a":[0,1], "b":[0,1]})
    response = pd.DataFrame({"subject":[1]*4, "profile_id":plan["profiles"].profile_id, "score":[0.,1.,2.,3.]})
    fitted = oe.conjoint_fit(plan, response)
    # Seed zero's first draw is [3,3,0,2]: a subsequent actual draw loses rank.
    with pytest.raises(AnalysisError) as error:
        oe.conjoint_bootstrap_fit(fitted, method="pairs", replications=20, seed=0)
    assert error.value.code == "unidentified_replica"
    assert "replica" in str(error.value)


@pytest.mark.parametrize("option", [{"replications":19}, {"replications":2000}, {"replications":True},
    {"seed":-1}, {"seed":2**32}, {"seed":False}, {"level":0}, {"level":1}, {"level":float("nan")}])
def test_invalid_monte_carlo_options_fail_explicitly(study, option):
    with pytest.raises(AnalysisError) as error:
        oe.conjoint_bootstrap_fit(study[0], **option)
    assert error.value.code == "invalid_option"


@pytest.mark.parametrize("procedure", ["fit", "importance", "shares"])
def test_work_memory_device_weights_integrity_and_seed_controls(study, procedure):
    fitted, plan, _, _, _, _ = study
    def call(result=fitted, **options):
        common = {"replications":39, **options}
        if procedure == "fit":
            return oe.conjoint_bootstrap_fit(result, **common)
        if procedure == "importance":
            return oe.conjoint_bootstrap_importance(result, **common)
        return oe.conjoint_bootstrap_shares(result, plan, method="logit", **common)
    for option, code in [({"max_work":1},"work_limit"), ({"device":"mps"},"unsupported_device"),
                         ({"weights":[1]},"unsupported_weights")]:
        with pytest.raises(AnalysisError) as error:
            call(**option)
        assert error.value.code == code
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        call(replications=1999)
    assert error.value.code == "workspace_limit"
    corrupted = copy.deepcopy(fitted)
    corrupted.attrs["state"]["subjects"][0]["parameters_scaled"][0] += 1
    with pytest.raises(AnalysisError) as error:
        call(corrupted)
    assert error.value.code == "invalid_result"
    first, second, changed = call(seed=0), call(seed=0), call(seed=1)
    assert oe.conjoint_save(first) == oe.conjoint_save(second)
    assert oe.conjoint_save(first) != oe.conjoint_save(changed)


def test_undefined_importance_nonpositive_btl_and_single_respondent_are_not_dropped(study):
    _, plan, responses, _, _, attributes = study
    zero = responses.copy()
    zero.score = 0.
    fitted = oe.conjoint_fit(plan, zero, attributes, factors={"price":"ideal"})
    for call, code in [(lambda:oe.conjoint_bootstrap_importance(fitted, replications=20),"undefined_importance"),
                       (lambda:oe.conjoint_bootstrap_shares(fitted, plan, method="btl", replications=20),"nonpositive_btl_score")]:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == code
    shares = oe.conjoint_bootstrap_shares(fitted, plan, replications=20)
    np.testing.assert_allclose(shares["estimates"].estimate, 1/len(plan))
    single = oe.conjoint_fit(plan, responses.iloc[:len(plan)], attributes, factors={"price":"ideal"})
    with pytest.raises(AnalysisError) as error:
        oe.conjoint_bootstrap_importance(single, replications=20)
    assert error.value.code == "insufficient_subjects"


def test_bootstrap_replay_never_refits_or_redraws(study, monkeypatch):
    fitted, plan, _, _, _, _ = study
    outputs = [oe.conjoint_bootstrap_fit(fitted, replications=20),
               oe.conjoint_bootstrap_importance(fitted, replications=20),
               oe.conjoint_bootstrap_shares(fitted, plan, method="logit", replications=20)]
    artifacts = [oe.conjoint_save(output) for output in outputs]
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay cannot fit or generate new random draws")
    monkeypatch.setattr(torch.linalg, "qr", forbidden)
    monkeypatch.setattr(random, "Random", forbidden)
    for output, artifact in zip(outputs, artifacts):
        restored = oe.conjoint_load(artifact)
        assert oe.conjoint_save(restored) == artifact
        assert restored.to_latex() == output.to_latex()


def test_exhaustive_rademacher_law_recovers_analytical_hc2_covariance(monkeypatch):
    plan = pd.DataFrame({"profile_id":["a","b","c","d"], "feature":[0,0,1,1]})
    response = pd.DataFrame({"subject":[7]*4, "profile_id":plan.profile_id, "score":[1.,3.,4.,7.]})
    fitted = oe.conjoint_fit(plan, response, {"feature":[0,1]})
    x = np.column_stack([np.ones(4), [1.,1.,-1.,-1.]])
    beta = np.linalg.lstsq(x, response.score, rcond=None)[0]
    residual = response.score.to_numpy()-x@beta
    bread = np.linalg.inv(x.T@x)
    h = np.einsum("ij,jk,ik->i", x, bread, x)
    analytical = bread@x.T@np.diag(residual**2/(1-h))@x@bread
    complete = list(itertools.product([-1.,1.], repeat=4))*2
    probabilities = iter([.25 if w == 1 else .75 for draw in complete for w in draw])
    class Exhaustive:
        def random(self):
            return next(probabilities)
    monkeypatch.setattr(random, "Random", lambda seed:Exhaustive())
    output = oe.conjoint_bootstrap_fit(fitted, method="rademacher", replications=32)
    actual = output["covariance"].iloc[:2,2:4].to_numpy(float)
    np.testing.assert_allclose(actual, analytical*32/31, rtol=1e-12, atol=1e-12)


def test_actual_respondent_procedure_matches_complete_enumerated_sampling_law(study, monkeypatch):
    _, plan, response, _, _, attributes = study
    response = response.iloc[:len(plan)*3]
    fitted = oe.conjoint_fit(plan, response, attributes, factors={"price":"ideal"})
    complete = list(itertools.product(range(3), repeat=3))
    positions = iter([position for draw in complete for position in draw])
    class Exhaustive:
        def randrange(self, size):
            assert size == 3
            return next(positions)
    monkeypatch.setattr(random, "Random", lambda seed:Exhaustive())
    output = oe.conjoint_bootstrap_importance(fitted, replications=27)
    individual = individual_targets(study, "importance")[:3]
    centered = individual-individual.mean(0)
    np.testing.assert_allclose(output["covariance"].iloc[:,1:].to_numpy(float),
                               centered.T@centered/9*27/26, rtol=2e-8, atol=1e-10)
    assert output.attrs["state"]["draw_subject_positions"] == [list(draw) for draw in complete]


@pytest.mark.parametrize("procedure", ["fit", "importance", "shares"])
def test_cumulative_admission_precedes_any_random_draw_or_refit(study, monkeypatch, procedure):
    fitted, plan, _, _, _, _ = study
    def forbidden(*args, **kwargs):
        raise AssertionError("Budget rejection must happen before random allocation or numerical fits")
    monkeypatch.setattr(random, "Random", forbidden)
    monkeypatch.setattr(torch.linalg, "qr", forbidden)
    with pytest.raises(AnalysisError) as error:
        if procedure == "fit":
            oe.conjoint_bootstrap_fit(fitted, replications=1999, max_work=1)
        elif procedure == "importance":
            oe.conjoint_bootstrap_importance(fitted, replications=1999, max_work=1)
        else:
            oe.conjoint_bootstrap_shares(fitted, plan, replications=1999, max_work=1)
    assert error.value.code == "work_limit"


def test_shuffled_keyed_rows_and_whole_subject_exclusion_keep_actual_source_provenance(study):
    _, plan, responses, x, mapping, attributes = study
    responses = responses.copy()
    responses.loc[responses.index[2], "score"] = np.nan  # drop numeric subject17, retain string"17"
    responses = responses.sample(frac=1, random_state=805)
    fitted = oe.conjoint_fit(plan, responses, attributes, factors={"price":"ideal"}, missing="drop_subjects")
    output = oe.conjoint_bootstrap_fit(fitted, method="residual", replications=20)
    assert len(output.attrs["state"]["subjects"]) == 3
    for i, record in enumerate(output.attrs["state"]["subjects"]):
        sample = responses.iloc[record["source_positions"]]
        assert sample.profile_id.tolist() == plan.profile_id.tolist()
        assert sample.index.tolist() == record["source_labels"]
        assert sample.subject.tolist() == [record["subject"]]*len(plan)
        beta = np.linalg.lstsq(x, sample.score.to_numpy(float), rcond=None)[0]
        np.testing.assert_allclose(output["estimates"].iloc[i*13:(i+1)*13].estimate, mapping@beta, rtol=2e-8)
    group = oe.conjoint_bootstrap_importance(fitted, replications=20)
    assert len(group["individual"]) == 3
    assert all(not isinstance(who, int) for who in group.attrs["state"]["subjects"])
