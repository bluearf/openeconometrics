"""Frequency expansion, per-copy refits and independent summary/density oracles."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import chi2, f as f_dist

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate import discrim_options as options
from openecon.econometrics.multivariate.discrim import discrim, discrim_predict, discrim_summary
from openecon.resources import use_workspace_budget

NAMES = ["x", "y", "z"]


def domain(seed=971, *, hetero=False, size=24, strings=False):
    rng = np.random.default_rng(seed)
    labels = ["alpha", "beta", "gamma"] if strings else [0, 1, 2]
    means = np.array([[0, 1, -.5], [1.3, -.7, 1.1], [-.9, .2, 1.8]])
    bases = [np.array([[1, .35, -.2], [.35, 1.4, .3], [-.2, .3, .8]])]*3
    if hetero:
        bases = [bases[0], np.array([[2, -.4, .6], [-.4, .8, .1], [.6, .1, 1.7]]),
                 np.array([[.65, .1, .3], [.1, 1.8, -.35], [.3, -.35, 1.2]])]
    blocks = [rng.multivariate_normal(mean, covariance, size+i*3)
              for i, (mean, covariance) in enumerate(zip(means, bases))]
    out = pd.DataFrame(np.concatenate(blocks), columns=NAMES)
    out["g"] = np.concatenate([[label]*len(block) for label, block in zip(labels, blocks)])
    out["w"] = rng.integers(1, 5, len(out))
    return out


def restored(result):
    payload = json.loads(json.dumps({"attrs": result.attrs, "tables": {
        key: value.astype(object).where(value.notna(), None).to_dict(orient="split")
        for key, value in result.items()}}, allow_nan=False))
    return TableSet({key: pd.DataFrame(**value) for key, value in payload["tables"].items()}, **payload["attrs"])


def summaries(frame, labels=None):
    labels = sorted(frame.g.unique()) if labels is None else labels
    blocks = [frame.loc[frame.g == label, NAMES].to_numpy() for label in labels]
    means = pd.DataFrame([block.mean(0) for block in blocks], index=labels, columns=NAMES)
    covariances = {label: pd.DataFrame(np.cov(block.T), index=NAMES, columns=NAMES)
                   for label, block in zip(labels, blocks)}
    return means, covariances, pd.Series([len(block) for block in blocks], index=labels)


def moments(x, groups, labels):
    # Separate NumPy computation: anchored sums, direct within-group deviations.
    anchor = x[0]
    z = x-anchor
    blocks = [z[groups == label] for label in labels]
    counts = np.array([len(block) for block in blocks])
    means = np.array([block.mean(0) for block in blocks])
    ss = [(block-mean).T @ (block-mean) for block, mean in zip(blocks, means)]
    within = np.sum(ss, axis=0)
    grand = np.average(means, weights=counts, axis=0)
    centered = means-grand
    between = centered.T @ (counts[:, None]*centered)
    return anchor, means, counts, ss, within, grand, between


def prior_vector(priors, counts):
    if isinstance(priors, str):
        return np.full(len(counts), 1/len(counts)) if priors == "equal" else counts/counts.sum()
    return np.asarray(priors)


def density(query, x, groups, labels, method, priors):
    anchor, means, counts, ss, within, _, _ = moments(x, groups, labels)
    scores = []
    for mean, ng, wg, prior in zip(means, counts, ss, priors):
        covariance = within/(len(x)-len(labels)) if method == "lda" else wg/(ng-1)
        delta = (query-anchor)-mean
        distance = np.einsum("ij,ij->i", delta, np.linalg.solve(covariance, delta.T).T)
        scores.append(-.5*distance + np.log(prior) - (.5*np.linalg.slogdet(covariance)[1] if method == "qda" else 0))
    scores = np.column_stack(scores)
    probabilities = np.exp(scores-scores.max(1, keepdims=True))
    return scores, probabilities/probabilities.sum(1, keepdims=True)


def declared_density(query, means, covariances, counts, method, priors):
    pooled = sum((count-1)*covariances[label].to_numpy()
                 for label, count in counts.items())/(counts.sum()-len(counts))
    scores = []
    for label, mean, prior in zip(means.index, means.to_numpy(), priors):
        covariance = pooled if method == "lda" else covariances[label].to_numpy()
        delta = query-mean
        distance = np.einsum("ij,ij->i", delta, np.linalg.solve(covariance, delta.T).T)
        scores.append(-.5*distance+np.log(prior)-(.5*np.linalg.slogdet(covariance)[1] if method == "qda" else 0))
    scores = np.column_stack(scores)
    probabilities = np.exp(scores-scores.max(1, keepdims=True))
    return probabilities/probabilities.sum(1, keepdims=True)


def confusion(scores, actual, labels, weights=None):
    weights = np.ones(len(actual), dtype=int) if weights is None else weights
    out = np.zeros((len(labels), len(labels)), dtype=np.int64)
    predicted = scores.argmax(1)
    for label, estimate, weight in zip(actual, predicted, weights):
        out[labels.index(label), estimate] += weight
    return out


def per_copy_loo(x, groups, labels, method, priors):
    out = []
    for i in range(len(x)):
        kept = np.arange(len(x)) != i
        scores, _ = density(x[i:i+1], x[kept], groups[kept], labels, method, priors)
        out.append(scores[0])
    return np.array(out)


def assert_canonical(fit, x, groups, labels):
    _, _, counts, _, within, grand, between = moments(x, groups, labels)
    n, p = x.shape
    q = min(p, len(labels)-1)
    chol = np.linalg.cholesky(within)
    inner = np.linalg.solve(chol, np.linalg.solve(chol, between).T)
    roots, vectors = np.linalg.eigh((inner+inner.T)/2)
    roots, vectors = roots[::-1][:q], vectors[:, ::-1][:, :q]
    coef = np.linalg.solve(chol.T, vectors)*np.sqrt(n-len(labels))
    std = np.sqrt(np.diag(within)/(n-len(labels)))
    signs = np.sign((coef*std[:, None])[np.abs(coef*std[:, None]).argmax(0), np.arange(q)])
    coef *= signs
    np.testing.assert_allclose(fit["canonical_functions"].eigenvalue, roots, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(fit["unstandardized_coefficients"].loc[NAMES], coef, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(fit["standardized_coefficients"], coef*std[:, None], rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(fit["structure_matrix"], (within/(n-len(labels))@coef)/std[:, None], rtol=2e-11, atol=2e-12)
    for k, row in enumerate(fit["canonical_functions"].itertuples()):
        log_lambda = -np.log1p(roots[k:]).sum()
        statistic = -(n-1-(p+len(labels))/2)*log_lambda
        np.testing.assert_allclose([row.wilks_lambda, row.chi2, row.p_value],
                                  [np.exp(log_lambda), statistic, chi2.sf(statistic, row.df)], rtol=2e-10, atol=1e-12)


@pytest.mark.parametrize("hetero,strings", [(False, False), (True, True)])
@pytest.mark.parametrize("method", ["lda", "qda"])
@pytest.mark.parametrize("priors", ["equal", "proportional", [.2, .3, .5]])
def test_frequency_literal_expansion_all_tables_and_per_copy_refits(hetero, strings, method, priors):
    frame = domain(hetero=hetero, strings=strings)
    expanded = frame.loc[frame.index.repeat(frame.w)].reset_index(drop=True)
    labels = sorted(frame.g.unique())
    fit = discrim(frame, "g", NAMES, method=method, priors=priors, weights="w", loo=True)
    raw = discrim(expanded, "g", NAMES, method=method, priors=priors, loo=True)
    # Every pre-existing table, not just classification coefficients.
    for key, expected in raw.items():
        actual = fit[key]
        if key == "groups":
            actual = actual.loc[:, ["n", "prior"]]
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=2e-9, atol=2e-10)
    x, group = expanded[NAMES].to_numpy(), expanded.g.to_numpy()
    prior = prior_vector(priors, np.array([sum(group == label) for label in labels]))
    scores, probability = density(frame[NAMES].to_numpy(), x, group, labels, method, prior)
    pred = discrim_predict(fit, frame)
    np.testing.assert_allclose(pred.iloc[:, 1:], probability, rtol=3e-11, atol=2e-12)
    np.testing.assert_array_equal(pred.predicted, np.array(labels)[scores.argmax(1)])
    np.testing.assert_array_equal(fit["classification_table"].iloc[:, :3], confusion(scores, frame.g.to_numpy(), labels, frame.w.to_numpy()))
    loo = per_copy_loo(x, group, labels, method, prior)
    np.testing.assert_array_equal(fit["classification_table_loo"].iloc[:, :3], confusion(loo, group, labels))
    np.testing.assert_array_equal(fit["classification_table_physical"].iloc[:, :3], confusion(scores, frame.g.to_numpy(), labels))
    assert fit.attrs["physical_rows"] == len(frame) and fit.attrs["n"] == len(expanded)
    assert fit.attrs["prior_policy_loo"] == "fixed full-fit priors"
    assert fit.attrs["loo_unit"] == "one expanded frequency observation"
    if method == "lda":
        assert_canonical(fit, x, group, labels)


@pytest.mark.parametrize("hetero,strings", [(False, False), (True, True)])
@pytest.mark.parametrize("method", ["lda", "qda"])
@pytest.mark.parametrize("priors", ["equal", "proportional", [.2, .3, .5]])
def test_declared_summary_all_recoverable_tables_density_and_canonical(hetero, strings, method, priors):
    frame = domain(913, hetero=hetero, strings=strings, size=45)
    means, covariances, counts = summaries(frame)
    fit = discrim_summary(means, covariances, counts, method=method, priors=priors, group="g")
    raw = discrim(frame, "g", NAMES, method=method, priors=priors)
    for key, expected in raw.items():
        if key != "classification_table":
            pd.testing.assert_frame_equal(fit[key], expected, check_dtype=False, rtol=2e-9, atol=2e-10)
    assert not any(key.startswith("classification_table") for key in fit)
    assert fit.attrs["percent_correct"] is None and fit.attrs["percent_correct_loo"] is None
    assert not fit.attrs["training_classification_available"] and not fit.attrs["loo_available"]
    assert fit.attrs["physical_rows"] is None and fit.attrs["input_kind"] == "group_summary"
    labels = list(means.index)
    query = domain(421, size=11)[NAMES]
    expected_scores, expected_prob = density(query.to_numpy(), frame[NAMES].to_numpy(), frame.g.to_numpy(), labels, method, prior_vector(priors, counts.to_numpy()))
    predicted = discrim_predict(fit, query)
    np.testing.assert_allclose(predicted.iloc[:, 1:], expected_prob, rtol=3e-11, atol=2e-12)
    np.testing.assert_array_equal(predicted.predicted, np.array(labels)[expected_scores.argmax(1)])
    if method == "lda":
        assert_canonical(fit, frame[NAMES].to_numpy(), frame.g.to_numpy(), labels)
    # Independent direct determinants/Box M, equality F and p-values.
    _, _, ns, ss, within, _, between = moments(frame[NAMES].to_numpy(), frame.g.to_numpy(), labels)
    df2 = len(frame)-len(labels)
    m = df2*np.linalg.slogdet(within/df2)[1]-sum((n-1)*np.linalg.slogdet(s/(n-1))[1] for n, s in zip(ns, ss))
    correction = (sum(1/(ns-1))-1/df2)*(2*3**2+3*3-1)/(6*(3+1)*(3-1))
    np.testing.assert_allclose(fit["box_m"].iloc[0][["statistic", "chi2", "chi2_p_value"]], [m, m*(1-correction), chi2.sf(m*(1-correction), 12)], rtol=2e-10, atol=1e-11)
    equality = np.diag(between)/np.diag(within)*df2/(len(labels)-1)
    np.testing.assert_allclose(fit["tests_of_equality"].statistic, equality, rtol=2e-11)
    np.testing.assert_allclose(fit["tests_of_equality"].p_value, f_dist.sf(equality, len(labels)-1, df2), rtol=2e-10, atol=1e-12)


@pytest.mark.parametrize("method", ["lda", "qda"])
@pytest.mark.parametrize("kind", ["frequency", "summary"])
def test_complete_saved_prediction_state_missing_alignment_and_offsets(method, kind):
    frame = domain(191, hetero=True, strings=True, size=30)
    frame[NAMES] += 1e10
    if kind == "frequency":
        frame.loc[5, "x"] = np.nan
        frame.loc[6, "w"] = np.nan
        frame.loc[7, "w"] = 0
        fit = discrim(frame, "g", NAMES, method=method, priors="proportional", weights="w", loo=True)
        clean = frame.dropna()
        expanded = clean.loc[clean.index.repeat(clean.w.astype(int))]
    else:
        declared = summaries(frame)
        fit = discrim_summary(*declared, method=method, priors="proportional")
        expanded = frame
    saved = restored(fit)
    assert saved.attrs == fit.attrs
    for name, value in fit.items():
        pd.testing.assert_frame_equal(saved[name], value, check_dtype=False, check_index_type=False,
                                      check_names=False, check_frame_type=False)
    query = domain(998, strings=True, size=5)[NAMES]+1e10
    query.index = [f"row{i//2}" for i in range(len(query))]
    query.iloc[1, 0] = np.nan
    one, two = discrim_predict(fit, query), discrim_predict(saved, query)
    pd.testing.assert_frame_equal(one, two)
    assert one.index.equals(query.index) and one.iloc[1].isna().all()
    labels = fit.attrs["groups"]
    valid = query.notna().all(axis=1)
    _, expected = density(query.loc[valid].to_numpy(), expanded[NAMES].to_numpy(), expanded.g.to_numpy(), labels, method, np.array(fit.attrs["priors"]))
    if kind == "summary":
        # At 1e10, declared means have already rounded their sub-ULP location.
        # The target is the actual supplied summary, not hidden original rows.
        expected = declared_density(query.loc[valid].to_numpy(), *declared, method, np.array(fit.attrs["priors"]))
    np.testing.assert_allclose(one.loc[valid].iloc[:, 1:], expected, rtol=3e-10, atol=2e-11)
    assert discrim_predict(saved, query*float("nan")).isna().all().all()
    bad = restored(fit)
    bad.attrs["discriminant_state"]["mean_offsets"][0][0] += .1
    with pytest.raises(AnalysisError, match="state"):
        discrim_predict(bad, query)
    bad = restored(fit)
    del bad.attrs["discriminant_state"]
    with pytest.raises(AnalysisError, match="state"):
        discrim_predict(bad, query)


def test_frequency_missing_zero_categories_and_input_forms():
    frame = domain(strings=True)
    frame["g"] = pd.Categorical(frame.g, categories=["gamma", "unused", "alpha", "beta"])
    frame.loc[1, "w"] = 0
    frame.loc[2, "x"] = np.nan
    frame.loc[3, "w"] = np.nan
    fit = discrim(frame, "g", NAMES, weights="w", loo=True)
    assert fit.attrs["groups"] == ["gamma", "alpha", "beta"]
    assert fit.attrs["n_missing"] == 2 and fit.attrs["n_zero_weight"] == 1
    assert fit.attrs["n"] == int(frame.dropna().w.sum())
    with pytest.raises(AnalysisError, match="missing"):
        discrim(frame, "g", NAMES, weights="w", missing="raise")
    clean = domain()
    one = discrim(clean.to_dict(orient="list"), "g", NAMES, weights="w")
    two = discrim(clean.to_dict(orient="records"), "g", NAMES, weights="w")
    for key in one:
        pd.testing.assert_frame_equal(one[key], two[key])
    zero_group = clean.copy()
    zero_group.loc[zero_group.g == 2, "w"] = 0
    assert discrim(zero_group, "g", NAMES, weights="w").attrs["groups"] == [0, 1]


@pytest.mark.parametrize("bad", [-1, .5, np.inf, -np.inf, 2**53+1, 2**54])
def test_bad_frequency_values_and_roles(bad):
    frame = domain()
    frame["w"] = frame.w.astype(float) if isinstance(bad, float) else frame.w
    frame.loc[0, "w"] = bad
    with pytest.raises(AnalysisError):
        discrim(frame, "g", NAMES, weights="w")


def test_frequency_precision_unsupported_and_zero_rank_failures():
    frame = domain()
    with pytest.raises(AnalysisError, match="precision"):
        discrim(frame.assign(w=2**52), "g", NAMES, weights="w")
    with pytest.raises(AnalysisError):
        discrim(frame.assign(w=0), "g", NAMES, weights="w")
    with pytest.raises(AnalysisError):
        discrim(frame, "g", NAMES, weights="g")
    with pytest.raises(AnalysisError):
        discrim(frame, "g", NAMES, weights="w", weight_type="aweight")
    streaming = Dataset.from_batches(lambda: iter([frame]), columns=list(frame), row_count=len(frame))
    with pytest.raises(AnalysisError, match="resident"):
        discrim(streaming, "g", NAMES, weights="w")
    with pytest.raises(AnalysisError):
        discrim(frame.assign(z=frame.x), "g", NAMES, weights="w")
    # Very large frequencies on identical rows cannot supply covariance rank.
    with pytest.raises(AnalysisError):
        discrim(frame.groupby("g").head(1).assign(w=10000), "g", NAMES, weights="w", method="qda")


def test_lda_singleton_and_loo_actual_rank_not_just_count():
    frame = domain(size=12)
    frame = pd.concat([frame[frame.g == 0].iloc[:1], frame[frame.g != 0]], ignore_index=True)
    frame.w = 1
    fit = discrim(frame, "g", NAMES, weights="w")
    assert fit["group_covariances"].iloc[:3, 2:].isna().all().all()
    assert fit.attrs["discriminant_state"]["sample_covariances"][0] is None
    pd.testing.assert_frame_equal(discrim_predict(fit, frame), discrim_predict(restored(fit), frame))
    with pytest.raises(AnalysisError):
        discrim(frame, "g", NAMES, weights="w", loo=True)
    # Full-rank two-dimensional covariance, deletion of one frequency-one point
    # collapses it, despite a nominal group count much larger than p+2.
    collapsing = pd.DataFrame({"g": [0]*3+[1]*5, "x": [0, 1, 0, 2, 3, 2, 3, 2.5],
                              "y": [0, 0, 1, 2, 2, 3, 3, 2.5], "w": [20, 20, 1, 2, 2, 2, 2, 2]})
    discrim(collapsing, "g", ["x", "y"], weights="w", method="qda")
    with pytest.raises(AnalysisError, match="singular"):
        discrim(collapsing, "g", ["x", "y"], weights="w", method="qda", loo=True)


def test_summary_individual_psd_pooled_pd_and_qda_refusal():
    means = pd.DataFrame([[0, 0], [1, 2]], columns=["x", "y"], index=["a", "b"])
    covs = {"a": [[1, 0], [0, 0]], "b": [[0, 0], [0, 2]]}
    fit = discrim_summary(means, covs, {"a": 3, "b": 3})
    assert "box_m" not in fit and fit["pooled_covariance"].iloc[0, 0] == .5
    assert np.isfinite(discrim_predict(restored(fit), pd.DataFrame([[.5, 1]], columns=["x", "y"])).iloc[:, 1:]).all().all()
    with pytest.raises(AnalysisError, match="QDA"):
        discrim_summary(means, covs, [3, 3], method="qda")
    with pytest.raises(AnalysisError, match="singular"):
        discrim_summary(means, {"a": covs["a"], "b": covs["a"]}, [3, 3])


def test_summary_psd_and_feasible_rank_are_invariant_to_measurement_units():
    means = pd.DataFrame([[0., 0], [1., 1e-10]], columns=["x", "y"], index=[0, 1])
    impossible = {0: [[1., 0], [0, 1e-20]], 1: np.eye(2)}
    with pytest.raises(AnalysisError, match="rank"):
        discrim_summary(means, impossible, [2, 20])
    indefinite = {0: [[1., 1e-8], [1e-8, 1e-20]], 1: np.eye(2)}
    with pytest.raises(AnalysisError, match="semidefinite"):
        discrim_summary(means, indefinite, [20, 20])
    feasible = {0: [[1., 1e-10], [1e-10, 1e-20]],
                1: [[1., -1e-10], [-1e-10, 1e-20]]}
    fit = discrim_summary(means, feasible, [2, 2])
    assert "box_m" not in fit
    assert np.isfinite(discrim_predict(restored(fit), means).iloc[:, 1:]).all().all()


@pytest.mark.parametrize("mutation", ["fraction", "bool", "small", "total", "count_order", "missing_group", "cov_order", "asymmetric", "indefinite", "nonfinite", "mean_nonfinite", "rank", "columns", "group_collision", "complex"])
def test_declared_summary_invalid_contracts(mutation):
    means, covs, counts = summaries(domain())
    kwargs = {}
    if mutation == "fraction":
        counts = [20., 20, 20]
    if mutation == "bool":
        counts = [True, 20, 20]
    if mutation == "small":
        counts = [1, 20, 20]
    if mutation == "total":
        counts = [2**52]*3
    if mutation == "count_order":
        counts = counts.iloc[::-1]
    if mutation == "missing_group":
        covs.pop(1)
    if mutation == "cov_order":
        covs[0] = covs[0].iloc[::-1]
    if mutation == "asymmetric":
        covs[0].iloc[0, 1] += .1
    if mutation == "indefinite":
        covs[0].iloc[0, 1] = covs[0].iloc[1, 0] = 5
    if mutation == "nonfinite":
        covs[0].iloc[0, 0] = np.inf
    if mutation == "mean_nonfinite":
        means.iloc[0, 0] = np.inf
    if mutation == "rank":
        counts = [2, 20, 20]
    if mutation == "columns":
        kwargs["columns"] = NAMES[::-1]
    if mutation == "group_collision":
        means.index = [1, "1", 2]
    if mutation == "complex":
        covs[0] = covs[0].to_numpy().astype(complex)
    with pytest.raises(AnalysisError):
        discrim_summary(means, covs, counts, **kwargs)


def test_resource_refusal_precedes_numerical_conversion(monkeypatch):
    frame = domain(size=60)
    def forbidden(*args, **kwargs):
        raise AssertionError("numerical conversion happened before admission")
    monkeypatch.setattr(options.c, "matrix", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        discrim(pd.concat([frame]*100), "g", NAMES, weights="w")
    means = pd.DataFrame(np.zeros((129, 3)), columns=NAMES)
    with pytest.raises(AnalysisError, match="128"):
        discrim_summary(means, {}, [])
    means = pd.DataFrame(np.zeros((2, 257)), columns=[f"v{i}" for i in range(257)])
    with pytest.raises(AnalysisError, match="256"):
        discrim_summary(means, {}, [])
    with pytest.raises(AnalysisError, match="work budget"):
        options._geometry(200, 120, 100000)


def test_summary_count_mapping_and_positional_covariances_equivalence():
    means, covs, counts = summaries(domain(strings=True))
    one = discrim_summary(means, covs, counts)
    two = discrim_summary(means, {g: c.to_numpy().tolist() for g, c in reversed(list(covs.items()))}, dict(counts))
    for key in one:
        pd.testing.assert_frame_equal(one[key], two[key])
    assert "declared, not empirically verified" in one.attrs["inferential_assumptions"]


@pytest.mark.parametrize("method", ["lda", "qda"])
def test_centered_lda_posterior_preserves_far_query_small_log_odds(method):
    means = pd.DataFrame([[-1., 0], [1., 0]], index=[0, 1], columns=["x", "y"])
    covariances = {0: np.eye(2), 1: np.eye(2)}
    fit = discrim_summary(means, covariances, [20, 20], method=method)
    query = pd.DataFrame({"x": [.001, -.001], "y": [1e12, -1e12]})
    result = discrim_predict(restored(fit), query)
    # With identity pooled covariance and means +-1 in x, log odds = 2x.
    # Computing common squared distances first loses these small differences.
    expected = 1/(1+np.exp(-2*query.x.to_numpy()))
    np.testing.assert_allclose(result.posterior_1, expected, atol=1e-14)
    assert list(result.predicted) == [1, 0]


@pytest.mark.parametrize("priors", ["equal", [.2, .3, .5]])
def test_far_and_near_group_means_lda_posteriors_and_one_copy_loo(priors):
    rng = np.random.default_rng(522)
    frame = pd.DataFrame({"x": np.concatenate([rng.normal(mean, 1, 80) for mean in [0, 1e10, 1e10+2]]),
                          "g": np.repeat([0, 1, 2], 80), "w": 2})
    fit = discrim(frame, "g", ["x"], weights="w", priors=priors, loo=True)
    expanded = frame.loc[frame.index.repeat(frame.w)].reset_index(drop=True)
    x, groups = expanded[["x"]].to_numpy(), expanded.g.to_numpy()
    prior = prior_vector(priors, np.array([160]*3))
    scores, probability = density(frame[["x"]].to_numpy(), x, groups, [0, 1, 2], "lda", prior)
    predicted = discrim_predict(restored(fit), frame)
    # Supplied float64 coordinates have spacing ~2e-6 at 1e10; two independent
    # group mean accumulation orders can differ by one unit at that spacing.
    np.testing.assert_allclose(predicted.iloc[:, 1:], probability, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(predicted.predicted, np.array([0, 1, 2])[scores.argmax(1)])
    loo = per_copy_loo(x, groups, [0, 1, 2], "lda", prior)
    np.testing.assert_array_equal(fit["classification_table_loo"].iloc[:, :3], confusion(loo, groups, [0, 1, 2]))
    if priors == "equal":
        np.testing.assert_array_equal(fit["classification_table_loo"].iloc[:, :3], [[160, 0, 0], [0, 132, 28], [0, 20, 140]])


def huge_frequency_outlier(x=.001, y=1e7):
    frequency = 2**50-1
    rows = [[g, center+dx, dy, frequency] for g, center in [(0, -1), (1, 1)]
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]]
    rows.append([0, x, y, 1])
    return pd.DataFrame(rows, columns=["g", "x", "y", "w"])


@pytest.mark.parametrize("x,y", [(.001, 1e7), (-.001, 1e8), (.001, 3e8)])
def test_one_copy_loo_pairwise_odds_near_exact_count_limit(x, y):
    frame = huge_frequency_outlier(x, y)
    fit = discrim(frame, "g", ["x", "y"], weights="w", loo=True)
    state = fit.attrs["discriminant_state"]
    n = fit.attrs["n"]
    assert n == 2**53-7 and fit.attrs["physical_rows"] == 9
    counts = torch.tensor(state["counts"], dtype=torch.float64)
    means = torch.tensor(state["mean_offsets"], dtype=torch.float64)
    values = torch.tensor(frame[["x", "y"]].to_numpy(), dtype=torch.float64)-torch.tensor(state["origin"])
    factor = torch.linalg.cholesky(torch.tensor(fit["pooled_covariance"].to_numpy())*(n-2))
    scores = options._linear_loo_scores(values, means, factor, n-3,
                                       torch.log(torch.tensor(state["priors"])),
                                       torch.tensor(frame.g.to_numpy()), counts)
    # Exact deletion leaves the eight corner rows with equal covariance and
    # means (+/-1,0); log odds = 2*x / [N_remaining/(N_remaining-2)].
    expected = 2*x*((n-3)/(n-1))
    np.testing.assert_allclose(float(scores[-1, 1]-scores[-1, 0]), expected, atol=2e-6, rtol=0)
    assert int(scores[-1].argmax()) == (1 if x > 0 else 0)
    restored(fit)                         # full near-limit counts remain finite JSON.


def test_qda_refuses_unresolved_unequal_covariance_large_logits():
    means = pd.DataFrame([[-1., 0], [1., 0]], index=[0, 1], columns=["x", "y"])
    fit = discrim_summary(means, {0: np.eye(2), 1: np.diag([1+1e-6, 1])}, [20, 20], method="qda")
    query = pd.DataFrame({"x": [.001], "y": [1e12]})
    with pytest.raises(AnalysisError, match="not resolved"):
        discrim_predict(restored(fit), query)
    # A representable top-score gap near five can preserve argmax while still
    # corrupting posterior digits; refusal must cover this case too.
    near_competitive = pd.DataFrame({"x": [2.5001], "y": [4e6]})
    with pytest.raises(AnalysisError, match="not resolved"):
        discrim_predict(restored(fit), near_competitive)
    # Strongly separated losing logits remain resolved, with saturated zero/one
    # posterior probabilities instead of a blanket far-query refusal.
    wide = discrim_summary(means, {0: np.eye(2), 1: 2*np.eye(2)}, [20, 20], method="qda")
    out = discrim_predict(wide, query)
    assert out.posterior_1.iloc[0] == 1. and out.posterior_0.iloc[0] == 0.


def test_qda_loo_refuses_unresolved_huge_count_quadratic_comparison():
    frame = huge_frequency_outlier()
    discrim(frame, "g", ["x", "y"], weights="w", method="qda")
    with pytest.raises(AnalysisError, match="not resolved"):
        discrim(frame, "g", ["x", "y"], weights="w", method="qda", loo=True)
