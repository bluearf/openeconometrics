"""Independent regression/projection audit of selection and separable hypotheses.

Selection candidates are tested by two dense NumPy regressions, not by the
production Schur-complement calculation. Joint hypotheses use direct residual
SSCP, a generalized SciPy eigenproblem and the documented classical F laws.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy import linalg, stats
from statsmodels.multivariate.manova import MANOVA

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate.discrim_selection import (
    discrim_stepwise, discrim_stepwise_predict,
)
from openecon.econometrics.stats.manova import manova
from openecon.econometrics.stats.manova_options import (
    manova_contrast, manova_oneway, manova_summary, rm_mtest,
)
from openecon.econometrics.stats.rm_anova import rm_anova
from openecon.econometrics.summary_state import restore_summary, summary_state


def selection_domain(which):
    rng = np.random.default_rng(9521 if which == "clinical" else 17523)
    groups = np.repeat(np.arange(3), [37, 43, 31])
    mixing = np.eye(6) + np.tril(np.ones((6, 6)), -1) * .17
    values = rng.normal(size=(len(groups), 6)) @ mixing.T
    offsets = np.array([[0, .3, 0, .2, 0, 0],
                        [1.2, -.6, .8, 0, 0, .1],
                        [-.7, .9, -.5, 0, .1, 0]])
    values += offsets[groups]
    if which == "process":
        values *= np.array([.03, 120, 2, .4, 8, 1])
        values += np.array([30, -800, 3, 20, 0, 1])
    names = ["signal_a", "signal_b", "signal_c", "nuisance_a", "nuisance_b", "nuisance_c"]
    frame = pd.DataFrame(values, columns=names)
    frame["group"] = groups
    frame["copies"] = rng.integers(0, 5, len(frame))
    return frame, names


def partial_f_by_regressions(values, groups, conditioning, candidate, frequencies=None):
    """Group adjustment of one outcome via independent nested regression fits."""
    labels = np.unique(groups)
    group_columns = np.column_stack([groups == label for label in labels[1:]])
    z = values - values[0]
    z /= z.std(axis=0, ddof=1)
    response = z[:, candidate]
    base = np.column_stack([np.ones(len(z)), z[:, conditioning]])
    full = np.column_stack([base, group_columns])
    if frequencies is not None:
        root_weights = np.sqrt(frequencies)
        response, base, full = (response*root_weights, base*root_weights[:, None],
                                full*root_weights[:, None])
    null_residual = response - base @ np.linalg.lstsq(base, response, rcond=None)[0]
    full_residual = response - full @ np.linalg.lstsq(full, response, rcond=None)[0]
    r, e = null_residual @ null_residual, full_residual @ full_residual
    n = len(z) if frequencies is None else int(sum(frequencies))
    df1, df2 = len(labels) - 1, n - len(labels) - len(conditioning)
    statistic = (r / e - 1) * df2 / df1
    return float(statistic), float(stats.f.sf(statistic, df1, df2)), df1, df2


def selection_reference(values, groups, method, *, p_enter=.05, p_remove=.1,
                        include=(), frequencies=None):
    selected = list(include) if method != "backward" else list(range(values.shape[1]))
    path, visited = [], set()
    for _ in range(128):
        key = tuple(sorted(selected))
        if key in visited:
            raise AssertionError("The independent fixture has a selection cycle.")
        visited.add(key)
        removal = [(j, partial_f_by_regressions(values, groups,
                    [i for i in selected if i != j], j, frequencies))
                   for j in range(values.shape[1]) if j in selected and j not in include]
        if method != "forward" and removal:
            j, test = min(removal, key=lambda item: (item[1][0], item[0]))
            if test[1] > p_remove:
                selected.remove(j)
                path.append(("remove", j, test))
                continue
        if method == "backward":
            return sorted(selected), path
        entry = [(j, partial_f_by_regressions(values, groups, selected, j, frequencies))
                 for j in range(values.shape[1]) if j not in selected]
        if entry:
            j, test = max(entry, key=lambda item: (item[1][0], -item[0]))
            if test[1] < p_enter:
                selected.append(j)
                path.append(("enter", j, test))
                continue
        return sorted(selected), path
    raise AssertionError("The independent fixture exhausted its selection path.")


def manova_domain(which, *, nuisance=False):
    rng = np.random.default_rng(1527 if which == "clinical" else 6528)
    groups = np.repeat(np.arange(3), [23, 31, 19])
    x = rng.normal(size=len(groups)) + .3 * groups
    covariance = np.array([[1.4, .35, -.25], [.35, .8, .12], [-.25, .12, 1.7]])
    means = np.array([[.2, .8, -.5], [1.1, -.2, .6], [-.6, .3, 1.2]])
    y = means[groups] + rng.multivariate_normal(np.zeros(3), covariance, len(groups))
    if nuisance:
        y += x[:, None] * np.array([.6, -.25, .4])
    if which == "process":
        y = y * np.array([.002, 40, .2]) + np.array([2, -150, 20])
    names = ["response_a", "response_b", "response_c"]
    frame = pd.DataFrame(y, columns=names)
    frame["group"] = groups
    frame["x"] = x
    frame["copies"] = rng.integers(1, 4, len(frame))
    return frame, names


def dense_geometry(frame, outcomes, *, nuisance=False):
    groups = frame["group"].to_numpy()
    labels = np.unique(groups)
    coding = np.vstack([np.eye(len(labels)-1), -np.ones(len(labels)-1)])
    columns = [np.ones(len(frame)), coding[groups]]
    if nuisance:
        columns.append(frame["x"].to_numpy())
    design = np.column_stack(columns)
    y = frame[outcomes].to_numpy()
    origin = y[0]
    beta_centered = np.linalg.lstsq(design, y-origin, rcond=None)[0]
    residual = (y-origin)-design @ beta_centered
    beta = beta_centered.copy()
    beta[0] += origin
    return design, beta, np.linalg.inv(design.T @ design), residual.T @ residual


def joint_reference(beta, bread, error, l_matrix, m_matrix, null, df):
    estimate = l_matrix @ beta @ m_matrix
    difference = estimate-null
    q = l_matrix @ bread @ l_matrix.T
    projected_error = m_matrix.T @ error @ m_matrix
    hypothesis = difference.T @ np.linalg.solve(q, difference)
    covariance = np.kron(q, projected_error/df)
    roots = linalg.eigh(hypothesis, projected_error, eigvals_only=True)[::-1]
    roots = np.maximum(roots, 0)[:min(l_matrix.shape[0], m_matrix.shape[1])]
    return estimate, covariance, hypothesis, projected_error, roots


def classical_multivariate_reference(roots, responses, hypotheses, df):
    s = min(responses, hypotheses)
    m = (abs(responses-hypotheses)-1)/2
    n = (df-responses-1)/2
    pillai = np.sum(roots/(1+roots))
    wilks = np.prod(1/(1+roots))
    hotelling, roy = np.sum(roots), roots[0]
    df1_p, df2_p = s*(2*m+s+1), s*(2*n+s+1)
    t = np.sqrt((responses**2*hypotheses**2-4)/(responses**2+hypotheses**2-5)) \
        if responses**2+hypotheses**2 > 5 else 1
    df1_w = responses*hypotheses
    df2_w = (df-(responses-hypotheses+1)/2)*t-(responses*hypotheses-2)/2
    df1_h, df2_h = s*(2*m+s+1), 2*(s*n+1)
    r = max(responses, hypotheses)
    rows = {
        "pillai": [pillai, (df2_p/df1_p)*pillai/(s-pillai), df1_p, df2_p],
        "wilks": [wilks, (wilks**(-1/t)-1)*df2_w/df1_w, df1_w, df2_w],
        "hotelling": [hotelling, df2_h*hotelling/(s*df1_h), df1_h, df2_h],
        "roy": [roy, roy*(df-r+hypotheses)/r, r, df-r+hypotheses],
    }
    return {name: [*row, stats.f.sf(row[1], row[2], row[3])] for name, row in rows.items()}


def portable(result):
    return json.loads(json.dumps({"attrs": copy.deepcopy(result.attrs), "tables": {
        key: value.astype(object).where(value.notna(), None).to_dict(orient="split")
        for key, value in result.items()}}, allow_nan=False))


def restored(result):
    portable(result)  # Independently require finite complete table/metadata JSON.
    return restore_summary(summary_state(result))


def rehash(state, field="sha256"):
    body = {key: value for key, value in state.items() if key != field}
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def check_scaled_matrix(actual, expected):
    """Keep full covariance/SSCP checks meaningful across measurement units."""
    scale = np.sqrt(np.maximum(np.abs(np.diag(expected)), np.finfo(float).tiny))
    divisor = np.outer(scale, scale)
    np.testing.assert_allclose(np.asarray(actual)/divisor, expected/divisor,
                               rtol=3e-9, atol=3e-11)


def check_screening(frame, names, result, method, *, include=(), frequencies=None):
    values, groups = frame[names].to_numpy(), frame["group"].to_numpy()
    expected, path = selection_reference(values, groups, method, include=include,
                                         frequencies=frequencies)
    assert result.attrs["variables"] == [names[j] for j in expected]
    assert result.attrs["post_selection_inference"] is False
    actual = result["selection_history"]
    assert list(zip(actual["action"], actual["variable"], strict=True)) == [
        (action, names[j]) for action, j, _ in path]
    selected = list(range(len(names))) if method == "backward" else list(include)
    history = {int(row.step): row for row in actual.itertuples()}
    for step, block in result["candidate_tests"].groupby("step", sort=True):
        for row in block.itertuples():
            j = names.index(row.variable)
            conditioning = [i for i in selected if i != j] if row.action == "remove" else selected
            f_value, p_value, df1, df2 = partial_f_by_regressions(values, groups, conditioning, j,
                                                               frequencies)
            np.testing.assert_allclose([row.statistic, row.reference_p], [f_value, p_value],
                                       rtol=3e-9, atol=2e-12)
            assert (row.df1, row.df2, row.model_size) == (df1, df2, len(selected))
            ratio = f_value*df1/df2
            np.testing.assert_allclose([row.partial_lambda, row.partial_r_squared],
                                       [1/(1+ratio), ratio/(1+ratio)], rtol=2e-10, atol=1e-12)
        if int(step) in history:
            row = history[int(step)]
            j = names.index(row.variable)
            if row.action == "enter":
                selected.append(j)
            else:
                selected.remove(j)
            selected.sort()
    forbidden = {"canonical", "equality", "box_m", "multivariate", "inference"}
    assert not forbidden.intersection(result)


@pytest.mark.parametrize("domain", ["clinical", "process"])
@pytest.mark.parametrize("method", ["forward", "backward", "stepwise"])
def test_every_selection_candidate_against_two_dense_regressions(domain, method):
    frame, names = selection_domain(domain)
    result = discrim_stepwise(frame, "group", names, method=method)
    check_screening(frame, names, result, method)
    predicted = discrim_stepwise_predict(restored(result), frame)
    direct = discrim_stepwise_predict(result, frame)
    pd.testing.assert_frame_equal(predicted, direct)
    posterior = predicted.filter(like="posterior_").to_numpy()
    np.testing.assert_allclose(posterior.sum(1), 1, atol=2e-15)


@pytest.mark.parametrize("domain", ["clinical", "process"])
@pytest.mark.parametrize("method", ["forward", "backward", "stepwise"])
def test_weighted_screening_matches_literal_independent_copies(domain, method):
    frame, names = selection_domain(domain)
    expanded = frame.loc[frame.index.repeat(frame["copies"])].reset_index(drop=True)
    result = discrim_stepwise(frame, "group", names, method=method, weights="copies")
    check_screening(expanded, names, result, method)
    literal = discrim_stepwise(expanded, "group", names, method=method)
    assert result.attrs["variables"] == literal.attrs["variables"]
    for key in ("candidate_tests", "selection_history"):
        pd.testing.assert_frame_equal(result[key], literal[key], rtol=2e-9, atol=2e-12)
    assert result.attrs["n"] == len(expanded)
    assert result.attrs["n_zero_weight"] == int((frame["copies"] == 0).sum())
    np.testing.assert_allclose(discrim_stepwise_predict(restored(result), frame).filter(like="posterior_"),
                               discrim_stepwise_predict(literal, frame).filter(like="posterior_"),
                               rtol=2e-10, atol=2e-12)


def test_bidirectional_selection_actually_removes_an_entered_proxy():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(3), 80)
    means = np.array([[-1, -1], [0, 1], [1, 0]])
    a, b = (rng.normal(size=(240, 2))+means[groups]).T
    frame = pd.DataFrame({"proxy": a+b+.5*rng.normal(size=240), "a": a, "b": b,
                          "noise": rng.normal(size=240), "group": groups})
    names = ["proxy", "a", "b", "noise"]
    result = discrim_stepwise(frame, "group", names)
    check_screening(frame, names, result, "stepwise")
    assert list(zip(result["selection_history"]["action"],
                    result["selection_history"]["variable"], strict=True)) == [
        ("enter", "proxy"), ("enter", "a"), ("enter", "b"), ("remove", "proxy")]


def test_large_frequency_zero_tails_keep_conditional_f_ordering():
    frame, original = selection_domain("clinical")
    names = [*original[3:], *original[:3][::-1]]
    frame["copies"] = 10**12
    result = discrim_stepwise(frame, "group", names, method="forward", weights="copies")
    check_screening(frame, names, result, "forward", frequencies=frame["copies"].tolist())
    first = result["candidate_tests"].query("step == 1")
    assert (first["reference_p"] == 0).all()
    assert first["statistic"].nunique() == len(names)
    assert result["selection_history"].iloc[0].variable == first.loc[first["statistic"].idxmax(), "variable"]
    assert result["selection_history"].iloc[0].variable != names[0]


def test_selection_fixed_pool_and_duplicate_prediction_alignment():
    frame, names = selection_domain("clinical")
    frame.loc[0, names[-1]] = np.nan
    result = discrim_stepwise(frame, "group", names, method="forward")
    assert result.attrs["n"] == len(frame)-1
    assert result.attrs["n_missing"] == 1
    check_screening(frame.iloc[1:], names, result, "forward")
    query = frame.iloc[[5, 8, 5]].copy()
    query.index = ["same", "second", "same"]
    query.iloc[1, query.columns.get_loc(result.attrs["variables"][0])] = np.nan
    prediction = discrim_stepwise_predict(restored(result), query)
    assert prediction.index.tolist() == query.index.tolist()
    assert prediction.iloc[1].isna().all()
    np.testing.assert_array_equal(prediction.iloc[0], prediction.iloc[2])


def test_prior_only_and_protected_selection_are_explicit():
    frame, names = selection_domain("clinical")
    result = discrim_stepwise(frame, "group", names, method="forward", p_enter=1e-100,
                             priors=[.2, .3, .5])
    assert result.attrs["variables"] == []
    prediction = discrim_stepwise_predict(restored(result), frame.iloc[:3])
    np.testing.assert_allclose(prediction.filter(like="posterior_").to_numpy(),
                               np.tile([.2, .3, .5], (3, 1)), atol=0)
    protected = discrim_stepwise(frame, "group", names, method="backward", include=[names[-1]])
    assert names[-1] in protected.attrs["variables"]
    assert names[-1] not in protected["selection_history"]["variable"].tolist()


def test_selection_offset_invariance_and_independent_state_refusals():
    frame, names = selection_domain("process")
    shifted = frame.copy()
    shifted[names] += np.array([1e7, -1e8, 1e6, 1e7, 1e8, -1e7])
    first = discrim_stepwise(frame, "group", names)
    second = discrim_stepwise(shifted, "group", names)
    assert first.attrs["variables"] == second.attrs["variables"]
    np.testing.assert_allclose(first["candidate_tests"]["statistic"],
                               second["candidate_tests"]["statistic"], rtol=2e-6, atol=2e-8)
    corrupted = restored(first)
    corrupted.attrs["selection_state"]["mean_offsets"][0][0] += .1
    with pytest.raises(AnalysisError):
        discrim_stepwise_predict(corrupted, frame)
    forged = restored(first)
    state = forged.attrs["selection_state"]
    state["selected"] = names[::-1]
    forged.attrs["selection_state_sha256"] = rehash(state, field="unused")
    with pytest.raises(AnalysisError):
        discrim_stepwise_predict(forged, frame)


def check_multivariate(result, roots, p, q, df):
    expected = classical_multivariate_reference(roots, p, q, df)
    for row in result["multivariate"].itertuples():
        np.testing.assert_allclose([row.value, row.statistic, row.df1, row.df2, row.p_value],
                                   expected[row.test], rtol=5e-9, atol=2e-11)
        expected_type = ("exact" if min(p, q) == 1 or
                         (row.test == "wilks" and min(p, q) <= 2) else
                         "upper_bound" if row.test == "roy" else "approximate")
        assert row.f_type == expected_type


def check_joint(result, expected, df, alpha=.05):
    estimate, covariance, hypothesis, error, roots = expected
    check_scaled_matrix(result["hypothesis_sscp"], hypothesis)
    check_scaled_matrix(result["error_sscp"], error)
    check_scaled_matrix(result["target_covariance"], covariance)
    np.testing.assert_allclose(result["roots"]["eigenvalue"][:len(roots)], roots,
                               rtol=3e-9, atol=2e-10)
    table = result["estimates"]
    null = table["null"].to_numpy()
    point = estimate.ravel()
    se = np.sqrt(np.diag(covariance))
    t_value = (point-null)/se
    critical = stats.t.ppf(1-alpha/2, df)
    expected_rows = np.column_stack([point, se, t_value, np.full(len(point), df),
                                     2*stats.t.sf(abs(t_value), df),
                                     point-critical*se, point+critical*se])
    np.testing.assert_allclose(table[["estimate", "std_error", "statistic", "df", "p_value",
                                     "ci_low", "ci_high"]], expected_rows, rtol=4e-9, atol=2e-10)
    check_multivariate(result, roots, estimate.shape[1], estimate.shape[0], df)
    assert result.attrs["familywise_intervals"] is False
    assert result.attrs["target_order"] == list(map(list, zip(table["contrast"], table["transform"], strict=True)))


@pytest.mark.parametrize("domain", ["clinical", "process"])
def test_weighted_and_summary_manova_full_moments_and_general_covariance(domain):
    frame, names = manova_domain(domain)
    expanded = frame.loc[frame.index.repeat(frame["copies"])].reset_index(drop=True)
    groups = sorted(expanded["group"].unique())
    means = expanded.groupby("group")[names].mean()
    covariances = {group: expanded.loc[expanded["group"] == group, names].cov()
                   for group in groups}
    counts = expanded.groupby("group").size()
    mean_values = means.to_numpy()
    error = sum((int(counts[group])-1)*covariances[group].to_numpy() for group in groups)
    centre = np.average(mean_values, axis=0, weights=counts)
    delta = mean_values-centre
    hypothesis = delta.T @ (counts.to_numpy()[:, None]*delta)
    df = len(expanded)-len(groups)
    roots = np.maximum(linalg.eigh(hypothesis, error, eigvals_only=True)[::-1], 0)[:2]
    results = [manova_oneway(frame, names, "group", weights="copies"),
               manova_summary(means, covariances, counts)]
    for result in results:
        check_scaled_matrix(result["error_sscp"], error)
        check_scaled_matrix(result["hypothesis_sscp"], hypothesis)
        check_scaled_matrix(result["coefficient_covariance"], np.kron(np.diag(1/counts), error/df))
        check_multivariate(result, roots, 3, 2, df)
        for row in result["univariate"].itertuples():
            j = names.index(row.outcome)
            f_value = hypothesis[j, j]/2/(error[j, j]/df)
            np.testing.assert_allclose([row.hypothesis_ss, row.error_ss, row.statistic, row.p_value],
                                       [hypothesis[j, j], error[j, j], f_value, stats.f.sf(f_value, 2, df)],
                                       rtol=3e-9, atol=2e-8)
            assert (row.df1, row.df2) == (2, df)
        coefficient_covariance = np.kron(np.diag(1/counts), error/df)
        se = np.sqrt(np.diag(coefficient_covariance))
        critical = stats.t.ppf(.975, df)
        point = mean_values.ravel()
        np.testing.assert_allclose(result["means"][["estimate", "std_error", "ci_low", "ci_high"]],
                                   np.column_stack([point, se, point-critical*se, point+critical*se]),
                                   rtol=3e-9, atol=2e-8)
        l_matrix = np.array([[1, -1, 0], [0, 1, -1.]])
        m_matrix = np.array([[1, 0], [-.5, 1], [.25, -.4]])
        null = np.array([[.01, .02], [-.03, .04]])
        contrast = manova_contrast(restored(result), l_matrix, M=m_matrix, null=null,
                                   contrast_names=["first", "second"], transform_names=["mix", "other"])
        expected = joint_reference(mean_values, np.diag(1/counts), error,
                                   l_matrix, m_matrix, null, df)
        check_joint(contrast, expected, df)
        pd.testing.assert_frame_equal(contrast["estimates"],
                                      manova_contrast(restored(contrast), l_matrix, M=m_matrix,
                                                      null=null, contrast_names=["first", "second"],
                                                      transform_names=["mix", "other"])["estimates"])


@pytest.mark.parametrize("domain", ["clinical", "process"])
def test_saved_mancova_general_hypothesis_matches_full_dense_fit(domain):
    frame, names = manova_domain(domain, nuisance=True)
    result = manova(frame, names, ["group"], covariates=["x"])
    x_matrix, beta, bread, error = dense_geometry(frame, names, nuisance=True)
    old_columns = ["Intercept", "group[1]", "group[2]", "x"]
    order = [old_columns.index(name) for name in result.attrs["manova_contrast_state"]["design_columns"]]
    beta, bread = beta[order], bread[np.ix_(order, order)]
    l_matrix = np.array([[0, .5, 1, -.2], [1, 0, .3, .7]])
    m_matrix = np.array([[1, -.2], [.2, 1], [-.4, .3]])
    null = np.array([[.05, -.01], [.02, -.03]])
    df = len(frame)-x_matrix.shape[1]
    contrast = manova_contrast(restored(result), l_matrix, M=m_matrix, null=null)
    check_joint(contrast, joint_reference(beta, bread, error, l_matrix, m_matrix, null, df), df)


@pytest.mark.parametrize("domain", ["clinical", "process"])
def test_general_hypothesis_against_statsmodels_independent_projection(domain):
    frame, names = manova_domain(domain, nuisance=True)
    result = manova(frame, names, ["group"], covariates=["x"])
    design, _, _, _ = dense_geometry(frame, names, nuisance=True)
    l_matrix = np.array([[0, 1, 0, 0], [0, 0, 1, 0.]])
    m_matrix = np.array([[1, 0], [-.5, 1], [.25, -.4]])
    null = np.array([[.1, .2], [-.1, 0]])
    external = MANOVA(frame[names].to_numpy(), design).mv_test([
        ("declared", l_matrix, m_matrix, null)]).results["declared"]
    old_columns = ["Intercept", "group[1]", "group[2]", "x"]
    order = [old_columns.index(name) for name in result.attrs["manova_contrast_state"]["design_columns"]]
    contrast = manova_contrast(result, l_matrix[:, order], M=m_matrix, null=null)
    check_scaled_matrix(contrast["hypothesis_sscp"], external["H"])
    check_scaled_matrix(contrast["error_sscp"], external["E"])
    mapping = {"pillai": "Pillai's trace", "wilks": "Wilks' lambda",
               "hotelling": "Hotelling-Lawley trace", "roy": "Roy's greatest root"}
    for row in contrast["multivariate"].itertuples():
        reference = external["stat"].loc[mapping[row.test]]
        np.testing.assert_allclose(row.value, reference["Value"], rtol=2e-9, atol=2e-10)
        if row.test != "hotelling":
            np.testing.assert_allclose([row.statistic, row.df1, row.df2, row.p_value],
                                       reference[["F Value", "Num DF", "Den DF", "Pr > F"]].astype(float),
                                       rtol=3e-9, atol=2e-10)
        else:
            # Explicitly retain the native/R classical higher-rank F convention.
            assert row.df2 != reference["Den DF"]


def rm_domain(domain):
    rng = np.random.default_rng(11528 if domain == "clinical" else 23528)
    groups = np.repeat(np.arange(3), [17, 21, 15])
    covariance = .6*np.eye(6)+.3*np.ones((6, 6))
    response = rng.multivariate_normal(np.zeros(6), covariance, len(groups))
    response += np.array([[0, .4, .9, .2, .5, 1.1], [.3, .1, .8, .6, .2, 1.3],
                          [-.2, .8, .3, .1, 1.1, .7]])[groups]
    if domain == "process":
        response = response*.035+80
    rows = [[subject, int(group), cell//3, cell % 3, response[subject, cell]]
            for subject, group in enumerate(groups) for cell in range(6)]
    return pd.DataFrame(rows, columns=["subject", "group", "phase", "time", "response"]), response, groups


@pytest.mark.parametrize("domain", ["clinical", "process"])
def test_joint_repeated_measures_from_direct_subject_cell_regression(domain):
    frame, response, groups = rm_domain(domain)
    result = rm_anova(frame, "response", "subject", ["phase", "time"], between=["group"])
    design = np.column_stack([np.ones(len(groups)), np.vstack([np.eye(2), -np.ones(2)])[groups]])
    centred = response-response[0]
    beta = np.linalg.lstsq(design, centred, rcond=None)[0]
    residual = centred-design @ beta
    beta[0] += response[0]
    bread, error = np.linalg.inv(design.T @ design), residual.T @ residual
    l_matrix = np.array([[0, 1, -1], [1, .2, .3]])
    m_matrix = np.array([[-1, 0], [1, 0], [0, -1], [0, 0], [0, 0], [0, 1.]])
    null = np.array([[.01, -.02], [.03, 0]])
    df = len(groups)-3
    contrast = rm_mtest(restored(result), l_matrix, M=m_matrix, null=null,
                        contrast_names=["between", "average"], transform_names=["early", "cross"])
    check_joint(contrast, joint_reference(beta, bread, error, l_matrix, m_matrix, null, df), df)
    assert contrast.attrs["sphericity_required"] is False
    pd.testing.assert_frame_equal(contrast["estimates"],
                                  rm_mtest(restored(contrast), l_matrix, M=m_matrix, null=null,
                                           contrast_names=["between", "average"],
                                           transform_names=["early", "cross"])["estimates"])


def test_saved_contrast_state_rank_label_and_tamper_refusal():
    frame, names = manova_domain("clinical")
    result = manova_oneway(frame, names, "group")
    l_matrix, m_matrix = [[1, -1, 0]], [[1], [-1], [0]]
    corrupted = restored(result)
    corrupted.attrs["manova_contrast_state"]["bread"][0][0] *= 2
    with pytest.raises(AnalysisError):
        manova_contrast(corrupted, l_matrix, M=m_matrix)
    forged = restored(result)
    state = forged.attrs["manova_contrast_state"]
    state["bread"][0][0] *= 2
    state["sha256"] = rehash(state)
    with pytest.raises(AnalysisError):
        manova_contrast(forged, l_matrix, M=m_matrix)
    for left, right in (([[1, -1, 0], [2, -2, 0]], m_matrix),
                        (l_matrix, [[1, 2], [-1, -2], [0, 0]])):
        with pytest.raises(AnalysisError):
            manova_contrast(result, left, M=right)
    labelled_l = pd.DataFrame(l_matrix, columns=["group[2]", "group[1]", "group[3]"])
    with pytest.raises(AnalysisError):
        manova_contrast(result, labelled_l, M=m_matrix)
    with pytest.raises(AnalysisError):
        manova_summary(pd.DataFrame([[0, 0], [1, 1]], index=["a", "b"], columns=["x", "y"]),
                       {"a": np.eye(2), "b": np.eye(2)}, [2, 2])


def test_huge_common_response_level_preserves_small_represented_contrast():
    rng = np.random.default_rng(78528)
    groups = np.repeat(np.arange(3), [21, 25, 19])
    y = 1e12+rng.normal(size=(len(groups), 3))
    y[:, 1] += .125
    frame = pd.DataFrame(y, columns=["a", "b", "c"])
    frame["group"] = groups
    result = manova_oneway(frame, ["a", "b", "c"], "group")
    l_matrix, m_matrix = [[1/3, 1/3, 1/3]], [[1], [-1], [0]]
    contrast = manova_contrast(restored(result), l_matrix, M=m_matrix)
    transformed = y[:, 0]-y[:, 1]
    means = np.array([transformed[groups == j].mean() for j in range(3)])
    variance = sum(np.sum((transformed[groups == j]-means[j])**2) for j in range(3))/(len(y)-3)
    expected = means.mean()
    expected_se = np.sqrt(variance*np.sum(1/np.bincount(groups))/9)
    np.testing.assert_allclose(contrast["estimates"][["estimate", "std_error"]].iloc[0],
                               [expected, expected_se], rtol=2e-10, atol=2e-11)


def test_rm_scalar_reduction_and_recomputed_checksum_order_refusal():
    frame, response, groups = rm_domain("clinical")
    result = rm_anova(frame, "response", "subject", ["phase", "time"], between=["group"])
    l_matrix, m_matrix = [[1, 0, 0]], [[1], [-1], [0], [0], [0], [0]]
    contrast = rm_mtest(restored(result), l_matrix, M=m_matrix, null=[[.01]])
    transformed = response[:, 0]-response[:, 1]
    group_means = np.array([transformed[groups == j].mean() for j in range(3)])
    error = sum(np.sum((transformed[groups == j]-group_means[j])**2) for j in range(3))
    df = len(groups)-3
    estimate = group_means.mean()
    se = np.sqrt(error/df*np.sum(1/np.bincount(groups))/9)
    np.testing.assert_allclose(contrast["estimates"][["estimate", "std_error"]].iloc[0],
                               [estimate, se], rtol=3e-11, atol=2e-12)
    expected_f = ((estimate-.01)/se)**2
    np.testing.assert_allclose(contrast["multivariate"]["statistic"], expected_f,
                               rtol=3e-11, atol=2e-12)
    assert (contrast["multivariate"]["f_type"] == "exact").all()
    for key in ("between_design_columns", "typed_cell_order"):
        forged = restored(result)
        state = forged.attrs["rm_contrast_state"]
        state[key][0], state[key][1] = state[key][1], state[key][0]
        state["sha256"] = rehash(state)
        with pytest.raises(AnalysisError):
            rm_mtest(forged, l_matrix, M=m_matrix)


def test_legacy_rm_rounded_cell_coefficients_are_accurate_or_refused():
    frame, _, groups = rm_domain("clinical")
    frame["response"] += 1e12
    response = frame.pivot(index="subject", columns=["phase", "time"], values="response").to_numpy()
    transformed = response[:, 0]-response[:, 1]
    expected = np.mean([transformed[groups == j].mean() for j in range(3)])
    result = rm_anova(frame, "response", "subject", ["phase", "time"], between=["group"])
    try:
        contrast = rm_mtest(restored(result), [[1, 0, 0]], M=[[1], [-1], [0], [0], [0], [0]])
    except AnalysisError as error:
        assert error.code == "unresolved_precision"
    else:
        np.testing.assert_allclose(contrast["estimates"]["estimate"], expected, rtol=2e-10, atol=2e-11)
