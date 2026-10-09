"""Independent nested-regression, replication and Gaussian classification oracles."""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import f as f_dist

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate import discrim_selection as s
from openecon.resources import use_workspace_budget


def domain(seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(3), 80)
    means = np.array([[-1, -1], [0, 1], [1, 0]])
    a, b = (rng.normal(size=(240, 2))+means[groups]).T
    proxy = a+b+.5*rng.normal(size=240)
    out = pd.DataFrame({"proxy": proxy, "a": a, "b": b, "noise": rng.normal(size=240), "g": groups})
    out["w"] = rng.integers(1, 5, len(out))
    return out, ["proxy", "a", "b", "noise"]


def biomarker_domain():
    rng = np.random.default_rng(901)
    groups = np.repeat(["healthy", "stage1", "stage2", "stage3"], [41, 47, 53, 59])
    codes = pd.Categorical(groups, categories=["healthy", "stage1", "stage2", "stage3"]).codes
    means = np.array([[0, 0, 0], [1, 0, .3], [0, 1.4, -.4], [-.8, -.5, 1]])
    covariance = np.array([[1, .2, -.15], [.2, 1.2, .1], [-.15, .1, .7]])
    markers = rng.multivariate_normal(np.zeros(3), covariance, len(groups))+means[codes]
    out = pd.DataFrame(markers, columns=["protein", "metabolite", "enzyme"])
    out["panel"] = .8*out.protein-.5*out.enzyme+rng.normal(scale=.7, size=len(out))
    out["age"] = rng.normal(size=len(out))
    out["g"] = groups
    out["w"] = rng.integers(1, 4, len(out))
    return out, ["panel", "protein", "metabolite", "enzyme", "age"]


def expanded(frame):
    return frame.loc[frame.index.repeat(frame.w.astype(int))].reset_index(drop=True)


def reference_test(frame, retained, candidate):
    # Solve nested raw-data regressions. Do not use production moment identities.
    x = frame[retained].to_numpy() if retained else np.empty((len(frame), 0))
    x = x-x.mean(0)
    response = frame[candidate].to_numpy()
    response = response-response.mean()
    base = np.column_stack([np.ones(len(frame)), x])
    groups = pd.get_dummies(frame.g, drop_first=True).to_numpy(dtype=float)
    full = np.column_stack([base, groups])
    resid0 = response-base@np.linalg.lstsq(base, response, rcond=None)[0]
    resid1 = response-full@np.linalg.lstsq(full, response, rcond=None)[0]
    reduced, error = resid0@resid0, resid1@resid1
    df1, df2 = groups.shape[1], len(frame)-full.shape[1]
    statistic = (reduced-error)/error*df2/df1
    return statistic, df1, df2, f_dist.sf(statistic, df1, df2), error/reduced


def reference_path(frame, names, method, include=(), p_enter=.05, p_remove=.1):
    current = names.copy() if method == "backward" else [name for name in names if name in include]
    moves, tests = [], []
    while True:
        chosen = None
        if method != "forward":
            for name in current:
                if name in include:
                    continue
                values = reference_test(frame, [other for other in current if other != name], name)
                tests.append((len(moves)+1, "remove", name, len(current), values))
                if values[3] > p_remove and (chosen is None or values[3] > chosen[2][3]):
                    chosen = ("remove", name, values)
        if chosen is None and method != "backward":
            for name in names:
                if name in current:
                    continue
                values = reference_test(frame, current, name)
                tests.append((len(moves)+1, "enter", name, len(current), values))
                if values[3] < p_enter and (chosen is None or values[3] < chosen[2][3]):
                    chosen = ("enter", name, values)
        if chosen is None:
            return current, moves, tests
        action, name, values = chosen
        current = [item for item in names if item in current or item == name] if action == "enter" \
            else [item for item in current if item != name]
        moves.append((action, name, values))


def probabilities(training, names, query, labels, priors):
    if not names:
        return np.tile(priors, (len(query), 1))
    anchor = training[names].to_numpy()[0]
    x = training[names].to_numpy()-anchor
    blocks = [x[training.g.to_numpy() == label] for label in labels]
    means = np.array([block.mean(0) for block in blocks])
    within = sum((block-mean).T@(block-mean) for block, mean in zip(blocks, means))
    covariance = within/(len(training)-len(labels))
    query = query[names].to_numpy()-anchor
    logits = np.column_stack([-.5*np.einsum("ij,ij->i", query-mean,
        np.linalg.solve(covariance, (query-mean).T).T)+np.log(prior)
        for mean, prior in zip(means, priors)])
    posterior = np.exp(logits-logits.max(1, keepdims=True))
    return posterior/posterior.sum(1, keepdims=True)


def restored(result):
    payload = json.loads(json.dumps({"attrs": result.attrs, "tables": {
        key: value.astype(object).where(value.notna(), None).to_dict(orient="split")
        for key, value in result.items()}, "axes": {
        key: [value.index.names, value.columns.names, value.attrs] for key, value in result.items()}}, allow_nan=False))
    tables = {key: pd.DataFrame(**value) for key, value in payload["tables"].items()}
    for key, frame in tables.items():
        frame.index.names, frame.columns.names, frame.attrs = payload["axes"][key]
    return TableSet(tables, **payload["attrs"])


@pytest.mark.parametrize("fixture", [domain, biomarker_domain])
@pytest.mark.parametrize("method", ["forward", "backward", "stepwise"])
@pytest.mark.parametrize("frequency", [False, True])
def test_complete_paths_reference_probabilities_and_gaussian_posterior(fixture, method, frequency):
    frame, names = fixture()
    include = [names[-1]]
    training = expanded(frame) if frequency else frame
    expected, moves, tests = reference_path(training, names, method, include)
    fit = s.discrim_stepwise(frame, "g", names, method=method, include=include,
                            weights="w" if frequency else None, priors="proportional")
    assert fit.attrs["variables"] == expected
    assert list(zip(fit["selection_history"].action, fit["selection_history"].variable)) == [(a, n) for a, n, _ in moves]
    assert len(fit["candidate_tests"]) == len(tests)
    for actual, (step, action, name, size, values) in zip(fit["candidate_tests"].itertuples(index=False), tests):
        assert (actual.step, actual.action, actual.variable, actual.model_size) == (step, action, name, size)
        statistic, df1, df2, probability, partial = values
        np.testing.assert_allclose([actual.statistic, actual.reference_p, actual.partial_lambda],
                                   [statistic, probability, partial], rtol=2e-9, atol=2e-13)
        assert (actual.df1, actual.df2) == (df1, df2)
    labels, priors = fit.attrs["groups"], fit.attrs["priors"]
    oracle = probabilities(training, expected, frame, labels, priors)
    predicted = s.discrim_stepwise_predict(fit, frame)
    np.testing.assert_allclose(predicted.iloc[:, 1:], oracle, rtol=2e-12, atol=2e-13)
    assert predicted.predicted.tolist() == [labels[i] for i in oracle.argmax(1)]
    assert fit.attrs["post_selection_inference"] is False and fit.attrs["cross_validation"] is False
    assert not any(key in fit for key in ("box_m", "canonical_functions", "tests_of_equality"))
    assert all("p_value" not in table.columns and "chi2" not in table.columns for table in fit.values())
    assert "literal independent replication" in fit.attrs["inferential_assumptions"]


def test_bidirectional_genuine_removal_and_univariate_conditional_df():
    frame, names = domain()
    fit = s.discrim_stepwise(frame, "g", names)
    assert list(fit["selection_history"].action) == ["enter", "enter", "enter", "remove"]
    assert list(fit["selection_history"].variable) == ["proxy", "a", "b", "proxy"]
    assert fit.attrs["variables"] == ["a", "b"]
    assert list(fit["selection_history"].df2) == [237, 236, 235, 235]


@pytest.mark.parametrize("fixture", [domain, biomarker_domain])
def test_literal_expansion_all_tables_accounting_and_roundtrip(fixture):
    frame, names = fixture()
    fit = s.discrim_stepwise(frame, "g", names, weights="w", priors="proportional")
    raw = s.discrim_stepwise(expanded(frame), "g", names, priors="proportional")
    for key in fit:
        if key in ("groups", "classification_table_physical"):
            continue
        if key in ("candidate_tests", "selection_history"):
            assert fit[key].iloc[:, :4].equals(raw[key].iloc[:, :4])
            np.testing.assert_allclose(fit[key].iloc[:, 4:].to_numpy(float), raw[key].iloc[:, 4:].to_numpy(float),
                                       rtol=2e-10, atol=2e-12)
        else:
            np.testing.assert_allclose(fit[key].to_numpy(float), raw[key].to_numpy(float), rtol=2e-10, atol=2e-12)
    assert fit.attrs["n"] == int(frame.w.sum())
    assert fit.attrs["n_physical"] == len(frame)
    saved = restored(fit)
    assert saved.attrs == fit.attrs
    for key in fit:
        pd.testing.assert_frame_equal(saved[key], fit[key], check_dtype=False, check_index_type=False,
                                      check_column_type=False, check_frame_type=False)
    pd.testing.assert_frame_equal(s.discrim_stepwise_predict(saved, frame), s.discrim_stepwise_predict(fit, frame))
    assert "tabular" in fit["candidate_tests"].to_latex()


def test_fixed_missing_sample_zero_frequency_and_selected_prediction_alignment():
    frame, names = domain()
    frame["w"] = 1
    frame.index = pd.Index(np.arange(len(frame))*7, name="person")
    frame.loc[frame.index[0], "noise"] = np.nan
    frame.loc[frame.index[1], "w"] = np.nan
    frame.loc[frame.index[2], "w"] = 0
    fit = s.discrim_stepwise(frame, "g", names, weights="w")
    assert fit.attrs["n_missing"] == 2 and fit.attrs["n_zero_weight"] == 1
    assert fit.attrs["n_physical"] == len(frame)-3
    raw = s.discrim_stepwise(expanded(frame.dropna()), "g", names)
    assert fit.attrs["variables"] == raw.attrs["variables"]
    pd.testing.assert_frame_equal(fit["selection_history"].iloc[:, :4], raw["selection_history"].iloc[:, :4])
    query = frame.copy()
    query.loc[query.index[3], fit.attrs["variables"][0]] = np.nan
    pred = s.discrim_stepwise_predict(restored(fit), query)
    assert pred.index.equals(query.index)
    assert pred.loc[query.index[3]].isna().all()
    assert pred.loc[query.index[0]].notna().all()  # unselected missing noise is irrelevant to projection
    with pytest.raises(AnalysisError, match="missing"):
        s.discrim_stepwise(frame, "g", names, weights="w", missing="raise")


@pytest.mark.parametrize("method", ["forward", "backward", "stepwise"])
def test_empty_prior_only_and_forced_variable(method):
    frame = pd.DataFrame({"g": np.repeat(["a", "b", "c"], 5), "x": np.tile(np.arange(-2, 3), 3)})
    fit = s.discrim_stepwise(frame, "g", ["x"], method=method, priors=[.2, .3, .5])
    assert fit.attrs["variables"] == []
    pred = s.discrim_stepwise_predict(restored(fit), pd.DataFrame(index=pd.Index(range(7), name="newrow")))
    assert len(pred) == 7 and pred.predicted.tolist() == ["c"]*7
    np.testing.assert_array_equal(pred.iloc[:, 1:], np.tile([.2, .3, .5], (7, 1)))
    assert fit["classification_functions"].index.tolist() == ["Intercept"]
    forced = s.discrim_stepwise(frame, "g", ["x"], method=method, include=["x"])
    assert forced.attrs["variables"] == ["x"] and len(forced["selection_history"]) == 0


@pytest.mark.parametrize("kwargs", [{"method": "qda"}, {"include": ["absent"]}, {"include": ["a", "a"]},
    {"p_enter": 0}, {"p_enter": 1}, {"p_remove": np.nan}, {"p_remove": 1}, {"max_steps": True},
    {"max_steps": 0}, {"max_steps": 129}, {"weight_type": "aweight"}, {"weights": "g"}])
def test_option_refusals(kwargs):
    frame, names = domain()
    with pytest.raises(AnalysisError):
        s.discrim_stepwise(frame, "g", names, **kwargs)


@pytest.mark.parametrize("weights", [-1, .5, np.inf, 2**53+1])
def test_invalid_frequencies_before_conversion(weights):
    frame, names = domain()
    frame["w"] = weights
    with pytest.raises(AnalysisError):
        s.discrim_stepwise(frame, "g", names, weights="w")


def test_total_frequency_limit_zero_group_and_no_expansion():
    frame, names = domain()
    with pytest.raises(AnalysisError):
        s.discrim_stepwise(frame.assign(w=2**52), "g", names, weights="w")
    with pytest.raises(AnalysisError):
        s.discrim_stepwise(frame.assign(w=0), "g", names, weights="w")
    zero = frame.copy()
    zero.loc[zero.g == 2, "w"] = 0
    fit = s.discrim_stepwise(zero, "g", names, weights="w")
    assert fit.attrs["groups"] == [0, 1]
    assert fit.attrs["n_zero_weight"] == 80
    large = frame.assign(w=1_000_000)
    fit = s.discrim_stepwise(large, "g", names, weights="w", include=names)
    assert fit.attrs["n"] == 240_000_000 and fit.attrs["n_physical"] == 240


def test_full_candidate_rank_scale_invariance_and_large_offsets():
    frame, names = domain()
    base = s.discrim_stepwise(frame, "g", names)
    scaled = frame.copy()
    scaled[names] = scaled[names]*[1e-6, 1e5, 2e-3, 1e8]+[1e3, 1e10, 10, 1e10]
    other = s.discrim_stepwise(scaled, "g", names)
    assert other.attrs["variables"] == base.attrs["variables"]
    np.testing.assert_allclose(other["candidate_tests"].statistic, base["candidate_tests"].statistic,
                               rtol=2e-6, atol=1e-7)
    query = s.discrim_stepwise_predict(restored(other), scaled)
    np.testing.assert_allclose(query.iloc[:, 1:], s.discrim_stepwise_predict(base, frame).iloc[:, 1:],
                               rtol=2e-6, atol=2e-7)
    with pytest.raises(AnalysisError, match="singular|redundant|condition"):
        s.discrim_stepwise(frame.assign(alias=frame.a), "g", [*names, "alias"])
    with pytest.raises(AnalysisError):
        s.discrim_stepwise(frame.assign(flat=frame.g), "g", [*names, "flat"])
    with pytest.raises(AnalysisError):
        s.discrim_stepwise(frame.assign(almost=frame.a+1e-8*frame.noise), "g", [*names, "almost"])


def test_max_steps_and_cycle_refuse_incomplete_selection(monkeypatch):
    frame, names = domain()
    with pytest.raises(AnalysisError, match="max_steps"):
        s.discrim_stepwise(frame, "g", names, max_steps=1)
    original = s._candidate
    def cycle(within, means, counts, retained, candidate):
        values = original(within, means, counts, retained, candidate)
        values[-1] = .07
        return values
    monkeypatch.setattr(s, "_candidate", cycle)
    with pytest.raises(AnalysisError, match="cycling"):
        s.discrim_stepwise(frame, "g", names, p_enter=.1, p_remove=.05)


def test_exact_input_order_ties(monkeypatch):
    frame, names = domain()
    original = s._candidate
    def tie(within, means, counts, retained, candidate):
        values = original(within, means, counts, retained, candidate)
        values[-1] = .01 if not retained else .5
        values[2] = 10 if not retained else 0
        return values
    monkeypatch.setattr(s, "_candidate", tie)
    fit = s.discrim_stepwise(frame, "g", names, method="forward")
    assert fit.attrs["variables"] == [names[0]]


@pytest.mark.parametrize("tamper", ["digest", "covariance", "selection", "counts", "accounting", "origin", "table", "history"])
def test_saved_state_tampering_even_with_new_digest_is_refused(tamper):
    frame, names = domain()
    fit = restored(s.discrim_stepwise(frame, "g", names))
    state = fit.attrs["selection_state"]
    if tamper == "digest":
        state["priors"][0] += .1
    elif tamper == "covariance":
        state["group_sscp"][0][0][0] += 1
    elif tamper == "selection":
        state["selected"] = [names[0]]
        fit.attrs["variables"] = [names[0]]
    elif tamper == "counts":
        state["counts"][0] = 2**53
    elif tamper == "accounting":
        state["n_zero_weight"] = 1
    elif tamper == "origin":
        state["origin"] = [[0]*len(names)]
    elif tamper == "table":
        fit["pooled_covariance"].iloc[0, 0] += 1
    else:
        state["history"][0][2] = "noise"
    if tamper != "digest":
        fit.attrs["selection_state_sha256"] = s.o._hash(state)
    with pytest.raises(AnalysisError, match="state|inconsistent"):
        s.discrim_stepwise_predict(fit, frame)


def test_empty_state_still_validates_all_unselected_moments():
    frame = pd.DataFrame({"g": np.repeat([0, 1, 2], 5), "x": np.tile(np.arange(-2, 3), 3)})
    fit = deepcopy(s.discrim_stepwise(frame, "g", ["x"]))
    fit.attrs["selection_state"]["group_sscp"][0][0][0] = -1
    fit.attrs["selection_state_sha256"] = s.o._hash(fit.attrs["selection_state"])
    with pytest.raises(AnalysisError):
        s.discrim_stepwise_predict(fit, pd.DataFrame(index=range(7)))


def test_resource_and_resident_refusals_precede_dense_work(monkeypatch):
    frame, names = domain()
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError):
            s.discrim_stepwise(pd.concat([frame]*100), "g", names)
    with pytest.raises(AnalysisError, match="resident|Dataset"):
        s.discrim_stepwise(Dataset.from_frame(frame), "g", names)
    monkeypatch.setattr(s, "MAX_WORK", 1)
    with pytest.raises(AnalysisError, match="work budget"):
        s.discrim_stepwise(frame, "g", names)


def test_geometry_bounds_before_conversion():
    with pytest.raises(AnalysisError):
        s.discrim_stepwise(pd.DataFrame(index=range(100_001)), "g", ["x"])
    with pytest.raises(AnalysisError):
        s.discrim_stepwise(pd.DataFrame(), "g", [f"x{i}" for i in range(33)])


def test_public_exports_and_mapping_resident_inputs():
    import openecon as oe
    frame, names = domain()
    assert callable(oe.discrim_stepwise) and callable(oe.discrim_stepwise_predict)
    one = oe.discrim_stepwise(frame.to_dict("list"), "g", names, weights="w")
    two = oe.discrim_stepwise(frame.to_dict("records"), "g", names, weights="w")
    assert one.attrs["selection_state_sha256"] == two.attrs["selection_state_sha256"]
