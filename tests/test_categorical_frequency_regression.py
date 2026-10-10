"""Independent repeated-case, quadratic and spectral weighted CATREG checks."""

import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import lsq_linear
import torch

from scripts.torch_test_state import preserve_torch_default_device

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.categorical import frequency_regression as m
from openecon.econometrics.categorical import frequency as shared
from openecon.econometrics.categorical.optimal import catreg_nominal, catreg_ordinal
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


def sample():
    rng = np.random.default_rng(928)
    frame = pd.DataFrame({"a": rng.choice(["red", "blue", "green"], 60),
                          "b": rng.choice(["low", "mid", "high"], 60),
                          "x": rng.normal(size=60), "v": rng.normal(size=60),
                          "f": rng.integers(1, 6, size=60)})
    frame["y"] = (frame.a.map({"red": 2., "blue": -1., "green": .2})
                  + frame.b.map({"low": 2., "mid": 1., "high": -2.})
                  + .8*frame.x + rng.normal(scale=.3, size=60))
    return frame


def fit(kind, frame=None, **kwargs):
    frame = sample() if frame is None else frame
    defaults = dict(frequency="f", n_starts=2, maxiter=300, tol=1e-10)
    defaults.update(kwargs)
    if kind == "nominal":
        return m.catreg_nominal_fweight(frame, "y", ["a", "b", "x"],
                                        scales={"a": "nominal", "b": "nominal", "x": "numeric"}, **defaults)
    if kind == "ordinal":
        return m.catreg_ordinal_fweight(frame, "y", ["a", "b", "x"],
            scales={"a": "nominal", "b": "ordinal", "x": "numeric"},
            orders={"b": ["low", "mid", "high"]}, **defaults)
    if kind == "nominal_response":
        return m.catreg_nominal_response_fweight(frame, "a", ["b", "x", "v"],
             scales={"b": "nominal", "x": "numeric", "v": "numeric"}, **defaults)
    return m.catreg_ordinal_response_fweight(frame, "b", ["a", "x", "v"],
        outcome_order=["low", "mid", "high"],
        scales={"a": "nominal", "x": "numeric", "v": "numeric"}, **defaults)


def weighted_ls(matrix, y, counts):
    root = np.sqrt(counts)
    return matrix@np.linalg.lstsq(matrix*root[:, None], np.asarray(y)*root, rcond=None)[0]


def dummy_oracle(frame):
    columns = [np.ones(len(frame)), frame.x.to_numpy()]
    for name in ("a", "b"):
        columns += [(frame[name] == value).to_numpy(float) for value in sorted(frame[name].unique())[1:]]
    return weighted_ls(np.column_stack(columns), frame.y, frame.f.to_numpy())


def quadratic_oracle(frame):
    # Separate nonnegative least-squares cones for signed monotone effects.
    columns = [np.ones(len(frame)), frame.x.to_numpy()]
    columns += [(frame.a == value).to_numpy(float) for value in ["green", "red"]]
    codes = frame.b.map({"low": 0, "mid": 1, "high": 2}).to_numpy()
    best = None
    root = np.sqrt(frame.f.to_numpy())
    for direction in (1, -1):
        design = np.column_stack(columns+[direction*(codes >= k) for k in (1, 2)])
        lower = np.array([-np.inf]*len(columns)+[0., 0.])
        result = lsq_linear(design*root[:, None], frame.y*root,
                            bounds=(lower, np.full(len(lower), np.inf)), tol=1e-13, lsmr_tol=1e-13)
        predicted = design@result.x
        loss = np.sum(frame.f*(frame.y-predicted)**2)
        if best is None or loss < best[0]:
            best = loss, predicted
    return best[1]


def partition_pava(values, counts):
    best = None
    for breaks in itertools.product((False, True), repeat=len(values)-1):
        stops = [i+1 for i, cut in enumerate(breaks) if cut]+[len(values)]
        projected, start = np.empty(len(values)), 0
        for stop in stops:
            projected[start:stop] = np.average(values[start:stop], weights=counts[start:stop])
            start = stop
        if np.any(np.diff(projected) < -1e-12):
            continue
        loss = np.sum(counts*(projected-values)**2)
        if best is None or loss < best[0]:
            best = loss, projected.copy()
    return best[1]


def response_spectral(frame, columns, groups):
    # Independent category Rayleigh problem after weighted QR projection.
    w = frame.f.to_numpy(float)
    total = w.sum()
    design = frame[columns].to_numpy(float)
    design -= np.average(design, axis=0, weights=w)
    q, _ = np.linalg.qr(design*np.sqrt(w)[:, None])
    codes = np.asarray(groups)
    indicator = np.eye(codes.max()+1)[codes]
    masses = indicator.T@w
    b = np.sqrt(w)[:, None]*indicator/np.sqrt(masses)
    constant = np.sqrt(masses/total)
    center = np.eye(len(masses))-np.outer(constant, constant)
    matrix = center@b.T@q@q.T@b@center
    values, vectors = np.linalg.eigh(matrix)
    score = np.sqrt(total)*vectors[:, -1]/np.sqrt(masses)
    return values[-1], score, weighted_ls(design, score[codes], w)


def check_geometry(result):
    w = result.attrs["frequency_state"]["counts"]
    z = result["transformed"].iloc[:, 2:].to_numpy(float)
    np.testing.assert_allclose(np.average(z, axis=0, weights=w), 0., atol=2e-13)
    np.testing.assert_allclose(np.average(z*z, axis=0, weights=w), 1., atol=2e-13)
    for _, rows in result["quantifications"].groupby("variable"):
        if rows.scale.iloc[0] == "ordinal":
            assert np.all(np.diff(rows.quantification.to_numpy(float)) >= -1e-12)
    for _, rows in result["iterations"].groupby("start"):
        assert np.all(np.diff(rows.objective.to_numpy(float)) <= 1e-9)
    assert "descriptive" in result.attrs["inference"]
    assert not {"p", "std_error", "covariance", "ci_lower"} & set(result["coefficients"].columns)


def test_numeric_limit_independent_weighted_least_squares():
    frame = sample()
    result = m.catreg_nominal_fweight(frame, "y", ["x", "v"], frequency="f",
                                     scales={"x": "numeric", "v": "numeric"})
    expected = weighted_ls(np.column_stack([np.ones(len(frame)), frame.x, frame.v]), frame.y, frame.f.to_numpy())
    np.testing.assert_allclose(result["fitted"].fitted, expected, atol=2e-14)
    assert result["fit"].frequency_n.iloc[0] == frame.f.sum()
    check_geometry(result)


def test_nominal_weighted_dummy_regression_oracle():
    frame = sample()
    result = fit("nominal", frame, tol=1e-12)
    np.testing.assert_allclose(result["fitted"].fitted, dummy_oracle(frame), atol=2e-7, rtol=1e-7)
    check_geometry(result)


def test_ordinal_independent_constrained_quadratic_oracle():
    frame = sample()
    result = fit("ordinal", frame, tol=1e-12)
    np.testing.assert_allclose(result["fitted"].fitted, quadratic_oracle(frame), atol=2e-7, rtol=1e-7)
    assert result["coefficients"].set_index("term").loc["b", "beta_descriptive"] < 0
    check_geometry(result)


def test_ordinal_category_ties_partition_oracle():
    means, counts = np.array([5., 3., 4., 1.]), np.array([2, 6, 4, 8])
    frame = pd.DataFrame({"a": ["a", "b", "c", "d"], "y": means, "f": counts})
    result = m.catreg_ordinal_fweight(frame, "y", ["a"], frequency="f", orders={"a": list("abcd")})
    expected = -partition_pava(-means, counts)
    np.testing.assert_allclose(result["fitted"].fitted, expected, atol=2e-14)
    assert result["fit"].weighted_sse.iloc[0] == pytest.approx(2.4)
    q = result["quantifications"].quantification.to_numpy()
    assert q[1] == pytest.approx(q[2])


def test_nominal_response_generalized_eigen_oracle():
    frame = sample()
    result = m.catreg_nominal_response_fweight(frame, "a", ["x", "v"], frequency="f",
                         scales={"x": "numeric", "v": "numeric"}, tol=1e-12, maxiter=600)
    state = result.attrs["frequency_state"]
    mapping = {label[1]: i for i, label in enumerate(state["response"]["levels"])}
    root, score, predicted = response_spectral(frame, ["x", "v"], frame.a.map(mapping).to_numpy())
    if score[np.argmax(np.abs(score))] < 0:
        score, predicted = -score, -predicted
    assert result["fit"].r_squared.iloc[0] == pytest.approx(root, abs=2e-11)
    np.testing.assert_allclose(state["response"]["quantifications"], score, atol=2e-6)
    np.testing.assert_allclose(result["fitted"].fitted, predicted, atol=2e-6)
    check_geometry(result)


def test_ordinal_response_face_enumeration_and_stationarity():
    # Exhaust all contiguous pooling faces and both eigenvector signs.
    frame = pd.DataFrame({"c": np.repeat(np.arange(4), 4),
       "x": [0., .1, -.1, .2, 2., 2.1, 1.9, 2.2, 1., 1.1, .9, 1.2, 3., 3.1, 2.9, 3.2],
       "f": [1, 2, 3, 1]*4})
    candidates = []
    for breaks in itertools.product((False, True), repeat=3):
        group = np.cumsum([0]+list(breaks))
        if max(group) == 0:
            continue
        root, score, predicted = response_spectral(frame, ["x"], group[frame.c.to_numpy()])
        for sign in (-1, 1):
            if np.all(np.diff(sign*score) >= -1e-12):
                candidates.append((root, sign*score[group], sign*predicted))
    best = max(candidates, key=lambda x: x[0])
    result = m.catreg_ordinal_response_fweight(frame, "c", ["x"], frequency="f",
                           scales={"x": "numeric"}, outcome_order=list(range(4)), tol=1e-12)
    assert result["fit"].r_squared.iloc[0] == pytest.approx(best[0], abs=2e-12)
    np.testing.assert_allclose(result.attrs["frequency_state"]["response"]["quantifications"], best[1], atol=2e-9)
    np.testing.assert_allclose(result["fitted"].fitted, best[2], atol=2e-9)
    check_geometry(result)


@pytest.mark.parametrize("kind", ["nominal", "ordinal", "nominal_response", "ordinal_response"])
def test_common_count_scaling_split_rows_and_literal_expansion(kind):
    frame = sample()
    result = fit(kind, frame)
    scaled = fit(kind, frame.assign(f=frame.f*7))
    expanded = frame.loc[frame.index.repeat(frame.f)].reset_index(drop=True).assign(f=1)
    repeated = fit(kind, expanded)
    predictions = m.catreg_fweight_predict(repeated, frame)["predictions"].iloc[:, 1]
    np.testing.assert_allclose(scaled["fitted"].fitted, result["fitted"].fitted, atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(predictions, result["fitted"].fitted, atol=3e-6, rtol=3e-6)
    assert repeated.attrs["frequency_n"] == result.attrs["frequency_n"]
    assert len(result["fitted"]) == len(frame)


@pytest.mark.parametrize("kind", ["nominal", "ordinal"])
def test_all_one_counts_match_existing_unweighted_estimators(kind):
    frame = sample().assign(f=1)
    result = fit(kind, frame, tol=1e-12)
    kwargs = dict(scales={"a": "nominal", "b": "nominal" if kind == "nominal" else "ordinal", "x": "numeric"}, n_starts=2, tol=1e-12, maxiter=300)
    old = (catreg_nominal if kind == "nominal" else catreg_ordinal)(frame, "y", ["a", "b", "x"],
                 **kwargs, **({} if kind == "nominal" else {"orders": {"b": ["low", "mid", "high"]}}))
    np.testing.assert_allclose(result["fitted"].fitted, old["fitted"].fitted, atol=2e-7)


@pytest.mark.parametrize("kind", ["nominal", "ordinal", "nominal_response", "ordinal_response"])
def test_complete_restore_latex_query_no_count_and_partial_positions(kind):
    frame = sample()
    result = fit(kind, frame)
    text = summary_state(result)
    restored = restore_summary(text)
    assert summary_state(restored) == text
    assert restored.to_latex() == result.to_latex()
    for name in result:
        assert restored[name].equals(result[name])
    variables = result.attrs["frequency_state"]["variables"]
    query = frame[variables].copy()
    query.index = ["same"]*len(query)
    query.iloc[1, 0] = None
    predictions = m.catreg_fweight_predict(text, query, missing="drop")
    assert predictions.attrs["sample_positions"] == [i for i in range(len(query)) if i != 1]
    np.testing.assert_array_equal(predictions["predictions"].iloc[:, 1], result["fitted"].fitted.drop(1))
    assert "frequency" not in predictions["predictions"].columns
    assert restore_summary(summary_state(predictions)).to_latex() == predictions.to_latex()


def test_zero_rows_excluded_before_invalid_feature_and_missing_counts():
    frame = sample()
    frame.index = [0]*len(frame)
    additions = pd.DataFrame({"a": [[{"unbounded": "x"}]], "b": [None], "x": [None], "v": [None], "y": [None], "f": [0]})
    extra = pd.concat([frame, additions, frame.iloc[[0]].assign(f=np.nan), frame.iloc[[0]].assign(y=np.nan)], ignore_index=True)
    result = fit("nominal", extra)
    assert result.attrs["sample_positions"] == list(range(60))
    assert result.attrs["frequency_state"]["zero_positions"] == [60]
    assert result.attrs["frequency_state"]["missing_positions"] == [61, 62]
    with pytest.raises(AnalysisError):
        fit("nominal", extra, missing="raise")


@pytest.mark.parametrize("count", [True, -1, .2, np.inf, 1e9+1, "2"])
def test_invalid_count_even_on_feature_missing_row(count):
    frame = sample()
    frame["f"] = frame.f.astype(object)
    frame.loc[0, ["y", "f"]] = [None, count]
    with pytest.raises(AnalysisError):
        fit("nominal", frame)


def test_total_guard_includes_later_dropped_counts():
    frame = sample()
    frame.loc[0, ["y", "f"]] = [None, 10**9]
    with pytest.raises(AnalysisError, match="total"):
        fit("nominal", frame)


@pytest.mark.parametrize("bad", ["unknown", [["nested"]], True])
def test_unknown_or_malformed_query_category(bad):
    result = fit("nominal")
    query = sample().iloc[:2].copy()
    query["a"] = query.a.astype(object)
    query.at[0, "a"] = bad
    with pytest.raises(AnalysisError):
        m.catreg_fweight_predict(result, query)


def test_typed_categories_and_huge_finite_category_label():
    frame = pd.DataFrame({"a": pd.Series([1, 1., True, "1", 1, 1., True, "1"], dtype=object),
                          "y": [1., 2., 3., 4., 1.2, 2.2, 3.2, 4.2], "f": [1, 2]*4})
    result = m.catreg_nominal_fweight(frame, "y", ["a"], frequency="f")
    assert len(result.attrs["frequency_state"]["descriptors"][0]["levels"]) == 4
    m.catreg_fweight_predict(summary_state(result), frame)
    huge = frame.assign(a=pd.Series([1e300, 2e300]*4, dtype=object))
    result = m.catreg_nominal_fweight(huge, "y", ["a"], frequency="f")
    m.catreg_fweight_predict(summary_state(result), huge)


@pytest.mark.parametrize("option", [dict(device="mps"), dict(device="cuda"), dict(n_starts=0), dict(maxiter=1001), dict(tol=0)])
def test_invalid_controls(option):
    with pytest.raises(AnalysisError):
        fit("nominal", **option)


def test_dataframe_only_distinct_columns_numeric_dtype_and_order_domain():
    with pytest.raises(AnalysisError):
        fit("nominal", sample().to_dict("list"))
    with pytest.raises(AnalysisError):
        fit("nominal", frequency="a")
    with pytest.raises(AnalysisError):
        fit("nominal", sample().assign(x=True))
    with pytest.raises(AnalysisError):
        m.catreg_ordinal_fweight(sample(), "y", ["a"], frequency="f", orders={"a": ["red", "blue"]})


def test_rank_nonconvergence_degeneracy_and_tied_nominal_roots_refused():
    frame = sample().assign(v=lambda d: d.x)
    with pytest.raises(AnalysisError, match="rank"):
        m.catreg_nominal_fweight(frame, "y", ["x", "v"], frequency="f", scales={"x": "numeric", "v": "numeric"})
    with pytest.raises(AnalysisError):
        fit("nominal", maxiter=1)
    with pytest.raises(AnalysisError):
        m.catreg_nominal_fweight(sample().assign(a="same"), "y", ["a"], frequency="f")
    c = np.tile(np.arange(3), 4)
    tied = pd.DataFrame({"c": c, "x": (c == 0).astype(float), "v": (c == 1).astype(float), "f": 1})
    with pytest.raises(AnalysisError, match="roots tie"):
        m.catreg_nominal_response_fweight(tied, "c", ["x", "v"], frequency="f", scales={"x": "numeric", "v": "numeric"})


def test_private_seed_and_default_device_resident_cpu():
    before = torch.random.get_rng_state().clone()
    with preserve_torch_default_device():
        torch.set_default_device("meta")
        result = fit("nominal")
        prediction = m.catreg_fweight_predict(result, sample())
    assert torch.equal(before, torch.random.get_rng_state())
    assert result.attrs["device"] == prediction.attrs["device"] == "cpu"


@pytest.mark.parametrize("budget", [dict(max_bytes=1), dict(max_work=1)])
def test_fit_admission_before_weights_or_preparation(monkeypatch, budget):
    def forbidden(*args, **kwargs):
        pytest.fail("Numerical materialization preceded admission")
    monkeypatch.setattr(shared, "_weights", forbidden)
    monkeypatch.setattr(shared, "_prepare", forbidden)
    with pytest.raises(AnalysisError):
        fit("nominal", **budget)
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        fit("nominal")


@pytest.mark.parametrize("mutation", ["nested", "long", "extra", "descriptor", "hugeint", "table"])
def test_untrusted_state_before_seal_and_tensor(monkeypatch, mutation):
    result = copy.deepcopy(fit("nominal"))
    state = result.attrs["frequency_state"]
    if mutation == "nested":
        state["raw"][0][0] = ["float", {"nested": [1]*100}]
    elif mutation == "long":
        state["raw"][0][1] = ["str", "x"*257]
    elif mutation == "extra":
        state["extra"] = [1]*1000
    elif mutation == "descriptor":
        state["descriptors"][0]["extra"] = [1]*1000
    elif mutation == "hugeint":
        state["beta"][0] = 10**10000
    else:
        result["fitted"] = pd.DataFrame({"bad": [[1, 2]]})
    def forbidden(*args, **kwargs):
        pytest.fail("Malformed state reached hashing or tensors")
    monkeypatch.setattr(shared, "_seal", forbidden)
    monkeypatch.setattr(shared, "_weights", forbidden)
    with pytest.raises(AnalysisError):
        m.catreg_fweight_predict(result, sample())


@pytest.mark.parametrize("field", ["counts", "raw", "beta", "objective", "trace", "table", "metadata", "response_roots"])
def test_semantic_tampering_rehashed_is_refused(field):
    result = copy.deepcopy(fit("nominal_response" if field == "response_roots" else "nominal"))
    state = result.attrs["frequency_state"]
    if field == "counts":
        state["counts"][0] += 1
        result.attrs["frequency_n"] += 1
    elif field == "raw":
        state["raw"][0][0][1] += 1
    elif field == "beta":
        state["beta"][0] += .1
    elif field == "objective":
        state["objective"] += .01
    elif field == "trace":
        state["histories"][-1][5] = 1.
    elif field == "table":
        result["fitted"].iat[0, 3] += .01
    elif field == "metadata":
        result.attrs["inference"] = "ordinary p-values"
    else:
        state["response_roots"][-1] += .01
    result.attrs["state_sha256"] = shared._seal(state)
    with pytest.raises(AnalysisError):
        m.catreg_fweight_predict(result, sample())


@pytest.mark.parametrize("field", [3, 4, 5])
def test_forged_trace_with_matching_full_tables_and_checksum_is_refused(field):
    result = copy.deepcopy(fit("nominal"))
    state = result.attrs["frequency_state"]
    index = max(i for i, row in enumerate(state["histories"]) if row[0] == state["chosen_start"])
    state["histories"][index][field] = 123.
    result["iterations"].iat[index, field] = 123.
    result.attrs["state_sha256"] = shared._seal(state)
    result = shared._save(result)
    with pytest.raises(AnalysisError, match="trace|convergence"):
        m.catreg_fweight_predict(result, sample())


def test_saved_resource_records_cannot_understate_dimension_cost():
    result = copy.deepcopy(fit("nominal"))
    result.attrs["resources"]["estimated_workspace_bytes"] = 1
    result.attrs["resources"]["buffers"]["physical_rows_and_numerical_workspace"] = 1
    result = shared._save(result)
    with pytest.raises(AnalysisError, match="admission"):
        m.catreg_fweight_predict(result, sample())


@pytest.mark.parametrize("field", ["converged", "settings", "sources", "method"])
def test_nonprimitive_metadata_refused_before_seal(monkeypatch, field):
    result = copy.deepcopy(fit("nominal"))
    result.attrs[field] = np.array([1, 2])
    def forbidden(*args, **kwargs):
        pytest.fail("Nonprimitive metadata reached hashing")
    monkeypatch.setattr(shared, "_seal", forbidden)
    with pytest.raises(AnalysisError):
        m.catreg_fweight_predict(result, sample())


def test_combined_query_and_validation_work_budget():
    result = fit("nominal")
    _, _, validation_work = m._checked(result, shared.BYTES, shared.WORK)
    query = pd.concat([sample()]*10, ignore_index=True)
    query_work = 64*len(query)*len(result.attrs["variables"])**2
    with pytest.raises(AnalysisError):
        m.catreg_fweight_predict(result, query, max_work=max(validation_work, query_work))
    m.catreg_fweight_predict(result, query, max_work=validation_work+query_work)


def test_saved_json_and_validation_budget_before_seal(monkeypatch):
    result = fit("nominal")
    text = summary_state(result)
    def forbidden(*args, **kwargs):
        pytest.fail("Saved materialization preceded admission")
    monkeypatch.setattr(shared, "_seal", forbidden)
    for value in (result, text):
        with pytest.raises(AnalysisError):
            m.catreg_fweight_predict(value, sample(), max_bytes=1)


def test_full_saved_json_rejects_preview_and_nonfinite_tokens():
    result = fit("nominal")
    payload = json.loads(summary_state(result))
    payload["tables"].pop("iterations")
    with pytest.raises(AnalysisError):
        m.catreg_fweight_predict(json.dumps(payload), sample())
    with pytest.raises(AnalysisError):
        m.catreg_fweight_predict(summary_state(result).replace('"float64"', 'NaN', 1), sample())


def test_delivery_fixture_all_four_full_restore():
    from benchmarks.categorical_frequency_eight import run_stage
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        for stage in range(4):
            result = run_stage(stage, Path("tests/fixtures/categorical_frequency"), Path(directory))
            assert result.attrs["n"] == 48
            assert result.attrs["frequency_n"] == 120
