"""Independent response-contrast and full-refit prediction bootstrap oracles."""

import copy
import itertools
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.categorical import optimal as base
from openecon.econometrics.categorical.optimal_options import (
    catreg_bootstrap, catreg_nominal_response, catreg_ordinal_response,
    catreg_outcome_predict,
)
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


def response_fixture():
    counts = np.array([6, 11, 8, 15])
    labels = np.repeat(np.arange(4), counts)
    rng = np.random.default_rng(417)
    means = np.array([[-2., 0.], [2., 1.], [-1., -1.], [3., .2]])
    x = means[labels]+rng.normal(size=(len(labels), 2))*[.7, 1.3]
    return pd.DataFrame({"y": labels, "x1": x[:, 0], "x2": x[:, 1]})


def projection(matrix):
    matrix = matrix-matrix.mean(0)
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    u = u[:, s > s[0]*1e-12]
    return u@u.T


def response_eigen_oracle(indicator, projector):
    # Solve a generalized Rayleigh problem on the weighted centered category
    # space. This uses a dense NumPy spectral oracle, independent of native ALS.
    counts = indicator.sum(0)
    n = indicator.shape[0]
    b = indicator/np.sqrt(counts)
    constant = np.sqrt(counts/n)
    center = np.eye(len(counts))-np.outer(constant, constant)
    k = center@(b.T@projector@b)@center
    values, vectors = np.linalg.eigh((k+k.T)/2)
    q = np.sqrt(n)*vectors[:, -1]/np.sqrt(counts)
    return values[-1], q, values


def ordinal_face_oracle(indicator, projector):
    best = None
    k = indicator.shape[1]
    for mask in range(1, 1 << (k-1)):
        cuts = [0]+[j+1 for j in range(k-1) if mask & (1 << j)]+[k]
        merge = np.zeros((k, len(cuts)-1))
        for block, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
            merge[lo:hi, block] = 1
        _, block_q, _ = response_eigen_oracle(indicator@merge, projector)
        for direction in (1, -1):
            q = direction*(merge@block_q)
            if np.all(np.diff(q) >= -1e-10):
                r2 = np.linalg.norm(projector@indicator@q)**2/len(indicator)
                if best is None or r2 > best[0]:
                    best = (r2, q)
    return best


def isotonic_partition_oracle(target, counts):
    best = None
    for breaks in itertools.product((False, True), repeat=len(target)-1):
        stops = [i+1 for i, cut in enumerate(breaks) if cut]+[len(target)]
        start, q = 0, np.empty(len(target))
        for stop in stops:
            q[start:stop] = np.average(target[start:stop], weights=counts[start:stop])
            start = stop
        if np.any(np.diff(q) < -1e-12):
            continue
        loss = np.sum(counts*(q-target)**2)
        if best is None or loss < best[0]:
            best = (loss, q)
    q = best[1]-np.average(best[1], weights=counts)
    return q/np.sqrt(np.average(q*q, weights=counts))


def assert_complete_restore(result):
    state = summary_state(result)
    restored = restore_summary(state)
    assert summary_state(restored) == state
    assert result.to_latex() == restored.to_latex()
    for name in result:
        assert result[name].to_latex() == restored[name].to_latex()
    return restored


def check_response_fit(result):
    y = result["fitted"].quantified_observed.to_numpy(float)
    fitted = result["fitted"].quantified_fitted.to_numpy(float)
    z = result["transformed"].iloc[:, 1:].to_numpy(float)
    beta = result["coefficients"].beta_quantified_response.to_numpy(float)[1:]
    np.testing.assert_allclose(y.mean(), 0, atol=1e-14)
    np.testing.assert_allclose(np.mean(y*y), 1, atol=1e-14)
    np.testing.assert_allclose(fitted, z@beta, atol=1e-13)
    np.testing.assert_allclose(z.T@(y-fitted), 0, atol=3e-12)
    for _, rows in result["iterations"].groupby("start"):
        assert np.all(np.diff(rows.objective) <= 1e-9)
    assert not {"std_error", "p", "ci_lower", "covariance"}.intersection(result["coefficients"].columns)


def test_nominal_response_generalized_eigenvalue_and_canonical_sign():
    frame = response_fixture()
    projector = projection(frame[["x1", "x2"]].to_numpy())
    indicator = np.eye(4)[frame.y]
    r2, q, values = response_eigen_oracle(indicator, projector)
    assert values[-1]-values[-2] > .7
    if q[np.argmax(np.abs(q))] < 0:
        q = -q
    result = catreg_nominal_response(frame, "y", ["x1", "x2"],
        scales={"x1": "numeric", "x2": "numeric"}, tol=1e-12)
    np.testing.assert_allclose(result["fit"].r_squared, r2, atol=2e-13)
    np.testing.assert_allclose(result.attrs["response_state"]["response"]["quantifications"], q, atol=3e-7)
    np.testing.assert_allclose(result["fitted"].quantified_fitted, projector@indicator@q, atol=3e-7)
    assert q[np.argmax(np.abs(q))] > 0
    check_response_fit(result)
    restored = assert_complete_restore(result)
    prediction = catreg_outcome_predict(restored, frame)
    np.testing.assert_allclose(prediction["predictions"].quantified_fitted, result["fitted"].quantified_fitted, atol=1e-14)


def test_ordinal_response_global_pooling_face_and_independent_stationarity():
    frame = response_fixture()
    projector = projection(frame[["x1", "x2"]].to_numpy())
    indicator = np.eye(4)[frame.y]
    r2, q = ordinal_face_oracle(indicator, projector)
    result = catreg_ordinal_response(frame, "y", ["x1", "x2"], outcome_order=[0, 1, 2, 3],
        scales={"x1": "numeric", "x2": "numeric"}, tol=1e-12)
    learned = np.array(result.attrs["response_state"]["response"]["quantifications"])
    np.testing.assert_allclose(result["fit"].r_squared, r2, atol=2e-13)
    np.testing.assert_allclose(learned, q, atol=3e-7)
    assert learned[1] == learned[2]
    counts = indicator.sum(0)
    target = indicator.T@result["fitted"].quantified_fitted.to_numpy()/counts
    np.testing.assert_allclose(learned, isotonic_partition_oracle(target, counts), atol=1e-7)
    check_response_fit(result)
    restored = assert_complete_restore(result)
    np.testing.assert_allclose(catreg_outcome_predict(restored, frame)["predictions"].quantified_fitted,
                               result["fitted"].quantified_fitted, atol=1e-14)


@pytest.mark.parametrize("ordinal", [False, True])
def test_binary_response_matches_standardized_binary_ols(ordinal):
    frame = response_fixture().assign(y=lambda x: (x.y >= 2).astype(int))
    z = frame[["x1", "x2"]].to_numpy()
    z = (z-z.mean(0))/z.std(0)
    y = frame.y.to_numpy(float)
    y = (y-y.mean())/y.std()
    expected = z@np.linalg.lstsq(z, y, rcond=None)[0]
    kwargs = {"outcome_order": [0, 1]} if ordinal else {}
    fun = catreg_ordinal_response if ordinal else catreg_nominal_response
    result = fun(frame, "y", ["x1", "x2"], scales={"x1": "numeric", "x2": "numeric"}, **kwargs)
    fitted = result["fitted"].quantified_fitted.to_numpy()
    sign = 1 if fitted@expected >= 0 else -1
    assert sign == 1 if ordinal else True
    np.testing.assert_allclose(sign*fitted, expected, atol=3e-13)
    np.testing.assert_allclose(result["fit"].r_squared, np.mean(expected**2), atol=1e-13)


def test_mixed_nominal_predictor_response_matches_independent_dummy_space():
    rng = np.random.default_rng(19)
    n = 160
    a = rng.integers(0, 3, n)
    x = rng.normal(size=n)
    latent = np.array([-2, .2, 2])[a]+x+rng.normal(size=n)*.5
    y = np.digitize(latent, [-1, .7, 2])
    frame = pd.DataFrame({"y": y, "a": a, "x": x})
    matrix = np.column_stack([a == 1, a == 2, x]).astype(float)
    expected, _, _ = response_eigen_oracle(np.eye(4)[y], projection(matrix))
    result = catreg_nominal_response(frame, "y", ["a", "x"], scales={"a": "nominal", "x": "numeric"},
                                    n_starts=4, maxiter=500, tol=1e-12)
    np.testing.assert_allclose(result["fit"].r_squared, expected, atol=2e-11)
    check_response_fit(result)
    assert_complete_restore(result)
    with pytest.raises(AnalysisError) as error:
        catreg_outcome_predict(result, pd.DataFrame({"a": [99], "x": [0.]}))
    assert error.value.code == "unknown_category"


def bootstrap_fixture():
    rng = np.random.default_rng(421)
    a = np.tile(["a", "b", "c"], 30)
    x = rng.normal(size=len(a))
    y = pd.Series(a).map({"a": -1., "b": 2., "c": .5}).to_numpy()+.7*x+rng.normal(size=len(a))*.4
    return pd.DataFrame({"y": y, "a": a, "x": x})


def dummy_prediction(sample, queries):
    def matrix(frame):
        return np.column_stack([np.ones(len(frame)), frame.x,
                                (frame.a == "b").astype(float), (frame.a == "c").astype(float)])
    return matrix(queries)@np.linalg.lstsq(matrix(sample), sample.y, rcond=None)[0]


def test_bootstrap_full_refit_matches_dummy_ols_draws_and_joint_percentiles():
    frame = bootstrap_fixture()
    queries = pd.DataFrame({"a": ["a", "b", "c", "b"], "x": [-1., 0., 1., 2.]})
    result = catreg_bootstrap(frame, "y", ["a", "x"], queries=queries,
        scales={"a": "nominal", "x": "numeric"}, seed=17, reps=29, n_starts=2, maxiter=50, tol=1e-12)
    expected = np.array([dummy_prediction(frame.iloc[indices], queries)
                         for indices in result.attrs["bootstrap_state"]["sampled_original_positions"]])
    np.testing.assert_allclose(result["draw_predictions"].iloc[:, 1:], expected, atol=3e-8)
    np.testing.assert_allclose(result["baseline_predictions"].fitted, dummy_prediction(frame, queries), atol=3e-9)
    np.testing.assert_allclose(result["prediction_covariance"], np.cov(expected, rowvar=False), atol=3e-10)
    np.testing.assert_allclose(result["percentile_intervals"][["percentile_lower", "percentile_upper"]],
                               np.quantile(expected, [.025, .975], axis=0).T, atol=3e-8)
    assert abs(result["prediction_covariance"].iloc[1, 3]) > 1e-4
    assert result.attrs["inference_available"] and result.attrs["successful_draws"] == 29
    for draw in result.attrs["bootstrap_state"]["draw_states"]:
        d = draw["optimal_state"]["descriptors"][0]
        counts, q = np.array(d["counts"]), np.array(d["quantifications"])
        np.testing.assert_allclose(np.average(q, weights=counts), 0, atol=1e-14)
        np.testing.assert_allclose(np.average(q*q, weights=counts), 1, atol=1e-14)
        assert len(draw["starts"]) == 2 and draw["iterations"]
    assert_complete_restore(result)
    again = catreg_bootstrap(frame, "y", ["a", "x"], queries=queries,
        scales={"a": "nominal", "x": "numeric"}, seed=17, reps=29, n_starts=2, maxiter=50, tol=1e-12)
    assert summary_state(again) == summary_state(result)


def test_bootstrap_ordinal_predictor_is_refitted_with_declared_order():
    frame = bootstrap_fixture()
    queries = frame[["a", "x"]].iloc[:3]
    result = catreg_bootstrap(frame, "y", ["a", "x"], queries=queries,
        predictor_scale="ordinal", scales={"a": "ordinal", "x": "numeric"}, orders={"a": ["a", "c", "b"]},
        reps=7, n_starts=1, maxiter=50, tol=1e-12)
    for draw in result.attrs["bootstrap_state"]["draw_states"]:
        assert np.all(np.diff(draw["optimal_state"]["descriptors"][0]["quantifications"]) >= 0)
    assert result.attrs["inference_available"]
    assert_complete_restore(result)


def test_bootstrap_failures_are_recorded_without_survivor_inference():
    frame = bootstrap_fixture().iloc[:16].copy()
    frame["a"] = ["common"]*15+["rare"]
    kwargs = dict(queries=frame[["a", "x"]].iloc[[0, 15]], scales={"a": "nominal", "x": "numeric"},
                  seed=9, reps=19, n_starts=1, maxiter=50)
    result = catreg_bootstrap(frame, "y", ["a", "x"], failure="record", **kwargs)
    assert result.attrs["failures"] and not result.attrs["inference_available"]
    assert not {"prediction_covariance", "percentile_intervals"}.intersection(result)
    assert len(result["draw_status"]) == 19
    assert len(result.attrs["bootstrap_state"]["sampled_original_positions"]) == 19
    assert any(x["code"] == "bootstrap_absent_category" for x in result.attrs["failures"])
    assert_complete_restore(result)
    with pytest.raises(AnalysisError, match="no surviving-draw inference") as err:
        catreg_bootstrap(frame, "y", ["a", "x"], failure="raise", **kwargs)
    assert err.value.code == "bootstrap_failure" and err.value.bootstrap_failures


def test_original_missing_positions_and_prediction_unknown_categories():
    frame = response_fixture()
    frame.loc[[1, 7], "x1"] = np.nan
    result = catreg_ordinal_response(frame, "y", ["x1", "x2"], outcome_order=[0, 1, 2, 3],
        scales={"x1": "numeric", "x2": "numeric"})
    assert result.attrs["sample_positions"] == [i for i in range(40) if i not in (1, 7)]
    query = frame.iloc[:4]
    prediction = catreg_outcome_predict(result, query, missing="drop")
    assert prediction.attrs["sample_positions"] == [0, 2, 3]
    with pytest.raises(AnalysisError):
        catreg_outcome_predict(result, query)
    b = bootstrap_fixture()
    b.loc[3, "x"] = np.nan
    boot = catreg_bootstrap(b, "y", ["a", "x"], queries=b[["a", "x"]].iloc[:2],
        scales={"a": "nominal", "x": "numeric"}, reps=3, maxiter=50)
    assert 3 not in boot.attrs["sample_positions"]
    assert all(3 not in draw for draw in boot.attrs["bootstrap_state"]["sampled_original_positions"])
    with pytest.raises(AnalysisError) as err:
        catreg_bootstrap(b, "y", ["a", "x"], queries=pd.DataFrame({"a": ["never"], "x": [1.]}),
                        scales={"a": "nominal", "x": "numeric"}, reps=3, maxiter=50)
    assert err.value.code == "unknown_category"


@pytest.mark.parametrize("mutation", ["beta", "map", "numeric", "roots", "positions"])
def test_guarded_saved_response_prediction_refuses_resealed_semantic_damage(mutation):
    frame = response_fixture()
    result = catreg_ordinal_response(frame, "y", ["x1", "x2"], outcome_order=[0, 1, 2, 3],
        scales={"x1": "numeric", "x2": "numeric"})
    result = restore_summary(summary_state(result))
    state = copy.deepcopy(result.attrs["response_state"])
    if mutation == "beta":
        state["beta"][0] += .1
    elif mutation == "map":
        state["response"]["quantifications"][0] += .1
    elif mutation == "numeric":
        state["descriptors"][0]["mean"] += .1
    elif mutation == "roots":
        state["conditional_response_roots"][-1] += .1
    else:
        state["sample_positions"][0] = 1
    result.attrs["response_state"] = state
    result.attrs["state_sha256"] = base._seal(state)
    with pytest.raises(AnalysisError) as err:
        catreg_outcome_predict(result, frame)
    assert err.value.code == "invalid_state"


@pytest.mark.parametrize("mutation", ["nested_raw", "long_raw", "extra_field", "extra_descriptor", "nested_beta", "nested_roots", "nested_code"])
def test_saved_response_preflights_primitives_and_fields_before_seal(monkeypatch, mutation):
    frame = response_fixture()
    result = catreg_ordinal_response(frame, "y", ["x1", "x2"], outcome_order=[0, 1, 2, 3],
        scales={"x1": "numeric", "x2": "numeric"})
    state = copy.deepcopy(result.attrs["response_state"])
    if mutation == "nested_raw":
        state["training_raw_predictors"][0][0] = ["float", [1.]*100]
    elif mutation == "long_raw":
        state["training_raw_predictors"][0][0] = ["str", "x"*257]
    elif mutation == "extra_field":
        state["unexpected"] = [1.]*100
    elif mutation == "extra_descriptor":
        state["descriptors"][0]["unexpected"] = [1.]*100
    elif mutation == "nested_beta":
        state["beta"][0] = [1.]*100
    elif mutation == "nested_roots":
        state["conditional_response_roots"][0] = [1.]*100
    else:
        state["training_response_codes"][0] = [1.]*100
    result.attrs["response_state"] = state
    def forbidden(*args, **kwargs):
        raise AssertionError("Unbounded state reached JSON hashing")
    monkeypatch.setattr(base, "_seal", forbidden)
    with pytest.raises(AnalysisError) as error:
        catreg_outcome_predict(result, frame)
    assert error.value.code == "invalid_state"


def test_saved_response_admits_json_and_numerical_buffers_before_seal(monkeypatch):
    frame = response_fixture()
    result = catreg_ordinal_response(frame, "y", ["x1", "x2"], outcome_order=[0, 1, 2, 3],
        scales={"x1": "numeric", "x2": "numeric"})
    def forbidden(*args, **kwargs):
        raise AssertionError("State reached JSON hashing before buffer admission")
    monkeypatch.setattr(base, "_seal", forbidden)
    with pytest.raises(AnalysisError) as error:
        catreg_outcome_predict(result, frame, max_bytes=1)
    assert error.value.code == "workspace_limit"


def test_degenerate_zero_association_tied_contrast_and_nonconvergence_refusals():
    frame = response_fixture()
    with pytest.raises(AnalysisError):
        catreg_ordinal_response(frame, "y", ["x1", "x2"], outcome_order=[0, 1, 2, 3],
            scales={"x1": "numeric", "x2": "numeric"}, maxiter=1)
    zero = pd.DataFrame({"y": np.repeat([0, 1], 10), "x": np.tile([-1., 1.], 10)})
    with pytest.raises(AnalysisError) as err:
        catreg_nominal_response(zero, "y", ["x"], scales={"x": "numeric"})
    assert err.value.code == "degenerate_transform"
    tied = pd.DataFrame({"y": np.tile([0, 1, 2], 10)})
    tied["x1"], tied["x2"] = (tied.y == 0).astype(float), (tied.y == 1).astype(float)
    with pytest.raises(AnalysisError) as err:
        catreg_nominal_response(tied, "y", ["x1", "x2"], scales={"x1": "numeric", "x2": "numeric"})
    assert err.value.code == "unidentified_response"
    with pytest.raises(AnalysisError):
        catreg_nominal_response(frame.assign(x2=frame.x1), "y", ["x1", "x2"],
                                scales={"x1": "numeric", "x2": "numeric"})


@pytest.mark.parametrize("option", [{"max_bytes": 1}, {"max_work": 1}, {"device": "cuda"}, {"weights": "w"},
                                  {"seed": True}, {"n_starts": 13}, {"tol": 0}, {"missing": "fill"}])
def test_outcome_input_and_budget_refusals(option):
    frame = response_fixture()
    with pytest.raises(AnalysisError):
        catreg_nominal_response(frame, "y", ["x1", "x2"], scales={"x1": "numeric", "x2": "numeric"}, **option)


def test_ordinal_order_and_bootstrap_domain_refusals():
    frame = response_fixture()
    for order in ([0, 1, 2], [0, 1, 1, 2, 3], [0, 1, 2, 3, 4]):
        with pytest.raises(AnalysisError):
            catreg_ordinal_response(frame, "y", ["x1"], outcome_order=order, scales={"x1": "numeric"})
    b = bootstrap_fixture()
    for extra in [{"max_bytes": 1}, {"max_work": 1}, {"reps": 1}, {"reps": 500},
                  {"failure": "drop"}, {"confidence": 1}, {"weights": "w"}, {"device": "mps"}]:
        with pytest.raises(AnalysisError):
            catreg_bootstrap(b, "y", ["a", "x"], queries=b[["a", "x"]].iloc[:2],
                            scales={"a": "nominal", "x": "numeric"}, **extra)
    with pytest.raises(AnalysisError):
        catreg_bootstrap(b, "y", ["a", "x"], queries=b[["a", "x"]].iloc[:65])
    with pytest.raises(AnalysisError):
        catreg_bootstrap(b.assign(y="category"), "y", ["a", "x"], queries=b[["a", "x"]].iloc[:2])
    predictors = [f"x{i}" for i in range(12)]
    wide = frame.assign(**{name: np.arange(40) for name in predictors})
    with pytest.raises(AnalysisError, match="11 predictors"):
        catreg_nominal_response(wide, "y", predictors)
    with pytest.raises(AnalysisError, match="2–32"):
        catreg_ordinal_response(frame, "y", ["x1"], outcome_order=list(range(33)))


def test_early_guard_projects_wide_mappings_before_coercion(monkeypatch):
    frame = response_fixture()
    data = frame.to_dict("list") | {"unused": object()}
    result = catreg_nominal_response(data, "y", ["x1", "x2"], scales={"x1": "numeric", "x2": "numeric"})
    assert result.attrs["n"] == 40
    def forbidden(*args, **kwargs):
        raise AssertionError("Input was materialized before budget refusal")
    monkeypatch.setattr(base, "_coerce_frame", forbidden)
    with pytest.raises(AnalysisError) as err:
        catreg_nominal_response(data, "y", ["x1", "x2"], max_bytes=1)
    assert err.value.code == "workspace_limit"
    with pytest.raises(AnalysisError) as err:
        catreg_bootstrap(data, "y", ["x1", "x2"], queries={"x1": [1.], "x2": [2.]}, max_bytes=1)
    assert err.value.code == "workspace_limit"


def test_all_public_routes_pin_cpu_and_global_workspace_is_admitted():
    frame = response_fixture()
    prior = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        result = catreg_nominal_response(frame, "y", ["x1", "x2"], scales={"x1": "numeric", "x2": "numeric"})
        assert catreg_outcome_predict(result, frame).attrs["device"] == "cpu"
        b = bootstrap_fixture()
        assert catreg_bootstrap(b, "y", ["a", "x"], queries=b[["a", "x"]].iloc[:2],
                               scales={"a": "nominal", "x": "numeric"}, reps=3, maxiter=30).attrs["device"] == "cpu"
    finally:
        torch.set_default_device(prior)
    large = pd.concat([frame]*50, ignore_index=True)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as err:
        catreg_nominal_response(large, "y", ["x1", "x2"], scales={"x1": "numeric", "x2": "numeric"})
    assert err.value.code == "workspace_limit"
    assert json.loads(summary_state(result))["schema"] == "openecon.summary.v1"
