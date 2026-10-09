"""Independent compressed-case resampling, constrained refit and replay checks."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import lsq_linear
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.categorical import frequency as f
from openecon.econometrics.categorical import frequency_bootstrap as m
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


def data():
    rng = np.random.default_rng(9842)
    frame = pd.DataFrame({"a": np.tile(["low", "mid", "high"], 16),
        "x": rng.normal(size=48), "frequency": rng.integers(3, 8, size=48)})
    frame["y"] = frame.a.map({"low": 2., "mid": .4, "high": -1.5})+.8*frame.x+rng.normal(scale=.3, size=48)
    return frame


def query():
    return pd.DataFrame({"a": ["low", "mid", "high"], "x": [-.8, .2, .9]}, index=[7, 7, -1])


def fit(kind="nominal", frame=None, **kwargs):
    frame = data() if frame is None else frame
    options = dict(frequency="frequency", queries=query(), scales={"a": kind, "x": "numeric"},
        reps=7, n_starts=2, maxiter=100, tol=1e-12, seed=74)
    options.update(kwargs)
    if kind == "nominal":
        return m.catreg_nominal_fweight_bootstrap(frame, "y", ["a", "x"], **options)
    return m.catreg_ordinal_fweight_bootstrap(frame, "y", ["a", "x"], orders={"a": ["low", "mid", "high"]}, **options)


@pytest.fixture(scope="module")
def nominal():
    return fit()


@pytest.fixture(scope="module")
def ordinal():
    return fit("ordinal")


def nominal_oracle(frame, counts, queries):
    # A fresh ordinary dummy regression on the literal repeated sample.
    design = np.column_stack([np.ones(len(frame)), frame.x,
        (frame.a == "mid").astype(float), (frame.a == "high").astype(float)])
    xq = np.column_stack([np.ones(len(queries)), queries.x,
        (queries.a == "mid").astype(float), (queries.a == "high").astype(float)])
    root = np.sqrt(counts)
    beta = np.linalg.lstsq(design*root[:, None], frame.y.to_numpy()*root, rcond=None)[0]
    return xq@beta


def ordinal_oracle(frame, counts, queries):
    codes = frame.a.map({"low": 0, "mid": 1, "high": 2}).to_numpy()
    qcode = queries.a.map({"low": 0, "mid": 1, "high": 2}).to_numpy()
    root = np.sqrt(counts)
    best = None
    for sign in [1, -1]:
        matrix = np.column_stack([np.ones(len(frame)), frame.x, sign*(codes >= 1), sign*(codes >= 2)])
        xq = np.column_stack([np.ones(len(queries)), queries.x, sign*(qcode >= 1), sign*(qcode >= 2)])
        result = lsq_linear(matrix*root[:, None], frame.y.to_numpy()*root,
            bounds=([-np.inf, -np.inf, 0, 0], [np.inf]*4), tol=1e-12, max_iter=1000)
        objective = np.sum(counts*(frame.y-matrix@result.x)**2)
        if best is None or objective < best[0]:
            best = objective, xq@result.x
    return best[1]


@pytest.mark.parametrize("kind", ["nominal", "ordinal"])
def test_every_full_refit_and_joint_uncertainty_against_independent_oracles(kind, nominal, ordinal):
    result = nominal if kind == "nominal" else ordinal
    state = result.attrs["bootstrap_state"]
    frame, queries = data(), query()
    oracle = nominal_oracle if kind == "nominal" else ordinal_oracle
    expected = np.array([oracle(frame, np.array(counts), queries) for counts in state["draw_counts"]])
    np.testing.assert_allclose(state["point"], oracle(frame, frame.frequency.to_numpy(), queries), atol=2e-8)
    np.testing.assert_allclose(state["draw_predictions"], expected, atol=3e-8)
    np.testing.assert_allclose(result["prediction_covariance"], np.cov(expected, rowvar=False, ddof=1), atol=1e-10)
    np.testing.assert_allclose(result["percentile_intervals"][["percentile_lower", "percentile_upper"]],
        np.quantile(expected, [.025, .975], axis=0, method="linear").T, atol=3e-8)
    assert np.abs(np.array(result["prediction_covariance"])[0, 1]) > 1e-6
    for i, text in enumerate(state["draw_fits"]):
        fit_state = json.loads(text)["attrs"]["frequency_state"]
        assert sum(fit_state["counts"]) == frame.frequency.sum()
        assert fit_state["controls"]["n_starts"] == 2
        assert len(fit_state["starts"]) == 2
        assert fit_state["controls"]["seed"] == (74+104729*(i+1)) % (2**31-1)
        assert fit_state["histories"] and fit_state["stationarity"] < 1e-5
    assert result.attrs["inference_available"]
    assert not {"parameters_covariance", "p_values", "ordinary_se"}&set(result)


def test_literal_expansion_oracle_for_all_saved_nominal_draws(nominal):
    frame = data()
    for counts, predictions in zip(nominal.attrs["bootstrap_state"]["draw_counts"], nominal.attrs["bootstrap_state"]["draw_predictions"]):
        expanded = frame.iloc[np.repeat(np.arange(len(frame)), counts)].copy()
        expected = nominal_oracle(expanded, np.ones(len(expanded)), query())
        np.testing.assert_allclose(predictions, expected, atol=2e-8)


def test_explicit_numeric_limit_is_ordinary_count_bootstrap_regression():
    frame = data()
    frame["z"] = np.linspace(-1, 1, len(frame))
    queries = frame[["x", "z"]].iloc[:4].copy()
    result = m.catreg_nominal_fweight_bootstrap(frame, "y", ["x", "z"], frequency="frequency",
        queries=queries, scales={"x": "numeric", "z": "numeric"}, reps=3, n_starts=1, maxiter=10, tol=1e-12)
    matrix = np.column_stack([np.ones(len(frame)), frame[["x", "z"]]])
    xq = np.column_stack([np.ones(len(queries)), queries])
    for counts, predictions in zip(result.attrs["bootstrap_state"]["draw_counts"], result.attrs["bootstrap_state"]["draw_predictions"]):
        root = np.sqrt(counts)
        beta = np.linalg.lstsq(matrix*root[:, None], frame.y.to_numpy()*root, rcond=None)[0]
        np.testing.assert_allclose(predictions, xq@beta, atol=1e-10)
    m.catreg_fweight_bootstrap_restore(summary_state(result))


def test_large_frequency_fit_and_replay_depend_on_physical_rows():
    frame = data()
    frame.frequency = 20_000_000  # 960 million literal cases, still 48 fitted rows.
    result = fit(frame=frame, reps=2)
    state = result.attrs["bootstrap_state"]
    assert all(len(counts) == 48 and sum(counts) == 960_000_000 for counts in state["draw_counts"])
    assert len(summary_state(result)) < 250_000
    m.catreg_fweight_bootstrap_restore(summary_state(result))


def test_ordinal_pooling_is_refitted_in_each_draw():
    frame = data()
    frame.y += frame.a.map({"low": 0., "mid": 2.3, "high": 0.})
    result = fit("ordinal", frame, reps=5)
    for counts, predictions in zip(result.attrs["bootstrap_state"]["draw_counts"], result.attrs["bootstrap_state"]["draw_predictions"]):
        np.testing.assert_allclose(predictions, ordinal_oracle(frame, np.array(counts), query()), atol=3e-7)
    assert any(np.isclose(*json.loads(text)["attrs"]["frequency_state"]["descriptors"][0]["quantifications"][:2])
        for text in result.attrs["bootstrap_state"]["draw_fits"])


def test_sequential_sampler_matches_independent_binomial_factorization_and_private_rng():
    counts = [1, 2, 7, 4]
    global_state = torch.get_rng_state().clone()
    actual = torch.Generator(device="cpu").manual_seed(1987)
    independent = torch.Generator(device="cpu").manual_seed(1987)
    for _ in range(100):
        trials, source_total = sum(counts), sum(counts)
        expected = []
        for original in counts[:-1]:
            value = int(torch.binomial(torch.tensor(float(trials), dtype=torch.float64),
                torch.tensor(original/source_total, dtype=torch.float64), generator=independent)) if trials else 0
            expected.append(value)
            trials -= value
            source_total -= original
        expected.append(trials)
        assert m._draw_counts(counts, actual) == expected
    assert torch.equal(torch.get_rng_state(), global_state)


def test_sampled_moments_follow_multinomial_not_equal_physical_row_law():
    counts = [1, 2, 3]
    rng = torch.Generator(device="cpu").manual_seed(553)
    values = np.array([m._draw_counts(counts, rng) for _ in range(6000)])
    p = np.array(counts)/sum(counts)
    np.testing.assert_allclose(values.mean(0), counts, atol=.06)
    np.testing.assert_allclose(np.cov(values, rowvar=False), sum(counts)*(np.diag(p)-np.outer(p, p)), atol=.065)
    assert np.all(values.sum(1) == sum(counts))


def test_billion_count_domain_uses_scalar_binomials_without_expansion(monkeypatch):
    original = torch.binomial
    calls = []
    def binomial(trials, p, *, generator):
        calls.append((trials.shape, p.shape, float(trials), float(p)))
        return original(trials, p, generator=generator)
    def refuse(*args, **kwargs):
        raise AssertionError("literal rows were allocated")
    monkeypatch.setattr(torch, "binomial", binomial)
    monkeypatch.setattr(torch, "multinomial", refuse)
    monkeypatch.setattr(torch, "repeat_interleave", refuse)
    counts = m._draw_counts([999_999_998, 1, 1], torch.Generator(device="cpu").manual_seed(17))
    assert sum(counts) == 1_000_000_000
    assert len(calls) <= 2 and all(shape == pshape == torch.Size([]) for shape, pshape, _, _ in calls)


@pytest.mark.parametrize("kind", ["nominal", "ordinal"])
def test_complete_state_restore_numeric_replay_and_exact_latex(kind, nominal, ordinal, monkeypatch):
    result = nominal if kind == "nominal" else ordinal
    full = summary_state(result)
    ordinary = restore_summary(full)
    assert summary_state(ordinary) == full and ordinary.to_latex() == result.to_latex()
    def refuse(*args, **kwargs):
        raise AssertionError("restore refitted an optimizer")
    monkeypatch.setattr(m.r, "_fit", refuse)
    replay = m.catreg_fweight_bootstrap_restore(full)
    assert summary_state(replay) == full and replay.to_latex() == result.to_latex()
    assert summary_state(m.catreg_fweight_bootstrap_restore(ordinary)) == full


def test_seed_repeatability_preserves_global_rng_and_input():
    frame = data()
    original = frame.copy(deep=True)
    rng = torch.get_rng_state().clone()
    left, right = fit(frame=frame, reps=3), fit(frame=frame, reps=3)
    assert summary_state(left) == summary_state(right)
    assert torch.equal(torch.get_rng_state(), rng)
    pd.testing.assert_frame_equal(frame, original)


def test_failed_draw_is_retained_without_redrawing_and_withholds_every_inference_table(monkeypatch):
    source = data()
    source.loc[:, "frequency"] = 1
    original = m._draw_counts
    called = []
    def omit(counts, generator):
        sampled = original(counts, generator)
        if not called:
            sampled = [0 if source.a.iloc[i] == "high" else int(v) for i, v in enumerate(sampled)]
            sampled[0] += sum(counts)-sum(sampled)
        called.append(sampled)
        return sampled
    monkeypatch.setattr(m, "_draw_counts", omit)
    result = fit(frame=source, failure="record", reps=5)
    state = result.attrs["bootstrap_state"]
    assert len(called) == len(state["draw_counts"]) == 5
    assert state["draw_counts"][0] == called[0]
    assert state["failures"][0][0:2] == [0, "bootstrap_absent_category"]
    assert state["draw_fits"][0] is state["draw_predictions"][0] is None
    assert not result.attrs["inference_available"]
    assert not m.INFERENCE_TABLES & set(result)


def test_natural_sparse_failure_policy_raise_and_record():
    frame = data().iloc[:10].copy()
    frame.loc[:, "frequency"] = 1
    frame.a = ["rare"]+["common"]*9
    options = dict(queries=pd.DataFrame({"a": ["rare"], "x": [0.]}), reps=19, seed=9)
    result = fit(frame=frame, failure="record", **options)
    state = result.attrs["bootstrap_state"]
    assert state["failures"] and len(state["draw_counts"]) == 19
    assert set(result).isdisjoint(m.INFERENCE_TABLES)
    assert m.catreg_fweight_bootstrap_restore(summary_state(result)).attrs["inference_available"] is False
    with pytest.raises(AnalysisError) as error:
        fit(frame=frame, failure="raise", **options)
    assert error.value.code == "bootstrap_failure"
    assert error.value.bootstrap_counts == state["draw_counts"][:len(error.value.bootstrap_counts)]


def test_original_positions_missing_and_zero_counts_preserved():
    frame = data()
    frame.index = [17]*len(frame)
    frame.loc[frame.index[:0], "x"] = np.nan  # Duplicate labels never select rows.
    frame.iloc[3, frame.columns.get_loc("frequency")] = 0
    frame.iloc[3, frame.columns.get_loc("a")] = "unseen-zero-row"
    frame.iloc[5, frame.columns.get_loc("x")] = np.nan
    frame.iloc[7, frame.columns.get_loc("frequency")] = np.nan
    result = fit(frame=frame, reps=3)
    baseline = json.loads(result.attrs["bootstrap_state"]["baseline"])["attrs"]["frequency_state"]
    assert baseline["zero_positions"] == [3]
    assert baseline["missing_positions"] == [5, 7]
    assert baseline["positions"] == [i for i in range(48) if i not in [3, 5, 7]]
    assert all(len(counts) == 45 and sum(counts) == sum(baseline["counts"])
        for counts in result.attrs["bootstrap_state"]["draw_counts"])
    m.catreg_fweight_bootstrap_restore(summary_state(result))


@pytest.mark.parametrize("changes", [dict(reps=1), dict(reps=200), dict(confidence=1.),
    dict(confidence=10**1000), dict(tol=10**1000), dict(seed=True), dict(n_starts=0),
    dict(device="cuda"), dict(failure="skip"), dict(missing="omit")])
def test_scalar_controls_refuse(changes):
    with pytest.raises(AnalysisError):
        fit(**changes)


@pytest.mark.parametrize("budget", [dict(max_bytes=1), dict(max_work=1)])
def test_admission_before_sampling_tensor_or_fitting(budget, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("materialized before admission")
    monkeypatch.setattr(torch, "tensor", refuse)
    monkeypatch.setattr(torch, "binomial", refuse)
    monkeypatch.setattr(m.r, "_fit", refuse)
    with pytest.raises(AnalysisError) as error:
        fit(**budget)
    assert error.value.code in {"resource_limit", "workspace_limit"}


def test_global_budget_before_materialization(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("sampled despite global budget")
    monkeypatch.setattr(torch, "binomial", refuse)
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        fit()


def test_artifact_and_cumulative_work_admission_precedes_first_fit(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("fit before cumulative admission")
    monkeypatch.setattr(m.r, "_fit", refuse)
    with pytest.raises(AnalysisError) as error:
        fit(reps=199, maxiter=1000)
    assert error.value.code == "resource_limit"
    with pytest.raises(AnalysisError) as error:
        fit(max_work=20_000)
    assert error.value.code == "resource_limit"


def test_query_unknown_missing_wide_or_too_many_refused_before_fit(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("fitted before invalid query refusal")
    monkeypatch.setattr(m.r, "_fit", refuse)
    for queries in [pd.DataFrame({"a": ["never"], "x": [1.]}), query().assign(x=np.nan),
                    pd.concat([query()]*22), query().rename(columns={"x": "absent"})]:
        with pytest.raises(AnalysisError):
            fit(queries=queries)


def test_frequency_input_domains_and_missing_policy():
    for invalid in [True, 1.2, np.inf, -1, 1_000_000_001]:
        frame = data().astype({"frequency": object})
        frame.loc[0, "frequency"] = invalid
        with pytest.raises(AnalysisError):
            fit(frame=frame)
    with pytest.raises(AnalysisError):
        fit(frame=data().assign(y="category"))
    frame = data()
    frame.loc[0, "x"] = np.nan
    with pytest.raises(AnalysisError) as error:
        fit(frame=frame, missing="raise")
    assert error.value.code == "missing_data"


def forged(result, mutate):
    state = copy.deepcopy(result.attrs["bootstrap_state"])
    mutate(state)
    return m._output(state, result.attrs["resources"], result.attrs["declared_work"], result.attrs["declared_artifact_bound_bytes"])


@pytest.mark.parametrize("mutation", [
    lambda s: s["draw_counts"][0].__setitem__(0, s["draw_counts"][0][0]+1),
    lambda s: s["point"].__setitem__(0, s["point"][0]+1),
    lambda s: s["draw_predictions"][0].__setitem__(0, s["draw_predictions"][0][0]+1),
    lambda s: s["receipts"][0].__setitem__(3, s["receipts"][0][3]+1),
    lambda s: s["query_raw"][0][1].__setitem__(1, 1e250),
])
def test_resealed_matching_tables_do_not_replace_numerical_replay(nominal, mutation):
    output = forged(nominal, mutation)
    with pytest.raises(AnalysisError) as error:
        m.catreg_fweight_bootstrap_restore(output)
    assert error.value.code == "invalid_state"


def test_resealed_full_draw_fit_tampering_refused(nominal):
    def change(state):
        payload = json.loads(state["draw_fits"][0])
        calibration = payload["attrs"]["frequency_state"]
        calibration["beta"][0] += .1
        payload["attrs"]["state_sha256"] = f._seal(calibration)
        state["draw_fits"][0] = json.dumps(payload)
    output = forged(nominal, change)
    with pytest.raises(AnalysisError) as error:
        m.catreg_fweight_bootstrap_restore(output)
    assert error.value.code == "invalid_state"


@pytest.mark.parametrize("mutation", [
    lambda s: s.__setitem__("extra", [0]*100),
    lambda s: s["query_raw"][0].__setitem__(0, ["str", "x"*257]),
    lambda s: s["query_raw"][0].__setitem__(0, ["str", {"nested": "x"}]),
    lambda s: s["draw_counts"][0].__setitem__(0, True),
    lambda s: s["receipts"][0].__setitem__(1, 1),
])
def test_primitive_refusal_before_seal_or_tensor(nominal, mutation, monkeypatch):
    output = copy.deepcopy(nominal)
    mutation(output.attrs["bootstrap_state"])
    def refuse(*args, **kwargs):
        raise AssertionError("unsafe primitive reached seal/tensor")
    monkeypatch.setattr(m.f, "_seal", refuse)
    monkeypatch.setattr(torch, "tensor", refuse)
    with pytest.raises(AnalysisError) as error:
        m.catreg_fweight_bootstrap_restore(output)
    assert error.value.code == "invalid_state"


def test_complete_table_and_metadata_tampering_refused(nominal):
    output = copy.deepcopy(nominal)
    output["prediction_covariance"].iloc[0, 1] += .01
    with pytest.raises(AnalysisError):
        m.catreg_fweight_bootstrap_restore(output)
    output = copy.deepcopy(nominal)
    output.attrs["inference_available"] = 1
    with pytest.raises(AnalysisError):
        m.catreg_fweight_bootstrap_restore(output)
    payload = json.loads(summary_state(nominal))
    payload["tables"]["draw_status"]["data"][0][0] = {"nested": 1}
    with pytest.raises(AnalysisError):
        m.catreg_fweight_bootstrap_restore(json.dumps(payload))


def test_restore_combined_budget_admitted_before_any_model_replay(nominal, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("model replay before aggregate budget")
    monkeypatch.setattr(m.r, "_checked", refuse)
    with pytest.raises(AnalysisError):
        m.catreg_fweight_bootstrap_restore(nominal, max_work=1)
    with pytest.raises(AnalysisError):
        m.catreg_fweight_bootstrap_restore(summary_state(nominal), max_bytes=1)


def test_cpu_selected_under_default_meta_device():
    previous = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        result = fit(reps=2)
        assert result.attrs["device"] == "cpu"
        m.catreg_fweight_bootstrap_restore(summary_state(result))
    finally:
        torch.set_default_device(previous)
