"""Future source-authored acceptance tests; never collected in source-only review."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.bayesian_var import api, draws, forecast, impulse
from openecon.econometrics.bayesian_var.admission import digest
from openecon.econometrics.bayesian_var_public import transport, tables


def test_actual_public_catalogue_listing_is_tensor_free_after_shared_integration():
    import openecon
    package_parent = str(Path(openecon.__file__).resolve().parent.parent)
    source = """
import builtins, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'pandas', 'numpy', 'scipy'):
        raise AssertionError('listing public BVAR names imported scientific runtime: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import openecon
from openecon.econometrics.bayesian_var_public import EXPORTS, ESTIMATORS
assert not ESTIMATORS and len(EXPORTS)==29
assert all(name in dir(openecon) and name in openecon.__all__ for name in EXPORTS)
assert not any(name in sys.modules for name in ('torch','pandas','numpy','scipy'))
"""
    subprocess.run([sys.executable, "-I", "-B", "-c", source, package_parent], check=True)


@pytest.fixture
def states():
    frame = pd.DataFrame({"period": list(range(16)),
                          "a": [float((i*3)%7)+.1*i for i in range(16)],
                          "b": [float((i*5)%11)-.2*i for i in range(16)]},
                         index=pd.Index(["row"+str(i) for i in range(16)], name="original"))
    prior = transport.bayes_var_prior_restore({"mean": [[0., 0.]]*3,
        "row_scale": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        "innovation_scale": [[2., .1], [.1, 2.]], "degrees_of_freedom": 16.})
    parent = api.bayes_var(frame, ["a", "b"], time="period", prior=prior)
    joint = draws.bayes_var_draws(parent, draws=4, seed=721)
    return [parent,
            api.bayes_var_predict(parent, [[1., .2, -.1], [1., -.3, .4]], terms=parent.terms),
            api.bayes_var_contrast(parent, [0., 1., .2], [.3, 1.], threshold=.1), joint,
            forecast.bayes_var_forecast(joint, horizon=1, innovation_seed=726),
            impulse.bayes_var_irf(joint, horizon=1, orthogonalized=True)]


def raw(value):
    from pydantic import BaseModel
    return BaseModel.model_dump(value, mode="python")


def reseal(value):
    if isinstance(value, dict):
        for item in value.values():
            reseal(item)
        if "integrity_sha256" in value:
            value["integrity_sha256"] = digest({k: v for k, v in value.items() if k != "integrity_sha256"})
    elif isinstance(value, (tuple, list)):
        for item in value:
            reseal(item)
    return value


def cold_guards(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Cold complete-state replay attempted fitting or RNG.")
    monkeypatch.setattr(api, "bayes_var", forbidden)
    monkeypatch.setattr(draws, "primitive_stream", forbidden)
    monkeypatch.setattr(forecast, "innovation_stream", forbidden)
    import torch
    for name in ("rand", "randn", "randint", "normal", "manual_seed", "_standard_gamma"):
        monkeypatch.setattr(torch, name, forbidden)


def test_all_six_complete_typed_replays_and_whole_tables(states, monkeypatch):
    cold_guards(monkeypatch)
    for state in states:
        restored = transport.bayes_var_restore(tables.bayes_var_state_json(state, indent=2))
        assert type(restored) is type(state)
        assert raw(restored) == raw(state)
        frames = tables.bayes_var_tables(state)
        assert frames.attrs["complete_state"] == json.loads(tables.bayes_var_state_json(state))
        assert "no seed-stream authentication" in frames.attrs["replay_contract"]
        for path, array, shape in tables.leaves(raw(state)):
            if shape:
                expected, _, _ = tables.array_rows(array, shape)
                frame = frames[".".join(path)]
                assert frame.values.tolist() == expected
                assert frame.attrs["complete_shape"] == shape
        assert "scalar_fields_and_availability" in frames


def test_resealed_full_covariance_source_and_cached_primitive_negatives(states, monkeypatch):
    cold_guards(monkeypatch)
    cases = []
    posterior = json.loads(tables.bayes_var_state_json(states[0]))
    posterior["algebra"]["joint_covariance"][0][1] += .5
    cases.append(posterior)
    prediction = json.loads(tables.bayes_var_state_json(states[1]))
    prediction["full_covariance"][0][3] += .5
    cases.append(prediction)
    joint = json.loads(tables.bayes_var_state_json(states[3]))
    joint["bartlett"][0][0][0] += .2
    cases.append(joint)
    stable = json.loads(tables.bayes_var_state_json(states[3]))
    stable["stable"][0] = not stable["stable"][0]
    cases.append(stable)
    outcome = json.loads(tables.bayes_var_state_json(states[4]))
    outcome["outcome_paths"][0][0][0] += .2
    cases.append(outcome)
    source = json.loads(tables.bayes_var_state_json(states[0]))
    source["source"]["source_index"]["values"][0] = "different_original_row"
    cases.append(source)
    for body in cases:
        with pytest.raises(AnalysisError):
            transport.bayes_var_restore(reseal(body))


def test_cached_replay_preserves_descriptive_seed_contract(states, monkeypatch):
    cold_guards(monkeypatch)
    original = json.loads(tables.bayes_var_state_json(states[3]))
    changed = copy.deepcopy(original)
    changed["seed"] += 1
    changed["torch_version"] = "recorded prior runtime"
    restored = transport.bayes_var_draws_restore(reseal(changed))
    assert restored.coefficients == states[3].coefficients
    assert restored.innovations == states[3].innovations
    assert restored.seed != states[3].seed


def test_public_admission_before_replay_frames_or_rng(states, monkeypatch):
    body = json.loads(tables.bayes_var_state_json(states[0]))
    cold_guards(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("Resource rejection occurred after replay/frame allocation.")
    monkeypatch.setattr(transport, "mapping", forbidden)
    monkeypatch.setattr(pd, "DataFrame", forbidden)
    for key, value in (("max_work", 1), ("max_bytes", 1)):
        malformed = copy.deepcopy(body)
        malformed[key] = value
        with pytest.raises(AnalysisError):
            tables.bayes_var_tables(malformed)
        with pytest.raises(AnalysisError):
            transport.bayes_var_restore(malformed)
    with pytest.raises(AnalysisError):
        tables.bayes_var_state_json(body, indent=10**8)
    bad = copy.deepcopy(body)
    bad["source"]["series"][0] = "☃"*10001
    with pytest.raises(AnalysisError):
        tables.bayes_var_tables(bad)


def test_proper_prior_geometry_and_exact_schema_routes(states):
    prior = raw(states[0].prior)
    prior["row_scale"] = ((1., 0., 0.), (0., -1., 0.), (0., 0., 1.))
    with pytest.raises(AnalysisError):
        transport.bayes_var_prior_restore(prior)
    with pytest.raises(AnalysisError):
        transport.bayes_var_irf_restore(raw(states[4]))
    with pytest.raises(AnalysisError):
        transport.bayes_var_restore({"schema_version": "openecon.ResultBundle"})


def test_unavailable_moments_remain_explicit_full_state_and_table_cells():
    frame = pd.DataFrame({"period": [0, 1, 2], "a": [0., 1., .1], "b": [.5, -.2, .4]})
    parent = api.bayes_var(frame, ["a", "b"], time="period", prior={
        "mean": [[0., 0.]]*3, "row_scale": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        "innovation_scale": [[1., 0.], [0., 1.]], "degrees_of_freedom": 1.1})
    assert parent.algebra.innovation_covariance is None
    shown = tables.bayes_var_tables(parent)
    cells = shown["scalar_fields_and_availability"].set_index("field")
    assert cells.loc["algebra.innovation_covariance", "availability"] == "unavailable"
    joint = draws.bayes_var_draws(parent, draws=4, seed=781)
    future = forecast.bayes_var_forecast(joint, horizon=3, innovation_seed=782)
    assert future.outcome_summary.sample_mean is None
    assert future.outcome_summary.covariance_mc_estimate is None
    assert future.outcome_summary.mean_mc_standard_error is None
    assert future.outcome_summary.covariance_mc_standard_error is None
    shown = tables.bayes_var_tables(future)
    cells = shown["scalar_fields_and_availability"].set_index("field")
    assert cells.loc["outcome_summary.sample_mean", "availability"] == "unavailable"
    assert "no joint coverage" in cells.loc["outcome_summary.interval_label", "value"]


def test_whole_source_tables_preserve_signed64_periods_before_native_safe_int_format():
    frame = pd.DataFrame({"period": [2**60+i for i in range(4)], "a": [0., 1., .1, -.2], "b": [.5, -.2, .4, .7]})
    prior = {"mean": [[0., 0.]]*3, "row_scale": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
             "innovation_scale": [[1., 0.], [0., 1.]], "degrees_of_freedom": 16.}
    parent = api.bayes_var(frame, ["a", "b"], time="period", prior=prior)
    shown = tables.bayes_var_tables(parent)["source.source_values"]
    assert [shown.iloc[0, i] for i in range(4)] == [2**60+i for i in range(4)]
    assert all(type(shown.iloc[0, i]) is int for i in range(4))
