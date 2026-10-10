"""Future v2 negative/call-count controls; not collected in source-only review."""
import copy
from collections import Counter
import json

from pydantic import BaseModel
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.bayesian_var import api, draws, forecast, impulse, kernels
from openecon.econometrics.bayesian_var.admission import digest
from openecon.econometrics.bayesian_var_public import tables, transport
from test_bayesian_var_public_integration import cold_guards, raw, reseal, states as states


def parent_body(body):
    if "joint_draws" in body:
        body = body["joint_draws"]
    return body.get("parent", body)


def unchecked_like(template, body, mode):
    """Forge nested typed instances without calling original validation methods."""
    if isinstance(template, BaseModel):
        fields = {key: unchecked_like(item, body[key], mode)
                  for key, item in template.__dict__.items()}
        if mode == "construct":
            return type(template).model_construct(**fields)
        return BaseModel.model_copy(template, update=fields)
    return body


def public_entries():
    return (tables.admission, transport.bayes_var_restore,
            tables.bayes_var_tables, tables.bayes_var_state_json)


def forbid_replay(monkeypatch):
    cold_guards(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("Malformed admission/provenance reached numerical replay.")
    for module, name in ((kernels, "posterior_algebra"), (api, "prediction_algebra"),
                         (api, "contrast_algebra"), (draws, "transform"),
                         (forecast, "paths_algebra"), (forecast, "summarize"),
                         (impulse, "impulse_algebra"), (impulse, "summarize")):
        monkeypatch.setattr(module, name, forbidden)


@pytest.mark.parametrize("columns", ([], (), [[]], [None]))
@pytest.mark.parametrize("mode", ("construct", "base_copy"))
def test_empty_malformed_source_refuses_before_mapping_replay_or_output(states, monkeypatch, columns, mode):
    cases = []
    for state in states:
        body = copy.deepcopy(raw(state))
        parent_body(body)["source"]["source_values"] = columns
        reseal(body)
        cases.extend((body, json.dumps(body), unchecked_like(state, body, mode)))
    forbid_replay(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("Empty BVAR source reached primitive/frame creation.")
    monkeypatch.setattr(transport, "mapping", forbidden)
    import pandas as pd
    monkeypatch.setattr(pd, "DataFrame", forbidden)
    for malformed in cases:
        for entry in public_entries():
            with pytest.raises(AnalysisError) as error:
                entry(malformed)
            assert error.value.code == "invalid_state"


@pytest.mark.parametrize("mode", ("construct", "base_copy"))
@pytest.mark.parametrize("field", ("source_dtypes", "source_index", "schema_version"))
def test_forged_nested_typed_provenance_reenters_original_before_validators(states, monkeypatch, mode, field):
    cases = []
    for state in states:
        body = copy.deepcopy(raw(state))
        parent = parent_body(body)
        source = parent["source"]
        if field == "source_dtypes":
            source["source_dtypes"] = [source["source_dtypes"][0], "int64", source["source_dtypes"][2]]
        elif field == "source_index":
            source["source_index"]["values"] = source["source_index"]["values"][:-1]
        else:
            parent["schema_version"] = "forged.unrecognized.posterior"
        source["source_sha256"] = digest({key: item for key, item in source.items() if key != "source_sha256"})
        reseal(body)
        cases.append(unchecked_like(state, body, mode))
    forbid_replay(monkeypatch)
    for forged in cases:
        for entry in (transport.bayes_var_restore, tables.bayes_var_tables, tables.bayes_var_state_json):
            with pytest.raises(AnalysisError):
                entry(forged)


@pytest.mark.parametrize("location", ("top", "nested"))
@pytest.mark.parametrize("storage", ("dict", "extra", "private"))
def test_typed_normalization_refuses_concealed_unknown_fields_instead_of_dropping_them(states, monkeypatch, location, storage):
    forged = unchecked_like(states[4], copy.deepcopy(raw(states[4])), "construct")
    target = forged if location == "top" else forged.joint_draws.parent.source
    if storage == "dict":
        target.__dict__["undeclared_metadata"] = "must never disappear"
    else:
        object.__setattr__(target, "__pydantic_" + storage + "__", {"undeclared_metadata": "must never disappear"})
    forbid_replay(monkeypatch)
    for entry in (transport.bayes_var_restore, tables.bayes_var_tables, tables.bayes_var_state_json):
        with pytest.raises(AnalysisError) as error:
            entry(forged)
        assert error.value.code == "invalid_state"


def test_plain_mapping_normalization_keeps_unknown_fields_for_semantic_refusal(states, monkeypatch):
    body = copy.deepcopy(raw(states[0]))
    body["undeclared_metadata"] = {"all_fields": ["preserved", {"nested": 7}]}
    normalized = transport.mapping(body)
    assert normalized == body
    assert normalized is not body
    cold_guards(monkeypatch)
    with pytest.raises(AnalysisError):
        transport.bayes_var_restore(body)


def test_complete_tree_normalization_does_not_call_original_checked_or_dump_overrides(states, monkeypatch):
    bodies = [raw(state) for state in states]
    cold_guards(monkeypatch)
    from openecon.econometrics.bayesian_var.posterior import FrozenModel
    def forbidden(*args, **kwargs):
        raise AssertionError("Complete primitive normalization recursively revalidated a checked/dump method.")
    for name in ("checked", "model_dump", "model_dump_json", "model_copy"):
        monkeypatch.setattr(FrozenModel, name, forbidden)
    for state, body in zip(states, bodies, strict=True):
        assert transport.mapping(state) == body
        assert tables.primitive(transport.bayes_var_restore(state)) == body


PHASES = ((kernels, "posterior_algebra"), (api, "prediction_algebra"),
          (api, "contrast_algebra"), (draws, "transform"),
          (forecast, "paths_algebra"), (forecast, "summarize"),
          (impulse, "impulse_algebra"), (impulse, "summarize"))
EXPECTED = ((1, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 0, 0, 0),
            (1, 0, 0, 1, 0, 0, 0, 0),
            (1, 0, 0, 1, 1, 2, 0, 0),
            (1, 0, 0, 1, 0, 0, 1, 1))


def require_phase_counts(counts, expected):
    assert tuple(counts[position] for position in range(len(PHASES))) == expected
    assert sum(expected) <= 5 < 12


def test_original_numerical_replay_multiplicity_is_complete_for_all_routes_inputs_and_outputs(states, monkeypatch):
    cold_guards(monkeypatch)
    counts = Counter()
    for position, (module, name) in enumerate(PHASES):
        original = getattr(module, name)
        def wrapper(*args, _position=position, _original=original, **kwargs):
            counts[_position] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, name, wrapper)
    for state, expected in zip(states, EXPECTED, strict=True):
        body = raw(state)
        # Each public operation includes its own normalization and one semantic
        # validation. Dict/JSON/typed inputs exercise every admitted transport path.
        for value in (body, json.dumps(body), state):
            for entry in (transport.bayes_var_restore, tables.bayes_var_tables, tables.bayes_var_state_json):
                counts.clear()
                entry(value)
                require_phase_counts(counts, expected)
                replay_plan = tables.admission(value)[1]
                assert replay_plan["estimated_work"] <= parent_body(body)["max_work"]
                # Negative sensitivity: one hidden duplicate phase fails this
                # receipt's count gate even if coefficients and full arrays match.
                duplicated = counts.copy()
                duplicated[0] += 1
                with pytest.raises(AssertionError):
                    require_phase_counts(duplicated, expected)


def test_combined_work_limit_still_refuses_before_normalization_and_numerics(states, monkeypatch):
    bodies = [copy.deepcopy(raw(state)) for state in states]
    for body in bodies:
        parent_body(body)["max_work"] = 1
        reseal(body)
    forbid_replay(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("Resource refusal followed complete primitive normalization.")
    monkeypatch.setattr(transport, "_primitive_tree", forbidden)
    for body in bodies:
        for entry in (transport.bayes_var_restore, tables.bayes_var_tables, tables.bayes_var_state_json):
            with pytest.raises(AnalysisError) as error:
                entry(body)
            assert error.value.code == "work_limit"
