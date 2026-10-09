"""Owned synthetic helper proofs; these tests never launch a native app."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from openecon.econometrics.core import TableSet, table
from openecon.output_latex import add_output_latex


SCRIPTS = Path(__file__).parents[1]/"scripts"


@pytest.fixture
def helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    modules = []
    for name in ("verify_rank_ordered_runtime", "verify_rank_ordered_source_bridge"):
        spec = importlib.util.spec_from_file_location("rank_acceptance_test_"+name, SCRIPTS/(name+".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


@pytest.fixture
def owned(tmp_path, helpers):
    verifier, _ = helpers
    outputs, methods = [], {}
    for name in verifier.METHODS:
        frames = {key: table([[1., None, ["A", "B"]]], columns=["value", "missing", "remaining"])
                  for key in verifier.expected_tables(name)}
        result = TableSet(frames, contract="rank_ordered_v1" if name in ("oim", "hc0", "cr0") else "rank_ordered_postestimation_v1")
        payload = {"attrs": result.attrs,
                   "tables": {key: value.astype(object).where(value.notna(), None).to_dict(orient="split") for key, value in frames.items()},
                   "latex": result.to_latex()}
        content = json.dumps(payload, sort_keys=True, allow_nan=False)
        path = tmp_path/(name+".json")
        path.write_text(content)
        key = "parameters" if name in ("oim", "hc0", "cr0") else "predictions" if name in ("first", "stages") else "margins"
        output = {"type": "table", "data": {"columns": ["value", "missing", "remaining"], "rows": [[1., None, "['A', 'B']"]],
                                               "index": [[0]], "index_names": [None], "total_rows": 1, "total_columns": 3}}
        add_output_latex(output)
        outputs.append(output)
        methods[name] = {"sha256": hashlib.sha256(content.encode()).hexdigest(),
                         "table_rows": {key: 1 for key in frames}, "displayed_keys": [key],
                         "displayed_latex_sha256": {key: hashlib.sha256(str(frames[key].to_latex()).encode()).hexdigest()}}
    proof = {"frozen": True, "all_four_scopes_verified": True, "full_state_replay_verified": True,
             "saved_tables": 76, "displayed_tables": 7, "third_party_estimation_imports": [], "methods": methods}
    run = {"status": "ok", "stdout": verifier.MARKER+json.dumps(proof, sort_keys=True), "outputs": outputs}
    return verifier, tmp_path, run, proof


def test_complete_files_and_exact_displayed_values_index_and_latex_bind(owned):
    verifier, directory, run, proof = owned
    assert verifier.verify_outputs(run, directory) == proof
    assert len(verifier.verify_files(directory, proof)) == 7


@pytest.mark.parametrize("alter", [
    lambda data: data["rows"][0].__setitem__(0, 2.),
    lambda data: data["columns"].__setitem__(0, "wrong"),
    lambda data: data["index"][0].__setitem__(0, 2),
    lambda data: data.__setitem__("total_columns", 4),
    lambda data: data.__setitem__("index_names", ["wrong"]),
])
def test_correct_shapes_do_not_admit_wrong_displayed_data(owned, alter):
    verifier, directory, run, _ = owned
    bad = copy.deepcopy(run)
    alter(bad["outputs"][0]["data"])
    with pytest.raises(AssertionError):
        verifier.verify_outputs(bad, directory)


def test_displayed_latex_with_valid_environment_but_wrong_content_is_refused(owned):
    verifier, directory, run, _ = owned
    run["outputs"][0]["latex"] = run["outputs"][0]["latex"].replace("1.0000", "2.0000")+"% forged"
    with pytest.raises(AssertionError, match="LaTeX"):
        verifier.verify_outputs(run, directory)


def test_self_reported_smaller_table_scope_is_not_complete_acceptance(owned):
    verifier, directory, run, proof = owned
    proof["saved_tables"] = 75
    run["stdout"] = verifier.MARKER+json.dumps(proof, sort_keys=True)
    with pytest.raises(AssertionError):
        verifier.verify_outputs(run, directory)


def test_resealed_saved_full_latex_must_match_every_complete_table(owned):
    verifier, directory, _, proof = owned
    path = directory/"oim.json"
    value = json.loads(path.read_text())
    value["latex"] += "% forged export"
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    proof["methods"]["oim"]["sha256"] = verifier.digest(path)
    with pytest.raises(AssertionError, match="full LaTeX"):
        verifier.verify_files(directory, proof)


def test_imported_scientific_modules_must_be_the_hashed_current_source(helpers, monkeypatch, tmp_path):
    _, bridge = helpers
    assert len(bridge.source_origins()) == 3
    import openecon.econometrics.discrete.rank_ordered as fitted
    other = tmp_path/"foreign-rank-ordered.py"
    other.write_text("# A foreign installed source path\n")
    monkeypatch.setattr(fitted, "__file__", str(other))
    with pytest.raises(AssertionError, match="source import origin"):
        bridge.source_origins()
