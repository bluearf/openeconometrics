"""Real console records survive desktop publication and team readback as data."""

from copy import deepcopy
import json

import pytest

from openecon.console import ConsoleSession
from openecon.workspace import Workspace
from test_desktop_cloud import api as api, headers


CASES = {
    "ordinary": "y='wage', x=['education', 'experience'], covariance='HC3'",
    "weighted": "y='wage', x=['education', 'experience'], covariance='HC2', weights='weight', weight_type='aweight'",
    "multi_cluster": "y='wage', x=['education', 'experience'], covariance='cluster', cluster=['firm', 'wave']",
    "hac": "y='wage', x=['education', 'experience'], covariance='hac', time='tick', lags=2, kernel='bartlett'",
    "intercept_only": "y='wage', x=[], covariance='nonrobust'",
}


@pytest.fixture(scope="module")
def console_records(tmp_path_factory):
    session = ConsoleSession(Workspace(tmp_path_factory.mktemp("contract-console")))
    records = {}
    try:
        for name, arguments in CASES.items():
            code = f"""import openecon as oe
frame = oe.example()
frame['weight'] = 1 + frame.index % 5
frame['tick'] = range(len(frame))
frame['firm'] = frame.index // 20
frame['wave'] = frame.index % 12
model = oe.ols(data=frame, {arguments})
model.title = 'Synthetic publication probe'
print('before-model')
display(model)
print('before-chart')
display(oe.plot.coefficients(model))
display(oe.Latex(model.to_latex()))
"""
            record = session.execute(code)
            assert record["status"] == "ok", record.get("error")
            assert [item["type"] for item in record["outputs"]] == ["model", "plot", "latex"]
            record["actor_uid"] = "editor"
            records[name] = record
        yield records
    finally:
        session.close()


@pytest.mark.parametrize("case", CASES)
def test_new_console_model_round_trips_through_shared_archive(api, console_records, case):
    submitted = deepcopy(console_records[case])
    path = f"/api/projects/{api.pid}/workspace/desktop/results"
    response = api.client.post(
        path, headers=headers("editor"), json={"record": submitted, "input_files": []}
    )
    assert response.status_code == 201, response.json()
    shared = response.json()
    assert (
        api.client.post(
            path, headers=headers("editor"), json={"record": submitted, "input_files": []}
        ).json()
        == shared
    )
    run = api.store.db.get(f"oe_projects/{api.pid}/runs/{shared['id']}")
    archived = json.loads(api.storage.get(run["result"]))
    expected_outputs = deepcopy(submitted["outputs"])
    expected_outputs[0]["data"]["display_omitted"] = ["covariance_matrix", "sample_positions"]
    assert archived["outputs"] == expected_outputs
    assert archived["events"] == submitted["events"]
    assert archived["stdout"] == submitted["stdout"]
    model = archived["outputs"][0]["data"]
    assert model["title"] == "Synthetic publication probe"
    assert model["tests"] == submitted["outputs"][0]["data"]["tests"]
    assert {"weights", "weight_type", "time", "panel", "options"} <= model["spec"].keys()
    for role in ("owner", "viewer"):
        response = api.client.get(
            f"/api/projects/{api.pid}/workspace/console", headers=headers(role)
        )
        assert response.status_code == 200
        saved = response.json()["history"][0]
        assert saved["outputs"] == archived["outputs"]
        assert saved["events"] == archived["events"]
        assert saved["execution_origin"] == "desktop"
        assert saved["actor_email"] == "editor@example.com"
    assert (
        api.client.get(
            f"/api/projects/{api.pid}/workspace/console", headers=headers("outsider")
        ).status_code
        == 404
    )
    api.runner.start.assert_not_called()


def test_previous_model_contract_is_still_archived_and_readable(api, console_records):
    submitted = deepcopy(console_records["ordinary"])
    model = submitted["outputs"][0]["data"]
    for name in ("title", "tests"):
        model.pop(name)
    for name in ("weights", "weight_type", "time", "panel", "options"):
        model["spec"].pop(name)
    response = api.client.post(
        f"/api/projects/{api.pid}/workspace/desktop/results",
        headers=headers("editor"),
        json={"record": submitted, "input_files": []},
    )
    assert response.status_code == 201, response.json()
    saved = api.client.get(
        f"/api/projects/{api.pid}/workspace/console", headers=headers("viewer")
    ).json()["history"][0]
    assert saved["outputs"][0]["data"]["coefficients"] == model["coefficients"]
    assert saved["outputs"][0]["data"]["spec"]["covariance"] == "HC3"
    assert saved["outputs"][0]["data"]["tests"] == {}
    api.runner.start.assert_not_called()


@pytest.mark.parametrize(
    "invalid", ["unknown_result_field", "unknown_option", "invalid_weight", "invalid_coefficient"]
)
def test_new_model_contract_remains_strict_before_storage(api, console_records, invalid):
    submitted = deepcopy(console_records["ordinary"])
    model = submitted["outputs"][0]["data"]
    if invalid == "unknown_result_field":
        model["unexpected"] = "rejected"
    elif invalid == "unknown_option":
        model["spec"]["options"]["unimplemented"] = True
    elif invalid == "invalid_weight":
        model["spec"]["weight_type"] = "aweight"
    else:
        model["coefficients"][0]["estimate"] = "invalid"
    response = api.client.post(
        f"/api/projects/{api.pid}/workspace/desktop/results",
        headers=headers("editor"),
        json={"record": submitted, "input_files": []},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_RESULT"
    assert not api.storage.objects
    api.runner.start.assert_not_called()


def test_control_validates_new_model_and_legacy_latex_without_loading_compute(
    console_records, tmp_path
):
    import os
    from pathlib import Path
    import subprocess
    import sys

    submitted = deepcopy(console_records["weighted"])
    # Exercise re-rendering as well as already-generated presentation fields.
    for key in ("latex", "latex_math", "latex_style"):
        submitted["outputs"][0].pop(key, None)
    payload = tmp_path / "synthetic-record.json"
    payload.write_text(json.dumps(submitted))
    program = """import importlib.abc, json, sys
blocked = ('torch', 'pandas', 'openecon.analysis', 'openecon.console', 'openecon.engines')
class NoCompute(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Control imported compute: ' + fullname)
sys.meta_path.insert(0, NoCompute())
from openecon.team_server import validate_worker_result
from openecon.output_latex import enrich_record
record = json.load(open(sys.argv[1]))
run = dict(id='f' * 32, code=record['code'], created_at='2026-10-03T00:00:00+00:00', generation=0, email='synthetic@example.com')
trusted, files = validate_worker_result(dict(execution_id=run['id'], record=record), run)
assert not files
restored = enrich_record(trusted)
model = restored['outputs'][0]
assert model['data']['spec']['weight_type'] == 'aweight'
assert model['data']['tests']
assert 'analytic weights' in model['latex']
assert not any(name in sys.modules for name in blocked)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", program, str(payload)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
