"""Schema 2 team HTTP persistence is inert and keeps legacy compatibility."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
from packaging.markers import Marker
import pytest

from openecon.team_auth import TeamAuth, TeamIdentity
from openecon.team_server import create_team_app
from openecon.team_storage import MemoryStorage
from openecon.team_store import MemoryDocuments, TeamStore

_SPEC = importlib.util.spec_from_file_location(
    "uv_manifest_live_contract", Path(__file__).parents[1] / "scripts/verify_uv_manifest_live.py")
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)
ORIGIN = "https://uv-manifest.example.com"
PROJECT = "openecon-test"


@pytest.fixture
def api(monkeypatch):
    from openecon import package_installer
    install = Mock(side_effect=AssertionError("Metadata must not install packages"))
    evaluate = Mock(side_effect=AssertionError("Metadata must not evaluate server platform markers"))
    monkeypatch.setattr(package_installer, "install", install)
    monkeypatch.setattr(Marker, "evaluate", evaluate)
    db, storage = MemoryDocuments(), MemoryStorage()
    users = {role: TeamIdentity(role, f"{role}@example.com", role, True)
             for role in ("owner", "viewer", "outsider")}
    store = TeamStore(db, owner_email=users["owner"].email, public_origin=ORIGIN)
    store.me(users["owner"])
    project = store.create_project(users["owner"], "Disposable uv manifest")
    invite = store.invite(project["id"], users["owner"], users["viewer"].email, "viewer")
    store.accept(invite["id"], users["viewer"])

    def verify(token):
        user = users[token]
        return {"uid": user.uid, "sub": user.uid, "email": user.email, "name": user.name,
                "email_verified": True, "aud": PROJECT,
                "iss": f"https://securetoken.google.com/{PROJECT}",
                "firebase": {"sign_in_provider": "password"}}

    runner = Mock()
    runner.start.side_effect = AssertionError("Manifest sharing must not execute code")
    app = create_team_app(store=store, storage=storage, runner=runner,
                          auth=TeamAuth(PROJECT, verifier=verify), public_origin=ORIGIN,
                          firebase_config={"projectId": PROJECT, "authDomain": PROJECT + ".firebaseapp.com"})
    with TestClient(app, base_url=ORIGIN) as client:
        def call(method, path, *, role="owner", expected=200, **kwargs):
            response = client.request(method, "/api" + path,
                                      headers={"Authorization": "Bearer " + role, "Origin": ORIGIN}, **kwargs)
            assert response.status_code == expected, response.text
            assert response.headers["cache-control"] == "no-store"
            return response.json()
        yield SimpleNamespace(client=client, call=call, store=store, db=db, storage=storage,
                              runner=runner, project=project["id"], install=install, evaluate=evaluate)
    install.assert_not_called()
    evaluate.assert_not_called()
    assert runner.mock_calls == []
    assert storage.objects == {}
    assert not any("/runs/" in path for path in db.data)


def path(api):
    return f"/projects/{api.project}/workspace/environment"


def test_reusable_live_contract_passes_against_in_memory_team_http(api):
    checks = probe.verify_contract(api.call, api.project)
    assert len(checks) == 13
    assert "legacy_schema1_exact_roundtrip" in checks
    assert "viewer_reads_200_writes_403_without_mutation" in checks
    expected = {"manifest": probe.modern_manifest(), "version": 3}
    assert api.call("GET", path(api), role="viewer") == expected
    persisted = api.db.get(f"oe_projects/{api.project}/workspace/environment")
    assert persisted["manifest"] == expected["manifest"] and persisted["version"] == 3


@pytest.mark.parametrize("installer", ["uv", "pip"])
def test_schema2_installer_specs_and_inactive_platform_roots_are_preserved(api, installer):
    manifest = probe.modern_manifest() | {"installer": installer}
    response = api.call("PUT", path(api), json={"manifest": manifest, "version": 0})
    assert response == {"manifest": manifest, "version": 1}
    # Darwin's locked root and Win32's absent root remain declarative on any host.
    assert len(response["manifest"]["specifications"]) == 2
    assert response["manifest"]["requirements"] == [{"name": "oe-qa-demo", "version": "1.2.3"}]
    assert api.call("GET", path(api), role="viewer") == response


@pytest.mark.parametrize("name,invalid", list(probe.invalid_manifests().items()))
def test_invalid_schema2_inputs_do_not_replace_existing_manifest(api, name, invalid):
    saved = api.call("PUT", path(api), json={"manifest": probe.modern_manifest(), "version": 0})
    before = deepcopy(api.db.data)
    rejected = api.call("PUT", path(api), expected=422, json={"manifest": invalid, "version": 1})
    assert rejected["detail"]["code"] == "INVALID_ENVIRONMENT"
    assert api.call("GET", path(api), role="viewer") == saved
    # Rejected mutations must not increment versions or emit update audit events.
    assert api.db.data == before


def test_viewer_write_and_optimistic_conflict_leave_exact_store_unchanged(api):
    api.call("PUT", path(api), json={"manifest": probe.modern_manifest(), "version": 0})
    before = deepcopy(api.db.data)
    api.call("PUT", path(api), role="viewer", expected=403,
             json={"manifest": probe.legacy_manifest(), "version": 1})
    assert api.db.data == before
    rejected = api.call("PUT", path(api), expected=409,
                        json={"manifest": probe.legacy_manifest(), "version": 0})
    assert rejected["detail"]["code"] == "VERSION_CONFLICT"
    assert api.db.data == before


def test_schema2_project_metadata_not_readable_by_outsider_or_anonymous(api):
    api.call("PUT", path(api), json={"manifest": probe.modern_manifest(), "version": 0})
    api.call("GET", path(api), role="outsider", expected=404)
    response = api.client.get("/api" + path(api))
    assert response.status_code == 401 and "oe-qa-demo" not in response.text


def test_clean_source_guard_refuses_previously_loaded_nonrelease_modules(monkeypatch, tmp_path):
    release = tmp_path / "release"
    expected = release / "src/openecon"
    expected.mkdir(parents=True)
    (expected / "team_store.py").write_text("# guard fixture, never imported\n")
    monkeypatch.setattr(probe, "RELEASE_SOURCE", release)
    original = list(probe.sys.path)
    monkeypatch.setattr(probe.sys, "path", original)
    with pytest.raises(probe.ProbeFailure, match="clean 0.3.7"):
        probe.configure_release_source()
