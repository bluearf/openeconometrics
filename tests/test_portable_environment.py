"""Portable imports are inert until an explicit validated restore."""

from copy import deepcopy
import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from openecon import portable_environment as portable
from openecon.project_packages import PackageError
from openecon.server import create_app
from test_project_packages import CORE, PYTHON, manifest, managers as package_managers, wait_job


@pytest.fixture
def portable_managers(monkeypatch, tmp_path):
    yield from package_managers.__wrapped__(monkeypatch, tmp_path)


def resign(document):
    payload = {k: v for k, v in document.items() if k != "sha256"}
    document["sha256"] = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return document


def test_export_preserves_declarations_extras_inactive_markers_and_exact_locks():
    value = {
        **manifest("demo"),
        "schema": 2,
        "installer": "uv",
        "specifications": [
            'demo[extra]>=1.2; python_version >= "3.11"',
            'windows-only==2.0; sys_platform == "win32"',
        ],
    }
    document = portable.export_document(value)
    decoded = json.loads(json.dumps(document))
    result = portable.validate_document(decoded, core=CORE, python=PYTHON)
    assert result["requirements"] == result["locked"] == value["locked"]
    assert result["installer"] == "uv"
    assert "demo[extra]" in result["specifications"][0]
    assert 'sys_platform == "win32"' in result["specifications"][1]
    assert document["runtime"] == portable.runtime_identity()
    assert portable.export_document(result) == document


def test_another_project_preview_never_installs_and_explicit_restore_reopens(portable_managers):
    source, target = portable_managers("source"), portable_managers("target")
    source.start_install("demo")
    assert wait_job(source)["job"]["state"] == "complete"
    exported = source.export_portable()
    before = target.snapshot()
    assert target.preview_portable(exported)["matches"] is False
    assert target.snapshot() == before
    assert not list(target.root.iterdir())
    target.restore_portable(exported)
    restored = wait_job(target)
    assert restored["job"]["state"] == "complete"
    assert restored["manifest"] == source.snapshot()["manifest"]
    assert target.export_portable() == exported
    assert (target.active_path() / "demo.py").read_text() == (
        source.active_path() / "demo.py"
    ).read_text()
    reopened = portable_managers("target")
    assert reopened.export_portable() == exported


@pytest.mark.parametrize(
    "variant,code",
    [
        ("edited", "MANIFEST_CHECKSUM"),
        ("platform", "INCOMPATIBLE_PLATFORM"),
        ("core", "INCOMPATIBLE_CORE"),
        ("python", "INCOMPATIBLE_PYTHON"),
        ("extras", "INVALID_MANIFEST"),
        ("checksum_unicode", "INVALID_MANIFEST"),
        ("schema", "INVALID_MANIFEST"),
        ("oversize", "INVALID_MANIFEST"),
    ],
)
def test_bad_documents_preserve_previous_active_generation(portable_managers, variant, code):
    manager = portable_managers()
    manager.start_install("demo")
    assert wait_job(manager)["job"]["state"] == "complete"
    before = manager.snapshot()
    active = (manager.root / "active.json").read_bytes()
    document = deepcopy(manager.export_portable())
    if variant == "edited":
        document["manifest"]["locked"][0]["version"] = "9.0"
    elif variant == "platform":
        document["runtime"]["machine"] = "wrong"
        resign(document)
    elif variant == "core":
        document["manifest"]["core"]["openecon"] = "999.0"
        resign(document)
    elif variant == "python":
        document["manifest"]["python"] = "3.99"
        resign(document)
    elif variant == "extras":
        document["surprise"] = "field"
    elif variant == "checksum_unicode":
        document["sha256"] = "\u00e9" * 64
    elif variant == "schema":
        document["format"] = "other"
    elif variant == "oversize":
        document["manifest"]["oversized"] = "x" * 65536
    for action in (manager.preview_portable, manager.restore_portable):
        with pytest.raises(PackageError) as error:
            action(document)
        assert error.value.code == code
        assert manager.snapshot() == before
        assert (manager.root / "active.json").read_bytes() == active


def test_api_preview_checks_checksum_without_changing_environment(tmp_path):
    with TestClient(create_app(tmp_path, project_packages=True)) as client:
        token = client.get("/api/session").json()["token"]
        headers = {"X-OpenEcon-Token": token}
        document = client.get("/api/environment/export", headers=headers).json()
        before = client.get("/api/environment", headers=headers).json()
        preview = client.post(
            "/api/environment/import/preview", json={"document": document}, headers=headers
        )
        assert preview.status_code == 200
        assert preview.json()["matches"] is True
        assert client.get("/api/environment", headers=headers).json() == before
        document["manifest"]["python"] = "3.99"
        rejected = client.post(
            "/api/environment/import/restore", json={"document": document}, headers=headers
        )
        assert rejected.status_code == 422
        assert rejected.json()["detail"]["code"] == "MANIFEST_CHECKSUM"
        assert client.get("/api/environment", headers=headers).json() == before
