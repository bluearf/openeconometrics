"""Lazy public delivery and persisted contracts for the assignment and identification extensions."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

NAMES = ('cluster_randomization', 'bernoulli_randomization', 'neyman_ate', 'stratified_neyman_ate', 'cluster_neyman_ate', 'paired_neyman_ate', 'manski_ate_inference', 'stratified_lee_bounds')


def test_causal_targets_manifests_are_torch_free():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys, openecon\n"
        "from openecon.econometrics.causal_design import EXPORTS\n"
        f"assert all(name in EXPORTS for name in {NAMES!r})\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'scipy' not in sys.modules\n"
        "assert 'statsmodels' not in sys.modules\n"
    )
    run = subprocess.run([sys.executable, "-c", code], cwd=root, text=True,
                         capture_output=True, env=dict(os.environ, PYTHONPATH=str(root / "src")))
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize("name", NAMES)
def test_public_causal_target_resolves_to_its_registered_source(name):
    import openecon as oe
    from openecon.econometrics.causal_design import EXPORTS

    function = getattr(oe, name)
    assert callable(function)
    assert function.__name__ == name
    assert function.__module__ == EXPORTS[name].split(":")[0]


def test_runtime_verifier_scopes_new_batch_without_changing_previous_batch():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import verify_causal_targets_runtime as old\n"
        "before = old.EXAMPLE, old.MODULES, old.NAMES, old.MARKER\n"
        "import verify_causal_assignment_runtime as new\n"
        "assert (old.EXAMPLE, old.MODULES, old.NAMES, old.MARKER) == before\n"
        f"assert new.NAMES == {NAMES!r}\n"
        "text = new.header('/synthetic-artifact-path')\n"
        "assert 'assignment_extensions' in text and 'identification_extensions' in text\n"
        "try:\n"
        "    with new.configured_engine():\n"
        "        assert old.NAMES == new.NAMES\n"
        "        raise ValueError('scope unwind')\n"
        "except ValueError:\n"
        "    pass\n"
        "assert (old.EXAMPLE, old.MODULES, old.NAMES, old.MARKER) == before\n"
    )
    run = subprocess.run([sys.executable, "-c", code], cwd=root, text=True,
                         capture_output=True, env=dict(os.environ, PYTHONPATH=str(root / "scripts")))
    assert run.returncode == 0, run.stderr


def test_full_frozen_receipt_rejects_previous_batch_or_incomplete_tables():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import verify_causal_assignment_runtime as v\n"
        "cases = [dict(status='ok', stdout='CAUSAL_TARGETS_EIGHT_OK {}', outputs=[]),\n"
        "         dict(status='ok', stdout=v.MARKER + '{}', outputs=[])]\n"
        "for case in cases:\n"
        "    try:\n"
        "        v.proof_from_execution(case)\n"
        "    except RuntimeError:\n"
        "        continue\n"
        "    raise AssertionError('incomplete or stale proof accepted')\n"
    )
    run = subprocess.run([sys.executable, "-c", code], cwd=root, text=True,
                         capture_output=True, env=dict(os.environ, PYTHONPATH=str(root / "scripts")))
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize("mode", ["--seed", "--ui-observed"])
@pytest.mark.parametrize("listener_kind", ["primary_app", "owned_path_prefix"])
def test_installed_stale_port_refuses_foreign_listener_before_any_api_request(
    tmp_path, monkeypatch, mode, listener_kind,
):
    import importlib.util
    import json
    import plistlib

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "causal_assignment_installed_guard", root / "scripts/verify_causal_assignment_installed.py"
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    app, data = tmp_path / "Owned QA.app", tmp_path / "owned-profile"
    (app / "Contents").mkdir(parents=True)
    data.mkdir()
    (app / "Contents/Info.plist").write_bytes(plistlib.dumps({
        "CFBundleIdentifier": "org.openecon.qa.causalassignment",
    }))
    (data / ".runtime-port.json").write_text(json.dumps({"port": 43123}))
    runtime = app / "Contents/Resources/runtime/openecon-runtime/openecon-runtime"
    foreign = (
        "/Applications/OpenEconometrics.app/Contents/Resources/runtime/openecon-runtime/openecon-runtime"
        if listener_kind == "primary_app" else str(runtime) + "-other"
    )
    calls, requests = [], []

    def fake_process(command, **options):
        calls.append(command)
        if command == ["lsof", "-t", "-nP", "-iTCP:43123", "-sTCP:LISTEN"]:
            return subprocess.CompletedProcess(command, 0, stdout="12345\n")
        assert command == ["ps", "-p", "12345", "-o", "command="]
        return subprocess.CompletedProcess(command, 0, stdout=foreign + " --port 43123\n")

    def forbidden_request(request, **options):
        requests.append(request)
        pytest.fail("A foreign runtime received an API read or mutation before its identity was checked")

    monkeypatch.setattr(verifier, "APP", app)
    monkeypatch.setattr(verifier, "DATA", data)
    monkeypatch.setattr(verifier.subprocess, "run", fake_process)
    monkeypatch.setattr(verifier, "urlopen", forbidden_request)
    monkeypatch.setattr(verifier, "source_identity", lambda *a, **k: pytest.fail("Foreign listener reached module verification"))
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(sys, "argv", ["verify_causal_assignment_installed.py", mode, "--receipt", str(receipt)])
    with pytest.raises(RuntimeError, match="does not belong to its installed runtime"):
        verifier.main()
    assert len(calls) == 2 and requests == []
    assert not receipt.exists()
    assert {path.name for path in data.iterdir()} == {".runtime-port.json"}
