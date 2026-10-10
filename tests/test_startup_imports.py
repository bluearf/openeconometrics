"""The web control process must not initialize the separate analysis runtime."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def fresh_process(source):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_desktop_bootstrap_does_not_import_torch_or_analysis(tmp_path):
    fresh_process('''
        import importlib.abc
        import sys
        import tempfile

        blocked = ("torch", "openecon.analysis", "openecon.engines")
        class NoTensorRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
                    raise AssertionError("Desktop bootstrap imported " + fullname)

        sys.meta_path.insert(0, NoTensorRuntime())
        import openecon as oe
        assert oe.capabilities()["precision"] == "float64"
        assert issubclass(oe.AnalysisError, ValueError)
        from openecon.desktop_runtime import create_desktop_app
        from fastapi.testclient import TestClient
        project = "a" * 32
        prefix = "/api/desktop/projects/" + project + "/workspace"
        app = create_desktop_app(tempfile.mkdtemp())
        with TestClient(app) as client:
            assert client.get("/api/auth/config").json() == {"mode": "desktop"}
            local_token = client.get("/api/session").json()["token"]
            local_headers = {"X-OpenEcon-Token": local_token}
            assert client.get("/api/capabilities", headers=local_headers).json() == oe.capabilities()
            assert client.get("/api/config", headers=local_headers).status_code == 200
            token = client.get(prefix + "/session").json()["token"]
            headers = {"X-OpenEcon-Token": token}
            assert client.put(prefix + "/console/script", headers=headers,
                              json={"code": "raise RuntimeError('must not run')"}).status_code == 200
            assert client.get(prefix + "/console", headers=headers).json()["status"]["pid"] is None
            assert client.get(prefix + "/desktop-cached-files", headers=headers).json() == {"files": [], "local_only": []}
            assert client.get(prefix + "/desktop-outbox", headers=headers).json() == {"items": []}
        assert not any(name in sys.modules for name in blocked)
    ''')


def test_desktop_entry_socket_startup_does_not_import_torch():
    fresh_process('''
        import asyncio
        import importlib.abc
        import sys
        import tempfile
        import threading
        from types import SimpleNamespace
        from unittest.mock import patch

        blocked = ("torch", "openecon.analysis", "openecon.engines")
        class NoTensorRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
                    raise AssertionError("Desktop entry startup imported " + fullname)
        sys.meta_path.insert(0, NoTensorRuntime())
        from openecon.desktop_entry import main
        gate = threading.Event()
        class NativeInput:
            def readline(self, maximum):
                gate.wait(10)
                return b'{"type":"shutdown"}\\n'

        def start_and_stop(server, sockets):
            async def lifecycle():
                server.config.load()
                server.lifespan = server.config.lifespan_class(server.config)
                await server.startup(sockets=sockets)
                assert server.started
                assert not any(name in sys.modules for name in blocked)
                await server.shutdown(sockets=sockets)
            asyncio.run(lifecycle())
            gate.set()

        # Run real app construction, socket binding, lifespan and ready output;
        # avoid a permanent HTTP loop in this isolated import boundary test.
        with patch("uvicorn.Server.run", start_and_stop), \\
             patch.object(sys, "stdin", SimpleNamespace(buffer=NativeInput())):
            assert main(["--port", "0", "--data-root", tempfile.mkdtemp()]) == 0
        assert not any(name in sys.modules for name in blocked)
    ''')


@pytest.mark.parametrize("module_name", ["openecon.workspace", "openecon.analysis"])
def test_workspace_analysis_retains_patchable_fit_hooks(tmp_path, monkeypatch, module_name):
    from importlib import import_module
    from openecon.analysis_contracts import AnalysisError
    from openecon.models import ModelSpec
    from openecon.workspace import Workspace

    workspace = Workspace(tmp_path)
    dataset = workspace.create_example()
    calls = []
    failure = AnalysisError("TEST_HOOK", "The patched fit was called.")

    def patched_fit(spec, *, data):
        calls.append((spec.outcome, len(data)))
        raise failure

    monkeypatch.setattr(import_module(module_name), "fit", patched_fit)
    with pytest.raises(AnalysisError) as raised:
        workspace.run_analysis(dataset["id"], ModelSpec(outcome="wage", predictors=["education"]))
    assert raised.value is failure
    assert calls == [("wage", 480)]


@pytest.mark.parametrize('legacy_runner', [None, 'jobs', 'sandbox'])
def test_team_cloud_bootstrap_serves_public_config_without_analysis_runtime(legacy_runner):
    fresh_process('''
        import importlib.abc
        import json
        import os
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        blocked = ("torch", "pandas", "openecon.analysis", "openecon.console",
                   "openecon.server", "openecon.data", "openecon.engines")

        class NoAnalysisRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
                    raise AssertionError("Control startup imported " + fullname)

        sys.meta_path.insert(0, NoAnalysisRuntime())
        import openecon
        assert all(name in dir(openecon) for name in openecon.__all__)
        from openecon.cloud import cloud_app
        from openecon.team_store import MemoryDocuments
        from fastapi.testclient import TestClient
        legacy_runner = LEGACY_PLACEHOLDER
        commit = "0123456789abcdef0123456789abcdef01234567"

        project = "openecon-test"
        origin = "https://openecon.example"
        config = {"projectId": project, "apiKey": "public-test-key",
                  "authDomain": project + ".firebaseapp.com", "appId": "test-app"}
        settings = {
            "OPENECON_MODE": "teams", "OPENECON_PROJECT_ID": project,
            "OPENECON_PUBLIC_ORIGIN": origin, "OPENECON_OWNER_EMAIL": "owner@example.com",
            "OPENECON_BUCKET": project + "-projects",
            "OPENECON_SIGNER_EMAIL": "signer@" + project + ".iam.gserviceaccount.com",
            "OPENECON_FIREBASE_CONFIG": json.dumps(config),
            "OPENECON_SOURCE_COMMIT": commit,
        }
        if legacy_runner:
            # Retired compute settings left on an old service are ignored.
            settings.update({
                "OPENECON_RUNNER": legacy_runner, "OPENECON_COMPUTE_JOB": "openecon-compute",
                "OPENECON_COMPUTE_EMAIL": "compute@" + project + ".iam.gserviceaccount.com",
                "OPENECON_COMPUTE_SERVICE": "openecon-sandbox",
                "OPENECON_COMPUTE_ORIGIN": "https://openecon-sandbox-123.us-central1.run.app",
                "OPENECON_COMPUTE_IMAGE": "image@sha256:" + "a" * 64,
            })
        # Replace external service clients only; run the real cloud branch,
        # configuration validation, auth middleware and HTTP bootstrap route.
        with patch.dict(os.environ, settings), \\
             patch("openecon.team_store.FirestoreDocuments", return_value=MemoryDocuments()), \\
             patch("openecon.team_storage.TeamStorage", return_value=SimpleNamespace()):
            app = cloud_app()
            with TestClient(app, base_url=origin) as client:
                response = client.get("/api/auth/config")
                assert response.status_code == 200
                body = response.json()
                assert body["firebase"]["projectId"] == project
                assert body["source_commit"] == commit
                assert body["cloud_execution_available"] is False
                assert client.get("/api/projects").status_code == 401
        assert not any(name in sys.modules for name in blocked)
        assert not any(name.startswith(("openecon.team_sandbox", "openecon.team_runner",
                                        "openecon.team_job", "google.cloud.run"))
                       for name in sys.modules)
    '''.replace('LEGACY_PLACEHOLDER', repr(legacy_runner)))


def test_lazy_public_api_keeps_exports_identity_and_import_star():
    fresh_process('''
        import openecon as oe
        import sys
        assert "torch" not in sys.modules and "pandas" not in sys.modules
        assert oe.__version__
        assert all(name in dir(oe) for name in oe.__all__)
        from openecon import ModelSpec, ResultBundle, plot
        assert "torch" not in sys.modules and "pandas" not in sys.modules
        from openecon.models import ModelSpec as ActualSpec, ResultBundle as ActualResult
        from openecon import plotting
        assert ModelSpec is ActualSpec and ResultBundle is ActualResult
        assert plot is plotting
        from openecon import *
        from openecon import analysis, data, console
        for name in ("AnalysisError", "capabilities", "fit", "ols", "logit", "probit"):
            assert getattr(oe, name) is getattr(analysis, name)
            assert globals()[name] is getattr(analysis, name)
        assert read is data.read and example is data.example_frame
        assert load_dataset is console.load_dataset
        assert oe.ols is ols
        try:
            oe.unknown_public_name
        except AttributeError as exc:
            assert "unknown_public_name" in str(exc)
        else:
            raise AssertionError("Unknown names must raise AttributeError")
    ''')
