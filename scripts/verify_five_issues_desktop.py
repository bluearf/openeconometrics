"""Verify five method additions in a disposable frozen macOS desktop runtime.

Uses loopback console APIs and synthetic data. No installed application, user
project, cloud service or browser UI is changed. Host orchestration is stdlib.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import tempfile
import time
from types import CodeType
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(runtime):
    runtime = runtime.resolve(strict=True)
    example = ROOT / "docs/examples/five_more_issues.py"
    fingerprint = digest(runtime)
    from PyInstaller.archive.readers import CArchiveReader

    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")

    def normalized(code):
        return code.replace(
            co_filename="<five-issues-bundle>",
            co_consts=tuple(
                normalized(value) if isinstance(value, CodeType) else value
                for value in code.co_consts
            ),
        )

    modules = [
        "openecon._network_cost_flow",
        "openecon.networks",
        "openecon.console_worker",
        "openecon.econometrics.postest.dependent_resampling",
        "openecon.econometrics.postest.scores",
        "openecon.econometrics.postest.suest",
        "openecon.econometrics.postest.resampling",
        "openecon.econometrics.teffects.csdid",
        "openecon.econometrics.teffects.stacked",
        "openecon.econometrics.teffects.matching",
        "openecon.econometrics.teffects.estimators",
        "openecon.econometrics.teffects",
        "openecon.econometrics.streaming_registry",
        "openecon.econometrics.streaming_teffects",
        "openecon.econometrics.streaming_csdid",
    ]
    source_hashes = {}
    for module in modules:
        source = ROOT / "src" / Path(*module.split("."))
        source = source / "__init__.py" if source.is_dir() else source.with_suffix(".py")
        assert normalized(archive.extract(module)) == normalized(
            compile(source.read_text(), str(source), "exec", dont_inherit=True)
        ), module
        source_hashes[module] = digest(source)
    with tempfile.TemporaryDirectory(prefix="openecon-five-frozen-") as temporary:
        root = Path(temporary)
        prefix = f"/api/desktop/projects/{uuid4().hex}/workspace"
        process = descriptor = token = None
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"

        def start():
            nonlocal process, descriptor, token
            process = subprocess.Popen(
                [str(runtime), "--port", "0", "--data-root", str(root)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=environment,
                start_new_session=True,
            )
            deadline = time.monotonic() + 60
            descriptor = None
            while time.monotonic() < deadline:
                if select.select([process.stdout], [], [], 0.25)[0]:
                    descriptor = json.loads(process.stdout.readline(8193))
                    break
                if process.poll() is not None:
                    raise RuntimeError("Frozen runtime exited before ready")
            assert descriptor and descriptor.get("type") == "ready"
            parsed = urlparse(descriptor["url"])
            assert parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
            assert parsed.port == descriptor["port"]
            token = None
            token = call("/session")["token"]

        def stop():
            if process is not None and process.poll() is None:
                try:
                    process.stdin.write(b'{"type":"shutdown"}\n')
                    process.stdin.flush()
                    process.wait(timeout=12)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
            if process is not None:
                process.stdin.close()
                process.stdout.close()

        def call(path, body=None):
            headers = {"Content-Type": "application/json"}
            if token:
                headers["X-OpenEcon-Token"] = token
            request = Request(
                descriptor["url"] + prefix + path,
                headers=headers,
                data=json.dumps(body).encode() if body is not None else None,
            )
            with urlopen(request, timeout=240) as response:
                return json.load(response)

        try:
            start()
            full_path = root / "full-results"
            code = (
                """import sys
import importlib.util
from pathlib import Path
import openecon as oe
import openecon._network_cost_flow as flow_module
import openecon.econometrics.postest.dependent_resampling as bootstrap_module
assert getattr(sys, 'frozen', False)
assert all(Path(module.__file__).is_relative_to(Path(sys._MEIPASS)) for module in [oe, flow_module, bootstrap_module])
assert importlib.util.find_spec('scipy') is None
assert importlib.util.find_spec('statsmodels') is None
"""
                + example.read_text()
                + f"""
full_path = Path({str(full_path)!r})
full_path.mkdir()
for i, model in enumerate(models):
    (full_path/f'{{i}}.json').write_text(model.model_dump_json())
print('FROZEN_FIVE:' + json.dumps({{'frozen_execution': True, 'modules_from_bundle': True,
    'external_oracle_packages_absent': True, 'saved_full_results': len(models)}}))
"""
            )
            run = call("/console/execute", dict(code=code, timeout_seconds=240))
            assert run["status"] == "ok", run.get("error")
            markers = {}
            for marker in ["FIVE_ISSUES_RECEIPT:", "FROZEN_FIVE:"]:
                values = [
                    json.loads(line[len(marker) :])
                    for line in run["stdout"].splitlines()
                    if line.startswith(marker)
                ]
                assert len(values) == 1
                markers.update(values[0])
            types = [item["type"] for item in run["outputs"]]
            assert types == ["model"] * 6 + ["table"] * 3, types
            assert all("latex" in item for item in run["outputs"])
            original_hashes = {str(i): digest(full_path / f"{i}.json") for i in range(6)}
            call("/console/reset", {})
            saved = next(item for item in call("/console")["history"] if item["id"] == run["id"])
            assert saved["outputs"] == run["outputs"] and saved["events"] == run["events"]
            stop()
            start()
            reopened = next(item for item in call("/console")["history"] if item["id"] == run["id"])
            assert reopened["outputs"] == run["outputs"] and reopened["events"] == run["events"]
            restored = call(
                "/console/execute",
                dict(
                    code=f"""import json
from pathlib import Path
import openecon as oe
from openecon.models import ResultBundle
full_path = Path({str(full_path)!r})
restored = [ResultBundle.model_validate_json((full_path/f'{{i}}.json').read_text()) for i in range(6)]
assert all(model.covariance_matrix and model.sample_positions for model in restored)
assert restored[0].provenance['estimator'] == 'suest'
assert oe.test(restored[0], {{'ols:x': 1, 'count:x': -1}})['statistic'] >= 0
assert restored[3].inference['families']['calendar']['critical_value'] > 0
assert restored[5].extra['frequency_duplication']['expanded_rows'] == 96
print('REOPENED_FULL_RESULTS:6')
""",
                    timeout_seconds=120,
                ),
            )
            assert restored["status"] == "ok", restored.get("error")
            assert "REOPENED_FULL_RESULTS:6" in restored["stdout"]
            assert original_hashes == {str(i): digest(full_path / f"{i}.json") for i in range(6)}
            assert digest(runtime) == fingerprint
            receipt = dict(
                markers,
                status="passed",
                runtime_executable_sha256=fingerprint,
                example_sha256=digest(example),
                verifier_sha256=digest(__file__),
                console_duration_ms=run["duration_ms"],
                ordered_outputs=types,
                publication_latex=True,
                history_after_worker_reset=True,
                history_after_runtime_restart=True,
                full_results_after_restart=True,
                full_result_hashes=original_hashes,
                frozen_modules_match_source=source_hashes,
                human_data_access=False,
                installed_application_modified=False,
                installed_ui_verified=False,
                browser_render_verified=False,
                release_delivered=False,
            )
        finally:
            stop()
        receipt["owned_runtime_stopped"] = process.poll() is not None
    receipt["owned_temporary_data_removed"] = not root.exists()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.runtime)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(dict(status=receipt["status"], outputs=receipt["ordered_outputs"], restart=True))
    )


if __name__ == "__main__":
    main()
