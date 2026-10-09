"""Owned frozen causal-design console, full artifacts, worker reset and restart.

This starts only a temporary, isolated local runtime. Native installed-window
evidence is checked separately by verify_causal_design_installed.py and the UI.
"""

from __future__ import annotations

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
EXAMPLE = ROOT / "docs/examples/causal_design_eight.py"
MODULES = (
    "openecon",
    "openecon.econometrics.causal_design",
    "openecon.econometrics.causal_design.common",
    "openecon.econometrics.causal_design.balance",
    "openecon.econometrics.causal_design.paired",
    "openecon.econometrics.causal_design.distribution",
    "openecon.econometrics.causal_design.survival",
    "openecon.econometrics.registry",
    "openecon.econometrics.core",
    "openecon.econometrics.multivariate.common",
    "openecon.analysis_contracts",
    "openecon.resources",
)
NAMES = (
    "ebalance",
    "cem",
    "balance",
    "rosenbaum_bounds",
    "paired_randomization",
    "treatment_cdf",
    "treatment_quantile",
    "treatment_rmst",
)
MARKER = "CAUSAL_DESIGN_EIGHT_OK "


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def normalized(code):
    return code.replace(
        co_filename="<verified-bundled-source>",
        co_consts=tuple(
            normalized(value) if isinstance(value, CodeType) else value for value in code.co_consts
        ),
    )


def source_identity(runtime):
    from PyInstaller.archive.readers import CArchiveReader

    archive = CArchiveReader(str(runtime)).open_embedded_archive("PYZ.pyz")
    sources = {}
    for module in MODULES:
        path = ROOT / "src" / Path(*module.split("."))
        path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        expected = compile(path.read_text(), str(path), "exec", dont_inherit=True)
        if normalized(archive.extract(module)) != normalized(expected):
            raise RuntimeError(f"Frozen bytecode differs from current owned source: {module}")
        sources[module] = digest(path)
    return sources


def header(directory):
    return (
        "import sys, importlib, importlib.util\nfrom pathlib import Path\n"
        "assert getattr(sys, 'frozen', False)\n"
        "assert importlib.util.find_spec('scipy') is None\n"
        "assert importlib.util.find_spec('statsmodels') is None\n"
        f"for module_name in {MODULES!r}:\n"
        "    assert Path(importlib.import_module(module_name).__file__).is_relative_to(Path(sys._MEIPASS))\n"
        f"ARTIFACT_DIRECTORY = {str(directory)!r}\n"
    )


def proof_from_execution(execution):
    if execution.get("status") != "ok":
        raise RuntimeError(f"Causal-design example failed: {execution.get('error')}")
    matches = [
        json.loads(line[len(MARKER) :])
        for line in execution.get("stdout", "").splitlines()
        if line.startswith(MARKER)
    ]
    if len(matches) != 1:
        raise RuntimeError("A unique causal-design example receipt is required")
    proof = matches[0]
    if (
        proof.get("nameorder") != list(NAMES)
        or proof.get("full_roundtrip_equal") is not True
        or proof.get("frozen") is not True
        or set(proof.get("artifact_sha256s", {})) != set(NAMES)
        or set(proof.get("table_counts", {})) != set(NAMES)
    ):
        raise RuntimeError("The full eight-artifact frozen receipt is incomplete")
    outputs = execution.get("outputs", [])
    if [item.get("type") for item in outputs] != ["table"] * 8:
        raise RuntimeError("Exactly eight representative table outputs are required")
    if not all(
        any("\\begin{" + kind + "}" in item.get("latex", "") for kind in ("tabular", "longtable"))
        for item in outputs
    ):
        raise RuntimeError("A representative output lacks its complete LaTeX table")
    return proof


def artifact_files(directory, proof):
    """Verify full payload, state and every table hash without executing a fit."""
    files = {}
    for name in NAMES:
        path = Path(directory) / (name + ".json")
        artifact = json.loads(path.read_text())
        payload = artifact["payload"]
        attrs = payload["attrs"]
        if (
            payload.get("schema") != "openecon.causal_design.artifact.v1"
            or fingerprint(payload) != artifact["sha256"]
            or artifact["sha256"] != proof["artifact_sha256s"][name]
            or attrs.get("procedure") != name
            or attrs["state"].get("procedure") != name
            or fingerprint(attrs["state"]) != attrs["state_sha256"]
            or fingerprint(payload["tables"]) != attrs["tables_sha256"]
            or len(payload["tables"]) != proof["table_counts"][name]
            or attrs.get("stata_parity_validated") is not False
        ):
            raise RuntimeError(f"Full causal-design artifact failed integrity checks: {name}")
        files[name] = digest(path)
    return files


def restore_code(directory, proof):
    return header(directory) + (
        "import openecon as oe\n"
        f"expected = {proof['artifact_sha256s']!r}\n"
        "for name, sha in expected.items():\n"
        "    artifact = oe.causal_design_save(oe.causal_design_load(Path(ARTIFACT_DIRECTORY)/(name+'.json')))\n"
        "    assert artifact['sha256'] == sha\n"
        "    restored = oe.causal_design_load(artifact)\n"
        "    assert oe.causal_design_save(restored) == artifact\n"
        "    assert restored.attrs['stata_parity_validated'] is False\n"
        "    assert '\\\\begin{tabular}' in restored.to_latex() or '\\\\begin{longtable}' in restored.to_latex()\n"
        "print('CAUSAL_DESIGN_ARTIFACTS_REOPENED')\n"
    )


def verify(runtime):
    runtime = Path(runtime).resolve(strict=True)
    runtime_hash = digest(runtime)
    sources = source_identity(runtime)
    with tempfile.TemporaryDirectory(prefix="openecon-causal-design-frozen-") as temporary:
        data = Path(temporary)
        directory = data / "complete-causal-design-artifacts"
        prefix = f"/api/desktop/projects/{uuid4().hex}/workspace"
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        process = descriptor = token = error_handle = None

        def call(path, body=None):
            headers = {"Content-Type": "application/json"}
            if token:
                headers["X-OpenEcon-Token"] = token
            request = Request(
                descriptor["url"] + prefix + path,
                headers=headers,
                data=json.dumps(body).encode() if body is not None else None,
            )
            with urlopen(request, timeout=180) as response:
                return json.load(response)

        def start():
            nonlocal process, descriptor, token, error_handle
            error_handle = (data / "runtime-errors.log").open("ab")
            process = subprocess.Popen(
                [str(runtime), "--port", "0", "--data-root", str(data)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=error_handle,
                env=environment,
                start_new_session=True,
            )
            descriptor = token = None
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if select.select([process.stdout], [], [], 0.25)[0]:
                    descriptor = json.loads(process.stdout.readline(8193))
                    break
                if process.poll() is not None:
                    raise RuntimeError("Owned runtime exited before readiness")
            if not descriptor or descriptor.get("type") != "ready":
                raise RuntimeError("Owned runtime did not become ready")
            parsed = urlparse(descriptor["url"])
            if (
                parsed.scheme != "http"
                or parsed.hostname != "127.0.0.1"
                or parsed.port != descriptor["port"]
            ):
                raise RuntimeError("The runtime descriptor must identify owned loopback HTTP")
            token = call("/session")["token"]

        def stop():
            if process and process.poll() is None:
                try:
                    process.stdin.write(b'{"type":"shutdown"}\n')
                    process.stdin.flush()
                    process.wait(timeout=12)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            if process:
                process.stdin.close()
                process.stdout.close()
            if error_handle:
                error_handle.close()

        def saved_execution(identifier):
            return next(row for row in call("/console")["history"] if row["id"] == identifier)

        def reopen():
            reopened = call(
                "/console/execute", {"code": restore_code(directory, proof), "timeout_seconds": 120}
            )
            if reopened.get(
                "status"
            ) != "ok" or "CAUSAL_DESIGN_ARTIFACTS_REOPENED" not in reopened.get("stdout", ""):
                raise RuntimeError(
                    f"Frozen full-artifact restoration failed: {reopened.get('error')}"
                )

        try:
            start()
            code = header(directory) + EXAMPLE.read_text()
            executed = call("/console/execute", {"code": code, "timeout_seconds": 180})
            proof = proof_from_execution(executed)
            hashes = artifact_files(directory, proof)
            pinned = saved_execution(executed["id"])
            if (
                pinned["code"] != code
                or pinned["outputs"] != executed["outputs"]
                or pinned["events"] != executed["events"]
            ):
                raise RuntimeError("The completed code/outputs/events were not persisted intact")
            pinned_hash = fingerprint(pinned)
            call("/console/reset", {})
            reopen()
            if fingerprint(saved_execution(executed["id"])) != pinned_hash:
                raise RuntimeError("Completed execution changed after worker reset")
            if artifact_files(directory, proof) != hashes:
                raise RuntimeError("Full artifact files changed after worker reset")
            stop()
            start()
            if fingerprint(saved_execution(executed["id"])) != pinned_hash:
                raise RuntimeError("Completed execution changed after runtime restart")
            reopen()
            if artifact_files(directory, proof) != hashes:
                raise RuntimeError("Full artifact files changed after runtime restart")
            if digest(runtime) != runtime_hash or source_identity(runtime) != sources:
                raise RuntimeError("Runtime or accepted current source changed during verification")
            receipt = dict(
                status="passed",
                frozen_execution=True,
                proof=proof,
                runtime_sha256=runtime_hash,
                compiled_modules_equal_source=sources,
                example_sha256=digest(EXAMPLE),
                verifier_sha256=digest(__file__),
                source_path_injected=False,
                external_oracle_packages_absent=True,
                execution_id=executed["id"],
                execution_sha256=pinned_hash,
                executed_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                console_duration_ms=executed["duration_ms"],
                ordered_output_types=["table"] * 8,
                representative_table_rows=[
                    len(item["data"]["rows"]) for item in executed["outputs"]
                ],
                representative_outputs_have_latex=True,
                full_artifact_payload_sha256=proof["artifact_sha256s"],
                artifact_files_sha256=hashes,
                eight_full_artifact_files_verified=True,
                worker_reset_readback_unchanged=True,
                runtime_restart_readback_unchanged=True,
                full_artifacts_restored_after_worker_reset=True,
                full_artifacts_restored_after_runtime_restart=True,
                native_window_verified=False,
                primary_app_untouched=True,
                human_project_untouched=True,
                public_release_delivered=False,
            )
        finally:
            stop()
        receipt["owned_runtime_stopped"] = process.poll() is not None
    receipt["owned_temporary_profile_removed"] = not data.exists()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new receipt path")
    receipt = verify(args.runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(args.output),
                "full_artifacts": len(receipt["artifact_files_sha256"]),
            }
        )
    )


if __name__ == "__main__":
    main()
