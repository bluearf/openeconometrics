"""Exercise a real bundled STDIO launcher and its loopback project UI API.

Only disposable synthetic projects are opened. The server receives no Python
path, system Python lookup, cloud token or client configuration mutation.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import time
import tempfile
from urllib.request import Request, urlopen
from uuid import uuid4

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openecon.desktop_runtime import worker_environment


class Runtime:
    def __init__(self, executable: Path, root: Path, log: Path):
        self.executable, self.root, self.log = executable, root, log

    def __enter__(self):
        self.errors = self.log.open("ab")
        environment = worker_environment()
        environment["PATH"] = "/usr/bin:/bin" if os.name != "nt" else environment.get("PATH", "")
        self.environment = environment
        self.process = subprocess.Popen(
            [str(self.executable), "--data-root", str(self.root)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors,
            env=environment, cwd=self.root,
        )
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Bundled runtime stopped before readiness; inspect the private log.")
            if select.select([self.process.stdout], [], [], .2)[0]:
                line = self.process.stdout.readline(4097)
                value = json.loads(line)
                if value.get("type") == "ready":
                    self.origin = value["url"]
                    return self
        self.process.terminate()
        self.process.wait(timeout=10)
        raise TimeoutError("Bundled runtime readiness timed out.")

    def request(self, path: str, *, token: str | None = None, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-OpenEcon-Token"] = token
        with urlopen(Request(self.origin + path, data=data, headers=headers), timeout=45) as response:
            return json.load(response)

    def __exit__(self, *args):
        self.process.stdin.write(b'{"type":"shutdown"}\n')
        self.process.stdin.flush()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.errors.close()


async def analyze(parameters, dataset_id):
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            initialized = await session.initialize()
            assert initialized.serverInfo.name == "OpenEconometrics"
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            assert names == {"list_datasets", "inspect_dataset", "create_example_dataset",
                             "run_analysis", "get_result", "list_results", "get_capabilities",
                             "start_analysis", "get_analysis_job", "cancel_analysis_job"}
            datasets = await session.call_tool("list_datasets")
            assert datasets.structuredContent["datasets"][0]["id"] == dataset_id
            request_id = str(uuid4())
            arguments = {
                "dataset_id": dataset_id,
                "spec": {"outcome": "wage", "predictors": ["education", "experience"], "covariance": "HC3"},
                "request_id": request_id,
            }
            result = await session.call_tool("start_analysis", arguments)
            assert not result.isError and result.structuredContent["state"] == "running", result
            job_id = result.structuredContent["id"]
            retry = await session.call_tool("start_analysis", arguments)
            assert not retry.isError and retry.structuredContent["id"] == job_id
            for _ in range(400):
                status = await session.call_tool("get_analysis_job", {"job_id": job_id})
                assert not status.isError, status
                if status.structuredContent["state"] == "completed":
                    break
                assert status.structuredContent["state"] in {"running", "publishing"}, status
                await asyncio.sleep(.05)
            assert status.structuredContent["state"] == "completed", status
            payload = status.structuredContent["result"]
            retry = await session.call_tool("start_analysis", arguments)
            assert retry.structuredContent["result_id"] == payload["id"]
            saved = await session.call_tool("list_results")
            assert len(saved.structuredContent["results"]) == 1
            cancel_start = await session.call_tool("start_analysis", {**arguments, "request_id": str(uuid4())})
            assert not cancel_start.isError and cancel_start.structuredContent["state"] == "running"
            cancel_id = cancel_start.structuredContent["id"]
            cancelled = await session.call_tool("cancel_analysis_job", {"job_id": cancel_id})
            assert not cancelled.isError
            for _ in range(100):
                cancelled = await session.call_tool("get_analysis_job", {"job_id": cancel_id})
                assert not cancelled.isError, cancelled
                if cancelled.structuredContent["state"] == "cancelled":
                    break
                assert cancelled.structuredContent["state"] == "cancelling", cancelled
                await asyncio.sleep(.05)
            assert cancelled.structuredContent["state"] == "cancelled", cancelled
            saved = await session.call_tool("list_results")
            assert len(saved.structuredContent["results"]) == 1
            assert payload["nobs"] == 480 and payload["observation_data_included"] is False
            assert payload["provenance"]["backend"] == "openecon.torch"
            assert not {"predictions", "sample_positions", "fitted_values", "residuals"} & payload.keys()
            return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    executable = arguments.runtime.resolve(strict=True)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    receipt = {"runtime_sha256": digest, "synthetic_only": True,
               "client_configs_changed": False, "system_python_required_by_server": False}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="OpenEconometrics MCP with spaces ") as temporary:
        root = Path(temporary).resolve()
        project = uuid4().hex
        prefix = f"/api/desktop/projects/{project}/workspace"
        with Runtime(executable, root, arguments.output.with_suffix(".private.log")) as server:
            token = server.request(prefix + "/session")["token"]
            config = server.request(prefix + "/config", token=token)
            assert config["mcp_server"] == {
                "command": str(executable), "args": ["mcp", "--workspace", str(root / "projects" / project)]}
            assert config["mcp_available"] is True
            profile = server.request(prefix + "/datasets/example", token=token, body={})
            previous = server.request(prefix + "/results/revision", token=token)["revision"]
            parameters = StdioServerParameters(**config["mcp_server"], env=server.environment, cwd=str(root))
            result = asyncio.run(asyncio.wait_for(analyze(parameters, profile["id"]), timeout=45))
            history = server.request(prefix + "/console", token=token)
            assert history["status"]["pid"] is None and history["variables"] == []
            agent = history["history"][0]
            assert agent["source"] == "mcp" and agent["id"] == result["id"]
            output = agent["outputs"][0]
            assert output["data"]["coefficients"] == result["coefficients"]
            assert "\\toprule" in output["latex"] and "\\bottomrule" in output["latex"]
            assert previous != server.request(prefix + "/results/revision", token=token)["revision"]
            receipt.update(stdio_discovery=True, project_dataset_shared=True,
                           background_job=True, idempotent_retry=True, cancellation_without_result=True,
                           synthetic_ols=True, publication_latex=True, console_worker_not_started=True,
                           copied_launcher_has_spaces=True, aggregate_default_response=True)
        with Runtime(executable, root, arguments.output.with_suffix(".private.log")) as server:
            token = server.request(prefix + "/session")["token"]
            assert server.request(prefix + "/console", token=token)["history"][0]["id"] == result["id"]
            other = f"/api/desktop/projects/{uuid4().hex}/workspace"
            token = server.request(other + "/session")["token"]
            assert server.request(other + "/console", token=token)["history"] == []
            assert server.request(other + "/datasets", token=token)["datasets"] == []
            receipt.update(reopened_history_preserved=True, project_isolation=True)
    receipt["status"] = "passed"
    arguments.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
