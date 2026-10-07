from __future__ import annotations

import asyncio
import json
import os
import sys
from uuid import uuid4
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openecon.mcp_server import compact_result
from openecon.workspace import Workspace


def _payload(result: Any) -> dict[str, Any]:
    assert not result.isError, result
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(next(item.text for item in result.content if item.type == "text"))


def test_compact_results_exclude_observation_level_values() -> None:
    compact = compact_result(
        {
            "id": "result_1",
            "nobs": 2,
            "coefficients": [{"term": "x", "estimate": 1.2}],
            "sample_positions": [3, 5],
            "fitted_values": [20, 30],
            "residuals": [0.1, 0.2],
            "diagnostics": {"durbin_watson": 1.9, "residuals": [0.1, 0.2]},
            "covariance_matrix": [[1.0]],
            "provenance": {
                "backend": "openecon.torch",
                "versions": {"torch": "test"},
                "categorical_encoding": {"private_category": ["private-value"]},
                "input_columns": ["private_column"],
            },
        },
        include_diagnostics=True,
    )
    assert compact["nobs"] == 2
    assert compact["observation_data_included"] is False
    assert compact["diagnostics"] == {"durbin_watson": 1.9}
    assert compact["covariance_matrix"] == [[1.0]]
    assert compact["provenance"] == {"backend": "openecon.torch", "versions": {"torch": "test"}}
    assert "private" not in json.dumps(compact)
    assert not {"sample_positions", "fitted_values", "residuals"}.intersection(compact)


@pytest.mark.parametrize("entrypoint", ["openecon.cli", "openecon.desktop_entry"])
def test_real_stdio_discovery_analysis_and_default_data_minimization(tmp_path: Path, entrypoint: str) -> None:
    workspace = tmp_path / "workspace"
    data_path = tmp_path / "private-data.csv"
    rng = np.random.default_rng(42)
    x = rng.normal(size=40)
    pd.DataFrame(
        {
            "x": x,
            "y": 4 + 2 * x + rng.normal(size=40),
            "confidential": [f"private-person-{index:03d}" for index in range(40)],
        }
    ).to_csv(data_path, index=False)
    metadata = Workspace(workspace).import_file(data_path)
    dataset_id = metadata["id"]
    source_root = Path(__file__).resolve().parents[1] / "src"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", entrypoint, "mcp", "--workspace", str(workspace)],
        env={**os.environ, "PYTHONPATH": str(source_root), "PYTHONUNBUFFERED": "1"},
    )

    async def exercise() -> None:
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                assert initialized.serverInfo.name == "OpenEconometrics"
                tools = await session.list_tools()
                assert {tool.name for tool in tools.tools} == {
                    "list_datasets",
                    "inspect_dataset",
                    "create_example_dataset",
                    "run_analysis",
                    "start_analysis",
                    "get_analysis_job",
                    "cancel_analysis_job",
                    "get_result",
                    "list_results",
                    "get_capabilities",
                }
                caps = _payload(await session.call_tool("get_capabilities"))
                assert "ols" in json.dumps(caps)
                assert "backends" not in caps
                listed = _payload(await session.call_tool("list_datasets"))
                assert listed["datasets"][0]["id"] == dataset_id
                assert "preview" not in listed["datasets"][0]
                inspected = _payload(
                    await session.call_tool("inspect_dataset", {"dataset_id": dataset_id})
                )
                assert inspected["row_count"] == 40
                assert "preview" not in inspected
                assert "private-person-" not in json.dumps(inspected)
                assert inspected["preview_included"] is False
                preview = _payload(
                    await session.call_tool(
                        "inspect_dataset", {"dataset_id": dataset_id, "include_preview": True}
                    )
                )
                assert len(preview["preview"]) == 10
                assert preview["preview"][0]["confidential"] == "private-person-000"
                analysis = _payload(
                    await session.call_tool(
                        "run_analysis",
                        {
                            "dataset_id": dataset_id,
                            "spec": {
                                "estimator": "ols",
                                "outcome": "y",
                                "predictors": ["x"],
                                "covariance": "HC3",
                                "missing": "raise",
                            },
                        },
                    )
                )
                assert analysis["nobs"] == 40
                assert "backend" not in analysis["spec"]
                assert "device" not in analysis["spec"]
                assert analysis["provenance"]["backend"] == "openecon.torch"
                assert analysis["observation_data_included"] is False
                assert "sample_positions" not in analysis
                assert "fitted_values" not in analysis
                assert "private-person-" not in json.dumps(analysis)
                result_id = analysis.get("id") or analysis["result_id"]
                saved = _payload(await session.call_tool("get_result", {"result_id": result_id}))
                assert saved["coefficients"] == analysis["coefficients"]
                unsupported = await session.call_tool(
                    "run_analysis",
                    {
                        "dataset_id": dataset_id,
                        "spec": {
                            "estimator": "eval",
                            "outcome": "y",
                            "predictors": ["x"],
                        },
                    },
                )
                assert unsupported.isError
                for obsolete_field, value in [("backend", "numpy"), ("device", "cuda")]:
                    obsolete = await session.call_tool(
                        "run_analysis",
                        {"dataset_id": dataset_id, "spec": {
                            "outcome": "y", "predictors": ["x"], obsolete_field: value,
                        }},
                    )
                    assert obsolete.isError, "Removed execution selectors must not be silently ignored."
                inaccessible = await session.call_tool(
                    "inspect_dataset", {"dataset_id": "../../private-data.csv"}
                )
                assert inaccessible.isError
                assert "[NOT_FOUND]" in " ".join(
                    item.text for item in inaccessible.content if item.type == "text"
                )
                missing_column = await session.call_tool(
                    "run_analysis",
                    {
                        "dataset_id": dataset_id,
                        "spec": {
                            "estimator": "ols",
                            "outcome": "y",
                            "predictors": ["unknown"],
                        },
                    },
                )
                assert missing_column.isError
                assert "[missing_columns]" in " ".join(
                    item.text for item in missing_column.content if item.type == "text"
                )
                results = _payload(await session.call_tool("list_results"))
                assert len(results["results"]) == 1
                example = _payload(await session.call_tool("create_example_dataset"))
                assert example["source"] == "example"
                assert "preview" not in example
                history = Workspace(workspace).display_history()
                assert len(history) == 1
                assert history[0]["id"] == result_id and history[0]["source"] == "mcp"
                assert history[0]["duration_ms"] > 0
                assert history[0]["outputs"][0]["data"]["predictions"] == []
                assert Workspace(workspace).console_history() == []

    asyncio.run(asyncio.wait_for(exercise(), timeout=45))


def test_real_stdio_background_retry_across_two_connections(tmp_path):
    workspace = tmp_path / "workspace"
    frame = pd.DataFrame({"x": [1., 2., 3., 4., 5., 6., 7.],
                          "y": [4., 5., 9., 10., 10., 14., 16.],
                          "weight": [1., 2., 1., 3., 2., 1., 4.],
                          "private": [f"secret-person-{i}" for i in range(7)]})
    frame.loc[2, "x"] = np.nan
    path = tmp_path / "weighted.csv"
    frame.to_csv(path, index=False)
    store = Workspace(workspace)
    dataset = store.import_file(path)
    spec = {"outcome": "y", "predictors": ["x"], "weights": "weight",
            "weight_type": "aweight", "covariance": "HC3", "missing": "drop"}
    reference = store.compute_analysis(dataset["id"], spec)
    parameters = StdioServerParameters(command=sys.executable,
        args=["-m", "openecon.desktop_entry", "mcp", "--workspace", str(workspace)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})

    async def exercise():
        async with stdio_client(parameters) as (read1, write1), stdio_client(parameters) as (read2, write2):
            async with ClientSession(read1, write1) as first, ClientSession(read2, write2) as second:
                await first.initialize()
                await second.initialize()
                arguments = {"dataset_id": dataset["id"], "spec": spec, "request_id": str(uuid4())}
                start = _payload(await first.call_tool("start_analysis", arguments))
                assert start["state"] == "running"
                retry = _payload(await second.call_tool("start_analysis", arguments))
                assert retry["id"] == start["id"]
                assert "ols" in json.dumps(_payload(await second.call_tool("get_capabilities")))
                for _ in range(400):
                    job = _payload(await second.call_tool("get_analysis_job", {"job_id": start["id"]}))
                    if job["state"] == "completed":
                        break
                    assert job["state"] in {"running", "publishing"}, job
                    await asyncio.sleep(.05)
                assert job["state"] == "completed", job
                assert "secret-person-" not in json.dumps(job)
                assert job["result"]["coefficients"] == reference["coefficients"]
                assert job["result"]["inference"] == reference["inference"]
                saved = store.get_result(job["result_id"])
                assert saved["sample_positions"] == reference["sample_positions"]
                assert saved["nobs"] == 6 and saved["dropped_rows"] == 1
                assert saved["spec"]["weight_type"] == "aweight"
                mtime = (store.result_path / f"{job['result_id']}.json").stat().st_mtime_ns
                assert _payload(await first.call_tool("start_analysis", arguments))["result_id"] == job["result_id"]
                assert _payload(await second.call_tool("cancel_analysis_job", {"job_id": job["id"]}))["state"] == "completed"
                assert (store.result_path / f"{job['result_id']}.json").stat().st_mtime_ns == mtime
                assert len(store.list_results()) == len(store.display_history()) == 1
                bad = await first.call_tool("start_analysis", {**arguments, "timeout_seconds": 0})
                assert bad.isError and "[JOB_LIMIT]" in " ".join(item.text for item in bad.content if item.type == "text")

    asyncio.run(asyncio.wait_for(exercise(), timeout=50))
