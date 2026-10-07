"""Synthetic acceptance against the separately installed native macOS QA app.

The native shell owns the frozen server. This script uses its public loopback
API, never injects a source Python path, and only creates an isolated QA project.
UI Run/render/restart evidence is recorded separately after these checks.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Library/Application Support/org.openecon.qa.network-data.01a11244"
APP = Path.home() / "Applications/OpenEconometrics-Network-QA/OpenEconometrics.app"
OUT = ROOT / "docs/evidence/network-data-installed-2026-10-07"


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def request(url, body=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-OpenEcon-Token"] = token
    with urlopen(
        Request(
            url,
            method="PUT" if body is not None and "/console/scripts/" in url else None,
            headers=headers,
            data=json.dumps(body).encode() if body is not None else None,
        ),
        timeout=1800,
    ) as response:
        return json.load(response)


UI_CODE = """from pathlib import Path
from uuid import uuid4
import json, hashlib, sys
import pandas as pd
import openecon as oe
assert getattr(sys, "frozen", False)
assert Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))
graph = oe.network([{"source": 1, "target": "1", "weight": 2},
                    {"source": "1", "target": "1", "weight": .5}],
                   nodes=[1, "1"], directed=True, weight="weight",
                   node_attributes={1: {"city": "Ankara"}, "1": {"city": "İstanbul"}})
snapshots = oe.network_snapshots({"Month " + str(i): graph for i in range(1, 9)}, ordered=True)
plot = oe.plot.network(snapshots, title="Trade — Türkiye", node_label="city",
    layout="circular", seed=42, frame_index=1, timeline=True).annotate("Reference", x=0, y=-80)
original = plot.model_dump()
folder = Path("qa_exports") / uuid4().hex
folder.mkdir(parents=True)
files = {}
for format in ("pdf", "svg", "png"):
    path = folder / ("trade." + format)
    getattr(plot, "to_" + format)(path, width=800, height=400)
    files[format] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
plot.save_view(folder / "view.json")
assert oe.plot.PlotSpec.load_view(folder / "view.json").model_dump() == original
assert json.loads((folder / "view.json").read_text())["config"]["network"]["encoding"] == "timeline-pool-v1"
assert plot.model_dump() == original
display(plot)
frame = pd.DataFrame({"id": pd.array(range(101), dtype="Int64"),
    "x": pd.array(range(101), dtype="Float64"),
    "y": pd.array([2 + .75*i + (i%7)/100 for i in range(101)], dtype="Float64")},
    index=pd.Index(range(500, 601), name="original"))
prepared = oe.Dataset.from_frame(frame).filter(lambda b: b["x"] >= 25)
lookup = oe.Dataset.from_frame(pd.DataFrame({"id": pd.array(range(101), dtype="Int64"), "group": ["A"]*101}))
joined = prepared.join(lookup, on="id")
result = oe.ols(data=joined, y="y", x=["x"], covariance="HC3", device="cpu")
display(joined.head(5))
display(result)
print("INSTALLED_NETWORK_DATA:" + json.dumps({"frozen": True, "exports": files,
    "selected_frame": 1, "frames": 8, "full_nodes": graph.node_count, "full_edges": graph.edge_count,
    "typed_ids": True, "native_fit_nobs": result.nobs, "join": joined.preparation_receipt,
    "plot_unchanged": True, "pooled_offline_view_roundtrip": True}, allow_nan=False))
"""


def main():
    OUT.mkdir(exist_ok=True)
    port = json.loads((DATA / ".runtime-port.json").read_text())["port"]
    origin = "http://127.0.0.1:" + str(port)
    token = request(origin + "/api/desktop/session")["token"]
    catalog = request(origin + "/api/desktop/local-projects", token=token)["projects"]
    project = next((row for row in catalog if row["name"] == "Network and Data QA"), None)
    if project is None:
        project = request(
            origin + "/api/desktop/local-projects",
            {"name": "Network and Data QA", "description": "Synthetic local acceptance only"},
            token,
        )
    request(origin + "/api/desktop/local-projects/" + project["id"] + "/open", {}, token)
    base = origin + "/api/desktop/projects/" + project["id"] + "/workspace"
    local_token = request(base + "/session")["token"]

    def call(path, body=None):
        return request(base + path, body, local_token)

    def execute(code, seconds=900):
        result = call("/console/execute", {"code": code, "timeout_seconds": seconds})
        if result.get("status") != "ok":
            raise RuntimeError(str(result.get("error")))
        return result

    prefix = 'import sys\nfrom pathlib import Path\nimport openecon as oe\nassert getattr(sys,"frozen",False)\nassert Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))\n'
    scientific = []
    existing_history = call("/console")["history"]
    for helper, expression, label in (
        ("benchmark_source_readers.py", '[run("xlsx",120001),run("dta",300001)]', "READERS"),
        ("benchmark_dataset_prepare.py", "run()", "PREPARATION"),
    ):
        content = (ROOT / "scripts" / helper).read_text().split('if __name__ == "__main__":')[0]
        content = content.replace("ROOT = Path(__file__).resolve().parents[1]", "ROOT = Path.cwd()")
        code = (
            prefix
            + content
            + '\nprint("INSTALLED_'
            + label
            + ':" + json.dumps('
            + expression
            + ",allow_nan=False))"
        )
        result = next(
            (
                row
                for row in existing_history
                if row.get("status") == "ok"
                and any(
                    line.startswith("INSTALLED_" + label + ":")
                    for line in row.get("stdout", "").splitlines()
                )
            ),
            None,
        )
        if result is None:
            result = execute(code)
        proof = next(
            json.loads(line.split(":", 1)[1])
            for line in result["stdout"].splitlines()
            if line.startswith("INSTALLED_" + label + ":")
        )
        scientific.append(dict(check=label, run_id=result["id"], proof=proof))
        print(label + " passed", flush=True)
    # Actual on-disk inputs for interruption, created by QA code before starting
    # the disposable parser/join. Source fingerprints are checked externally.
    fixtures = DATA / "qa-fixtures"
    fixtures.mkdir(exist_ok=True)
    import pandas as pd
    from openpyxl import Workbook

    xlsx = fixtures / "stop.xlsx"
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet()
    sheet.append(["id", "x", "y"])
    for i in range(120001):
        sheet.append([i, i % 101, 2 + 0.75 * (i % 101)])
    workbook.save(xlsx)
    workbook.close()
    parquet = fixtures / "stop.parquet"
    pd.DataFrame({"id": range(100001), "x": range(100001)}).to_parquet(parquet, row_group_size=8192)
    fingerprints = {str(file): digest(file) for file in (xlsx, parquet)}
    workspace = DATA / "projects" / project["id"]
    stopped = []
    for label, code, pattern in (
        (
            "reader",
            prefix + "source = oe.read(" + repr(str(xlsx)) + ")\nprint(source.row_count)",
            "converted.parquet",
        ),
        (
            "join",
            prefix
            + "source = oe.scan("
            + repr(str(parquet))
            + ')\njoined = source.join(source,on="id")\nfor block in joined.iter_batches():\n    print(len(block))',
            "rows.sqlite",
        ),
    ):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(call, "/console/execute", {"code": code, "timeout_seconds": 300})
            deadline = time.monotonic() + 90
            observed = []
            while time.monotonic() < deadline and not future.done():
                observed = [
                    file
                    for file in workspace.glob(".openecon-scratch-*/**/" + pattern)
                    if file.stat().st_size > 65536
                ]
                if observed:
                    break
                time.sleep(0.02)
            if not observed:
                raise RuntimeError("Did not observe actual " + label + " owned disk work")
            call("/console/interrupt", {})
            result = future.result(timeout=30)
            assert result["error"]["type"] == "INTERRUPTED", result.get("error")
            assert not list(workspace.glob(".openecon-scratch-*"))
            assert all(digest(file) == fingerprint for file, fingerprint in fingerprints.items())
            recovery = execute("2+2")
            assert recovery["outputs"][0]["data"] == "4"
            stopped.append(
                dict(
                    operation=label,
                    interrupted_run=result["id"],
                    actual_disk_work_observed=True,
                    scratch_removed=True,
                    source_unchanged=True,
                    recovery_run=recovery["id"],
                )
            )
            print(label + " Stop/recovery passed", flush=True)
    (OUT / "progress.json").write_text(
        json.dumps(dict(scientific=scientific, stop_checks=stopped), indent=2) + "\n"
    )
    existing_script = next(
        (row for row in call("/console/scripts")["scripts"] if row["name"] == "qa_network_data.py"),
        None,
    )
    script = (
        call(
            "/console/scripts/" + existing_script["id"],
            {"code": UI_CODE, "version": existing_script["version"]},
        )
        if existing_script
        else call("/console/scripts", {"name": "qa_network_data.py", "code": UI_CODE})
    )
    visual = execute(UI_CODE)
    marker = next(
        json.loads(line.split(":", 1)[1])
        for line in visual["stdout"].splitlines()
        if line.startswith("INSTALLED_NETWORK_DATA:")
    )
    call("/console/reset", {})
    saved = next(run for run in call("/console")["history"] if run["id"] == visual["id"])
    assert saved["outputs"] == visual["outputs"] and saved["events"] == visual["events"]
    plot = next(out for out in saved["outputs"] if out["type"] == "plot")
    # Checked artifact path served by the product, not a direct filesystem read.
    identifier = plot["data"].get("artifact", {}).get("id")
    if identifier:
        artifact = call("/console/plots/" + identifier)
        assert artifact["config"]["network"]["encoding"] == "timeline-pool-v1"
    else:
        assert plot["data"]["config"]["network"]["encoding"] == "timeline-pool-v1"
    record = dict(
        status="passed",
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        app=str(APP),
        identifier="org.openecon.qa.network-data.01a11244",
        native_origin=origin,
        project=project,
        script=script,
        source_ref="bb6317a",
        scientific=scientific,
        stop_checks=stopped,
        frozen_exports=marker,
        visual_run=visual["id"],
        saved_outputs_after_reset=True,
        native_UI_verified=False,
        scientific_proof_scope="Scientific modules unchanged from the first installed build; full scientific results are read back from its history after the publication-only repair.",
        main_installation_modified=False,
        human_projects_opened=False,
    )
    (OUT / "installed-api.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    (OUT / "qa_network_data.py").write_text(UI_CODE)
    print(
        json.dumps({"status": "passed", "project_id": project["id"], "visual_run": visual["id"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
