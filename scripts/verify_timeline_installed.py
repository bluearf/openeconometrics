"""Real checked pooled artifact, typed identities, offline view/export and missing-file recovery."""

import json
from urllib.error import HTTPError
from verify_network_data_installed import APP, DATA, OUT, request

CODE = """import sys, json
from pathlib import Path
import openecon as oe
assert getattr(sys, "frozen", False)
assert Path(oe.__file__).is_relative_to(Path(sys._MEIPASS))
nodes=[0,1,"1",*range(3,600)]
edges=[{"source":nodes[i],"target":nodes[(i+d)%600],"weight":1.}
       for i in range(600) for d in range(1,11)]
graph=oe.network(edges,nodes=nodes,directed=True,weight="weight",
    node_attributes={node:{"sector":str(i%8),"description":"Stable immutable attributes — Türkiye. "*30}
                     for i,node in enumerate(nodes)})
snapshots=oe.network_snapshots({"Month "+str(i):graph for i in range(1,9)},ordered=True)
plot=oe.plot.network(snapshots,title="Pooled timeline — 8 frames",max_nodes=600,max_edges=6000,
                     layout="circular",seed=42,frame_index=2,timeline=True)
expanded=plot.model_dump()
wire=plot.transport_dump()
assert wire["config"]["network"]["encoding"]=="timeline-pool-v1"
folder=Path("qa_timeline")
folder.mkdir(exist_ok=True)
plot.save_view(folder/"view.json")
assert oe.plot.PlotSpec.load_view(folder/"view.json").model_dump()==expanded
plot.save_html(folder/"offline.html")
plot.to_svg(folder/"selected.svg",width=800,height=400,overwrite=True)
display(plot)
print("INSTALLED_TIMELINE:"+json.dumps({"nodes":graph.node_count,"edges":graph.edge_count,
    "frames":8,"selected_frame":2,"frame_order":[x["label"] for x in expanded["config"]["network"]["frames"]],
    "expanded_bytes":len(json.dumps(expanded).encode()),"pooled_bytes":len(json.dumps(wire).encode()),
    "typed_ids_distinct":graph.node_count==600 and 1 in graph.nodes()["node"].tolist() and "1" in graph.nodes()["node"].tolist(),
    "offline_view_roundtrip":True,"offline_html":str((folder/"offline.html").resolve()),
    "offline_svg":str((folder/"selected.svg").resolve())},allow_nan=False))
"""


def main():
    origin = "http://127.0.0.1:" + str(
        json.loads((DATA / ".runtime-port.json").read_text())["port"]
    )
    token = request(origin + "/api/desktop/session")["token"]
    project = next(
        row
        for row in request(origin + "/api/desktop/local-projects", token=token)["projects"]
        if row["name"] == "Network and Data QA"
    )
    base = origin + "/api/desktop/projects/" + project["id"] + "/workspace"
    token = request(base + "/session")["token"]

    def call(path, body=None):
        return request(base + path, body, token)

    script = call("/console/scripts", {"name": "qa_timeline_pool.py", "code": CODE})
    result = call("/console/execute", {"code": CODE, "timeout_seconds": 180})
    assert result["status"] == "ok", result.get("error")
    proof = next(
        json.loads(line.split(":", 1)[1])
        for line in result["stdout"].splitlines()
        if line.startswith("INSTALLED_TIMELINE:")
    )
    (output,) = result["outputs"]
    assert output["type"] == "plot"
    reference = output["data"]["artifact"]
    artifact = call("/console/plots/" + reference["id"])
    assert artifact["config"]["network"]["encoding"] == "timeline-pool-v1"
    file = DATA / "projects" / project["id"] / "plots" / (reference["id"] + ".json")
    # Discover only this exact, product-authorized artifact in our own project.
    if not file.exists():
        file = next((DATA / "projects" / project["id"]).glob("**/" + reference["id"] + ".json"))
    held = file.with_suffix(".qa-held")
    file.rename(held)
    try:
        try:
            call("/console/plots/" + reference["id"])
            raise AssertionError("Missing artifact was served")
        except HTTPError as error:
            assert error.code in {404, 409, 422}
    finally:
        held.rename(file)
    assert call("/console/plots/" + reference["id"]) == artifact
    call("/console/reset", {})
    saved = next(row for row in call("/console")["history"] if row["id"] == result["id"])
    assert saved["outputs"] == result["outputs"] and saved["events"] == result["events"]
    record = dict(
        status="passed",
        app=str(APP),
        project_id=project["id"],
        run_id=result["id"],
        script=script,
        proof=proof,
        checked_artifact=reference,
        missing_artifact_refused=True,
        restored_artifact_identical=True,
        outputs_after_reset_identical=True,
        native_UI_verified=False,
    )
    (OUT / "installed-timeline.json").write_text(json.dumps(record, indent=2) + "\n")
    (OUT / "qa_timeline_pool.py").write_text(CODE)
    print(
        json.dumps(
            {"status": "passed", "checked_artifact_bytes": reference["bytes"], "proof": proof}
        )
    )


if __name__ == "__main__":
    main()
