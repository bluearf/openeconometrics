"""Compare the same expanded/pooled timeline in disposable real Chromium processes."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.request import ProxyHandler, build_opener

from openecon_charts import network
from openecon_charts.network_export import _browser, _CDP

ROOT = Path(__file__).resolve().parents[1]


def fixture(nodes=600, edges=6000, frames=8):
    class Graph:
        def to_plot_data(self, **kwargs):
            records = [
                {
                    "id": i,
                    "identity": {"type": "integer" if i % 2 else "string", "value": str(i)},
                    "label": f"Firm {i} — Türkiye",
                    "degree": 20,
                    "group": i % 6,
                    "attrs": {
                        "sector": f"Sector {i % 6}",
                        "region": "Türkiye",
                        "description": "Stable enterprise attributes",
                    },
                    "x": (i % 30) * 25,
                    "y": (i // 30) * 25,
                }
                for i in range(nodes)
            ]
            links = [
                {
                    "source": i % nodes,
                    "target": (i * 37 + 1) % nodes,
                    "weight": i % 7 + 1,
                    "attrs": {"type": "production", "description": "Stable link metadata"},
                }
                for i in range(edges)
            ]
            base = dict(
                nodes=records,
                edges=links,
                directed=True,
                node_count=nodes,
                edge_count=edges,
                shown_node_count=nodes,
                shown_edge_count=edges,
                sampled=False,
                selection="All nodes and edges",
            )
            snapshots = []
            for frame in range(frames):
                changed = [
                    dict(edge, weight=edge["weight"] + frame / 10) if i % 20 == frame else edge
                    for i, edge in enumerate(links)
                ]
                snapshots.append(
                    {"label": f"Month {frame + 1}", "network": dict(base, edges=changed)}
                )
            return dict(base, frames=snapshots)

    return network(
        Graph(),
        layout="fixed",
        title="Production timeline",
        timeline=True,
        seed=42,
        max_nodes=nodes,
        max_edges=edges,
    )


def observe(document, directory):
    profile = directory / "profile"
    page = directory / "timeline.html"
    page.write_text(document)
    process = subprocess.Popen(
        [
            _browser(None),
            "--headless",
            "--no-first-run",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-sync",
            "--remote-debugging-port=0",
            "--enable-precise-memory-info",
            f"--user-data-dir={profile}",
            "about:blank",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cdp = None
    try:
        deadline = time.monotonic() + 60
        descriptor = profile / "DevToolsActivePort"
        while not descriptor.exists():
            if time.monotonic() > deadline or process.poll() is not None:
                raise RuntimeError("Disposable browser did not start")
            time.sleep(0.03)
        port = int(descriptor.read_text().splitlines()[0])
        with build_opener(ProxyHandler({})).open(f"http://127.0.0.1:{port}/json/list") as response:
            pages = json.load(response)
        cdp = _CDP(next(p["webSocketDebuggerUrl"] for p in pages if p["type"] == "page"), deadline)
        cdp.call(
            "Emulation.setDeviceMetricsOverride",
            width=960,
            height=900,
            deviceScaleFactor=1,
            mobile=False,
        )
        cdp.call("Performance.enable")
        cdp.call("Page.navigate", url=page.as_uri())
        result = cdp.call(
            "Runtime.evaluate",
            awaitPromise=True,
            returnByValue=True,
            expression="""(async()=>{
          const end=performance.now()+45000;
          while (!window.OpenEconNetworkCharts?.get(document.getElementById('chart'))) {
            if (performance.now()>end) throw new Error('Mount timeout');
            await new Promise(r=>setTimeout(r,16));
          }
          await document.fonts.ready;
          const chart=OpenEconNetworkCharts.get(document.getElementById('chart'));
          const ready=performance.now();
          const before=performance.memory.usedJSHeapSize;
          const switches=[];
          for (const index of [1,7,2,6,0,7,3,1]) {
            const start=performance.now(); chart.setFrame(index);
            await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
            if(chart.frameIndex!==index || chart.config.nodes.length!==600 || chart.config.edges.length!==6000)
              throw new Error('Wrong selected frame/counts');
            switches.push(performance.now()-start);
          }
          const after=performance.memory.usedJSHeapSize;
          const first=chart.config.nodes[0].identity;
          if(first.type!=='string'||first.value!=='0') throw new Error('Typed identity changed');
          const saved=chart.toSpec();
          if(saved.config.options.frame_index!==1 || saved.config.network.frames.length!==8)
            throw new Error('Frame export changed');
          OpenEconNetworkCharts.unmount(document.getElementById('chart'));
          if(OpenEconNetworkCharts.get(document.getElementById('chart'))) throw new Error('Unmount retained chart');
          return {ready_ms:ready, heap_ready_bytes:before, heap_after_switches_bytes:after,
            switch_ms:switches, selected_frame:1, frame_order:saved.config.network.frames.map(f=>f.label),
            counts_and_typed_identity:true, unmounted:true};
        })()""",
        )
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"]["value"]
    finally:
        if cdp:
            cdp.sock.close()
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def main():
    plot = fixture()
    compact = plot.transport_dump()
    public = plot.model_dump()
    html = plot.to_html()
    wire = json.dumps(compact, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    legacy = json.dumps(public, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    assert wire in html
    expanded_html = html.replace(wire, legacy, 1)
    receipts = []
    with tempfile.TemporaryDirectory(prefix="openecon-timeline-benchmark-") as tmp:
        for repeat in range(3):
            for name, document in [("expanded", expanded_html), ("pooled", html)]:
                directory = Path(tmp) / f"{repeat}-{name}"
                directory.mkdir()
                receipts.append(dict(repeat=repeat, encoding=name, **observe(document, directory)))
    record = dict(
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        nodes=600,
        edges_per_frame=6000,
        frames=8,
        expanded_payload_bytes=len(legacy.encode()),
        pooled_payload_bytes=len(wire.encode()),
        payload_reduction=1 - len(wire) / len(legacy),
        same_fixture=True,
        measurements=receipts,
        limits=dict(
            aggregate_nodes=100000,
            aggregate_edges=1000000,
            frames=60,
            payload_bytes=128 * 1024 * 1024,
            network_records_cache="one per unique pooled record within unchanged display caps",
        ),
        limitations="Local shared-machine browser measurements, no universal timing guarantee. Synchronous pooled frames avoid new fetch/cache invalidation races. All frames remain local/offline; no lazy external frame files.",
        source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "packages/openecon-charts/src/openecon_charts/timeline.py",
                "packages/openecon-charts/src/openecon_charts/assets/network-renderer.js",
                "src/openecon/plot_artifacts.py",
                "scripts/benchmark_network_timeline_transport.py",
            )
        },
    )
    path = ROOT / "docs/evidence/market-72-timeline-transport-2026-10-07.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: record[k]
                for k in ("expanded_payload_bytes", "pooled_payload_bytes", "payload_reduction")
            }
        )
    )


if __name__ == "__main__":
    main()
