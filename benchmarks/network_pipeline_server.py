"""Loopback-only browser benchmark server; serves only owned artifacts and chart assets.

python benchmarks/network_pipeline_server.py --input /tmp/owned-network-run
Open the printed URL in headed Chrome. The page runs automatically and saves a receipt.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
from urllib.parse import unquote, urlparse
import uuid

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "packages/openecon-charts/src/openecon_charts/assets"
PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>Network pipeline benchmark</title>
<link rel="stylesheet" href="/assets/charts.css"><link rel="stylesheet" href="/assets/network.css">
<style>body{margin:24px;font-family:Barlow,sans-serif;color:#14263d}#chart{width:1100px;max-width:100%}
textarea{width:100%;height:80px}h1{font-size:24px}</style></head><body>
<h1>Network pipeline benchmark</h1><p id="state">Loading recorded artifacts...</p><main id="chart"></main>
<label>Saved measurement receipt<textarea id="result" readonly></textarea></label>
<script src="/assets/d3.min.js"></script><script src="/assets/network-webgl.js"></script>
<script src="/assets/network-font.js"></script><script src="/assets/network-pdf.js"></script>
<script src="/assets/network-renderer.js"></script><script src="/benchmark.js"></script></body></html>"""


def chrome_rss():
    """Observed summed RSS, not an exact OS lifetime peak or private-page memory."""
    rows = subprocess.check_output(["ps", "-axo", "pid=,rss=,comm="], text=True)
    processes = []
    for line in rows.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and "Google Chrome" in parts[2]:
            processes.append(dict(pid=int(parts[0]), rss_bytes=int(parts[1]) * 1024))
    return dict(processes=processes, available=bool(processes),
                summed_rss_bytes=sum(item["rss_bytes"] for item in processes) if processes else None)


class Sampler:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.samples = []
        self.finished = threading.Event()
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while not self.finished.wait(0.2):
            with self.lock:
                if self.active:
                    self.samples.append(chrome_rss())

    def start(self):
        with self.lock:
            if self.active:
                raise ValueError("A measurement is already active")
            self.samples = [chrome_rss()]
            self.active = True
            return self.samples[0]

    def stop(self):
        with self.lock:
            self.samples.append(chrome_rss())
            self.active = False
            observed = [item["summed_rss_bytes"] for item in self.samples if item["available"]]
            return dict(sample_interval_ms=200, samples=len(self.samples),
                observed_peak_summed_rss_bytes=max(observed) if observed else None,
                final=self.samples[-1], scope="All Google Chrome OS processes on this host, including other tabs and shared GPU/browser processes. Summed RSS may double count shared pages; 200ms sampling can miss short peaks. Not private tab memory, exact lifetime peak RSS, or a guaranteed memory bound.")


def safe_file(root, suffix):
    path = (root / unquote(suffix)).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("File is outside the allowed root or missing")
    return path


def handler(input_directory, sampler):
    source_paths = [ROOT / "benchmarks/network_pipeline_browser.js", Path(__file__),
                    *sorted(ASSETS.glob("*.js")), *sorted(ASSETS.glob("*.css"))]
    sources = {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest() for file in source_paths}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, value, status=200, content_type="application/json"):
            body = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/":
                    return self.respond(PAGE, content_type="text/html; charset=utf-8")
                if path == "/manifest.json":
                    manifest = json.loads((input_directory / "manifest.json").read_text())
                    manifest["browser_source_sha256"] = sources
                    return self.respond(manifest)
                elif path == "/benchmark.js":
                    file = ROOT / "benchmarks/network_pipeline_browser.js"
                elif path.startswith("/assets/"):
                    file = safe_file(ASSETS, path.removeprefix("/assets/"))
                elif path.startswith("/data/"):
                    file = safe_file(input_directory, path.removeprefix("/data/"))
                else:
                    return self.respond({"error": "Unknown route"}, 404)
                suffix = file.suffix
                kind = {".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2"}.get(suffix, "application/json")
                self.respond(file.read_bytes(), content_type=kind)
            except (ValueError, OSError) as error:
                self.respond({"error": str(error)}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                if path == "/memory/start":
                    return self.respond(sampler.start())
                if path == "/memory/stop":
                    return self.respond(sampler.stop())
                if path != "/results":
                    return self.respond({"error": "Unknown route"}, 404)
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 2 * 1024 * 1024:
                    raise ValueError("Result must be between 1 byte and 2 MiB")
                body = self.rfile.read(length)
                report = json.loads(body)
                if type(report) is not dict or report.get("schema") != 1 or not isinstance(report.get("reports"), list):
                    raise ValueError("Invalid report")
                if sources != {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest() for file in source_paths}:
                    raise ValueError("Browser sources changed during measurement")
                name = f"browser-{uuid.uuid4().hex}.json"
                with (input_directory / name).open("xb") as stream:
                    stream.write(body)
                self.respond(dict(file=name, bytes=len(body), sha256=hashlib.sha256(body).hexdigest()))
            except (ValueError, OSError) as error:
                self.respond({"error": str(error)}, 400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    directory = args.input.resolve()
    if not (directory / "manifest.json").is_file():
        parser.error("Run network_pipeline.py first")
    sampler = Sampler()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(directory, sampler))
    print(f"http://127.0.0.1:{server.server_port}/?repeats=3", flush=True)
    try:
        server.serve_forever()
    finally:
        sampler.finished.set()
        server.server_close()


if __name__ == "__main__":
    main()
