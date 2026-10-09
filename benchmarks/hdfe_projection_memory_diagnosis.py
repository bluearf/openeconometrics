"""Phase-instrumented, source-pinned repeat of the frozen HDFE CSV RSS case.

All original receipts remain untouched; this diagnostic is an additional ABBA
replication. The companion repeat-protocol was saved before these outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

from hdfe_projection_benchmark import run_case, save, sha


def peak():
    return int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--directory", required=True, type=Path)
    p.add_argument("--baseline-root", required=True, type=Path)
    p.add_argument("--optimized-root", required=True, type=Path)
    p.add_argument("--worker", type=int)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.worker is None:
        args.directory.mkdir(parents=True, exist_ok=False)
        outcomes = []
        for i in range(len(protocol["order"])):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--manifest",
                str(args.manifest.resolve()),
                "--protocol",
                str(args.protocol.resolve()),
                "--directory",
                str(args.directory.resolve()),
                "--baseline-root",
                str(args.baseline_root.resolve()),
                "--optimized-root",
                str(args.optimized_root.resolve()),
                "--worker",
                str(i),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                env={**os.environ, "PYTHONNOUSERSITE": "1"},
            )
            (args.directory / f"worker-{i}.log").write_text(result.stdout + result.stderr)
            outcomes.append(
                {"worker": i, "source": protocol["order"][i], "exit_code": result.returncode}
            )
            print(json.dumps(outcomes[-1]), flush=True)
        save(args.directory / "workers.json", outcomes)
        return int(any(x["exit_code"] for x in outcomes))
    kind = protocol["order"][args.worker]
    root = args.baseline_root if kind == "baseline" else args.optimized_root
    sys.path[:0] = [str(root / "src"), str(root / "packages/openecon-charts/src")]
    phases = [{"phase": "pre_openecon_import", "peak_process_rss_bytes": peak()}]
    import openecon as oe

    phases.append({"phase": "post_openecon_import", "peak_process_rss_bytes": peak()})
    import torch

    phases.append({"phase": "post_torch_import", "peak_process_rss_bytes": peak()})
    assert Path(oe.__file__).resolve().is_relative_to(root.resolve()), "wrong source SDK loaded"
    original = oe.fit
    phases.append({"phase": "post_resolve_fit", "peak_process_rss_bytes": peak()})

    def fit(*a, **kw):
        phases.append({"phase": "pre_fit", "peak_process_rss_bytes": peak()})
        value = original(*a, **kw)
        phases.append({"phase": "post_fit", "peak_process_rss_bytes": peak()})
        return value

    oe.fit = fit
    case = next(
        x for x in json.loads(args.manifest.read_text())["cases"] if x["id"] == protocol["case"]
    )
    directory = args.directory / f"worker-{args.worker}-{kind}" / case["id"]
    result = run_case(case, directory)
    result.update(
        source_pin=protocol["source_" + kind],
        source_kind=kind,
        phase_high_water_RSS=phases,
        torch_threads=torch.get_num_threads(),
        imported_modules={
            prefix: sorted(
                name for name in sys.modules if name == prefix or name.startswith(prefix + ".")
            )
            for prefix in ("torch", "numpy", "pandas", "pyarrow", "scipy", "statsmodels")
        },
        feature_sha256={
            name: sha(root / name)
            for name in (
                "src/openecon/econometrics/streaming_hdfe.py",
                "src/openecon/econometrics/streaming_ppml.py",
            )
        },
        repeat_protocol_sha256=sha(args.protocol),
    )
    save(directory / "receipt.json", result)
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
