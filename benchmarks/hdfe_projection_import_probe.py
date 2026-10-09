"""Fresh-process CPU import timings, separate from physical fit measurements."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import resource
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--source-pin", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    sys.path[:0] = [
        str(args.source_root / "src"),
        str(args.source_root / "packages/openecon-charts/src"),
    ]
    stages = []

    def stage(name, function):
        before = time.perf_counter()
        value = function()
        stages.append(
            {
                "stage": name,
                "seconds": time.perf_counter() - before,
                "parent_peak_RSS_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024)
                ),
            }
        )
        return value

    oe = stage("openecon_import", lambda: importlib.import_module("openecon"))
    torch = stage("torch_import", lambda: importlib.import_module("torch"))
    stage("ModelSpec_import", lambda: importlib.import_module("openecon.models"))
    stage("torch_two_threads", lambda: torch.set_num_threads(2))
    stage("fit_lazy_resolution", lambda: oe.fit)
    assert Path(oe.__file__).resolve().is_relative_to(args.source_root.resolve()), (
        "wrong SDK loaded"
    )
    result = {
        "schema": 1,
        "source_pin": args.source_pin,
        "sdk_path": oe.__file__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "stages": stages,
        "scope": "Fresh process with retained OS caches on shared host; import only, no model fit. Parent high-water RSS, not whole-machine/parser memory. Stages are sequential, not nested.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
