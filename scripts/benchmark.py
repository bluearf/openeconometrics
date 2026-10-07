"""Display archived OpenEcon 0.2 measurements; this is not a current benchmark.

The 0.3 workbench has one required PyTorch core and a changed public API. The
former dual-engine benchmark cannot validly measure that implementation. Its
source and results are preserved under benchmarks/ for historical inspection.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-report", action="store_true", help="Print the historical 0.2 JSON measurement; run no computation.")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if options.show_report:
        print((root / "benchmarks/results/2026-09-30-native-cpu.json").read_text(encoding="utf-8"))
    else:
        print("Historical OpenEcon 0.2 benchmark only. No current performance measurement was run.\n"
              "Read docs/performance.md or use --show-report to inspect the archived JSON.\n"
              "The single-core workbench needs a new validated benchmark before speed claims.")


if __name__ == "__main__":
    main()
