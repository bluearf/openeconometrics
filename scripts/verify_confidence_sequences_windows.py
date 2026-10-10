"""Actual Windows source science and isolated installed-wheel acceptance.

This does not claim frozen/native Windows UI acceptance. Full files and original
JUnit/logs are retained, bound to the tested commit and PR head/base.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def run(command, log, *, cwd=ROOT):
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=600, check=False)
    if result.returncode:
        raise RuntimeError(f"Acceptance failed; read {log.name}")
    return round(time.monotonic()-started, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("Windows acceptance must actually run on Windows.")
    directory = args.directory.resolve()
    if directory.exists():
        raise RuntimeError("Use an entirely new acceptance directory.")
    directory.mkdir(parents=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert source == os.environ["GITHUB_SHA"]
    receipt = {"status": "running", "source_commit": source,
        "pr_head": os.environ.get("CS_PR_HEAD"), "pr_base": os.environ.get("CS_PR_BASE"),
        "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "platform": platform.platform(), "python": sys.version, "scope": "Windows source science and installed wheels",
        "frozen_runtime": False, "native_window": False, "public_release_delivered": False}
    try:
        xml = directory/"source-tests.xml"
        run([sys.executable, "-m", "pytest", "tests/test_confidence_sequences.py", "-q",
             "--junitxml="+str(xml)], directory/"source-tests.log")
        suites = list(ET.parse(xml).getroot().iter("testsuite"))
        counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
        assert counts["tests"] >= 66 and all(counts[k] == 0 for k in ("failures", "errors", "skipped"))
        receipt["science"] = counts
        example = ROOT/"docs/examples/confidence_sequences_eight.py"
        for phase in ("source", "wheel"):
            work = directory/phase
            work.mkdir()
            if phase == "wheel":
                wheels = directory/"wheels"
                run(["uv", "build", "--wheel", "--out-dir", str(wheels)], directory/"build-sdk.log")
                run(["uv", "build", "packages/openecon-charts", "--wheel", "--out-dir", str(wheels)], directory/"build-charts.log")
                wheel_paths = sorted(wheels.glob("*.whl"))
                assert len(wheel_paths) == 2
                run(["uv", "pip", "install", "--python", sys.executable, "--reinstall", "--no-deps",
                     *map(str, wheel_paths)], directory/"install-wheels.log")
                guard = "from pathlib import Path; import openecon,sys; assert Path(openecon.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); assert not any(Path(p or '.').resolve()==Path("+repr(str(ROOT/"src"))+").resolve() for p in sys.path)"
                run([sys.executable, "-c", guard], directory/"wheel-isolation.log", cwd=work)
            run([sys.executable, str(example)], directory/(phase+"-example.log"), cwd=work)
            proof = json.loads((work/"confidence_sequence_acceptance.json").read_text())
            assert proof["stages"] == proof["full_states"] == 8 and len(proof["files"]) == 16
            for name, item in proof["files"].items():
                raw = (work/"confidence_sequence_results"/name).read_bytes()
                assert len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"]
            receipt[phase] = proof
        assert receipt["source"]["files"] == receipt["wheel"]["files"]
        receipt.update(status="passed", full_source_installed_wheel_bytes_equal=True,
                       example_sha256=hashlib.sha256(example.read_bytes()).hexdigest())
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (directory/"acceptance.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("status", "source_commit", "science", "full_source_installed_wheel_bytes_equal")}))


if __name__ == "__main__":
    main()
