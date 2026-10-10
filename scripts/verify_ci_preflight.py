"""Reject an obsolete CI merge identity before provisioning SDK runners.

This reuses the final gate's exact source/parent/tree check. Passing this check
does not replace any SDK, publication, package or aggregate validation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def verify(root, environment):
    spec = importlib.util.spec_from_file_location(
        "preflight_merge_validator", Path(__file__).with_name("verify_parallel_merge_gate.py")
    )
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    expected = {
        "source_commit": environment.get("GITHUB_SHA", ""),
        "pull_request_head": environment.get("CI_PR_HEAD") or None,
        "pull_request_base": environment.get("CI_PR_BASE") or None,
        "event": environment.get("GITHUB_EVENT_NAME", ""),
        "run_id": environment.get("GITHUB_RUN_ID", ""),
        "run_attempt": environment.get("GITHUB_RUN_ATTEMPT", ""),
    }
    return {
        "status": "passed",
        "scope": "Source identity only; the daily full gate remains required.",
        "expected": expected,
        "verified_git": validator.expected_binding(expected, root),
    }


def annotation(value):
    return str(value).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify(ROOT, os.environ)
    except (ValueError, subprocess.CalledProcessError) as error:
        message = (
            f"{error}. Integrate the current base into the PR and push a new head "
            "before running the daily full gate; rerunning an obsolete event does not refresh its pins."
        )
        report = {"status": "failed", "scope": "Source identity only", "error": message}
        print("::error title=CI source preflight failed::" + annotation(message))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
