"""Bounded single original-author Octave reference; stdlib orchestration only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/reference/bvar_chan2020_deterministic_fixture_v1.json"
ARCHIVE = ROOT / "tests/reference/bvar_chan2020_original_archive_v1.json"
DRIVER = ROOT / "scripts/bvar_chan2020_original_posterior_octave.m"
WORKER = ROOT / "scripts/compare_bvar_chan2020_original_posterior.py"
TOTAL_SECONDS = 600
ARCHIVE_SECONDS = 60
NATIVE_SECONDS = 120
OWN_SECONDS = 300
MAX_JSON = 32 * 1024**2
MAX_ZIP = 32 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write(path, value):
    data = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    require(len(data) <= MAX_JSON, "Complete JSON exceeds fixed32MiB cap")
    with path.open("xb") as stream:
        stream.write(data)


def read(path):
    require(path.stat().st_size <= MAX_JSON, "JSON exceeds fixed32MiB cap")
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def authenticate_archive(path, expected, source):
    raw = path.read_bytes()
    require(len(raw) == expected["archive"]["bytes"] <= MAX_ZIP, "Original ZIP length differs")
    require(hashlib.sha256(raw).hexdigest() == expected["archive"]["sha256"], "Original ZIP SHA differs")
    require(hashlib.md5(raw).hexdigest() == expected["archive"]["md5"], "Original official provenance MD5 differs")
    required = {row["path"]: row for row in expected["all_members"]}
    require(len(required) == len(expected["all_members"]) == 39, "Full original39 inventory differs")
    source.mkdir()
    checked = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        require(len(infos) == 39 and len({item.filename for item in infos}) == 39,
                "Original ZIP duplicate/member geometry differs")
        require({item.filename for item in infos} == set(required), "Original full member set differs")
        for item in infos:
            name = PurePosixPath(item.filename)
            require(not item.is_dir() and not name.is_absolute() and ".." not in name.parts,
                    "Unsafe original archive member")
            require(item.file_size == required[item.filename]["bytes"], "Original member size differs")
            data = archive.read(item)  # Full decode also verifies the ZIP member CRC.
            require(hashlib.sha256(data).hexdigest() == required[item.filename]["sha256"],
                    "Original member SHA differs")
            blob = hashlib.sha1(("blob " + str(len(data)) + "\0").encode() + data).hexdigest()
            require(blob == required[item.filename]["Git_blob_sha1"], "Original member Git blob differs")
            checked.append(required[item.filename])
            # Only these unchanged author sources enter the numerical load path.
            # Original external data stay in the authenticated ZIP, unused.
            if item.filename in ("prior_NC.m", "forecast_BVAR_NCP.m"):
                target = source / item.filename
                with target.open("xb") as stream:
                    stream.write(data)
                target.chmod(0o444)
    return checked


def archive_stage(output):
    expected = read(ARCHIVE)
    url = expected["source_url"]
    require(url == "https://joshuachan.org/code/large_BVAR_code.zip", "Unregistered author origin")
    request = urllib.request.Request(url, headers={"User-Agent": "OpenEconometrics-original-author-reference"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(MAX_ZIP+1)
        headers, resolved = dict(response.headers), response.geturl()
    require(len(raw) <= MAX_ZIP, "Author ZIP exceeds fixed cap")
    path = output / "original-author-large_BVAR_code.zip"
    with path.open("xb") as stream:
        stream.write(raw)
    checked = authenticate_archive(path, expected, output / "unchanged-original-source")
    write(output / "complete-authentic-original-archive.json", {
        "status": "FULL39_ORIGINAL_CRC_SHA_GIT_BLOB_MATCH", "archive": pin(path),
        "origin": url, "resolved": resolved, "headers": headers, "all_members": checked,
        "only_two_original_sources_staged_readonly": True})


def source_inventory():
    paths = subprocess.check_output(["git", "ls-files", "-z", "src", "scripts",
        "tests/reference", ".github/workflows/bvar-original-author-octave.yml",
        "docs/reference-licenses", "uv.lock", "pyproject.toml"], cwd=ROOT).split(b"\0")
    return [pin(ROOT / os.fsdecode(path)) for path in paths if path]


def hosted_event(output, head):
    event_name = os.environ.get("GITHUB_EVENT_NAME")
    require(event_name in {"pull_request", "workflow_dispatch"}, "Explicit hosted event required")
    require(os.environ.get("GITHUB_SHA") == head, "Hosted source identity differs")
    path = Path(os.environ["GITHUB_EVENT_PATH"])
    event = read(path)
    raw = path.read_bytes()
    with (output / "complete-authentic-runner-event.json").open("xb") as stream:
        stream.write(raw)
    result = {"event_name": event_name, "complete_event_pin": pin(output / "complete-authentic-runner-event.json"),
              "repository": os.environ["GITHUB_REPOSITORY"], "run_id": os.environ["GITHUB_RUN_ID"],
              "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"], "workflow_sha": os.environ.get("GITHUB_WORKFLOW_SHA"),
              "actual_source": head,
              "actual_source_commit_raw": subprocess.check_output(["git", "cat-file", "-p", head], cwd=ROOT, text=True)}
    if event_name == "pull_request":
        pr = event["pull_request"]
        parents = subprocess.check_output(["git", "show", "-s", "--format=%P", head], cwd=ROOT, text=True).split()
        require(parents == [pr["base"]["sha"], pr["head"]["sha"]], "Exact event base/head synthetic merge differs")
        result.update(PR_number=pr["number"], event_head=pr["head"]["sha"], event_base=pr["base"]["sha"], actual_parents=parents)
    else:
        require(event.get("inputs", {}).get("authorization") == "RUN_FROZEN_THREE_CASE_OCTAVE_POSTERIOR_REFERENCE_ONCE",
                "Exact manual reviewed authorization required")
        result["qualification"] = "Manual exact-source reference; no PR head/base pair inferred"
    return result


def owned_stage(name, argv, output, deadline, cap):
    remaining = min(cap, deadline-time.monotonic())
    require(remaining > 0, "Original reference OS deadline exhausted")
    started = time.monotonic()
    terminal = {"stage": name, "argv": [str(value) for value in argv],
                "started_monotonic": started, "OS_seconds_cap": remaining}
    with (output / (name+".stdout")).open("xb") as stdout, (output / (name+".stderr")).open("xb") as stderr:
        process = subprocess.Popen(argv, cwd=ROOT, env={**os.environ,
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=stdout, stderr=stderr, start_new_session=False)
        terminal["PID"] = process.pid
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            terminal["timed_out"] = True
            process.kill()
            code = process.wait(timeout=15)
        terminal.update(OS_exit=code, child_exited_and_reaped=True,
                        finished_monotonic=time.monotonic(), seconds=time.monotonic()-started)
    write(output / (name+"-complete-OS-terminal.json"), terminal)
    require(code == 0 and not terminal.get("timed_out", False), name+" genuinely failed; no retry")
    return terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--octave", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--ROOT-authorized-original-author-comparison", action="store_true", dest="authorized")
    parser.add_argument("--archive-stage", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.archive_stage:
        archive_stage(output)
        return
    require(args.authorized, "ROOT review and once execution authorization required")
    require(args.python and args.octave and args.source_commit, "Explicit actual runtime and source required")
    require(os.getsid(0) == os.getpid() == os.getpgrp(),
            "Run only inside the external600second watchdog-owned process group")
    require(not output.exists(), "Fresh original-author output required; no resume or retry")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(head == args.source_commit, "Actual source head changed")
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT),
            "Committed source must be clean")
    require(shutil.disk_usage(output.parent).free >= 256*1024**2, "Original reference requires256MiB free")
    output.mkdir()
    started = time.monotonic()
    deadline = started+TOTAL_SECONDS
    before = source_inventory()
    registration = {"schema": "openecon.original-author-Octave.once-registration.v1",
        "source_commit": head, "tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT,text=True).strip(),
        "complete_actual_Git_inventory": subprocess.check_output(["git", "ls-files", "--stage", "-z"],cwd=ROOT).decode().split("\0"),
        "source_before": before, "fixture": pin(FIXTURE), "archive_manifest": pin(ARCHIVE),
        "driver": pin(DRIVER), "worker": pin(WORKER), "controller": pin(Path(__file__).resolve()),
        "python": str(args.python.absolute()), "resolved_python_binary": pin(args.python.resolve()), "octave": pin(args.octave.resolve()),
        "host": platform.platform(), "hosted_event": hosted_event(output, head),
        "owned_process_group": os.getpgrp(), "all_inner_stages_inherit_owned_process_group": True,
        "started_monotonic": started,
        "whole_OS_seconds": TOTAL_SECONDS, "stage_caps": {"archive":60,"native":120,"own":300},
        "fixed_tolerances": read(FIXTURE)["tolerance"], "fits":3, "native_sampling_iterations":0,
        "qualifications": "Octave original-author deterministic posterior only; no MATLAB/vendor/installed/full forecasting parity."}
    write(output / "prospective-registration-before-outcomes.json", registration)
    status, failure = "FAILED_OR_INCOMPLETE_ORIGINAL_AUTHOR_REFERENCE", None
    try:
        owned_stage("native-runtime-version", [str(args.octave.resolve()), "--version"], output, deadline, 15)
        version_text = (output / "native-runtime-version.stdout").read_text()
        require("GNU Octave, version 8.4.0" in version_text, "Prospective Octave8.4.0 runtime differs")
        owned_stage("native-dependency-inventory", ["dpkg-query", "-W"], output, deadline, 15)
        owned_stage("native-linked-libraries", ["ldd", str(args.octave.resolve())], output, deadline, 15)
        owned_stage("authenticated-original-source", [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                    "--output", str(output), "--archive-stage"], output, deadline, ARCHIVE_SECONDS)
        native = output / "native-full-posterior"
        native.mkdir()
        owned_stage("original-author-native-posterior", [str(args.octave.resolve()), "--no-gui", "--no-window-system",
            "--no-history", "--no-init-file", "--no-site-file", "--quiet", str(DRIVER),
            str(output / "unchanged-original-source"), str(FIXTURE), str(native)], output, deadline, NATIVE_SECONDS)
        comparison = output / "own-full-posterior-comparison"
        comparison.mkdir()
        owned_stage("three-own-full-array-comparisons", [str(args.python.absolute()), "-I", "-B", str(WORKER),
            "--root", str(ROOT), "--native", str(native), "--fixture", str(FIXTURE), "--output", str(comparison)],
            output, deadline, OWN_SECONDS)
        result = read(comparison / "complete-original-author-comparison.json")
        require(result["status"] == "THREE_FIXED_ORIGINAL_AUTHOR_POSTERIOR_COMPARISONS_PASS" and len(result["cases"]) == 3,
                "All three preregistered original-author cases required")
        require(time.monotonic() <= deadline, "Original whole600second reference deadline exceeded")
        status = "PASS_ORIGINAL_AUTHOR_OCTAVE_DETERMINISTIC_POSTERIOR_THREE_CASES"
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        after = source_inventory()
        unchanged = after == before
        if not unchanged:
            status = "FAILED_SOURCE_PRESERVATION"
        files = [pin(path) for path in sorted(output.rglob("*")) if path.is_file()]
        write(output / "complete-once-reference-terminal.json", {
            "schema": "openecon.original-author-Octave.once-terminal.v1", "status":status,
            "failure": failure, "source_commit":head, "source_before":before,"source_after":after,
            "all_source_bytes_preserved":unchanged, "all_actual_outputs":files,
            "started_monotonic":started,"finished_monotonic":time.monotonic(),
            "seconds":time.monotonic()-started,"whole_OS_seconds_cap":TOTAL_SECONDS,
            "no_retry_or_native_sampling":True})
        require(unchanged, "Original source bytes changed")


if __name__ == "__main__":
    main()
