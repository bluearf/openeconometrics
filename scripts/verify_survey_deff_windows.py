"""Retain source-bound Windows survey DEFF SDK acceptance, without native claims."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import chdir
import hashlib
import importlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import platform
import re
import runpy
import subprocess
import sys
import tempfile
import time
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_PIN = "b97a503224c126f03b8238d87a8670cf9e9d112b"
SCIENTIFIC_FILES = {
    "src/openecon/econometrics/survey/common.py":
        "bf599e6fc4bcf021ddc7dcea41d35755757a3801e138da2f040738971bcfde02",
    "src/openecon/econometrics/survey/targets.py":
        "916ce28fdb38f07c2b7552e8a705ec4cff07b22b48dd2eda4185e04136c3c989",
}
TESTS = {"tests/test_survey_deff.py": 84, "tests/test_survey_inference.py": 108}
EXAMPLE = "docs/examples/survey_deff.py"
HELPER = "scripts/verify_survey_deff_windows.py"
WORKFLOW = ".github/workflows/survey-deff-windows.yml"
NAMES = tuple(f"{variant}_{kind}"
              for variant in ("unequal_full", "unequal_fixed", "equal_legacy")
              for kind in ("mean", "total", "ratio", "proportion"))
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def json_read(data):
    def invalid(value):
        raise ValueError("Nonfinite JSON token: " + value)
    return json.loads(data, parse_constant=invalid)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def committed_bytes(source, paths):
    requests = "".join(f"{source}:{path}\n" for path in paths).encode()
    raw = subprocess.run(["git", "cat-file", "--batch"], input=requests,
                         capture_output=True, cwd=ROOT, check=True).stdout
    stream, result = io.BytesIO(raw), {}
    for path in paths:
        header = stream.readline().decode().strip().split()
        require(len(header) == 3 and header[1] == "blob", "Missing committed blob: " + path)
        size = int(header[2])
        result[path] = stream.read(size)
        require(len(result[path]) == size and stream.read(1) == b"\n", "Invalid Git blob framing")
    require(stream.read() == b"", "Unexpected Git blob output")
    return result


def source_manifest(source):
    modules = [path for path in git("ls-tree", "-r", "--name-only", source, "--",
                                   "src/openecon", "packages/openecon-charts/src")
               .decode().splitlines() if path.endswith(".py")]
    paths = sorted(set(modules) | set(TESTS) | {
        EXAMPLE, HELPER, WORKFLOW, "pyproject.toml", "uv.lock",
        "packages/openecon-charts/pyproject.toml"})
    blobs, manifest = committed_bytes(source, paths), {}
    for path, original in blobs.items():
        require((ROOT / path).read_bytes() == original, "Checkout differs from Git bytes: " + path)
        manifest[path] = {"bytes": len(original), "sha256": digest(original)}
    scientific = committed_bytes(SCIENTIFIC_PIN, list(SCIENTIFIC_FILES))
    for path, expected in SCIENTIFIC_FILES.items():
        require(digest(scientific[path]) == expected, "Original scientific identity differs: " + path)
        require(manifest[path]["sha256"] == expected, "Scoped scientific source changed: " + path)
    return {"execution_source_pin": source, "python_module_count": len(modules),
            "files": manifest, "original_scientific_pin": SCIENTIFIC_PIN,
            "original_scientific_file_sha256": SCIENTIFIC_FILES}


def test_environment():
    env = os.environ.copy()
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST"):
        env.pop(name, None)
    env.update(PYTHONPATH=os.pathsep.join([str(ROOT / "src"),
                                         str(ROOT / "packages/openecon-charts/src")]),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONUTF8="1")
    env.update({name: "1" for name in THREADS})
    return env


def validate_collection(log):
    nodes = [line.strip() for line in log.splitlines()
             if any(line.startswith(path + "::") for path in TESTS)]
    counts = Counter(node.split("::", 1)[0] for node in nodes)
    require(dict(counts) == TESTS and len(set(nodes)) == sum(TESTS.values()),
            "Collection must contain exactly 84 DEFF and 108 inference cases")
    return nodes


def validate_junit(path, collected):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    suites = [suite for suite in suites if not suite.findall("testsuite")]
    totals = {key: sum(int(suite.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    expected = {"tests": sum(TESTS.values()), "failures": 0, "errors": 0, "skipped": 0}
    require(totals == expected, "Actual JUnit totals must be 192 passing and zero skipped")
    class_files = {"tests." + Path(path).stem: path for path in TESTS}
    cases, identities = list(root.iter("testcase")), []
    for case in cases:
        require(case.get("classname") in class_files, "Unexpected JUnit test class")
        require(not any(case.find(tag) is not None for tag in ("failure", "error", "skipped")),
                "A JUnit case did not pass")
        identities.append(class_files[case.get("classname")] + "::" + case.get("name", ""))
    require(len(cases) == totals["tests"] and len(set(identities)) == len(cases)
            and set(identities) == set(collected), "JUnit must replay the complete collected test set")
    return {**totals, "test_files": TESTS, "testcase_identities": identities}


def validate_artifacts(directory, stdout):
    lines = [line.removeprefix("SURVEY_DEFF_RECEIPT:") for line in stdout.splitlines()
             if line.startswith("SURVEY_DEFF_RECEIPT:")]
    require(len(lines) == 1, "One original example receipt is required")
    marker = json_read(lines[0])
    require({path.name for path in directory.iterdir()} ==
            {"survey-deff-artifacts.json", *(name + ".json" for name in NAMES)},
            "All 13 complete example JSON files are required")
    data = (directory / "survey-deff-artifacts.json").read_bytes()
    packet = json_read(data)
    require(packet["schema"] == "survey-deff-complete-artifacts-v1"
            and packet["status"] == "passed" and packet["result_names"] == list(NAMES)
            and packet["output_count"] == 12 and packet["table_count"] == 24
            and set(packet["artifacts"]) == set(NAMES)
            and packet["stata_parity_validated"] is False, "Incomplete example packet")
    fields = {"name", "state", "table", "covariance", "srs_reference_covariance",
              "reference_state", "design_effect", "contrast_coefficients", "contrast",
              "table_latex", "contrast_latex"}
    for name in NAMES:
        record = packet["artifacts"][name]
        require(set(record) == fields and record["name"] == name, "Incomplete result: " + name)
        require(record["table_latex"] and record["contrast_latex"], "Missing LaTeX output")
        canonical = json.dumps(record, sort_keys=True, allow_nan=False,
                               separators=(",", ":"), ensure_ascii=False).encode()
        require((directory / (name + ".json")).read_bytes() == canonical,
                "Individual result differs from full packet: " + name)
    for key, expected in {"schema": packet["schema"], "status": "passed",
                          "result_names": list(NAMES), "output_count": 12, "table_count": 24,
                          "artifact_filename": "survey-deff-artifacts.json",
                          "artifact_sha256": digest(data), "artifact_bytes": len(data),
                          "full_covariance_saved": True, "restored_equal": True,
                          "stata_parity_validated": False}.items():
        require(marker.get(key) == expected, "Example marker mismatch: " + key)
    return marker


def example_child(directory):
    require(sys.flags.isolated == 1, "The production example requires an isolated subprocess")
    source_roots = (ROOT / "src", ROOT / "packages/openecon-charts/src")
    sys.path[:0] = [str(path) for path in source_roots]
    blocked = []

    class BlockDevelopmentImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".", 1)[0] in {"scipy", "statsmodels"}:
                blocked.append(fullname)
                raise ImportError("Explicitly blocked development dependency: " + fullname)
            return None

    sys.meta_path.insert(0, BlockDevelopmentImports())
    for name in ("scipy", "statsmodels"):
        try:
            importlib.import_module(name)
        except ImportError:
            pass
        else:
            raise ValueError("Import blocking probe failed")
    import torch
    require(torch.get_num_threads() == 1, "Torch must observe the gate's single-thread recipe")
    with tempfile.TemporaryDirectory(prefix="survey-deff-example-") as cwd, chdir(cwd):
        os.environ["OPENECON_ACCEPTANCE_DIR"] = str(directory / "example")
        runpy.run_path(str(ROOT / EXAMPLE), run_name="__main__")
        importlib.import_module("openecon_charts")
        modules = {}
        for name, module in sorted(sys.modules.items()):
            if name.split(".", 1)[0] in {"openecon", "openecon_charts"}:
                location = Path(module.__file__).resolve()
                require(any(location.is_relative_to(path) for path in source_roots),
                        "Loaded a package module outside the current checkout: " + name)
                modules[name] = str(location)
        require(not any(name.split(".", 1)[0] in {"scipy", "statsmodels"}
                        for name in sys.modules), "A blocked development module was loaded")
        write_json(directory / "example-runtime.json", {
            "isolated_flag": sys.flags.isolated, "temporary_cwd": cwd,
            "torch_version": torch.__version__, "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "torch_cuda_available": torch.cuda.is_available(),
            "native_thread_environment": {name: os.environ.get(name) for name in THREADS},
            "blocked_import_probes": blocked[:2], "blocked_import_attempts": blocked,
            "loaded_package_modules": modules, "all_modules_from_execution_checkout": True,
        })


def run_command(command, log, env):
    started = time.monotonic()
    with log.open("wb") as output:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                stderr=subprocess.STDOUT, timeout=600)
    return {"command": command, "exit_code": result.returncode,
            "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--example-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.example_child:
        example_child(directory)
        return 0
    require(directory.is_relative_to(ROOT / "artifacts"), "Use an owned artifact directory")
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "survey-deff-windows-v1", "status": "failed",
               "scope": "Windows 2025 Python 3.13 source SDK tests and production example",
               "native_windows_acceptance": False, "public_release_acceptance": False,
               "platform": platform.platform(), "system": platform.system(),
               "python": sys.version, "interpreter": sys.executable,
               "github": {key: os.environ.get(key) for key in (
                   "GITHUB_SHA", "GITHUB_EVENT_NAME", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT",
                   "GITHUB_REPOSITORY", "RUNNER_OS", "ImageOS", "ImageVersion",
                   "DEFF_PR_HEAD", "DEFF_PR_BASE")}, "steps": {}}
    try:
        require(os.name == "nt" and platform.system() == "Windows"
                and sys.version_info[:2] == (3, 13) and os.environ.get("GITHUB_ACTIONS") == "true"
                and os.environ.get("RUNNER_OS") == "Windows", "A genuine hosted Windows Python 3.13 run is required")
        source = git("rev-parse", "HEAD").decode().strip()
        require(re.fullmatch(r"[0-9a-f]{40}", source) is not None
                and source == os.environ.get("GITHUB_SHA"), "Checkout must equal GITHUB_SHA")
        manifest = source_manifest(source)
        write_json(directory / "source-manifest.json", manifest)
        receipt.update(execution_source_pin=source, original_scientific_pin=SCIENTIFIC_PIN,
                       helper_sha256=manifest["files"][HELPER]["sha256"],
                       source_parent_pins=git("show", "-s", "--format=%P", source).decode().split())
        env = test_environment()
        collection = [sys.executable, "-m", "pytest", "-o", "addopts=", "--collect-only", "-q", *TESTS]
        receipt["steps"]["collection"] = run_command(collection, directory / "collection.log", env)
        require(receipt["steps"]["collection"]["exit_code"] == 0, "Source collection failed")
        nodes = validate_collection((directory / "collection.log").read_text(encoding="utf-8"))
        tests = [sys.executable, "-m", "pytest", "-o", "addopts=", "-q",
                 "--junitxml", str(directory / "tests.xml"), *TESTS]
        receipt["steps"]["tests"] = run_command(tests, directory / "tests.log", env)
        receipt["junit"] = validate_junit(directory / "tests.xml", nodes)
        require(receipt["steps"]["tests"]["exit_code"] == 0, "Windows source tests failed")
        command = [sys.executable, "-I", str(ROOT / HELPER), "--example-child",
                   "--directory", str(directory)]
        receipt["steps"]["example"] = run_command(command, directory / "example.log", env)
        require(receipt["steps"]["example"]["exit_code"] == 0, "Isolated production example failed")
        receipt["example"] = validate_artifacts(directory / "example",
                                                (directory / "example.log").read_text(encoding="utf-8"))
        receipt["runtime"] = json_read((directory / "example-runtime.json").read_bytes())
        for path, identity in manifest["files"].items():
            require(digest((ROOT / path).read_bytes()) == identity["sha256"],
                    "Source changed during execution: " + path)
        receipt["status"] = "passed"
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, ET.ParseError) as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        receipt["retained_files"] = {path.relative_to(directory).as_posix(): {
            "sha256": digest(path.read_bytes()), "bytes": path.stat().st_size}
            for path in sorted(directory.rglob("*")) if path.is_file()}
        write_json(directory / "receipt.json", receipt)
        print(json.dumps({"status": receipt["status"], "receipt": str(directory / "receipt.json"),
                          "native_windows_acceptance": False, "public_release_acceptance": False}))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
