"""Admit the a2 description refresh only after authenticated a1 byte comparison.

The historical scientific gate remains evidence for a1. This verifier binds the
unchanged a2 runtime to that baseline and requires fresh Linux package acceptance;
it does not represent a new complete numerical or native acceptance run.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import email
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shutil
import stat
import tarfile
import tomllib
import urllib.request
import xml.etree.ElementTree as ET
import zipfile


REPOSITORY = "bluearf/openeconometrics"
RELEASE = "python-v0.3.20a2"
BASELINE_RELEASE = "python-v0.3.20a1"
SOURCE_COMMIT = "a805870d971c0d571f6d5dec79e914b91d6479aa"
PUBLIC_BASELINE = "d11ed130feb4ad8e6e990e61bb0bd8a9101cb981"
PUBLIC_METADATA_BASELINE = "f202f44f0f4d0c5539105c950e7e3646abc9713f"
CAPABILITY_METADATA_REF = "35ee0ce87b6d773f9623a26a186bda539dc5322a"
RUNTIME_INVENTORY = "48685ba004b4a00c971583f1cbfe0d171ca4ce060150d9df783e0c7614fdfe35"
VERSIONS = {"openecon": "0.3.20a2", "openecon-charts": "0.3.2a2"}
OLD_VERSIONS = {"openecon": "0.3.20a1", "openecon-charts": "0.3.2a1"}
WORKFLOW = ".github/workflows/python-sdk-patch.yml"
DISTRIBUTIONS = {
    f"{name.replace('-', '_')}-{version}{suffix}"
    for name, version in VERSIONS.items()
    for suffix in ("-py3-none-any.whl", ".tar.gz")
}
PROOFS = {
    "PACKAGE-PAIR-PROVENANCE.json", "BUILD-EXECUTION.json", "SOURCE-IDENTITY.json",
    "linux-package-3.11-evidence.zip", "linux-package-3.13-evidence.zip",
}
ASSETS = DISTRIBUTIONS | PROOFS
VERSION_FILES = {
    "openecon/__init__.py": ("0.3.20a1", "0.3.20a2"),
    "openecon_charts/__init__.py": ("0.3.2a1", "0.3.2a2"),
}
SOURCE_VERSION_FILES = {
    "src/openecon/__init__.py": "openecon/__init__.py",
    "packages/openecon-charts/src/openecon_charts/__init__.py": "openecon_charts/__init__.py",
}
SOURCE_CHANGES = {
    "README.md", "PYPI-README.md", "packages/openecon-charts/PYPI-README.md",
    "pyproject.toml", "packages/openecon-charts/pyproject.toml", "uv.lock",
    "docs/distribution.md", "PUBLIC-SOURCE.md", "SOURCE-MANIFEST.json", *SOURCE_VERSION_FILES,
    "scripts/verify_public_package_pair.py", "tests/test_public_package_pair_metadata.py",
    "docs/capabilities.md", "docs/econometrics/capabilities.generated.json",
}
SUITES = {
    "installed-inventory-guards.xml": 33, "sdk-regressions.xml": 324,
    "network-export-wheel.xml": 88, "network-export-sdist.xml": 88,
}
MAX_ASSET = 128 * 1024 * 1024
MAX_ARCHIVE = 192 * 1024 * 1024
MAX_MEMBER = 64 * 1024 * 1024
BUILD_STEPS = {
    "Bind metadata refresh to the published runtime source",
    "Build source-pinned frontend and exactly four distributions",
    "Record immutable build execution and retain its exact bytes",
}
PACKAGE_STEPS = {
    "Independently verify the same metadata-only runtime identity",
    "Verify the same four distributions against source and build execution",
    "Select provisioned Chrome with its sandbox enabled",
    "Install and verify dependency-free charts wheel and sdist",
    "Install and test fresh SDK wheel and sdist environments",
    "Run bounded metadata, parser and OLS guard regressions",
    "Verify standalone wheel and sdist sandboxed Chrome exports",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(data: bytes):
    def unique(rows):
        result = {}
        for key, value in rows:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def exact_keys(value: dict, keys: set[str], message: str) -> None:
    require(isinstance(value, dict) and set(value) == keys, message)


def canonical(name: str) -> str:
    path = PurePosixPath(name)
    require(
        bool(name) and bool(path.parts) and path.as_posix() == name and not path.is_absolute()
        and ".." not in path.parts and "\\" not in name and "\0" not in name,
        f"Unsafe archive path: {name}",
    )
    return name


def regular(path: Path) -> None:
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), f"Symlink in verification path: {parent}")
    require(path.is_file(), f"Missing regular verification file: {path}")


def asset(data: bytes, expected: dict, name: str) -> None:
    require(len(data) == expected["bytes"] and sha(data) == expected["sha256"],
            f"Reviewed asset bytes differ: {name}")


def check_profile(profile: dict, baseline_profile_bytes: bytes) -> None:
    exact_keys(profile, {"schema_version", "repository", "release", "versions", "baseline",
                         "validation", "assets"}, "Unknown metadata refresh profile fields")
    require(profile["schema_version"] == 1 and type(profile["schema_version"]) is int,
            "Unknown metadata refresh profile version")
    require(profile["repository"] == REPOSITORY and profile["release"] == RELEASE
            and profile["versions"] == VERSIONS, "Metadata refresh identity differs")
    require(profile["baseline"] == {"release": BASELINE_RELEASE,
                                    "profile_sha256": sha(baseline_profile_bytes)},
            "Historical review profile differs")
    expected = profile["assets"]
    exact_keys(expected, ASSETS, "Missing or foreign metadata refresh assets")
    for name, row in expected.items():
        exact_keys(row, {"bytes", "sha256"}, "Unknown asset fields")
        require(canonical(name) == PurePosixPath(name).name
                and type(row["bytes"]) is int and 0 < row["bytes"] < MAX_ASSET
                and isinstance(row["sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "Unsafe asset bound/hash")
    validation = profile["validation"]
    exact_keys(validation, {"run_id", "run_attempt", "api_head_sha", "tested_checkout_commit",
                            "event", "workflow_path"}, "Unknown CI validation fields")
    require(all(type(validation[k]) is int and validation[k] > 0
                for k in ("run_id", "run_attempt")), "Invalid CI run identity")
    require(all(isinstance(validation[k], str) and re.fullmatch(r"[0-9a-f]{40}", validation[k])
                for k in ("api_head_sha", "tested_checkout_commit")), "Invalid CI commit pin")
    require(validation["event"] in {"pull_request", "push"}
            and validation["workflow_path"] == WORKFLOW, "Unaccepted package workflow/event")
    if validation["event"] == "push":
        require(validation["api_head_sha"] == validation["tested_checkout_commit"],
                "Push checkout and API head differ")


def request_bytes(url: str, limit: int, *, api: bool = False) -> bytes:
    headers = {"User-Agent": "OpenEconometrics-release-verifier"}
    if api:
        headers["Accept"] = "application/vnd.github+json"
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        result = response.read(limit + 1)
    require(len(result) <= limit, "Downloaded response exceeds its bound")
    return result


def check_ci(validation: dict, api_reader=None) -> None:
    if api_reader is None:
        def api_reader(url):
            return json_bytes(request_bytes(url, 2 * 1024 * 1024, api=True))
    base = f"https://api.github.com/repos/{REPOSITORY}/actions/runs/{validation['run_id']}"
    run = api_reader(base)
    require(run["id"] == validation["run_id"]
            and run["run_attempt"] == validation["run_attempt"]
            and run["repository"]["full_name"] == REPOSITORY
            and run["head_repository"]["full_name"] == REPOSITORY
            and run["head_sha"] == validation["api_head_sha"]
            and run["event"] == validation["event"] and run["path"] == WORKFLOW
            and run["status"] == "completed" and run["conclusion"] == "success",
            "Fresh public package run identity/conclusion differs")
    response = api_reader(base + f"/attempts/{validation['run_attempt']}/jobs?per_page=100")
    jobs = response["jobs"]
    names = {"Build one fixed SDK and charts pair", "Verify SDK patch package (3.11)",
             "Verify SDK patch package (3.13)"}
    require(response["total_count"] == len(jobs) == len(names)
            and Counter(row["name"] for row in jobs) == Counter(names),
            "Missing, duplicate or foreign package CI jobs")
    require(all(row["run_id"] == validation["run_id"]
                and row["head_sha"] == validation["api_head_sha"]
                and row["status"] == "completed" and row["conclusion"] == "success"
                and bool(row["steps"])
                and all(step["status"] == "completed"
                        and step["conclusion"] in {"success", "skipped"}
                        for step in row["steps"])
                for row in jobs), "Package CI job or step did not succeed")
    for row in jobs:
        required = BUILD_STEPS if row["name"] == "Build one fixed SDK and charts pair" else PACKAGE_STEPS
        steps = [step for step in row["steps"] if step["name"] in required]
        require(Counter(step["name"] for step in steps) == Counter(required)
                and all(step["conclusion"] == "success" for step in steps),
                "Required package build/acceptance step missing or skipped")


def zip_files(path: Path, *, limit: int = MAX_ARCHIVE) -> dict[str, bytes]:
    regular(path)
    result = {}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(0 < len(entries) < 20000 and sum(row.file_size for row in entries) < limit,
                "Excessive/empty ZIP archive")
        seen = set()
        for row in entries:
            name = canonical(row.filename.removesuffix("/") if row.is_dir() else row.filename)
            require(name not in seen, "Duplicate ZIP archive entry")
            seen.add(name)
            mode = row.external_attr >> 16
            require(not row.flag_bits & 1 and not stat.S_ISLNK(mode)
                    and (stat.S_IFMT(mode) in {0, stat.S_IFDIR} if row.is_dir()
                         else stat.S_IFMT(mode) in {0, stat.S_IFREG}),
                    "Encrypted or nonregular ZIP member")
            require(0 <= row.file_size < MAX_MEMBER, "Excessive ZIP member")
            if not row.is_dir():
                result[name] = archive.read(row)
    require(bool(result), "Empty ZIP archive payload")
    return result


def tar_files(path: Path, prefix: str) -> dict[str, bytes]:
    regular(path)
    result = {}
    seen = set()
    size = 0
    class BoundedStream:
        consumed = 0

        def __init__(self, stream):
            self.stream = stream

        def read(self, length):
            # Bound headers/PAX data too, before tarfile can allocate an
            # arbitrarily expanded metadata block from a compressed input.
            data = self.stream.read(min(length, MAX_ARCHIVE - self.consumed + 1))
            self.consumed += len(data)
            require(self.consumed <= MAX_ARCHIVE, "Excessive decompressed sdist stream")
            return data

    with gzip.open(path, "rb") as compressed, tarfile.open(
        fileobj=BoundedStream(compressed), mode="r|"
    ) as archive:
        for row in archive:
            name = canonical(row.name)
            require(name not in seen and (name == prefix or name.startswith(prefix + "/")),
                    "Duplicate/foreign sdist member")
            seen.add(name)
            require(row.isfile() or row.isdir(), "Nonregular sdist member")
            size += row.size
            require(len(seen) < 20000 and size < MAX_ARCHIVE
                    and 0 <= row.size < MAX_MEMBER, "Excessive sdist member/archive")
            if row.isfile():
                require(name != prefix, "sdist root is a file")
                stream = archive.extractfile(row)
                require(stream is not None, "Missing sdist member payload")
                result[name.removeprefix(prefix + "/")] = stream.read()
    require(bool(result), "Empty sdist")
    return result


def compare_runtime(old: dict[str, bytes], new: dict[str, bytes], prefix: str) -> dict:
    before = {name: data for name, data in old.items() if name.startswith(prefix)}
    after = {name: data for name, data in new.items() if name.startswith(prefix)}
    require(bool(before) and set(before) == set(after), "Runtime file inventory changed")
    for name, data in before.items():
        expected = data
        if name in VERSION_FILES:
            old_version, new_version = VERSION_FILES[name]
            literal = f'__version__ = "{old_version}"'.encode()
            require(data.count(literal) == 1, "Published runtime version literal differs")
            expected = data.replace(literal, f'__version__ = "{new_version}"'.encode(), 1)
        require(after[name] == expected, f"Runtime changed beyond exact version literal: {name}")
    return {name: sha(data) for name, data in sorted(after.items())}


def compare_project(old: bytes, new: bytes, name: str) -> dict:
    before, after = tomllib.loads(old.decode()), tomllib.loads(new.decode())
    expected = copy.deepcopy(before)
    require(expected["project"]["name"] == name
            and expected["project"]["version"] == OLD_VERSIONS[name],
            "Historical project identity differs")
    expected["project"]["version"] = VERSIONS[name]
    if name == "openecon":
        dependencies = expected["project"]["dependencies"]
        require(dependencies.count("openecon-charts==0.3.2a1") == 1,
                "Historical charts pin differs")
        expected["project"]["dependencies"] = [
            "openecon-charts==0.3.2a2" if value == "openecon-charts==0.3.2a1" else value
            for value in dependencies
        ]
    require(after == expected, f"Unreviewed dependency/build/project configuration: {name}")
    return after["project"]


def compare_metadata_headers(before: bytes, after: bytes, name: str) -> None:
    old, new = email.message_from_bytes(before), email.message_from_bytes(after)
    expected = []
    for key, value in old.items():
        if key == "Version":
            value = VERSIONS[name]
        elif key == "Requires-Dist" and value == "openecon-charts==0.3.2a1":
            value = "openecon-charts==0.3.2a2"
        expected.append((key, value))
    require(Counter(new.items()) == Counter(expected),
            f"Metadata headers changed beyond version/chart pin: {name}")


def compare_sdists(old: dict[str, bytes], new: dict[str, bytes], *, charts: bool) -> None:
    require(set(old) == set(new), "sdist source file inventory changed")
    for name, data in old.items():
        source_name = "packages/openecon-charts/" + name if charts else name
        if name != "PKG-INFO" and source_name not in SOURCE_CHANGES:
            require(new[name] == data, f"Unreviewed sdist source change: {source_name}")
    if not charts:
        expected = old["uv.lock"].replace(b'"0.3.20a1"', b'"0.3.20a2"')
        expected = expected.replace(b'"0.3.2a1"', b'"0.3.2a2"')
        expected = expected.replace(b'"==0.3.2a1"', b'"==0.3.2a2"')
        require(new["uv.lock"] == expected, "Lockfile changed beyond exact version/pin refresh")


def compare_capability_docs(old: dict[str, bytes], new: dict[str, bytes]) -> None:
    path = "docs/econometrics/capabilities.generated.json"
    before, after = json_bytes(old[path]), json_bytes(new[path])
    expected = copy.deepcopy(before)
    require(before["versions"]["sdk"] == "0.3.20a1"
            and before["versions"]["charts"] == "0.3.2a1", "Capability baseline versions differ")
    expected["source_ref"] = CAPABILITY_METADATA_REF
    expected["versions"].update(sdk="0.3.20a2", charts="0.3.2a2")
    dependencies = expected["dependencies"]["runtime"]
    require(dependencies.count("openecon-charts==0.3.2a1") == 1, "Capability baseline charts pin differs")
    expected["dependencies"]["runtime"] = [
        "openecon-charts==0.3.2a2" if value == "openecon-charts==0.3.2a1" else value
        for value in dependencies
    ]
    for source in ("pyproject.toml", "packages/openecon-charts/pyproject.toml", "src/openecon/__init__.py"):
        require(before["source_fingerprints"][source] == sha(old[source]),
                "Capability baseline fingerprint differs")
        expected["source_fingerprints"][source] = sha(new[source])
    require(after == expected, "Capability methods/domains changed beyond exact metadata fields")
    expected_md = old["docs/capabilities.md"]
    old_ref = f"Source reference: `{before['source_ref']}`".encode()
    require(expected_md.count(old_ref) == 1, "Capability baseline source reference differs")
    expected_md = expected_md.replace(old_ref, f"Source reference: `{CAPABILITY_METADATA_REF}`".encode(), 1)
    for before_version, after_version in ((b"`0.3.20a1`", b"`0.3.20a2`"),
                                          (b"`0.3.2a1`", b"`0.3.2a2`"),
                                          (b"`openecon-charts==0.3.2a1`", b"`openecon-charts==0.3.2a2`")):
        require(expected_md.count(before_version) == 1, "Capability baseline version label differs")
        expected_md = expected_md.replace(before_version, after_version, 1)
    require(new["docs/capabilities.md"] == expected_md,
            "Capability Markdown changed beyond exact metadata labels")


def check_manifest(old_data: bytes, data: bytes, source: dict[str, bytes],
                   old_source: dict[str, bytes]) -> dict:
    old, current = json_bytes(old_data), json_bytes(data)
    require(current["source_commit"] == old["source_commit"] == SOURCE_COMMIT,
            "Inherited scientific baseline source differs")
    old_rows = {row["path"]: row for row in old["files"]}
    rows = current["files"]
    current_rows = {canonical(row["path"]): row for row in rows}
    require(len(current_rows) == len(rows) and set(current_rows) == set(old_rows),
            "Source manifest inventory changed")
    require(sha(json.dumps(rows, sort_keys=True).encode()) == current["source_files_sha256"],
            "Refreshed source inventory digest differs")
    require(set(current) == set(old) | {"metadata_refresh"}
            and {key: value for key, value in current.items()
                 if key not in {"files", "source_files_sha256", "metadata_refresh"}}
            == {key: value for key, value in old.items()
                if key not in {"files", "source_files_sha256"}},
            "Historical scientific manifest claims changed")
    for name, row in current_rows.items():
        require(type(row["bytes"]) is int and row["bytes"] >= 0
                and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "Invalid manifest file")
        if name in source:
            require(row["bytes"] == len(source[name]) and row["sha256"] == sha(source[name]),
                    f"sdist bytes differ from manifest: {name}")
        if name not in SOURCE_CHANGES:
            require(row == old_rows[name], f"Unreviewed source inventory change: {name}")
        else:
            require(set(row) == set(old_rows[name])
                    and row["transformed"] == (row["sha256"] != row["original_sha256"]),
                    "Source transformation audit differs")
            for key in ("original_sha256", "original_bytes"):
                if key in old_rows[name]:
                    require(row.get(key) == old_rows[name][key], "Original source audit changed")
    refresh = current["metadata_refresh"]
    keys = {"schema_version", "baseline_public_commit", "baseline_runtime_public_commit",
            "inherited_private_source_commit", "previous_versions", "versions", "scope",
            "runtime_identity_verifier", "runtime_identity_receipt", "baseline_runtime_inventory_sha256",
            "new_full_scientific_gate_claimed", "registry_publication", "changed_files", "generated_documents",
            "packaging_verification_transforms", "generated_capability_transforms"}
    exact_keys(refresh, keys, "Unknown metadata refresh manifest fields")
    require(refresh["schema_version"] == 1
            and refresh["baseline_public_commit"] == PUBLIC_METADATA_BASELINE
            and refresh["baseline_runtime_public_commit"] == PUBLIC_BASELINE
            and refresh["inherited_private_source_commit"] == SOURCE_COMMIT
            and refresh["previous_versions"] == {"sdk": "0.3.20a1", "charts": "0.3.2a1"}
            and refresh["versions"] == {"sdk": "0.3.20a2", "charts": "0.3.2a2"}
            and refresh["runtime_identity_verifier"] == ".github/checks/verify_metadata_refresh_source.py"
            and refresh["runtime_identity_receipt"] == "source-identity.json"
            and refresh["baseline_runtime_inventory_sha256"] == RUNTIME_INVENTORY
            and refresh["new_full_scientific_gate_claimed"] is False
            and all(isinstance(refresh[key], str) and bool(refresh[key])
                    for key in ("scope", "registry_publication", "generated_documents")),
            "Metadata refresh broadened or lost its baseline identity")
    changes = []
    for name in sorted(SOURCE_CHANGES - {"SOURCE-MANIFEST.json"}):
        if name in source and source[name] != old_source.get(name):
            require(name in old_source, "Metadata refresh added source")
            changes.append({"path": name, "baseline_sha256": sha(old_source[name]),
                            "refreshed_sha256": sha(source[name]),
                            "baseline_bytes": len(old_source[name]), "refreshed_bytes": len(source[name])})
    require(refresh["changed_files"] == changes, "Metadata refresh changed-file audit differs")
    transforms = refresh["packaging_verification_transforms"]
    require(isinstance(transforms, list) and len(transforms) == 2
            and [row.get("path") for row in transforms]
            == ["scripts/verify_public_package_pair.py", "tests/test_public_package_pair_metadata.py"],
            "Package verification transformation scope differs")
    for row in transforms:
        exact_keys(row, {"path", "reason"}, "Unknown packaging transformation fields")
        require(isinstance(row["reason"], str) and bool(row["reason"]),
                "Packaging transformation reason missing")
    capability_transforms = refresh["generated_capability_transforms"]
    require(isinstance(capability_transforms, list) and len(capability_transforms) == 2
            and [row.get("path") for row in capability_transforms]
            == ["docs/capabilities.md", "docs/econometrics/capabilities.generated.json"],
            "Capability documentation transformation scope differs")
    for row in capability_transforms:
        exact_keys(row, {"path", "reason"}, "Unknown capability transformation fields")
        require(isinstance(row["reason"], str) and bool(row["reason"]),
                "Capability transformation reason missing")
    return current


def check_identity(identity: dict, validation: dict, old_wheels: dict) -> None:
    require(type(identity.get("schema_version")) is int
            and identity.get("new_full_scientific_gate_claimed") is False,
            "Runtime source identity proof has invalid claim types")
    expected = {
        "schema_version": 1, "status": "passed_metadata_only_runtime_identity",
        "baseline_repository": REPOSITORY, "baseline_public_commit": PUBLIC_BASELINE,
        "public_checkout_commit": validation["tested_checkout_commit"],
        "event": validation["event"], "run_id": str(validation["run_id"]),
        "run_attempt": str(validation["run_attempt"]),
        "scopes": ["src/openecon", "packages/openecon-charts/src/openecon_charts", "web"],
        "excluded_generated_directories": ["src/openecon/static", "web/node_modules", "web/dist"],
        "generated_frontend": "separately bound by verify_public_package_pair.py",
        "new_full_scientific_gate_claimed": False,
        "runtime_source_files": 1040, "unchanged_runtime_source_files": 1038,
        "baseline_inventory_sha256": RUNTIME_INVENTORY,
        "normalized_current_inventory_sha256": RUNTIME_INVENTORY,
        "version_only_files": [],
    }
    for source_name, wheel_name in sorted(SOURCE_VERSION_FILES.items()):
        before, after = VERSION_FILES[wheel_name]
        data = old_wheels[wheel_name]
        changed = data.replace(f'__version__ = "{before}"'.encode(),
                               f'__version__ = "{after}"'.encode(), 1)
        expected["version_only_files"].append({
            "path": source_name, "before": before, "after": after,
            "before_sha256": sha(data), "after_sha256": sha(changed),
        })
    require(identity == expected, "CI runtime source identity differs from pinned a1 source")


def check_xml(data: bytes, count: int) -> None:
    require(b"<!DOCTYPE" not in data and b"<!ENTITY" not in data, "Unsafe XML declarations")
    root = ET.fromstring(data)
    require(root.tag in {"testsuite", "testsuites"}, "Unknown acceptance XML root")
    cases = list(root.iter("testcase"))
    require(len(cases) == count and not any(list(root.iter(tag))
                for tag in ("failure", "error", "skipped")),
            "Failed, missing or skipped package acceptance cases")
    require(len({(case.get("classname"), case.get("name")) for case in cases}) == count,
            "Duplicate package acceptance cases")
    require(all(suite.get(key, "0") == "0"
                for suite in root.iter("testsuite") for key in ("failures", "errors", "skipped")),
            "Package acceptance XML summary did not pass")


def check_linux(files: dict[str, bytes], minor: str, build: dict, execution: dict,
                identity: bytes, provenance: bytes, descriptions: dict[str, bytes]) -> None:
    # upload-artifact finds /home/runner/work as the common ancestor of the
    # checkout and RUNNER_TEMP. Preserve the original ZIP and admit this exact
    # workspace prefix, rather than searching for trusted basenames anywhere.
    workspace = "openeconometrics/openeconometrics/"
    require(all(name.startswith("_temp/") or name.startswith(workspace + "pair-build/")
                for name in files), "Original Linux artifact workspace paths differ")
    normalized = {name.removeprefix(workspace): value for name, value in files.items()}
    require(len(normalized) == len(files), "Colliding Linux evidence paths")
    files = normalized
    expected_names = {
        "pair-build/build-execution.json", "pair-build/package-pair-provenance.json",
        "pair-build/source-identity.json", "_temp/pair-execution.json",
        "_temp/source-identity.json", *{f"_temp/{name}" for name in SUITES},
        *{f"_temp/{package}-{mode}-{kind}.json" for package in ("sdk", "charts")
          for mode in ("wheel", "sdist") for kind in ("installed", "smoke", "readme")},
    }
    require(set(files) == expected_names, "Missing or foreign Linux evidence files")
    require(json_bytes(files["pair-build/build-execution.json"]) == execution
            and files["pair-build/package-pair-provenance.json"] == provenance
            and files["pair-build/source-identity.json"] == files["_temp/source-identity.json"]
            == identity, "Original Linux build/source identity evidence differs")
    pair = json_bytes(files["_temp/pair-execution.json"])
    require(pair.get("same_four_distributions") is True
            and pair == {"status": "passed", **execution, "artifacts": build["artifacts"],
                     "generated_frontend": build["generated_frontend"],
                     "same_four_distributions": True}, "Linux did not test these four archives")
    for name, count in SUITES.items():
        check_xml(files["_temp/" + name], count)
    for package, projects in (("sdk", set(VERSIONS)), ("charts", {"openecon-charts"})):
        name = "openecon" if package == "sdk" else "openecon-charts"
        for mode in ("wheel", "sdist"):
            prefix = f"_temp/{package}-{mode}-"
            installed = json_bytes(files[prefix + "installed.json"])
            require(installed["status"] == "passed" and installed["source_commit"] == SOURCE_COMMIT
                    and installed["source_files_sha256"] == build["source_files_sha256"]
                    and installed["provenance_sha256"] == sha(provenance)
                    and installed["python"].startswith(minor + ".")
                    and all(installed[key] is True for key in
                            ("isolated", "workspace_paths_absent", "editable_installations_absent"))
                    and set(installed["packages"]) == projects,
                    "Fresh installed package identity/isolation differs")
            for project, record in installed["packages"].items():
                require(record["version"] == VERSIONS[project]
                        and record["production_files"] == build["production_files"][project]
                        and bool(record["imported_modules"])
                        and all(value in record["production_files"]
                                for value in record["imported_modules"].values()),
                        "Installed production inventory/origins differ")
            smoke = json_bytes(files[prefix + "smoke.json"])
            require(smoke["status"] == "passed", "Package smoke did not pass")
            if package == "sdk":
                require(smoke["sdk_version"] == VERSIONS["openecon"]
                        and smoke["charts_version"] == VERSIONS["openecon-charts"]
                        and smoke["registry_estimators"] == 171
                        and smoke["registry_exports"] == 995 and smoke["referenced_modules"] == 440
                        and all(smoke[key] is True for key in (
                            "lazy_registry_no_heavy_imports", "ols_independent_algebra",
                            "hc3_independent_full_covariance", "guarded_parquet_after_torch",
                            "exact_fit_guard_and_real_noise", "poisson_score_check",
                            "json_restore_predictions_duplicate_index", "latex_and_chart_protocol")),
                        "SDK smoke identity/scope differs")
            else:
                require(smoke["version"] == VERSIONS["openecon-charts"]
                        and all(smoke[key] is True for key in ("standalone_no_heavy_dependencies",
                                    "offline_html_latex", "embedded_woff2_and_ttf_fonts")),
                        "Charts smoke identity/scope differs")
            readme = descriptions[name]
            blocks = re.findall(r"```python\r?\n(.*?)\r?\n```", readme.decode(), flags=re.DOTALL)
            receipt = json_bytes(files[prefix + "readme.json"])
            require(bool(blocks) and receipt.get("isolated_installed_examples") is True
                    and receipt.get("offline") is True and receipt == {
                "status": "passed", "package": name, "python_blocks": len(blocks),
                "description_sha256": sha(readme),
                "code_sha256": [sha(block.encode()) for block in blocks],
                "isolated_installed_examples": True, "offline": True,
            }, "Installed description/offline example evidence differs")


def verify(directory: Path, baseline: Path, profile: dict, baseline_profile_bytes: bytes,
           *, api_reader=None, checker=None) -> dict:
    check_profile(profile, baseline_profile_bytes)
    historical = json_bytes(baseline_profile_bytes)
    require(historical["schema_version"] == 4 and historical["release"] == BASELINE_RELEASE
            and historical["repository"] == REPOSITORY, "Historical baseline profile identity differs")
    require(not directory.is_symlink() and not baseline.is_symlink(), "Linked evidence directory")
    require({path.name for path in directory.iterdir()} == ASSETS,
            "Missing or foreign downloaded refresh assets")
    for name, expected in profile["assets"].items():
        path = directory / name
        regular(path)
        asset(path.read_bytes(), expected, name)
    old_names = {
        name.replace(VERSIONS[project], OLD_VERSIONS[project])
        for project in VERSIONS for name in DISTRIBUTIONS
        if name.startswith(project.replace("-", "_") + "-")
    }
    for name in old_names | {"PACKAGE-PAIR-PROVENANCE.json"}:
        path = baseline / name
        regular(path)
        asset(path.read_bytes(), historical["assets"][name], name)
    check_ci(profile["validation"], api_reader)
    build_bytes = (directory / "PACKAGE-PAIR-PROVENANCE.json").read_bytes()
    build = json_bytes(build_bytes)
    old_build = json_bytes((baseline / "PACKAGE-PAIR-PROVENANCE.json").read_bytes())
    execution = json_bytes((directory / "BUILD-EXECUTION.json").read_bytes())
    identity_bytes = (directory / "SOURCE-IDENTITY.json").read_bytes()
    validation = profile["validation"]
    require(execution == {
        "public_checkout_commit": validation["tested_checkout_commit"],
        "private_source_commit": SOURCE_COMMIT, "source_files_sha256": build["source_files_sha256"],
        "event": validation["event"], "run_id": str(validation["run_id"]),
        "run_attempt": str(validation["run_attempt"]), "sdk_version": VERSIONS["openecon"],
        "charts_version": VERSIONS["openecon-charts"],
    }, "Accepted package execution differs")
    if checker is None:
        checker = runpy.run_path(str(Path(__file__).resolve().parents[2]
                                     / "scripts/verify_public_package_pair.py"))
    production, metadata, descriptions, old_wheels = {}, {}, {}, {}
    new_sdists, old_sdists = {}, {}
    for name, version in VERSIONS.items():
        basename = name.replace("-", "_")
        old_version = OLD_VERSIONS[name]
        old_wheel = zip_files(baseline / f"{basename}-{old_version}-py3-none-any.whl")
        wheel = zip_files(directory / f"{basename}-{version}-py3-none-any.whl")
        old_wheels.update(old_wheel)
        production[name] = compare_runtime(old_wheel, wheel, basename + "/")
        old_info, info = f"{basename}-{old_version}.dist-info/", f"{basename}-{version}.dist-info/"
        require(wheel[info + "WHEEL"] == old_wheel[old_info + "WHEEL"],
                "Wheel build metadata changed; review the exact Generator/header difference")
        compare_metadata_headers(old_wheel[old_info + "METADATA"], wheel[info + "METADATA"], name)
        old_sdist = tar_files(baseline / f"{basename}-{old_version}.tar.gz",
                              f"{basename}-{old_version}")
        sdist = tar_files(directory / f"{basename}-{version}.tar.gz", f"{basename}-{version}")
        compare_sdists(old_sdist, sdist, charts=name == "openecon-charts")
        compare_metadata_headers(old_sdist["PKG-INFO"], sdist["PKG-INFO"], name)
        old_sdists[name], new_sdists[name] = old_sdist, sdist
        project = compare_project(old_sdist["pyproject.toml"], sdist["pyproject.toml"], name)
        project["readme_bytes"] = sdist[project["readme"]]
        project["license_bytes"] = {notice: old_sdist[notice] for notice in project["license-files"]}
        project["charts_version"] = VERSIONS["openecon-charts"]
        descriptions[name] = project["readme_bytes"]
        links = re.findall(r"(?<!!)\[[^\]]+\]\(([^\s)]+)\)", descriptions[name].decode())
        require(all(link.startswith("https://") for link in links), "Relative description link")
        checker["check_wheel_layout"](wheel, project, production[name])
        for path, digest in production[name].items():
            source_path = "src/" + path
            require(sdist.get(source_path) == wheel[path] and sha(wheel[path]) == digest,
                    f"sdist runtime differs from wheel: {source_path}")
        metadata[name] = {"wheel": checker["check_metadata"](wheel, project, wheel=True),
                          "sdist": checker["check_metadata"](sdist, project, wheel=False)}
    sdk_source = new_sdists["openecon"]
    compare_capability_docs(old_sdists["openecon"], sdk_source)
    for path, data in new_sdists["openecon-charts"].items():
        if path != "PKG-INFO":
            require(sdk_source.get("packages/openecon-charts/" + path) == data,
                    "SDK sdist bundled a different charts source")
    manifest_data = sdk_source["SOURCE-MANIFEST.json"]
    manifest = check_manifest(old_sdists["openecon"]["SOURCE-MANIFEST.json"],
                              manifest_data, sdk_source, old_sdists["openecon"])
    generated = {"src/" + name: digest for name, digest in production["openecon"].items()
                 if name.startswith("openecon/static/")}
    require(generated == old_build["generated_frontend"], "Generated frontend changed")
    expected_build = {
        "status": "passed",
        "scope": "Frozen source and distribution byte verification; no publication or method parity claim.",
        "source_commit": SOURCE_COMMIT, "source_manifest_sha256": sha(manifest_data),
        "source_files_sha256": manifest["source_files_sha256"], "versions": VERSIONS,
        "artifacts": {name: profile["assets"][name] for name in sorted(DISTRIBUTIONS)},
        "production_files": production, "metadata": metadata, "generated_frontend": generated,
        "external_fixture_test_modules": old_build["external_fixture_test_modules"],
    }
    require(build == expected_build, "Complete refreshed package provenance differs")
    check_identity(json_bytes(identity_bytes), validation, old_wheels)
    for minor in ("3.11", "3.13"):
        files = zip_files(directory / f"linux-package-{minor}-evidence.zip", limit=4 * 1024 * 1024)
        check_linux(files, minor, build, execution, identity_bytes, build_bytes, descriptions)
    # Publish outputs appear only after every independent check succeeds.
    for project in VERSIONS:
        require(not (directory / project).exists(), "Refresh output must be fresh")
    for project in VERSIONS:
        destination = directory / project / "dist"
        destination.mkdir(parents=True)
        for name in sorted(DISTRIBUTIONS):
            if name.startswith(project.replace("-", "_") + "-"):
                shutil.copyfile(directory / name, destination / name)
    return {
        "status": "passed_metadata_only_refresh", "versions": VERSIONS,
        "baseline_release": BASELINE_RELEASE, "runtime_source_commit": SOURCE_COMMIT,
        "package_run_id": validation["run_id"],
        "tested_checkout_commit": validation["tested_checkout_commit"],
        "distributions": build["artifacts"], "new_full_scientific_gate_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--directory", type=Path, default=Path("reviewed-refresh"))
    args = parser.parse_args()
    regular(args.profile)
    historical = Path(__file__).with_name("reviewed-package-release.json")
    regular(historical)
    baseline_profile_bytes = historical.read_bytes()
    profile = json_bytes(args.profile.read_bytes())
    check_profile(profile, baseline_profile_bytes)
    args.directory.mkdir(exist_ok=False)
    base = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE}/"
    for name, expected in profile["assets"].items():
        data = request_bytes(base + name, expected["bytes"])
        asset(data, expected, name)
        (args.directory / name).write_bytes(data)
    print(json.dumps(verify(args.directory, args.baseline, profile, baseline_profile_bytes)))


if __name__ == "__main__":
    main()
