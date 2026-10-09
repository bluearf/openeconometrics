"""Verify immutable reviewed release bytes before protected Trusted Publishing."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy
import shutil
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(directory, profile):
    require(profile["schema_version"] == 3, "Unknown review profile")
    require(
        profile["versions"] == {"sdk": "0.3.19a1", "charts": "0.3.1a1"},
        "Package version tuple differs",
    )
    require(profile["release"] == "python-v0.3.19a1", "Release identity differs")
    require(profile["repository"] == "bluearf/openeconometrics", "Repository differs")
    assets = profile["assets"]
    require(
        {p.name for p in directory.iterdir()} == set(assets), "Missing or foreign reviewed assets"
    )
    for name, expected in assets.items():
        path = directory / name
        require(
            PurePosixPath(name).name == name and not path.is_symlink() and path.is_file(),
            "Unsafe reviewed file",
        )
        data = path.read_bytes()
        require(
            len(data) == expected["bytes"] and sha(data) == expected["sha256"],
            f"Reviewed asset bytes differ: {name}",
        )
    provenance = json.loads((directory / "RELEASE-PROVENANCE.json").read_bytes())
    require(provenance == profile["release_provenance"], "Complete reviewed provenance differs")
    distribution_names = {
        "openecon-0.3.19a1-py3-none-any.whl",
        "openecon-0.3.19a1.tar.gz",
        "openecon_charts-0.3.1a1-py3-none-any.whl",
        "openecon_charts-0.3.1a1.tar.gz",
    }
    build = json.loads((directory / "PACKAGE-PAIR-PROVENANCE.json").read_bytes())
    require(
        build["status"] == "passed"
        and build["artifacts"] == {n: assets[n] for n in distribution_names},
        "Fixed build distributions differ",
    )
    require(
        build["source_commit"] == provenance["private_package_source_commit"],
        "Package source pin differs",
    )
    require(
        build["source_files_sha256"] == provenance["source_files_sha256"],
        "Source inventory differs",
    )
    require(
        provenance["linux_fixed_pair"]["run_id"] == "37894416408",
        "Accepted fixed build run differs",
    )
    for minor in ("3.11", "3.13"):
        with zipfile.ZipFile(directory / f"linux-package-{minor}-evidence.zip") as evidence:
            require(
                sum(i.file_size for i in evidence.infolist()) < 4 * 1024 * 1024,
                "Excessive package evidence",
            )
            execution = json.loads(evidence.read("_temp/pair-execution.json"))
            require(
                execution["status"] == "passed"
                and execution["run_id"] == "37894416408"
                and execution["run_attempt"] == "1"
                and execution["artifacts"] == build["artifacts"],
                "Accepted Linux execution differs",
            )
            require(
                execution["public_checkout_commit"]
                == provenance["linux_fixed_pair"]["tested_public_merge_commit"],
                "Tested public tree differs",
            )
            for suite, count in [
                ("installed-inventory-guards", 33),
                ("sdk-regressions", 290),
                ("network-export-wheel", 53),
                ("network-export-sdist", 53),
            ]:
                tree = ET.fromstring(evidence.read(f"_temp/{suite}.xml"))
                require(
                    len(list(tree.iter("testcase"))) == count
                    and not any(list(tree.iter(tag)) for tag in ("failure", "error", "skipped")),
                    "Failed, missing or skipped accepted Linux cases",
                )
    mac = json.loads((directory / "MACOS-PACKAGE-ACCEPTANCE.json").read_bytes())
    require(
        mac["status"] == "passed"
        and {r["environment"] for r in mac["records"]}
        == {"sdk-wheel", "sdk-sdist", "charts-wheel", "charts-sdist"}
        and len(mac["records"]) == 4,
        "Mac installation scope differs",
    )
    require(
        all(
            r["isolated"]
            and r["workspace_paths_absent"]
            and r["editable_installations_absent"]
            and r["source_commit"] == build["source_commit"]
            and r["source_files_sha256"] == build["source_files_sha256"]
            and r["package_smoke"]["status"] == "passed"
            and all(
                x["status"] == "passed" and x["isolated_installed_examples"] and x["offline"]
                for x in r["package_readmes"]
            )
            for r in mac["records"]
        ),
        "Mac installed proof differs",
    )
    require(
        len(mac["sandboxed_browser_export_cases"]) == 2
        and all(
            r["status"] == "passed"
            and r["tests"] == 53
            and r["failures"] == r["errors"] == r["skipped"] == 0
            and r["temporary_root_fresh_before_launch"]
            and r["temporary_root_removed_after_process_exit"]
            for r in mac["sandboxed_browser_export_cases"]
        ),
        "Mac browser proof differs",
    )
    require(
        not mac["native_desktop_acceptance"] and not mac["cuda_acceptance"],
        "Package proof broadened to native/CUDA",
    )
    current = provenance["private_current_source_validation"]
    require(
        current["status"] == "passed"
        and current["job_result"] == "success"
        and current["timeout_seconds"] == 900,
        "Current private scientific gate has not passed",
    )
    require(
        current["production_bridge"]["complete_changed_paths"]
        == [
            ".github/workflows/ci.yml",
            "scripts/run_parallel_sdk_groups.py",
            "scripts/verify_merge_candidate.py",
            "scripts/verify_parallel_merge_gate.py",
            "tests/test_merge_gate.py",
            "tests/test_parallel_merge_gate.py",
            "tests/test_parallel_sdk_groups.py",
        ]
        and current["production_bridge"][
            "package_web_native_and_numerical_production_byte_identical"
        ]
        is True,
        "Unreviewed current-source production change",
    )
    require(
        all(
            r["tests"] == 10280 and r["failures"] == r["errors"] == r["skipped"] == 0
            for r in current["complete_sdk_unions"]
        )
        and len(current["complete_sdk_unions"]) == 2,
        "Complete current source scope differs",
    )
    require(
        len(current["sdk_cohorts"]) == 2
        and all(0 < r["seconds"] <= 900 for r in current["sdk_cohorts"]),
        "Whole cohort deadline differs",
    )
    source = directory / "_source"
    require(not source.exists(), "Source verification destination must be fresh")
    source.mkdir()
    with tarfile.open(directory / "openeconometrics-source-a8eb8799-public.tar.gz") as archive:
        members = archive.getmembers()
        require(
            len(members) == 2322 and sum(m.size for m in members) < 80 * 1024 * 1024,
            "Source archive scope differs",
        )
        names = set()
        for entry in members:
            path = PurePosixPath(entry.name)
            require(
                entry.isfile()
                and path.as_posix() == entry.name
                and not path.is_absolute()
                and ".." not in path.parts
                and path.parts[0] == "openeconometrics"
                and entry.name not in names
                and entry.mode in (0o644, 0o755)
                and entry.uid == entry.gid == entry.mtime == 0,
                "Unsafe or noncanonical source member",
            )
            names.add(entry.name)
            target = source.joinpath(*path.parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(entry).read())
            target.chmod(entry.mode)
    require(
        sha((source / "SOURCE-MANIFEST.json").read_bytes()) == build["source_manifest_sha256"],
        "Published manifest differs",
    )
    with zipfile.ZipFile(directory / "openecon-0.3.19a1-py3-none-any.whl") as wheel:
        for name in wheel.namelist():
            if name.startswith("openecon/static/") and not name.endswith("/"):
                path = PurePosixPath(name)
                require(
                    path.as_posix() == name and ".." not in path.parts, "Unsafe frontend member"
                )
                target = source / "src" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(wheel.read(name))
    dist = directory / "_dist"
    dist.mkdir()
    for name in distribution_names:
        shutil.copyfile(directory / name, dist / name)
    checker = runpy.run_path(str(source / "scripts/verify_public_package_pair.py"))["verify"]
    require(
        checker(source, dist, "0.3.19a1", "0.3.1a1", build["source_commit"]) == build,
        "Complete source, package contents, dependency metadata or notices differ",
    )
    for project, prefix in [("openecon", "openecon-"), ("openecon-charts", "openecon_charts-")]:
        destination = Path(project) / "dist"
        require(not destination.exists(), "Publishing destination must be fresh")
        destination.mkdir(parents=True)
        for name in distribution_names:
            if name.startswith(prefix):
                shutil.copyfile(directory / name, destination / name)
    return {
        "status": "passed",
        "distributions": build["artifacts"],
        "source_commit": build["source_commit"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--directory", type=Path, default=Path("reviewed"))
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    args.directory.mkdir(exist_ok=False)
    base = "https://github.com/bluearf/openeconometrics/releases/download/python-v0.3.19a1/"
    for name, expected in profile["assets"].items():
        require(
            PurePosixPath(name).name == name
            and type(expected["bytes"]) is int
            and 0 < expected["bytes"] < 32 * 1024 * 1024,
            "Unsafe reviewed asset bound",
        )
        with urllib.request.urlopen(base + name, timeout=60) as response:
            data = response.read(expected["bytes"] + 1)
        require(
            len(data) == expected["bytes"] and sha(data) == expected["sha256"],
            f"Public release bytes differ: {name}",
        )
        (args.directory / name).write_bytes(data)
    print(json.dumps(verify(args.directory, profile)))


if __name__ == "__main__":
    main()
