"""Bind four public SDK/chart distributions to their frozen source inventory.

This checks packaging provenance and bytes, not scientific method acceptance or
successful publication. Run after building the frontend and both packages from
the prepared public source snapshot, never from the private development tree.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import configparser
import csv
import email
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import runpy
import tarfile
import tomllib
import zipfile

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def relative_name(name: str) -> str:
    path = PurePosixPath(name)
    require(
        bool(name) and not path.is_absolute() and ".." not in path.parts,
        f"Unsafe archive/source path: {name}",
    )
    require(path.as_posix() == name, f"Noncanonical archive/source path: {name}")
    return name


def wheel_files(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        require(len(names) == len(set(names)), f"Duplicate wheel entries: {path.name}")
        return {relative_name(name): archive.read(name) for name in names}


def sdist_files(path: Path, prefix: str) -> dict[str, bytes]:
    files = {}
    with tarfile.open(path) as archive:
        for entry in archive:
            name = relative_name(entry.name)
            if name == prefix and entry.isdir():
                continue
            require(name.startswith(prefix + "/"), f"Unexpected sdist root: {name}")
            require(entry.isdir() or entry.isfile(), f"Nonregular sdist entry: {name}")
            if entry.isfile():
                relative = name.removeprefix(prefix + "/")
                require(relative not in files, f"Duplicate sdist entry: {relative}")
                handle = archive.extractfile(entry)
                require(handle is not None, f"Missing sdist bytes: {relative}")
                files[relative] = handle.read()
    return files


def match_package(files: dict[str, bytes], expected: dict[str, bytes], prefix: str) -> None:
    actual = {name: data for name, data in files.items() if name.startswith(prefix)}
    require(
        set(actual) == set(expected),
        f"Production file set differs for {prefix}: "
        f"missing={sorted(set(expected) - set(actual))}, "
        f"extra={sorted(set(actual) - set(expected))}",
    )
    for name, data in expected.items():
        require(actual[name] == data, f"Production bytes differ: {name}")


def check_wheel_layout(files: dict[str, bytes], project: dict, production: dict[str, bytes]) -> None:
    """Admit only source-bound production and the declared wheel metadata."""
    info = f"{project['name'].replace('-', '_')}-{project['version']}.dist-info/"
    entry_points = dict(project.get("entry-points", {}))
    require(
        not {"console_scripts", "gui_scripts"}.intersection(entry_points),
        "Reserved entry-point groups must use project scripts/gui-scripts",
    )
    for source_group, wheel_group in (("scripts", "console_scripts"), ("gui-scripts", "gui_scripts")):
        if project.get(source_group):
            entry_points[wheel_group] = project[source_group]
    allowed = set(production) | {
        info + name for name in ("METADATA", "WHEEL", "RECORD")
    } | {info + "licenses/" + name for name in project["license-files"]}
    if entry_points:
        allowed.add(info + "entry_points.txt")
    require(
        set(files) == allowed,
        f"Wheel file set differs: {project['name']}; "
        f"missing={sorted(allowed - set(files))}; extra={sorted(set(files) - allowed)}",
    )
    wheel = email.message_from_bytes(files[info + "WHEEL"])
    for header, value in (
        ("Wheel-Version", "1.0"), ("Root-Is-Purelib", "true"), ("Tag", "py3-none-any")
    ):
        require(wheel.get_all(header, []) == [value], f"Wheel {header} differs")
    require(
        set(wheel.keys()) <= {"Wheel-Version", "Generator", "Root-Is-Purelib", "Tag"}
        and len(wheel.get_all("Generator", [])) == 1 and bool(wheel["Generator"])
        and not wheel.get_payload(),
        "Unexpected wheel metadata header or payload",
    )
    if entry_points:
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.optionxform = str
        parser.read_string(files[info + "entry_points.txt"].decode("utf-8"))
        actual = {group: dict(parser.items(group)) for group in parser.sections()}
        require(not parser.defaults() and actual == entry_points, "Wheel entry points differ")

    record_name = info + "RECORD"
    rows = list(csv.reader(io.StringIO(files[record_name].decode("utf-8")), strict=True))
    require(all(len(row) == 3 for row in rows), "Invalid wheel RECORD row")
    names = [relative_name(row[0]) for row in rows]
    require(
        len(names) == len(set(names)) and set(names) == set(files),
        "Wheel RECORD file set differs",
    )
    for name, hash_value, size in rows:
        if name == record_name:
            require(not hash_value and not size, "Wheel RECORD must leave its own hash/size empty")
        else:
            expected_hash = "sha256=" + base64.urlsafe_b64encode(
                hashlib.sha256(files[name]).digest()
            ).rstrip(b"=").decode("ascii")
            require(
                hash_value == expected_hash and size == str(len(files[name])),
                f"Wheel RECORD bytes differ: {name}",
            )


def check_sdist_source(
    files: dict[str, bytes], source: dict[str, bytes], *, prefix: str = "",
    generated: dict[str, bytes] | None = None,
) -> None:
    """Bind every build, document and source file to the frozen public inventory."""
    generated = generated or {}
    require("pyproject.toml" in files, "sdist lost its source build configuration")
    for name, data in files.items():
        if name == "PKG-INFO":
            continue  # Core metadata is checked separately against that source configuration.
        if name in generated:
            require(data == generated[name], f"sdist generated provenance differs: {name}")
            continue
        source_name = prefix + name
        require(source_name in source, f"Unrecorded sdist file: {source_name}")
        require(data == source[source_name], f"sdist source bytes differ: {source_name}")
    require(set(generated) <= set(files), "sdist lost generated source provenance")


def canonical_requirement(value: str) -> str:
    requirement = Requirement(value)
    requirement.name = canonicalize_name(requirement.name)
    requirement.extras = {canonicalize_name(extra) for extra in requirement.extras}
    return str(requirement)


def check_dependency_metadata(metadata, project: dict) -> dict:
    """Compare declarative PEP 508 roots, never evaluate host-specific markers."""
    expected = sorted({canonical_requirement(value) for value in project.get("dependencies", [])})
    source_extras = project.get("optional-dependencies", {})
    expected_extras = [canonicalize_name(extra) for extra in source_extras]
    require(len(expected_extras) == len(set(expected_extras)), "Duplicate normalized source extra")
    extras = {canonicalize_name(extra): values for extra, values in source_extras.items()}

    def expanded(extra: str, parents: tuple = ()) -> set[str]:
        # Hatch expands this project's app/desktop/cloud self-extra references
        # into the inherited leaf requirements, preserving leaf platform markers.
        require(extra in extras, f"Unknown source extra: {extra}")
        require(extra not in parents, f"Circular source extra: {extra}")
        values = set()
        for value in extras[extra]:
            requirement = Requirement(value)
            if canonicalize_name(requirement.name) == canonicalize_name(project["name"]):
                require(
                    not requirement.url and not requirement.specifier and not requirement.marker,
                    "Self-extra references must not discard a version, URL or platform marker",
                )
                for inherited in requirement.extras:
                    values.update(expanded(canonicalize_name(inherited), (*parents, extra)))
            else:
                values.add(canonical_requirement(value))
        return values

    for extra in extras:
        extra_marker = Marker("extra == " + json.dumps(extra))
        for value in expanded(extra):
            requirement = Requirement(value)
            requirement.marker = (
                Marker(f"({requirement.marker}) and ({extra_marker})")
                if requirement.marker else extra_marker
            )
            expected.append(canonical_requirement(str(requirement)))
    actual = [canonical_requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    require(
        Counter(actual) == Counter(expected),
        f"Full dependency metadata differs: {project['name']}; "
        f"missing={list((Counter(expected) - Counter(actual)).elements())}; "
        f"extra={list((Counter(actual) - Counter(expected)).elements())}",
    )
    actual_extras = [canonicalize_name(extra) for extra in metadata.get_all("Provides-Extra", [])]
    require(
        Counter(actual_extras) == Counter(expected_extras),
        f"Declared extras metadata differs: {project['name']}",
    )
    return {"requires_dist": sorted(expected), "provides_extra": sorted(expected_extras)}


def check_metadata(files: dict[str, bytes], project: dict, *, wheel: bool) -> dict:
    candidates = [name for name in files if name.endswith(".dist-info/METADATA")]
    if not wheel:
        candidates = ["PKG-INFO"]
    require(len(candidates) == 1, "Distribution metadata must be unique")
    metadata_path = candidates[0]
    metadata = email.message_from_bytes(files[metadata_path])
    require(
        metadata["Name"] == project["name"] and metadata["Version"] == project["version"],
        f"Distribution identity differs: {project['name']}",
    )
    require(metadata["License-Expression"] == "Apache-2.0", "License expression differs")
    require(
        sorted(metadata["Requires-Python"].replace(" ", "").split(","))
        == sorted(project["requires-python"].replace(" ", "").split(",")),
        "Python range differs",
    )
    require(metadata["Description-Content-Type"] == "text/markdown", "Description is not Markdown")
    require(
        metadata.get_payload().encode("utf-8") == project["readme_bytes"],
        f"Package description differs: {project['name']}",
    )
    dependencies = metadata.get_all("Requires-Dist", [])
    dependency_receipt = check_dependency_metadata(metadata, project)
    if project["name"] == "openecon":
        require(
            "openecon-charts==" + project["charts_version"] in dependencies,
            "SDK does not pin the reviewed charts version",
        )
    else:
        require(not dependencies, "Standalone charts acquired a Python dependency")
    for name in project["license-files"]:
        archive_name = (
            metadata_path.removesuffix("METADATA") + "licenses/" + name if wheel else name
        )
        require(
            files.get(archive_name) == project["license_bytes"][name],
            f"License notice differs: {project['name']}/{name}",
        )
    return dependency_receipt


def verify(
    root: Path, dist: Path, sdk_version: str, charts_version: str, source_commit: str | None = None
) -> dict:
    manifest_bytes = (root / "SOURCE-MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    commit = manifest["source_commit"]
    require(
        isinstance(commit, str) and bool(re.fullmatch(r"[0-9a-f]{40}", commit)),
        "A frozen source commit is required; derivative/unfrozen manifests are not accepted",
    )
    require(source_commit is None or commit == source_commit, "Frozen source commit differs")
    records = manifest["files"]
    require(
        digest(json.dumps(records, sort_keys=True).encode()) == manifest["source_files_sha256"],
        "Source inventory hash differs",
    )
    included = runpy.run_path(str(root / "scripts/prepare_public_source.py"))["included"]
    source = {}
    for row in records:
        name = relative_name(row["path"])
        require(
            name not in source and included(name), f"Duplicate/excluded source inventory: {name}"
        )
        require(not (root / name).is_symlink(), f"Symlink in source inventory: {name}")
        data = (root / name).read_bytes()
        require(
            len(data) == row["bytes"] and digest(data) == row["sha256"],
            f"Source inventory bytes differ: {name}",
        )
        source[name] = data

    definitions = (
        ("openecon", root, sdk_version, "src/openecon/", "openecon/"),
        (
            "openecon-charts",
            root / "packages/openecon-charts",
            charts_version,
            "packages/openecon-charts/src/openecon_charts/",
            "openecon_charts/",
        ),
    )
    expected_names = {
        f"{name.replace('-', '_')}-{version}{suffix}"
        for name, _, version, _, _ in definitions
        for suffix in ("-py3-none-any.whl", ".tar.gz")
    }
    actual_names = {path.name for path in dist.iterdir() if path.is_file()}
    require(
        actual_names == expected_names,
        f"Expected exactly four distributions: {sorted(actual_names)}",
    )

    static = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "src/openecon/static").rglob("*")
        if path.is_file()
    }
    require(
        "src/openecon/static/index.html" in static
        and any(name.endswith(".js") for name in static)
        and any(name.endswith(".css") for name in static),
        "Built frontend assets are required",
    )
    source.update(static)
    production_inventories = {}
    metadata_receipts = {}
    for name, package_root, version, source_prefix, wheel_prefix in definitions:
        project = tomllib.loads((package_root / "pyproject.toml").read_text())["project"]
        require(
            project["name"] == name and project["version"] == version,
            "Source version tuple differs",
        )
        readme = package_root / project["readme"]
        project["readme_bytes"] = readme.read_bytes()
        links = re.findall(r"(?<!!)\[[^\]]+\]\(([^\s)]+)\)", readme.read_text())
        require(
            all(link.startswith("https://") for link in links),
            "Package descriptions need absolute HTTPS links",
        )
        project["license_bytes"] = {
            notice: (package_root / notice).read_bytes() for notice in project["license-files"]
        }
        project["charts_version"] = charts_version
        expected_source = {
            path: data for path, data in source.items() if path.startswith(source_prefix)
        }
        disk_paths = {
            path.relative_to(root).as_posix()
            for path in (root / source_prefix).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        require(disk_paths == set(expected_source), f"Unrecorded production files: {name}")
        for row in records:
            if row["path"].startswith(source_prefix):
                require(
                    not row["transformed"] and row["original_sha256"] == row["sha256"],
                    f"Production source was transformed: {row['path']}",
                )
        expected_wheel = {
            wheel_prefix + path.removeprefix(source_prefix): data
            for path, data in expected_source.items()
        }
        basename = f"{name.replace('-', '_')}-{version}"
        wheel = wheel_files(dist / (basename + "-py3-none-any.whl"))
        sdist = sdist_files(dist / (basename + ".tar.gz"), basename)
        match_package(wheel, expected_wheel, wheel_prefix)
        check_wheel_layout(wheel, project, expected_wheel)
        if name == "openecon":
            match_package(sdist, expected_source, source_prefix)
            check_sdist_source(
                sdist, source,
                generated={
                    path: (root / path).read_bytes()
                    for path in ("SOURCE-MANIFEST.json", "PUBLIC-SOURCE.md")
                },
            )
        else:
            expected_chart_sdist = {"src/" + path: data for path, data in expected_wheel.items()}
            match_package(sdist, expected_chart_sdist, "src/openecon_charts/")
            check_sdist_source(sdist, source, prefix="packages/openecon-charts/")
        require(
            sdist.get(project["readme"]) == project["readme_bytes"],
            "sdist lost package description",
        )
        metadata_receipts[name] = {
            "wheel": check_metadata(wheel, project, wheel=True),
            "sdist": check_metadata(sdist, project, wheel=False),
        }
        production_inventories[name] = {
            path: digest(data) for path, data in sorted(expected_wheel.items())
        }

    return {
        "status": "passed",
        "scope": "Frozen source and distribution byte verification; no publication or method parity claim.",
        "source_commit": commit,
        "source_manifest_sha256": digest(manifest_bytes),
        "source_files_sha256": manifest["source_files_sha256"],
        "versions": {"openecon": sdk_version, "openecon-charts": charts_version},
        "artifacts": {
            name: {
                "bytes": (dist / name).stat().st_size,
                "sha256": digest((dist / name).read_bytes()),
            }
            for name in sorted(expected_names)
        },
        "production_files": production_inventories,
        "metadata": metadata_receipts,
        "generated_frontend": {name: digest(data) for name, data in sorted(static.items())},
        "external_fixture_test_modules": manifest["external_fixture_test_modules"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-version", default="0.3.19a1")
    parser.add_argument("--charts-version", default="0.3.1a1")
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    record = verify(
        args.root.resolve(),
        args.dist.resolve(),
        args.sdk_version,
        args.charts_version,
        args.source_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in record.items()
                if key
                not in {"production_files", "generated_frontend", "external_fixture_test_modules"}
            }
        )
    )


if __name__ == "__main__":
    main()
