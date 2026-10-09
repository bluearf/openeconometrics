"""Distribution acceptance binds metadata and complete archives to frozen source."""

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import runpy
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = runpy.run_path(str(ROOT / "scripts/verify_public_package_pair.py"))
CHECK = VERIFIER["check_metadata"]


@pytest.fixture
def project():
    return {
        "name": "openecon", "version": "0.3.19a1", "requires-python": ">=3.11,<3.15",
        "dependencies": ["openecon-charts==0.3.1a1", "Core_Lib>=2,<3"],
        "optional-dependencies": {
            "Research_Tools": ["optional-lib[CSV]>=1; python_version < '3.13'"],
            "app": ["openecon[research-tools]"],
        },
        "charts_version": "0.3.1a1", "readme_bytes": b"Exact description\n",
        "license-files": [], "license_bytes": {},
    }


def archive_metadata(project, wheel, *, requirements=None, extras=None):
    requirements = requirements if requirements is not None else [
        "openecon-charts==0.3.1a1", "core-lib<3,>=2",
        'Optional_Lib[csv]>=1; python_version < "3.13" and extra == "research-tools"',
        'optional-lib[csv]>=1; python_version < "3.13" and extra == "app"',
    ]
    extras = extras if extras is not None else ["research-tools", "app"]
    headers = [
        "Metadata-Version: 2.4", "Name: openecon", "Version: 0.3.19a1",
        "Requires-Python: <3.15,>=3.11", "License-Expression: Apache-2.0",
        "Description-Content-Type: text/markdown",
        *["Requires-Dist: " + value for value in requirements],
        *["Provides-Extra: " + value for value in extras],
    ]
    name = "openecon-0.3.19a1.dist-info/METADATA" if wheel else "PKG-INFO"
    return {name: ("\n".join(headers) + "\n\n").encode() + project["readme_bytes"]}


@pytest.mark.parametrize("wheel", [True, False], ids=["wheel", "sdist"])
def test_full_dependency_and_extra_metadata_accepts_canonical_pep508(project, wheel):
    receipt = CHECK(archive_metadata(project, wheel), project, wheel=wheel)
    assert len(receipt["requires_dist"]) == 4
    assert receipt["provides_extra"] == ["app", "research-tools"]


@pytest.mark.parametrize("wheel", [True, False], ids=["wheel", "sdist"])
@pytest.mark.parametrize("mutation", [
    "missing-root", "changed-bound", "unrequested-extra-root", "changed-platform-marker",
    "changed-extra-marker", "duplicate-root", "missing-provided-extra", "unknown-provided-extra",
    "duplicate-provided-extra",
])
def test_full_metadata_rejects_dependency_or_extra_mismatch(project, wheel, mutation):
    files = archive_metadata(project, wheel)
    name, = files
    metadata = files[name]
    changes = {
        "missing-root": (b"Requires-Dist: core-lib<3,>=2\n", b""),
        "changed-bound": (b"core-lib<3,>=2", b"core-lib<4,>=2"),
        "unrequested-extra-root": (b"Requires-Dist: core-lib<3,>=2\n",
                                  b"Requires-Dist: core-lib<3,>=2\nRequires-Dist: rogue-lib\n"),
        "changed-platform-marker": (b'python_version < "3.13"', b'python_version < "3.14"'),
        "changed-extra-marker": (b'extra == "research-tools"', b'extra == "app"'),
        "duplicate-root": (b"Requires-Dist: core-lib<3,>=2\n",
                           b"Requires-Dist: core-lib<3,>=2\nRequires-Dist: Core_Lib>=2,<3\n"),
        "missing-provided-extra": (b"Provides-Extra: research-tools\n", b""),
        "unknown-provided-extra": (b"Provides-Extra: app\n",
                                   b"Provides-Extra: app\nProvides-Extra: unknown\n"),
        "duplicate-provided-extra": (b"Provides-Extra: app\n",
                                     b"Provides-Extra: app\nProvides-Extra: app\n"),
    }
    original, replacement = changes[mutation]
    assert original in metadata
    files[name] = metadata.replace(original, replacement)
    with pytest.raises(ValueError, match="(?:Full dependency|Declared extras) metadata differs"):
        CHECK(files, project, wheel=wheel)


@pytest.mark.parametrize("wheel", [True, False], ids=["wheel", "sdist"])
@pytest.mark.parametrize("reference,message", [
    ("openecon[unknown]", "Unknown source extra"),
    ("openecon[app]", "Circular source extra"),
    ("openecon[research-tools]; sys_platform == 'win32'", "Self-extra references"),
])
def test_self_extra_scope_cannot_be_unknown_circular_or_silently_unconditional(
    project, wheel, reference, message
):
    project["optional-dependencies"]["app"] = [reference]
    with pytest.raises(ValueError, match=message):
        CHECK(archive_metadata(project, wheel), project, wheel=wheel)


def write_record(files, info):
    record_name = info + "RECORD"
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    for name, data in sorted(files.items()):
        if name != record_name:
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
            writer.writerow((name, "sha256=" + encoded.decode(), len(data)))
    writer.writerow((record_name, "", ""))
    files[record_name] = output.getvalue().encode()


@pytest.fixture
def public_pair(tmp_path):
    """Tiny complete four-archive build without installing/building either package."""
    root, dist = tmp_path / "source", tmp_path / "dist"
    root.mkdir()
    dist.mkdir()
    definitions = (("openecon", "0.3.19a1", ""),
                   ("openecon-charts", "0.3.1a1", "packages/openecon-charts/"))
    source = {"scripts/prepare_public_source.py":
              (ROOT / "scripts/prepare_public_source.py").read_bytes()}
    wheels, sdists = {}, {}
    for name, version, prefix in definitions:
        package = name.replace("-", "_")
        requirements = '["openecon-charts==0.3.1a1"]' if not prefix else "[]"
        config = (f'[project]\nname="{name}"\nversion="{version}"\n'
                  f'requires-python=">=3.11"\nreadme="PYPI-README.md"\n'
                  f'license-files=["LICENSE"]\ndependencies={requirements}\n')
        if not prefix:
            config += '[project.scripts]\nopenecon="openecon.entrypoints:main"\n'
        local = {
            "pyproject.toml": config.encode(), "PYPI-README.md": b"Public description\n",
            "LICENSE": b"Source license\n", "docs/guide.md": b"Source guide\n",
            "tests/test_example.py": b"def test_example(): assert True\n",
            f"src/{package}/__init__.py": f'__version__="{version}"\n'.encode(),
        }
        source.update({prefix + path: data for path, data in local.items()})
        headers = ["Metadata-Version: 2.4", f"Name: {name}", f"Version: {version}",
                   "Requires-Python: >=3.11", "License-Expression: Apache-2.0",
                   "Description-Content-Type: text/markdown"]
        if not prefix:
            headers.append("Requires-Dist: openecon-charts==0.3.1a1")
        metadata = ("\n".join(headers) + "\n\n").encode() + local["PYPI-README.md"]
        info = f"{package}-{version}.dist-info/"
        wheel = {path.removeprefix("src/"): data for path, data in local.items()
                 if path.startswith("src/")}
        wheel.update({
            info + "METADATA": metadata,
            info + "WHEEL": (b"Wheel-Version: 1.0\nGenerator: review-fixture\n"
                             b"Root-Is-Purelib: true\nTag: py3-none-any\n"),
            info + "licenses/LICENSE": local["LICENSE"],
        })
        if not prefix:
            wheel[info + "entry_points.txt"] = (
                b"[console_scripts]\nopenecon = openecon.entrypoints:main\n"
            )
        wheels[package] = wheel
        sdists[package] = {**local, "PKG-INFO": metadata}
    static = {"src/openecon/static/index.html": b"<html/>",
              "src/openecon/static/app.js": b"void 0;",
              "src/openecon/static/app.css": b"body{}"}
    for path, data in {**source, **static}.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    records = [{"path": path, "bytes": len(data), "sha256": VERIFIER["digest"](data),
                "original_sha256": VERIFIER["digest"](data), "transformed": False}
               for path, data in sorted(source.items())]
    manifest = {"source_commit": "f" * 40, "files": records,
                "source_files_sha256": VERIFIER["digest"](json.dumps(records, sort_keys=True).encode()),
                "external_fixture_test_modules": []}
    generated = {"SOURCE-MANIFEST.json": json.dumps(manifest).encode(),
                 "PUBLIC-SOURCE.md": b"Prepared source\n"}
    for path, data in generated.items():
        (root / path).write_bytes(data)
    sdists["openecon"] = {**source, **static, **generated,
                          "PKG-INFO": sdists["openecon"]["PKG-INFO"]}
    wheels["openecon"].update({path.removeprefix("src/"): data
                              for path, data in static.items()})
    for name, version, _ in definitions:
        package = name.replace("-", "_")
        write_record(wheels[package], f"{package}-{version}.dist-info/")

    def verify():
        for name, version, _ in definitions:
            package = name.replace("-", "_")
            basename = f"{package}-{version}"
            with zipfile.ZipFile(dist / (basename + "-py3-none-any.whl"), "w") as archive:
                for path, data in wheels[package].items():
                    archive.writestr(path, data)
            with tarfile.open(dist / (basename + ".tar.gz"), "w:gz") as archive:
                for path, data in sdists[package].items():
                    row = tarfile.TarInfo(basename + "/" + path)
                    row.size = len(data)
                    archive.addfile(row, io.BytesIO(data))
        return VERIFIER["verify"](root, dist, "0.3.19a1", "0.3.1a1")

    return wheels, sdists, verify


def test_complete_public_pair_accepts_source_bound_archives(public_pair):
    _, _, verify = public_pair
    assert verify()["status"] == "passed"


@pytest.mark.parametrize("package,version", [("openecon", "0.3.19a1"),
                                           ("openecon_charts", "0.3.1a1")])
@pytest.mark.parametrize("mutation", [
    "private-json", "top-level-pth", "unknown-dist-info", "data-script", "changed-tag",
    "duplicate-tag", "non-purelib", "changed-production", "changed-license", "wrong-info-root",
    "record-missing", "record-duplicate", "record-wrong-hash", "record-wrong-size",
])
def test_complete_pair_rejects_unbound_wheel_files_and_records(
    public_pair, package, version, mutation
):
    wheels, _, verify = public_pair
    wheel = wheels[package]
    info = f"{package}-{version}.dist-info/"
    if mutation == "private-json":
        wheel["credentials.json"] = b"placeholder private bytes"
    elif mutation == "top-level-pth":
        wheel["unreviewed.pth"] = b"import unreviewed\n"
    elif mutation == "unknown-dist-info":
        wheel[info + "private-note.json"] = b"unrecorded"
    elif mutation == "data-script":
        wheel[f"{package}-{version}.data/scripts/unreviewed"] = b"unrecorded"
    elif mutation == "changed-tag":
        wheel[info + "WHEEL"] = wheel[info + "WHEEL"].replace(b"py3-none-any", b"cp311-none-any")
    elif mutation == "duplicate-tag":
        wheel[info + "WHEEL"] += b"Tag: cp311-none-any\n"
    elif mutation == "non-purelib":
        wheel[info + "WHEEL"] = wheel[info + "WHEEL"].replace(b"true", b"false")
    elif mutation == "changed-production":
        wheel[package + "/__init__.py"] += b"# changed source\n"
    elif mutation == "changed-license":
        wheel[info + "licenses/LICENSE"] += b"changed notice\n"
    elif mutation == "wrong-info-root":
        old = info + "METADATA"
        wheel[f"{package}-9.9.9.dist-info/METADATA"] = wheel.pop(old)
    elif mutation.startswith("record-"):
        lines = wheel[info + "RECORD"].splitlines(keepends=True)
        if mutation == "record-missing":
            lines.pop(0)
        elif mutation == "record-duplicate":
            lines.append(lines[0])
        else:
            row = next(csv.reader([lines[0].decode()]))
            row[1 if mutation == "record-wrong-hash" else 2] = "incorrect"
            output = io.StringIO()
            csv.writer(output).writerow(row)
            lines[0] = output.getvalue().encode()
        wheel[info + "RECORD"] = b"".join(lines)
    if not mutation.startswith("record-"):
        # A self-consistent RECORD must not make unreviewed contents acceptable.
        write_record(wheel, info)
    with pytest.raises(ValueError):
        verify()


@pytest.mark.parametrize("mutation", ["changed-target", "unknown-group", "unknown-command"])
def test_complete_pair_binds_entry_points_to_source(public_pair, mutation):
    wheels, _, verify = public_pair
    wheel = wheels["openecon"]
    info = "openecon-0.3.19a1.dist-info/"
    entry = wheel[info + "entry_points.txt"]
    if mutation == "changed-target":
        entry = entry.replace(b"openecon.entrypoints:main", b"openecon.entrypoints:unreviewed")
    elif mutation == "unknown-group":
        entry += b"\n[unreviewed]\ncommand = openecon.entrypoints:main\n"
    else:
        entry += b"unreviewed = openecon.entrypoints:main\n"
    wheel[info + "entry_points.txt"] = entry
    write_record(wheel, info)
    with pytest.raises(ValueError, match="Wheel entry points differ"):
        verify()


@pytest.mark.parametrize("package", ["openecon", "openecon_charts"])
@pytest.mark.parametrize("mutation", ["private-json", "changed-build-config", "missing-build-config",
                                      "changed-document", "changed-test"])
def test_complete_pair_binds_sdist_build_documents_and_tests(public_pair, package, mutation):
    _, sdists, verify = public_pair
    sdist = sdists[package]
    if mutation == "private-json":
        sdist["scripts/private-note.json" if package == "openecon" else "service-account.json"] = (
            b"placeholder private bytes"
        )
    elif mutation == "missing-build-config":
        del sdist["pyproject.toml"]
    else:
        path = {"changed-build-config": "pyproject.toml", "changed-document": "docs/guide.md",
                "changed-test": "tests/test_example.py"}[mutation]
        sdist[path] += b"# changed source bytes\n"
    with pytest.raises(ValueError, match="sdist"):
        verify()


@pytest.mark.parametrize("mutation", ["changed-manifest", "changed-boundary", "missing-manifest"])
def test_complete_pair_binds_generated_sdk_provenance(public_pair, mutation):
    _, sdists, verify = public_pair
    sdist = sdists["openecon"]
    if mutation == "missing-manifest":
        del sdist["SOURCE-MANIFEST.json"]
    else:
        path = "SOURCE-MANIFEST.json" if mutation == "changed-manifest" else "PUBLIC-SOURCE.md"
        sdist[path] += b"changed provenance\n"
    with pytest.raises(ValueError, match="sdist"):
        verify()
