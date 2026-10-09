"""Actual distribution bytes retain source and saved-state replay fixtures.

The build is isolated in pytest's temporary directory. The evidence exclusion
must leave every selected tracked non-evidence file and current static artifact
unchanged; historical JSON/gzip/tar fixtures remain at their original paths.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tomllib
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = {
    "docs/evidence/survey-inference-2026-10-07/persisted-states.json",
    "docs/evidence/irt-eight-2026-10-07/complete-results.json.gz",
    "docs/evidence/reliability-eight-2026-10-07/complete-results.tar.gz",
    "docs/evidence/missing-data-eight-2026-10-07/persisted-states.json",
    "docs/evidence/survey-regression-2026-10-07/persisted-states.json",
    "docs/evidence/survey-regression-replication-2026-10-07/persisted-states.json",
    "docs/evidence/survey-probit-poisson-replication-2026-10-07/persisted-states.json",
    "docs/evidence/control-functions-eight-2026-10-07/persisted-states.json.gz",
    "docs/evidence/market-131-mgarch/published-reference.json",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def selected(path, includes):
    # Current explicit includes contain literal paths, not wildcard patterns.
    # A slash anchors a tree; a bare filename/directory matches path components.
    return any(
        path == item or path.startswith(item + "/")
        if "/" in item else item in PurePosixPath(path).parts
        for item in includes
    )


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    output = tmp_path_factory.mktemp("source-distributions")
    subprocess.run(
        ["uv", "build", "--all-packages", "--out-dir", str(output)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return output


def test_core_sdist_complete_source_static_docs_and_nine_replay_fixtures(distributions):
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = config["project"]
    target = config["tool"]["hatch"]["build"]["targets"]["sdist"]
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    expected = {
        name for name in tracked
        if name and not name.startswith("docs/evidence/") and selected(name, target["include"])
    }
    expected |= FIXTURES | {".gitignore"}
    expected |= {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src/openecon/static").rglob("*") if path.is_file()
    }
    path = distributions / f"openecon-{project['version']}.tar.gz"
    with tarfile.open(path, "r:gz") as archive:
        members = {
            entry.name.split("/", 1)[1]: entry for entry in archive if entry.isfile()
        }
        assert expected <= members.keys(), sorted(expected - members.keys())
        assert {name for name in members if name.startswith("docs/evidence/")} == FIXTURES
        for name, entry in members.items():
            if name == "PKG-INFO":
                continue
            assert name == ".gitignore" or selected(name, target["include"]), name
            source = ROOT / name
            assert source.is_file() and not source.is_symlink(), name
            actual = archive.extractfile(entry).read()
            assert len(actual) == source.stat().st_size, name
            assert digest(actual) == digest(source.read_bytes()), name
        metadata = archive.extractfile(members["PKG-INFO"]).read()
        assert f"Name: {project['name']}\n".encode() in metadata
        assert f"Version: {project['version']}\n".encode() in metadata


@pytest.mark.parametrize("project_path,package,relative", [
    ("pyproject.toml", "openecon", "src/openecon"),
    ("packages/openecon-charts/pyproject.toml", "openecon_charts",
     "packages/openecon-charts/src/openecon_charts"),
])
def test_actual_wheel_and_sdist_preserve_every_package_python_file(
    distributions, project_path, package, relative,
):
    project = tomllib.loads((ROOT / project_path).read_text())["project"]
    distribution = project["name"].replace("-", "_") + "-" + project["version"]
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", relative], cwd=ROOT,
    ).decode().split("\0")
    sources = {
        Path(name).relative_to(relative).as_posix(): (ROOT / name).read_bytes()
        for name in tracked if name.endswith(".py")
    }
    with zipfile.ZipFile(distributions / (distribution + "-py3-none-any.whl")) as wheel:
        names = {name for name in wheel.namelist()
                 if name.startswith(package + "/") and name.endswith(".py")}
        assert names == {package + "/" + name for name in sources}
        for name, raw in sources.items():
            assert wheel.read(package + "/" + name) == raw
    with tarfile.open(distributions / (distribution + ".tar.gz"), "r:gz") as sdist:
        prefix = distribution + "/src/" + package + "/"
        names = {entry.name for entry in sdist if entry.isfile()
                 and entry.name.startswith(prefix) and entry.name.endswith(".py")}
        assert names == {prefix + name for name in sources}
        for name, raw in sources.items():
            assert sdist.extractfile(prefix + name).read() == raw
