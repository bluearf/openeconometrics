"""Actual isolated venv guards with small source-bound installed-file fixtures."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import py_compile
import runpy
import subprocess
import venv

import pytest

HELPER = Path(__file__).resolve().parent / "verify_installed_package_pair.py"
VERIFY = runpy.run_path(str(HELPER))["verify"]
VERSIONS = {"openecon": "0.3.19a1", "openecon-charts": "0.3.1a1"}


@pytest.fixture
def installed(tmp_path):
    environment = tmp_path / "env"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / "bin/python"
    site = Path(subprocess.check_output(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    files = {
        "openecon/__init__.py": b"from . import support\n__version__ = '0.3.19a1'\n",
        "openecon/support.py": b"VALUE = 2\n",
        "openecon/static/index.html": b"<html>Source-bound fixture</html>\n",
        "openecon_charts/__init__.py": b"__version__ = '0.3.1a1'\n",
        "openecon_charts/assets/font.woff2": b"wOF2Fixture font bytes",
    }
    for relative, data in files.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    infos = {}
    for name, version in VERSIONS.items():
        module = name.replace("-", "_")
        info = site / f"{module}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\nFixture\n"
        )
        recorded = [relative for relative in files if relative.startswith(module + "/")]
        recorded += [str((info / "METADATA").relative_to(site)),
                     str((info / "RECORD").relative_to(site))]
        (info / "RECORD").write_text("".join(f"{relative},,\n" for relative in recorded))
        infos[name] = info
    binding = {"status": "passed", "source_commit": "f" * 40,
               "source_files_sha256": "e" * 64, "versions": VERSIONS,
               "production_files": {
                   name: {relative: hashlib.sha256(data).hexdigest()
                          for relative, data in files.items()
                          if relative.startswith(name.replace("-", "_") + "/")}
                   for name in VERSIONS}}
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(binding))
    return {"python": python, "site": site, "workspace": workspace,
            "binding": binding, "provenance": provenance, "infos": infos}


def check(installed, packages=("openecon", "openecon-charts")):
    return VERIFY(installed["python"], installed["provenance"], list(packages), installed["workspace"])


def change_bound_init(installed, code):
    path = installed["site"] / "openecon/__init__.py"
    path.write_text(code)
    installed["binding"]["production_files"]["openecon"]["openecon/__init__.py"] = (
        hashlib.sha256(path.read_bytes()).hexdigest()
    )
    installed["provenance"].write_text(json.dumps(installed["binding"]))


@pytest.mark.parametrize("packages", [("openecon", "openecon-charts"), ("openecon-charts",)])
def test_exact_installed_production_and_imported_origins_pass(installed, packages):
    record = check(installed, packages)
    assert record["status"] == "passed" and record["isolated"]
    assert record["workspace_paths_absent"] and record["editable_installations_absent"]
    assert set(record["packages"]) == set(packages)
    for name in packages:
        assert record["packages"][name]["production_files"] == installed["binding"]["production_files"][name]
        assert record["packages"][name]["imported_modules"]


@pytest.mark.parametrize("package", ["openecon", "openecon-charts"])
@pytest.mark.parametrize("mutation", ["changed", "missing", "unexpected", "symlink-file", "symlink-directory"])
def test_installed_file_drift_is_refused(installed, package, mutation):
    root = installed["site"] / package.replace("-", "_")
    path = root / ("static/index.html" if package == "openecon" else "assets/font.woff2")
    if mutation == "changed":
        path.write_bytes(path.read_bytes() + b"changed byte")
    elif mutation == "missing":
        path.unlink()
    elif mutation == "unexpected":
        (root / "unexpected.py").write_text("UNREVIEWED = True\n")
    elif mutation == "symlink-file":
        target = installed["workspace"] / "source-copy"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    else:
        path.parent.rename(path.parent.with_name(path.parent.name + "-foreign"))
        path.parent.symlink_to(path.parent.with_name(path.parent.name + "-foreign"), target_is_directory=True)
    with pytest.raises(ValueError, match="production (?:file set differs|bytes differ)|symlink"):
        check(installed)


@pytest.mark.parametrize("mutation", ["editable-metadata", "recorded-pth", "workspace-path", "workspace-module", "foreign-package", "foreign-metadata"])
def test_editable_source_and_foreign_installations_are_refused(installed, mutation, tmp_path):
    info = installed["infos"]["openecon"]
    site = installed["site"]
    if mutation == "editable-metadata":
        (info / "direct_url.json").write_text(json.dumps({"dir_info": {"editable": True}}))
    elif mutation == "recorded-pth":
        (site / "source-editable.pth").write_text("# Editable-path file is not a production wheel file\n")
        with (info / "RECORD").open("a") as handle:
            handle.write("source-editable.pth,,\n")
    elif mutation == "workspace-path":
        (site / "source.pth").write_text(str(installed["workspace"]) + "\n")
    elif mutation == "workspace-module":
        (site / "source.pth").write_text(
            "import sys, types; m = types.ModuleType('source_fixture'); "
            f"m.__file__ = {str(installed['workspace'] / 'fixture.py')!r}; "
            "sys.modules['source_fixture'] = m\n"
        )
    else:
        foreign = tmp_path / "foreign-site"
        foreign.mkdir()
        source = site / "openecon" if mutation == "foreign-package" else info
        source.rename(foreign / source.name)
        (site / "foreign.pth").write_text(str(foreign) + "\n")
    with pytest.raises(ValueError, match="editable|Source workspace|outside the venv"):
        check(installed)


def test_foreign_interpreter_prefix_is_refused(installed):
    request = {"binding": installed["binding"], "provenance_sha256": "a" * 64,
               "packages": ["openecon"], "prefix": str(installed["workspace"]),
               "workspace": str(installed["workspace"])}
    result = subprocess.run(
        [str(installed["python"]), "-I", str(HELPER), "--child"],
        input=json.dumps(request), text=True, capture_output=True,
    )
    assert result.returncode != 0 and "foreign prefix" in result.stderr


@pytest.mark.parametrize("mutation", ["path", "module"])
def test_unrelated_foreign_source_paths_and_modules_are_refused(installed, mutation, tmp_path):
    foreign = tmp_path / "foreign-source"
    foreign.mkdir()
    hook = installed["site"] / "foreign.pth"
    if mutation == "path":
        hook.write_text(str(foreign) + "\n")
    else:
        hook.write_text(
            "import sys, types; m = types.ModuleType('foreign_fixture'); "
            f"m.__file__ = {str(foreign / 'fixture.py')!r}; "
            "sys.modules['foreign_fixture'] = m\n"
        )
    with pytest.raises(ValueError, match="Foreign .* outside the venv/runtime"):
        check(installed)


def test_installed_distribution_version_drift_is_refused(installed):
    metadata = installed["infos"]["openecon"] / "METADATA"
    metadata.write_text(metadata.read_text().replace("Version: 0.3.19a1", "Version: 0.3.20a1"))
    with pytest.raises(ValueError, match="distribution version differs"):
        check(installed)


@pytest.mark.parametrize("escape", [False, True])
def test_metadata_record_parent_alias_and_escape_are_refused(installed, tmp_path, escape):
    info = installed["infos"]["openecon"]
    relative = (info / "METADATA").relative_to(installed["site"]).as_posix()
    if escape:
        foreign = tmp_path / "foreign-site" / info.name / "METADATA"
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes((info / "METADATA").read_bytes())
        aliased = os.path.relpath(foreign, installed["site"])
    else:
        aliased = info.name + "/../" + relative
    record = info / "RECORD"
    record.write_text(record.read_text().replace(relative + ",,", aliased + ",,"))
    with pytest.raises(ValueError, match="parent traversal"):
        check(installed)


@pytest.mark.parametrize("mutation", ["version", "workspace-path", "foreign-submodule", "foreign-search-path", "changed-after-import"])
def test_source_verified_imports_still_require_bound_origins(installed, mutation, tmp_path):
    code = "__version__ = '0.3.19a1'\n"
    if mutation == "version":
        code = "__version__ = '0.3.20a1'\n"
    elif mutation == "workspace-path":
        code += f"import sys\nsys.path.insert(0, {str(installed['workspace'])!r})\n"
    elif mutation in {"foreign-submodule", "foreign-search-path"}:
        foreign = tmp_path / "foreign-module"
        foreign.mkdir()
        (foreign / "external.py").write_text("VALUE = 4\n")
        code += f"__path__.append({str(foreign)!r})\n"
        if mutation == "foreign-submodule":
            code += "from . import external\n"
    else:
        code += "with open(__file__, 'a') as handle:\n    handle.write('# changed while importing\\n')\n"
    change_bound_init(installed, code)
    with pytest.raises(ValueError, match="version differs|Source workspace|outside the venv|not bound"):
        check(installed)


def test_normal_bytecode_cache_is_allowed_but_symlink_cache_is_refused(installed, tmp_path):
    cache = installed["site"] / "openecon/__pycache__"
    py_compile.compile(str(installed["site"] / "openecon/support.py"), doraise=True)
    check(installed)
    for child in cache.iterdir():
        child.unlink()
    cache.rmdir()
    other = tmp_path / "foreign-cache"
    other.mkdir()
    cache.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        check(installed)


@pytest.mark.parametrize("filename", ["unreviewed.py", "foreign.cpython-313.pyc", "foreign.pyc"])
def test_only_normal_source_bound_bytecode_caches_are_ignored(installed, filename):
    cache = installed["site"] / "openecon/__pycache__"
    cache.mkdir()
    (cache / filename).write_bytes(b"Unbound extra cache contents")
    with pytest.raises(ValueError, match="cache file|bytecode cache"):
        check(installed)
