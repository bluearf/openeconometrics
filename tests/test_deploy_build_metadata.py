"""Required package descriptions, notices and frontend tools reach Docker stages."""

import importlib.util
import json
from pathlib import Path
import shlex
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("versions", [["2.14.0"], ["2.14.0", "2.14.0+cpu"]])
def test_windows_runtime_export_retains_locked_markers_and_cpu_version(
    tmp_path, monkeypatch, capsys, versions,
):
    spec = importlib.util.spec_from_file_location(
        "runtime_export", ROOT / "desktop/scripts/export_locked_runtime.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "uv.lock").write_text("\n".join(
        f'[[package]]\nname = "torch"\nversion = "{version}"\n' for version in versions
    ))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "DESKTOP", tmp_path / "desktop")
    calls = []

    def locked_export(command, *, cwd, check):
        calls.append(command)
        assert cwd == tmp_path and check is True
        assert command[2:6] == ["uv", "export", "--locked", "--no-dev"]
        assert command[command.index("--extra") + 1] == "desktop"
        assert command[command.index("--no-emit-package") + 1] == "torch"
        output = Path(command[command.index("--output-file") + 1])
        output.write_text('colorama==0.4.6 ; sys_platform == "win32"\n')

    monkeypatch.setattr(module.subprocess, "run", locked_export)
    module.main()
    result = json.loads(capsys.readouterr().out)
    assert result["torch_version"] == "2.14.0" and len(calls) == 1
    assert Path(result["requirements"]).read_text() == (
        'colorama==0.4.6 ; sys_platform == "win32"\n'
    )


@pytest.mark.parametrize("versions", [[], ["2.14.0", "2.15.0+cpu"]])
def test_windows_runtime_export_rejects_ambiguous_lock_before_export(
    tmp_path, monkeypatch, versions,
):
    spec = importlib.util.spec_from_file_location(
        "runtime_export_invalid", ROOT / "desktop/scripts/export_locked_runtime.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "uv.lock").write_text("package = []\n" if not versions else "\n".join(
        f'[[package]]\nname = "torch"\nversion = "{version}"\n' for version in versions
    ))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "DESKTOP", tmp_path / "desktop")

    def unexpected_export(*args, **kwargs):
        pytest.fail("An ambiguous lock must not start dependency export")

    monkeypatch.setattr(module.subprocess, "run", unexpected_export)
    with pytest.raises(SystemExit, match="one unambiguous PyTorch version"):
        module.main()
    assert not (tmp_path / "desktop/build").exists()


def test_default_deny_deploy_context_retains_package_build_metadata():
    docker = (ROOT / ".dockerignore").read_text()
    cloud = (ROOT / ".gcloudignore").read_text()
    assert docker == cloud
    rules = [
        line.strip() for line in docker.splitlines() if line.strip() and not line.startswith("#")
    ]
    assert rules[0] == "**"
    required = {"scripts/generate_frontend_notices.mjs"}
    for folder in (Path(), Path("packages/openecon-charts")):
        project = tomllib.loads((ROOT / folder / "pyproject.toml").read_text())["project"]
        for name in (project["readme"], *project["license-files"]):
            path = folder / name
            assert (ROOT / path).is_file()
            required.add(path.as_posix())
    for path in required:
        assert "!" + path in rules, f"Required build input excluded: {path}"
    assert "!scripts/**" not in rules
    for exclusion in ("**/.env", "**/.env.*", "**/*credentials*.json", "**/*.key"):
        assert exclusion in rules


def test_builder_copies_package_metadata_and_frontend_notice_generator():
    stages = {}
    stage = None
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        if line.startswith("FROM "):
            words = shlex.split(line)
            stage = words[-1] if "AS" in words else None
            stages[stage] = []
        elif line.startswith("COPY ") and not line.startswith("COPY --from="):
            stages[stage].append(shlex.split(line)[1:])
    root_project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    builder_inputs = {
        item
        for instruction in stages["builder"]
        if instruction[-1] == "./"
        for item in instruction[:-1]
    }
    assert {root_project["readme"], *root_project["license-files"]} <= builder_inputs
    assert ["packages/openecon-charts/", "./packages/openecon-charts/"] in stages["builder"]
    assert [
        "scripts/generate_frontend_notices.mjs",
        "/build/scripts/generate_frontend_notices.mjs",
    ] in stages["frontend"]


def test_control_image_is_sync_only_and_bound_to_its_source_commit():
    docker = (ROOT / "Dockerfile").read_text()
    stages = [shlex.split(line)[-1] for line in docker.splitlines()
              if line.startswith("FROM ") and " AS " in line]
    assert stages == ["frontend", "python-base", "builder", "runtime"]
    assert "sandbox" not in docker.lower() and "netns" not in docker
    final = docker.rsplit("\nFROM ", 1)[1]
    assert 'ARG OPENECON_SOURCE_COMMIT=""' in final
    assert "ENV OPENECON_SOURCE_COMMIT=${OPENECON_SOURCE_COMMIT}" in final
    assert "USER 10001:10001" in final
    assert final.rstrip().endswith('CMD ["python", "-m", "openecon.cloud"]')
    # One control build only: no compute image, broker target or second config.
    assert sorted(path.name for path in (ROOT / "deploy").iterdir()) == ["cloudbuild-control.yaml"]
    build = (ROOT / "deploy/cloudbuild-control.yaml").read_text()
    assert "      - --build-arg\n      - OPENECON_SOURCE_COMMIT=${_SOURCE_COMMIT}\n" in build
    assert "  _SOURCE_COMMIT: ${COMMIT_SHA}\n" in build
    assert "images:\n  - ${_IMAGE}\nsubstitutions:" in build
    assert "--target" not in build and "_COMPUTE_IMAGE" not in build
    # The desktop result-schema gate stays part of every control build.
    for marker in ("DesktopResult.model_validate_json", "validate_worker_result(",
                   "trusted['events'] == events", "PUBLICATION_STYLE"):
        assert marker in build
