"""Required package descriptions, notices and frontend tools reach Docker stages."""

from pathlib import Path
import shlex
import tomllib


ROOT = Path(__file__).resolve().parents[1]


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
