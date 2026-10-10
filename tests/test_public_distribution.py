"""Publication boundaries and retained asset-license bytes."""

from pathlib import Path
import hashlib
import json
import os
import runpy
import subprocess
import tarfile
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
included = runpy.run_path(str(ROOT / "scripts/prepare_public_source.py"))["included"]
archive_source = runpy.run_path(str(ROOT / "scripts/prepare_public_source.py"))["archive_source"]
public_evidence_links = runpy.run_path(str(ROOT / "scripts/prepare_public_source.py"))[
    "public_evidence_links"
]
public_build_configuration = runpy.run_path(str(ROOT / "scripts/prepare_public_source.py"))[
    "public_build_configuration"
]


def test_public_build_omits_only_private_identity_mapped_sdist_fixtures():
    source = (ROOT / "pyproject.toml").read_bytes()
    transformed, omissions = public_build_configuration("pyproject.toml", source)
    before, after = tomllib.loads(source.decode()), tomllib.loads(transformed.decode())
    if "force-include" not in before["tool"]["hatch"]["build"]["targets"]["sdist"]:
        manifest = json.loads((ROOT / "SOURCE-MANIFEST.json").read_text())
        recorded = manifest["source_distribution_private_fixture_omissions"]
        assert len(recorded) == 9
        assert all(row["source"] == row["destination"] and not included(row["source"])
                   for row in recorded)
        assert transformed == source and omissions == []
        assert next(row for row in manifest["files"] if row["path"] == "pyproject.toml")["transformed"]
        return
    forced = before["tool"]["hatch"]["build"]["targets"]["sdist"].pop("force-include")
    assert len(omissions) == len(forced) == 9
    assert omissions == [{"source": name, "destination": name} for name in sorted(forced)]
    assert all(not included(name) for name in forced)
    assert before == after
    assert public_build_configuration("pyproject.toml", transformed) == (transformed, [])


def test_public_build_retains_included_mappings_and_all_unrelated_toml_bytes():
    data = b'''[project]\nname = "example"\nversion = "1.2.3"\n[tool.hatch.build.targets.sdist.force-include]\n"docs/evidence/a.json" = "docs/evidence/a.json"\n"src/a.py" = "src/a.py"\n[tool.other]\nvalue = "unchanged"\n'''
    result, omitted = public_build_configuration("pyproject.toml", data)
    assert result == data.replace(b'"docs/evidence/a.json" = "docs/evidence/a.json"\n', b"")
    assert omitted == [{"source": "docs/evidence/a.json", "destination": "docs/evidence/a.json"}]
    assert public_build_configuration("packages/other/pyproject.toml", data) == (data, [])


def test_public_build_refuses_private_fixture_remapping_to_published_source():
    data = b'''[tool.hatch.build.targets.sdist.force-include]\n"docs/evidence/private.json" = "src/public.json"\n'''
    with pytest.raises(ValueError, match="identity mapping"):
        public_build_configuration("pyproject.toml", data)


def test_public_capability_transformation_updates_only_the_derived_build_fingerprint():
    transform = runpy.run_path(str(ROOT / "scripts/prepare_public_source.py"))["public_capability_fingerprint"]
    original, published = b"private build", b"reviewed public build"
    value = {"source_fingerprints": {"pyproject.toml": hashlib.sha256(original).hexdigest(),
                                     "src/engine.py": "retained"},
             "versions": {"sdk": "unchanged"}, "scientific_scope": ["retained"]}
    raw = json.dumps(value).encode()
    assert transform(raw, original, original) == (raw, [])
    result, receipt = transform(raw, original, published)
    expected = json.loads(raw)
    expected["source_fingerprints"]["pyproject.toml"] = hashlib.sha256(published).hexdigest()
    assert json.loads(result) == expected
    assert len(receipt) == 1
    assert receipt[0]["original"] == value["source_fingerprints"]["pyproject.toml"]
    assert receipt[0]["published"] == expected["source_fingerprints"]["pyproject.toml"]
    with pytest.raises(ValueError, match="reviewed source configuration"):
        transform(raw, b"unreviewed source", published)


def test_public_evidence_links_preserve_historical_scope_without_missing_targets():
    text = (
        "0.3.27 contained 230 modules and 80 estimators: "
        "[release record](../docs/evidence/desktop-0.3.27.json#fit). "
        "[source](../docs/capabilities.md) "
        "[public release](https://github.com/bluearf/openeconometrics/releases) "
        "[HTTP evidence](https://example.com/docs/evidence/a.json)"
    )
    transformed, records = public_evidence_links("desktop/README.md", text.encode())
    assert len(records) == 1
    assert records[0]["target"] == "docs/evidence/desktop-0.3.27.json"
    assert b"230 modules and 80 estimators" in transformed
    assert b"release record (internal evidence excluded from this public snapshot)" in transformed
    for target in (
        "../docs/capabilities.md",
        "https://github.com/bluearf/openeconometrics/releases",
        "https://example.com/docs/evidence/a.json",
    ):
        assert target.encode() in transformed
    assert public_evidence_links("desktop/README.md", transformed) == (transformed, [])
    assert public_evidence_links("data.json", b"[record](docs/evidence/private.json)") == (
        b"[record](docs/evidence/private.json)",
        [],
    )


def test_all_published_markdown_has_explicit_internal_evidence_boundaries():
    import re
    from urllib.parse import urlsplit
    import posixpath

    records = []
    manifest = ROOT / "SOURCE-MANIFEST.json"
    paths = (
        [
            row["path"]
            for row in json.loads(manifest.read_text())["files"]
            if row["path"].endswith(".md")
        ]
        if manifest.exists()
        else subprocess.check_output(["git", "ls-files", "*.md"], cwd=ROOT, text=True).splitlines()
    )
    for name in paths:
        if not included(name):
            continue
        transformed, rows = public_evidence_links(name, (ROOT / name).read_bytes())
        records.extend(rows)
        for target in re.findall(rb"\]\(([^)\s]+)\)", transformed):
            parsed = urlsplit(target.decode().strip("<>"))
            if parsed.scheme or parsed.netloc or parsed.path.startswith("/"):
                continue
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), parsed.path))
            assert not resolved.startswith("docs/evidence/"), (name, resolved)
    # Prepared archives use their explicit transformation receipt, never a
    # coincidentally discoverable parent Git checkout.
    receipt = (
        json.loads(manifest.read_text())["internal_evidence_link_boundaries"]
        if manifest.exists()
        else records
    )
    assert {row["target"] for row in receipt} >= {
        "docs/evidence/desktop-0.3.27.json",
        "docs/evidence/desktop-0.3.37.json",
        "docs/evidence/desktop-0.3.7.json",
    }


def test_public_snapshot_keeps_source_and_excludes_private_history_artifacts():
    for name in (
        "src/openecon/analysis.py",
        "src/openecon/examples/wages.csv",
        "desktop/src-tauri/icons/icon.icns",
        "packages/openecon-charts/src/openecon_charts/assets/BARLOW-OFL.txt",
        "LICENSE",
        "NOTICE",
        "AGENTS.md",
        "uv.lock",
    ):
        assert included(name), name
    for name in (
        "reports/private.docx",
        "docs/evidence/project.json",
        "tests/fixtures/stata/auto-numeric.csv",
        "desktop/runtime/openecon-runtime",
        "web/node_modules/react/index.js",
        ".env",
        "src/openecon/.env",
        "desktop/service-account.json",
        "src/openecon/private-key.pem",
        ".git/config",
        "credentials.json",
    ):
        assert not included(name), name


def test_public_snapshot_includes_declared_original_teaching_workbooks_and_plugin():
    workbooks = []
    for course in ("", "statistics", "microeconomics", "advanced-econometrics"):
        directory = ROOT / "docs/teaching" / course
        for lab in json.loads((directory / "catalog.json").read_text())["labs"]:
            if lab["data_mode"] == "prepared_excel":
                name = (directory / "labs" / lab["slug"] / lab["data_file"]).relative_to(ROOT).as_posix()
                assert included(name), name
                workbooks.append(name)
    assert len(workbooks) == 79
    plugin_files = runpy.run_path(str(ROOT / "scripts/prepare_public_source.py"))["PLUGIN_FILES"]
    assert len(plugin_files) == 11
    assert all(included(name) and (ROOT / name).is_file() for name in plugin_files)
    for name in (
        "docs/teaching/statistics/labs/01-measurement/customer.xlsx",
        "docs/teaching/statistics/labs/01-measurement/../household_measurements.xlsx",
        "docs/teaching/unknown/labs/01-measurement/household_measurements.xlsx",
        "docs/teaching/statistics/labs/01-measurement/customer.csv",
        "docs/customer.xlsx", "plugins/openeconometrics/credentials.json",
        "plugins/other/plugin.json", ".agents/plugins/private-key.json",
        "../src/openecon/examples/wages.csv", "",
    ):
        assert not included(name), name


def test_public_snapshot_does_not_publish_nonoriginal_or_undeclared_workbooks(tmp_path):
    module = runpy.run_path(str(ROOT / "scripts/prepare_public_source.py"))
    workbook = module["teaching_workbook"]
    workbook.__globals__["ROOT"] = tmp_path
    directory = tmp_path / "docs/teaching/statistics"
    lab = directory / "labs/01-measurement"
    lab.mkdir(parents=True)
    path = "docs/teaching/statistics/labs/01-measurement/household_measurements.xlsx"
    (directory / "catalog.json").write_text(json.dumps({"labs": [{
        "slug": "01-measurement", "data_mode": "prepared_excel",
        "data_file": "household_measurements.xlsx",
    }]}))
    assert not workbook(path)
    metadata = lab / "dataset.json"
    metadata.write_text(json.dumps({"filename": "household_measurements.xlsx", "source": "Customer data"}))
    assert not workbook(path)
    metadata.write_text(json.dumps({"filename": "household_measurements.xlsx",
                                    "source": "Original OpenEconometrics teaching simulation. Not empirical observations."}))
    assert workbook(path)
    assert not workbook(path.replace("household_measurements", "undeclared"))


def test_full_notices_match_locked_dependencies_and_upstream_font_bytes():
    lock = json.loads((ROOT / "web/package-lock.json").read_text())
    manifest = json.loads((ROOT / "web/public/third-party-manifest.json").read_text())
    expected = [value for key, value in lock["packages"].items() if key and not value.get("dev")]
    assert len(manifest["packages"]) == len(expected) > 100
    assert (
        manifest["package_lock_sha256"]
        == hashlib.sha256((ROOT / "web/package-lock.json").read_bytes()).hexdigest()
    )
    text = (ROOT / "web/public/THIRD-PARTY-NOTICES.txt").read_text()
    for record in manifest["packages"]:
        assert record["name"] + " " + record["version"] in text
        assert record["notices"]
    for row in json.loads((ROOT / "web/public/licenses/upstream.json").read_text()):
        path = ROOT / "web/public/licenses" / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    assert (
        "SIL OPEN FONT LICENSE"
        in (ROOT / "web/public/licenses/KATEX-FONTS-LICENSE.txt").read_text()
    )
    assert "Copyright" in (ROOT / "web/public/assets/BARLOW-OFL.txt").read_text()


def test_notice_generator_rejects_stale_outputs():
    subprocess.run(
        ["node", str(ROOT / "scripts/generate_frontend_notices.mjs"), "--check"],
        check=True,
        cwd=ROOT,
    )


def test_source_archive_is_reproducible_and_omits_local_owner_metadata(tmp_path):
    tree = tmp_path / "source"
    tree.mkdir()
    source = tree / "run.sh"
    source.write_text("#!/bin/sh\nexit 0\n")
    source.chmod(0o755)
    one, two = tmp_path / "one.gz", tmp_path / "two.gz"
    archive_source(tree, one)
    os.utime(source, (1, 1))
    archive_source(tree, two)
    assert one.read_bytes() == two.read_bytes()
    with tarfile.open(one) as archive:
        row = archive.getmember("openeconometrics/run.sh")
        assert (row.uid, row.gid, row.uname, row.gname, row.mtime, row.mode) == (
            0,
            0,
            "",
            "",
            0,
            0o755,
        )
