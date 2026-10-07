"""Publication boundaries and retained asset-license bytes."""
from pathlib import Path
import hashlib
import json
import os
import runpy
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
included = runpy.run_path(str(ROOT / "scripts/prepare_public_source.py"))["included"]
archive_source = runpy.run_path(str(ROOT / "scripts/prepare_public_source.py"))["archive_source"]


def test_public_snapshot_keeps_source_and_excludes_private_history_artifacts():
    for name in ("src/openecon/analysis.py", "src/openecon/examples/wages.csv", "desktop/src-tauri/icons/icon.icns", "packages/openecon-charts/src/openecon_charts/assets/BARLOW-OFL.txt", "LICENSE", "NOTICE", "uv.lock"):
        assert included(name), name
    for name in ("reports/private.docx", "docs/evidence/project.json", "tests/fixtures/stata/auto-numeric.csv", "desktop/runtime/openecon-runtime", "web/node_modules/react/index.js", ".env", "src/openecon/.env", "desktop/service-account.json", "src/openecon/private-key.pem", ".git/config", "credentials.json"):
        assert not included(name), name


def test_full_notices_match_locked_dependencies_and_upstream_font_bytes():
    lock = json.loads((ROOT / "web/package-lock.json").read_text())
    manifest = json.loads((ROOT / "web/public/third-party-manifest.json").read_text())
    expected = [value for key, value in lock["packages"].items() if key and not value.get("dev")]
    assert len(manifest["packages"]) == len(expected) > 100
    assert manifest["package_lock_sha256"] == hashlib.sha256((ROOT / "web/package-lock.json").read_bytes()).hexdigest()
    text = (ROOT / "web/public/THIRD-PARTY-NOTICES.txt").read_text()
    for record in manifest["packages"]:
        assert record["name"] + " " + record["version"] in text
        assert record["notices"]
    for row in json.loads((ROOT / "web/public/licenses/upstream.json").read_text()):
        path = ROOT / "web/public/licenses" / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    assert "SIL OPEN FONT LICENSE" in (ROOT / "web/public/licenses/KATEX-FONTS-LICENSE.txt").read_text()
    assert "Copyright" in (ROOT / "web/public/assets/BARLOW-OFL.txt").read_text()


def test_notice_generator_rejects_stale_outputs():
    subprocess.run(["node", str(ROOT / "scripts/generate_frontend_notices.mjs"), "--check"], check=True, cwd=ROOT)


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
        assert (row.uid, row.gid, row.uname, row.gname, row.mtime, row.mode) == (0, 0, "", "", 0, 0o755)
