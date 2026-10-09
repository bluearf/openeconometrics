"""Prepare a reviewable source snapshot without publishing private Git history.

Internal QA receipts/reports and externally sourced research fixture bytes are
excluded. The original repository is read-only. No GitHub visibility, account or
credential is changed. The output owns a new directory and records every byte.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import posixpath
import re
import subprocess
import tarfile
import tomllib
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    "README.md",
    "PYPI-README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitignore",
}
DIRECTORIES = {
    "src",
    "packages",
    "web",
    "desktop",
    "scripts",
    "tests",
    "docs",
    "benchmarks",
    "website",
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def included(path):
    """Positive source allowlist, with publication boundaries made explicit."""
    if path in ROOT_FILES:
        return True
    parts = Path(path).parts
    if parts[0] not in DIRECTORIES:
        return False
    if any(part == ".env" or part.startswith(".env.") for part in parts):
        return False
    if re.search(
        r"(?:credentials|service[-_]account|private[-_]key)\.(?:json|ya?ml)$", parts[-1], re.I
    ):
        return False
    if path.startswith(("docs/evidence/", "tests/fixtures/")):
        return False
    if any(
        part in {"node_modules", ".toolchains", "runtime", "target", "build", "dist", "__pycache__"}
        for part in parts
    ):
        return False
    if Path(path).suffix.lower() in {
        ".csv",
        ".parquet",
        ".dta",
        ".xlsx",
        ".docx",
        ".pdf",
        ".gz",
        ".sqlite",
        ".db",
        ".log",
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".dmg",
    }:
        return path == "src/openecon/examples/wages.csv"
    return True


def public_evidence_links(path: str, data: bytes):
    """Replace links to deliberately excluded internal evidence, never its bytes.

    Preserve dated measurement prose. This transformation makes its publication
    boundary explicit instead of suggesting the public snapshot contains a
    private receipt. HTTP links and included source-document links are unchanged.
    """
    if not path.endswith(".md"):
        return data, []
    records = []

    def replace(match):
        label, target = match.group(1), match.group(2)
        parsed = urlsplit(target.strip("<>"))
        if parsed.scheme or parsed.netloc or parsed.path.startswith("/"):
            return match.group(0)
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), unquote(parsed.path)))
        if not resolved.startswith("docs/evidence/") or included(resolved):
            return match.group(0)
        records.append({"document": path, "target": resolved, "label": label})
        return f"{label} (internal evidence excluded from this public snapshot)"

    text = re.sub(r"(?<!!)\[([^\]]+)\]\((<?[^\s)]+>?)\)", replace, data.decode("utf-8"))
    return text.encode("utf-8"), records


def public_build_configuration(path: str, data: bytes):
    """Omit only identity-mapped private sdist fixtures excluded from the snapshot.

    The private sdist retains its replay fixtures. A public build cannot force
    those absent files into the archive. All other TOML values remain identical.
    """
    if path != "pyproject.toml":
        return data, []
    original = tomllib.loads(data.decode("utf-8"))
    target = original.get("tool", {}).get("hatch", {}).get("build", {}).get(
        "targets", {}).get("sdist", {})
    forced = target.get("force-include", {})
    omitted = {}
    for source, destination in forced.items():
        if source.startswith(("docs/evidence/", "tests/fixtures/")) and not included(source):
            if source != destination:
                raise ValueError("Private sdist fixture must have an identity mapping")
            omitted[source] = destination
    if not omitted:
        return data, []
    text = data.decode("utf-8")
    header = re.search(r"(?m)^\[tool\.hatch\.build\.targets\.sdist\.force-include\][ \t]*\r?\n", text)
    if header is None:
        raise ValueError("Private sdist fixture table must be explicit")
    next_table = re.search(r"(?m)^\[", text[header.end():])
    end = header.end() + next_table.start() if next_table else len(text)
    retained = []
    removed = set()
    for line in text[header.end():end].splitlines(keepends=True):
        if not line.strip() or line.lstrip().startswith("#"):
            retained.append(line)
            continue
        entry = tomllib.loads(line)
        if len(entry) != 1:
            raise ValueError("Unsupported sdist fixture mapping")
        source = next(iter(entry))
        if source in omitted:
            if entry[source] != omitted[source]:
                raise ValueError("Private sdist fixture mapping changed")
            removed.add(source)
        else:
            retained.append(line)
    if removed != set(omitted):
        raise ValueError("Incomplete private sdist fixture omission")
    replacement = text[header.start():header.end()] + "".join(retained) if len(forced) > len(omitted) else ""
    transformed = text[:header.start()] + replacement + text[end:]
    for source in omitted:
        del forced[source]
    if not forced:
        del target["force-include"]
    if tomllib.loads(transformed) != original:
        raise ValueError("Public build transformation changed unrelated configuration")
    return transformed.encode("utf-8"), [
        {"source": source, "destination": destination}
        for source, destination in sorted(omitted.items())
    ]


def archive_source(tree: Path, archive: Path):
    """Reproducible regular files; never publish local user/group metadata."""
    with (
        archive.open("xb") as output,
        gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed,
    ):
        with tarfile.open(fileobj=compressed, mode="w|") as handle:
            for path in sorted(tree.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Public source archives cannot contain symlinks")
                if path.is_file():
                    data = path.read_bytes()
                    entry = tarfile.TarInfo("openeconometrics/" + path.relative_to(tree).as_posix())
                    entry.size = len(data)
                    entry.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                    entry.mtime = entry.uid = entry.gid = 0
                    entry.uname = entry.gname = ""
                    handle.addfile(entry, io.BytesIO(data))


PUBLIC_TEST_POLICY = r"""

# Public snapshot policy: external research fixture bytes are distributed by
# their original providers. These fixture-backed modules remain in source for
# review, but do not enter the default public collection without those files.
# The explicit excluded module list is recorded in SOURCE-MANIFEST.json.
def pytest_ignore_collect(collection_path, config):
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / 'SOURCE-MANIFEST.json').read_text())
    try:
        relative = Path(collection_path).resolve().relative_to(root).as_posix()
    except ValueError:
        return False
    return relative in record['external_fixture_test_modules']
"""


def prepare(output: Path, ref="HEAD"):
    if ref != "HEAD":
        raise ValueError("Prepare the reviewed HEAD; use an isolated checkout for another ref")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    tree = output / "source"
    tree.mkdir(mode=0o700)
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=ROOT, text=True
    ).splitlines()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    records, excluded, external_modules, evidence_links = [], [], [], []
    private_sdist_fixtures = []
    for path in paths:
        if not included(path):
            excluded.append(path)
            continue
        data = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)
        original_hash = digest(data)
        # Retain source-project attribution without publishing host usernames.
        if path.startswith("docs/") and path.endswith(".md"):
            data = re.sub(
                rb"/Users/[^/\s`]+/(?:Documents/GitHub|\.codex/worktrees)/",
                b"source-checkouts/",
                data,
            )
        data, transformed_links = public_evidence_links(path, data)
        evidence_links.extend(transformed_links)
        data, omitted_fixtures = public_build_configuration(path, data)
        private_sdist_fixtures.extend(omitted_fixtures)
        if (
            path.startswith("tests/test_")
            and path.endswith(".py")
            and b"fixtures" in data
            and path != "tests/test_public_distribution.py"
        ):
            external_modules.append(path)
        if path == "tests/conftest.py":
            data += PUBLIC_TEST_POLICY.encode()
        target = tree / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        mode = subprocess.check_output(
            ["git", "ls-tree", "HEAD", "--", path], cwd=ROOT, text=True
        ).split()[0]
        if mode == "100755":
            target.chmod(0o755)
        elif mode != "100644":
            raise ValueError(f"Unexpected non-regular source entry: {path}")
        records.append(
            {
                "path": path,
                "bytes": len(data),
                "sha256": digest(data),
                "original_sha256": original_hash,
                "transformed": original_hash != digest(data),
            }
        )
    if not (tree / "tests/conftest.py").exists():
        (tree / "tests/conftest.py").write_text(PUBLIC_TEST_POLICY)
        records.append(
            {
                "path": "tests/conftest.py",
                "bytes": len(PUBLIC_TEST_POLICY.encode()),
                "sha256": digest(PUBLIC_TEST_POLICY.encode()),
                "original_sha256": None,
                "transformed": True,
            }
        )
    manifest = {
        "schema_version": 1,
        "source_commit": commit,
        "publication_state": "prepared source snapshot; public access requires separate verification",
        "history": "new snapshot only; private Git history, branches, PRs and issues excluded",
        "research_data": "external fixture bytes excluded; obtain separately from original providers under their terms",
        "external_fixture_test_modules": sorted(external_modules),
        "files": records,
        "internal_evidence_link_boundaries": evidence_links,
        "source_distribution_private_fixture_omissions": private_sdist_fixtures,
        "excluded_paths": excluded,
        "source_files_sha256": digest(json.dumps(records, sort_keys=True).encode()),
    }
    (tree / "SOURCE-MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (tree / "PUBLIC-SOURCE.md").write_text(
        "# Public source snapshot\n\nThis candidate retains all OpenEconometrics production Python, chart, web and desktop source modules from the recorded source commit. It contains no private Git history, internal QA evidence/report files or external research fixture data. Original license and asset attribution files remain included.\n\n"
        "Install with `uv sync --frozen --extra app`, `npm --prefix web ci` and `npm --prefix web run build`. The initial installation requires internet. `SOURCE-MANIFEST.json` records every file and the explicit list of fixture-backed test modules omitted from default collection. Their source remains available for review; this does not claim those external-data scientific checks have run in this snapshot. Synthetic tests and native kernels remain included.\n\n"
        "Links to internal evidence omitted by the publication policy are replaced with an explicit boundary, listed in `SOURCE-MANIFEST.json`. Dated measurement statements retain their original method, source and version scope. They are historical source documentation, not independently accessible evidence of this snapshot's installed or public-release behavior.\n\n"
        "The public source build configuration omits force-include mappings for the internal replay fixtures that this snapshot excludes. The manifest records those exact identity mappings and original/transformed configuration hashes. Package metadata, dependencies and all other build settings are retained.\n\n"
        "The archive records the prepared source snapshot. Its presence alone does not prove public GitHub access, PyPI publication, a public Mac download, Developer ID signing or notarization. Verify those delivery layers separately.\n"
    )
    archive = output / "OpenEconometrics-source.tar.gz"
    archive_source(tree, archive)
    receipt = {
        "source_commit": commit,
        "archive": archive.name,
        "archive_sha256": digest(archive.read_bytes()),
        "archive_bytes": archive.stat().st_size,
        "file_count": len(records),
        "excluded_count": len(excluded),
        "external_fixture_test_modules": len(external_modules),
        "source_files_sha256": manifest["source_files_sha256"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output.resolve())))
