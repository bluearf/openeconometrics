"""Read-only hashes and selected preferences for a disposable macOS QA profile.

Refuses production identifiers. Reads no Firebase token or IndexedDB contents.
Compare snapshots before and after an app replacement, before executing code or
changing packages again. Core package markers may migrate; overlays, declared
requirements, data, sources, history, port and preferences must be preserved.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def digest(path):
    if path.is_symlink():
        raise ValueError("QA profile snapshots do not follow file aliases.")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(identifier, project):
    if not re.fullmatch(r"org\.openecon\.qa\.[a-z0-9-]+", identifier):
        raise ValueError("Only a disposable org.openecon.qa identifier is allowed.")
    if not re.fullmatch(r"[a-f0-9]{32}", project):
        raise ValueError("A synthetic local project ID is required.")
    root = Path.home() / "Library/Application Support" / identifier
    workspace = root / "projects" / project
    assert workspace.is_dir() and not workspace.is_symlink()
    files = {}
    for path in sorted(workspace.rglob("*")):
        if ".packages" in path.parts or "__pycache__" in path.parts:
            continue
        if path.is_file():
            files[path.relative_to(workspace).as_posix()] = digest(path)
    package = workspace / ".packages/active.json"
    active = json.loads(package.read_text())
    overlay = package.parent / ("generation-" + active["generation"]) / "site-packages"
    assert overlay.is_dir() and not overlay.is_symlink()
    overlay_files = {p.relative_to(overlay).as_posix(): digest(p)
                     for p in sorted(overlay.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    store = Path.home() / "Library/WebKit" / identifier
    preferences, uids = {}, []
    for database in store.rglob("localstorage.sqlite3"):
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            for key, raw in connection.execute(
                    "SELECT key,value FROM ItemTable WHERE key LIKE 'openecon:panes:%' "
                    "OR key LIKE 'openecon-desktop-profile:%'"):
                value = json.loads(raw.decode("utf-16-le") if isinstance(raw, bytes) else raw)
                if key.startswith("openecon:panes:"):
                    preferences[key] = value
                else:
                    uids.append(hashlib.sha256(value["user"]["uid"].encode()).hexdigest())
    manifest = active["manifest"]
    # Only the fixed-core markers are expected to change during a runtime update.
    package_contract = {k: v for k, v in manifest.items() if k != "core"}
    return {"identifier": identifier, "project_id": project,
            "port_file_sha256": digest(root / ".runtime-port.json"),
            "catalog_sha256": digest(root / "local-projects.json"),
            "workspace_files": files, "package_generation": active["generation"],
            "package_contract": package_contract, "overlay_files": overlay_files,
            "preferences": preferences, "cached_identity_sha256": sorted(set(uids)),
            "scope": "Synthetic source/data/history/overlay bytes and selected metadata; auth token contents excluded."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identifier", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = snapshot(args.identifier, args.project)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"workspace_files": len(result["workspace_files"]),
                      "overlay_files": len(result["overlay_files"]),
                      "pane_preferences": len(result["preferences"]),
                      "cached_identities": len(result["cached_identity_sha256"])}))
