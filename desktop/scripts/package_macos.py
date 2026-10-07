"""Preserve runtime library aliases in the installed app and compressed DMG."""

from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from package_runtime import optimize_runtime, prepare_distribution_notices

DESKTOP = Path(__file__).resolve().parents[1]


def finalize_app(app: Path) -> dict:
    if sys.platform != "darwin":
        raise RuntimeError("macOS disk images must be created on macOS.")
    resources = app / "Contents" / "Resources" / "runtime"
    runtime = resources / "openecon-runtime"
    prepare_distribution_notices(runtime)
    # Tauri copies resource symlinks as files. Restore the verified relative
    # aliases after bundling, before making the installer from the final app.
    optimization = optimize_runtime(runtime)
    manifest_path = resources / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["installed_optimization"] = optimization
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Tauri may re-sign nested resources. Restore the exact pre-signed engine
    # bytes whose manifest was compiled into the native executable.
    prepared = DESKTOP / "suggestions"
    engine_manifest = json.loads((prepared / "manifest.json").read_text())
    expected = {item["name"] for item in engine_manifest["files"]} | {"manifest.json"}
    if {path.name for path in prepared.iterdir()} != expected:
        raise RuntimeError("The prepared suggestion engine contains unexpected resources.")
    for item in engine_manifest["files"]:
        path = prepared / item["name"]
        if path.is_symlink() or path.stat().st_size != item["size_bytes"]:
            raise RuntimeError("The prepared suggestion engine does not match its manifest.")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise RuntimeError("The prepared suggestion engine does not match its manifest.")
    target = app / "Contents/Resources/suggestions"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(prepared, target)
    # Nested binaries already have signatures; signing only the outer bundle
    # preserves the hashes that native startup checks before execution.
    subprocess.run(["codesign", "--force", "--sign", "-", str(app)], check=True)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)
    return optimization


def package(app: Path, output: Path) -> dict:
    if app.name not in {"OpenEconometrics.app", "OpenEcon.app"}:
        raise RuntimeError("The installer source is not a recognized application bundle.")
    optimization = finalize_app(app)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A renamed build can share its parent with a retained legacy app. Package
    # only the selected app, never its siblings, and leave those siblings intact.
    with tempfile.TemporaryDirectory(prefix=".openeconometrics-dmg-", dir=output.parent) as temporary:
        staging = Path(temporary)
        shutil.copytree(app, staging / app.name, symlinks=True)
        (staging / "Applications").symlink_to("/Applications")
        subprocess.run(
            [
                "hdiutil",
                "create",
                "-ov",
                "-volname",
                app.stem,
                "-srcfolder",
                str(staging),
                "-format",
                "UDZO",
                "-imagekey",
                "zlib-level=6",
                str(output),
            ],
            check=True,
        )
    return {"app": str(app), "dmg": str(output), "installed_optimization": optimization}


def main():
    config = json.loads((DESKTOP / "src-tauri/tauri.conf.json").read_text())
    product, version = config["productName"], config["version"]
    if product not in {"OpenEconometrics", "OpenEcon"}:
        raise RuntimeError("The desktop product name is not a recognized bundle identity.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--app", type=Path, default=DESKTOP / f"src-tauri/target/release/bundle/macos/{product}.app"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DESKTOP / f"src-tauri/target/release/bundle/dmg/{product}_{version}_aarch64.dmg",
    )
    args = parser.parse_args()
    print(json.dumps(package(args.app.resolve(), args.output.resolve())))


if __name__ == "__main__":
    main()
