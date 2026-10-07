"""Build OpenEconometrics artifacts with a bundled runtime; no global install."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

DESKTOP = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", default="nsis" if os.name == "nt" else "app,dmg")
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--python", type=Path)
    parser.add_argument(
        "--target-dir", type=Path, help="Separate build output, including the app and installer."
    )
    args = parser.parse_args()
    config = json.loads((DESKTOP / "src-tauri/tauri.conf.json").read_text())
    product, version = config["productName"], config["version"]
    if product not in {"OpenEconometrics", "OpenEcon"}:
        raise SystemExit("The desktop product name is not a recognized bundle identity.")
    env = os.environ.copy()
    configured_target = args.target_dir or env.get("CARGO_TARGET_DIR")
    target = (
        Path(configured_target).resolve() if configured_target else DESKTOP / "src-tauri" / "target"
    )
    env["CARGO_TARGET_DIR"] = str(target)
    # Skip Finder automation for deterministic disk-image creation. The app and
    # Applications shortcut remain in the image without cosmetic positioning.
    env["CI"] = "true"
    cargo = DESKTOP / ".toolchains" / "cargo"
    if (cargo / "bin" / "cargo").is_file():
        env["CARGO_HOME"] = str(cargo)
        env["RUSTUP_HOME"] = str(DESKTOP / ".toolchains" / "rustup")
        env["PATH"] = str(cargo / "bin") + os.pathsep + env["PATH"]
    if not args.skip_runtime:
        command = [
            str(args.python or sys.executable),
            str(DESKTOP / "scripts" / "package_runtime.py"),
        ]
        subprocess.run(command, check=True, env=env)
    runtime = (
        DESKTOP
        / "runtime"
        / "openecon-runtime"
        / ("openecon-runtime.exe" if os.name == "nt" else "openecon-runtime")
    )
    if not runtime.is_file():
        raise SystemExit(
            "Build the self-contained Python runtime before packaging the desktop app."
        )
    subprocess.run(
        [sys.executable, str(DESKTOP / "scripts/prepare_local_suggestions.py")], check=True, env=env
    )
    bundles = (
        "app" if sys.platform == "darwin" and "dmg" in args.bundles.split(",") else args.bundles
    )
    subprocess.run(
        [
            "npm.cmd" if os.name == "nt" else "npm",
            "exec",
            "--",
            "tauri",
            "build",
            "--bundles",
            bundles,
        ],
        cwd=DESKTOP,
        env=env,
        check=True,
    )
    if sys.platform == "darwin" and "dmg" not in args.bundles.split(","):
        from package_macos import finalize_app

        finalize_app(target / f"release/bundle/macos/{product}.app")
    if sys.platform == "darwin" and "dmg" in args.bundles.split(","):
        bundle = target / "release" / "bundle"
        subprocess.run(
            [
                sys.executable,
                str(DESKTOP / "scripts/package_macos.py"),
                "--app",
                str(bundle / "macos" / f"{product}.app"),
                "--output",
                str(bundle / "dmg" / f"{product}_{version}_aarch64.dmg"),
            ],
            cwd=DESKTOP,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
