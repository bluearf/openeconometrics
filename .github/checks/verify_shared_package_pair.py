"""Bind both Python jobs to the same four source-verified build artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil


def verify(root: Path, build: Path, expected_execution: dict) -> dict:
    execution = json.loads((build / "build-execution.json").read_text())
    if execution != expected_execution:
        raise ValueError("Shared build execution differs from this exact checkout/run")
    expected = json.loads((build / "package-pair-provenance.json").read_text())
    frontend = build / "frontend"
    if not frontend.is_dir() or any(path.is_symlink() for path in build.rglob("*")):
        raise ValueError("Shared build must contain regular owned frontend files")
    target = root / "src/openecon/static"
    if target.is_symlink():
        raise ValueError("Frontend destination cannot be a symlink")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(frontend, target)
    checker = runpy.run_path(str(root / "scripts/verify_public_package_pair.py"))["verify"]
    actual = checker(root, build / "dist", execution["sdk_version"], execution["charts_version"],
                     execution["private_source_commit"])
    if actual != expected:
        raise ValueError("Shared distribution hashes, inventory or metadata differ from build")
    return {"status": "passed", **execution, "artifacts": actual["artifacts"],
            "generated_frontend": actual["generated_frontend"],
            "same_four_distributions": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "SOURCE-MANIFEST.json").read_text())
    expected = {
        "public_checkout_commit": os.environ["GITHUB_SHA"],
        "private_source_commit": manifest["source_commit"],
        "source_files_sha256": manifest["source_files_sha256"],
        "event": os.environ["GITHUB_EVENT_NAME"],
        "run_id": os.environ["GITHUB_RUN_ID"],
        "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "sdk_version": os.environ["SDK_VERSION"],
        "charts_version": os.environ["CHARTS_VERSION"],
    }
    record = verify(root, args.build.absolute(), expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "artifacts": record["artifacts"]}))


if __name__ == "__main__":
    main()
