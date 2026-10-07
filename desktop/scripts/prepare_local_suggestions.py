"""Prepare pinned llama.cpp resources. Model weights are downloaded by Settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

DESKTOP = Path(__file__).resolve().parents[1]
ENGINE_TAG = "b11146"
ENGINES = {
    "arm64": (11189714, "1ad3f9eff80edb9dbef4259ad564d1720612ef7eea48fa4afed0e54f5f3d5711"),
    "x64": (11237237, "305f0e3a17d2c01eb205cd0a62128357f1ec3b55329cb084d94e5ec0115d7a3b"),
}
MODEL = {
    "id": "qwen2.5-coder-0.5b-q8_0",
    "name": "Qwen2.5-Coder 0.5B",
    "revision": "5788aee90e725673490be18f553d234feaa8303d",
    "url": "https://huggingface.co/ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF/resolve/5788aee90e725673490be18f553d234feaa8303d/qwen2.5-coder-0.5b-q8_0.gguf",
    "size_bytes": 531068128,
    "sha256": "d0f8cd6c49bab52a0abdbe47948518b1f4d9b1a8a2a6825099cea31cea10ac56",
    "license": "Apache-2.0",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, target: Path, size: int, checksum: str) -> None:
    if target.is_file() and target.stat().st_size == size and digest(target) == checksum:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < size + 128 * 1024**2:
        raise RuntimeError("There is insufficient space for the verified download.")
    partial = target.with_suffix(target.suffix + ".partial")
    total = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
            while chunk := response.read(1024**2):
                total += len(chunk)
                if total > size:
                    raise RuntimeError("The download exceeds its pinned size.")
                stream.write(chunk)
        if total != size or digest(partial) != checksum:
            raise RuntimeError("The download does not match its pinned hash and size.")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def prepare(destination: Path, arch: str) -> dict:
    if sys.platform != "darwin" or arch != "arm64":
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {
            "engine": "llama.cpp",
            "version": ENGINE_TAG,
            "arch": arch,
            "supported": False,
            "license": "MIT",
            "model": MODEL,
            "files": [],
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    size, checksum = ENGINES[arch]
    name = f"llama-{ENGINE_TAG}-bin-macos-{arch}.tar.gz"
    archive = DESKTOP / "build" / "local-suggestions" / name
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{ENGINE_TAG}/{name}"
    download(url, archive, size, checksum)
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    with tarfile.open(archive) as bundle:
        members = {member.name: member for member in bundle.getmembers()}
        prefix = f"llama-{ENGINE_TAG}/"
        for path, member in members.items():
            basename = PurePosixPath(path).name
            if path != prefix + basename or not (
                basename in {"llama-server", "LICENSE"} or basename.endswith(".dylib")
            ):
                continue
            source = member
            visited = set()
            while source.issym():
                if source.name in visited or PurePosixPath(source.linkname).name != source.linkname:
                    raise RuntimeError("Unsafe archive library alias.")
                visited.add(source.name)
                source = members[prefix + source.linkname]
            if not source.isfile() or source.size > 16 * 1024**2:
                raise RuntimeError("Unsafe archive resource.")
            resource = destination / basename
            if resource.is_symlink():
                raise RuntimeError("The resource destination cannot be a symbolic link.")
            with bundle.extractfile(source) as stream, resource.open("wb") as output:
                shutil.copyfileobj(stream, output)
            resource.chmod(0o755 if basename == "llama-server" else 0o644)
            # Freeze hashes after ad-hoc signing; the final app signs its outer
            # bundle without rewriting these already verified native resources.
            if sys.platform == "darwin" and basename != "LICENSE":
                subprocess.run(
                    ["codesign", "--force", "--sign", "-", str(resource)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    ["codesign", "--verify", "--strict", str(resource)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            files.append(
                {
                    "name": basename,
                    "size_bytes": resource.stat().st_size,
                    "sha256": digest(resource),
                }
            )
    manifest = {
        "engine": "llama.cpp",
        "version": ENGINE_TAG,
        "arch": arch,
        "archive_url": url,
        "archive_sha256": checksum,
        "license": "MIT",
        "model": MODEL,
        "files": files,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESKTOP / "suggestions")
    parser.add_argument(
        "--arch", choices=list(ENGINES), default="arm64" if platform.machine() == "arm64" else "x64"
    )
    parser.add_argument(
        "--benchmark-model",
        type=Path,
        help="Download model separately for local inference validation; never bundle weights.",
    )
    args = parser.parse_args()
    manifest = prepare(args.destination, args.arch)
    if args.benchmark_model:
        download(MODEL["url"], args.benchmark_model, MODEL["size_bytes"], MODEL["sha256"])
    print(
        json.dumps(
            {
                "status": "prepared",
                "destination": str(args.destination),
                "engine": manifest["version"],
                "files": len(manifest["files"]),
                "model_bundled": False,
                "model_download_bytes": MODEL["size_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
