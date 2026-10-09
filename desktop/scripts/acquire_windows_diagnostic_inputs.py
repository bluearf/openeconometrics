"""Acquire one immutable failed Windows producer for a diagnostic-only replay.

No arbitrary run, repository, artifact, installer or source override is accepted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
import uuid
import zipfile

REPOSITORY = "bluearf/openecon"
RUN_ID = 37868521174
SOURCE_SHA = "e281bcf7b98c8b3cbb673a9fd749a054af495703"
INSTALLER_SHA = "df7b47988f548b018695bc4d05f0819e9be9b734787a058bb4d744c8e93c1a05"
ARTIFACTS = (
    (11590451069, "Windows-built-unverified-37868521174", 192410655,
     "760217ad849b72437ce386bb10986becc6b7881488a3d71f74d09ff99f8c73fc"),
    (11590201566, "Windows-acceptance-37868521174-1", 4121139,
     "562c49358ea4edd96c050bdbc5baf9525a7ea5f6530803b78a84b7cc7496052a"),
)
EVIDENCE = {
    "windows-installed-acceptance.json": (12360, "a99c8ab3f6868017cc38a809ee1d52d4e2d0e0170eb2ef2c3e9ed2a806896a56"),
    "native-build-reference.exe": (11279360, "54b0f185654ec45fb1da626ae71a2921e9df511358ee91b8005022376b22788d"),
    "native-build-reference.json": (321, "085f3f3a62280565e47d9a2b76885cf5470532868f119e8837669223a043523b"),
    "build-inputs.json": (1278, "8fba0eb81051d1e6c3fc32ce045ac82af047f2fa7bb483df3b25fa109903e612"),
}
PREVIOUS_URL = "https://github.com/bluearf/openeconometrics/releases/download/v0.3.44-windows-alpha.1/OpenEconometrics_0.3.44_x64-setup.exe"
PREVIOUS_SHA = "6a6587e4b570f57ae783ce37cfbdc383d6862eadae0b74f648b08cd77a403938"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def guarded_workspace() -> tuple[Path, Path]:
    if (platform.system() != "Windows" or os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
            or not os.environ.get("ImageOS", "").startswith("win")
            or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
            or os.environ.get("GITHUB_WORKFLOW") != "OpenEconometrics Windows UI diagnostic replay"
            or os.environ.get("GITHUB_EVENT_NAME") not in {"workflow_dispatch", "push"}
            or os.environ.get("GITHUB_REF") != "refs/heads/codex/windows-hosted-acceptance"):
        raise RuntimeError("Use only the separate fixed manual GitHub-hosted Windows diagnostic workflow")
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
    temporary = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    if Path.cwd().resolve(strict=True) != workspace:
        raise RuntimeError("Diagnostic checkout identity changed")
    return workspace, temporary


def github_request(path: str) -> Request:
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("A scoped Actions-read token is required")
    return Request("https://api.github.com/repos/" + REPOSITORY + path,
                   headers={"Authorization": "Bearer " + token, "User-Agent": "OpenEcon-fixed-Windows-diagnostic",
                            "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})


def api_json(path: str) -> dict:
    with build_opener(NoRedirect).open(github_request(path), timeout=60) as response:
        data = response.read(1024 * 1024 + 1)
        if response.status != 200 or len(data) > 1024 * 1024:
            raise RuntimeError("GitHub diagnostic metadata is unavailable or oversized")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise RuntimeError("GitHub diagnostic metadata is not an object")
    return value


def download(url: str, target: Path, expected_sha: str, maximum: int, expected_bytes: int | None = None) -> int:
    # The API request with a token is never automatically redirected. Artifact
    # storage receives a separate anonymous request without that header.
    if not url.startswith("https://"):
        raise RuntimeError("The diagnostic input download must use HTTPS")
    checksum, size = hashlib.sha256(), 0
    try:
        with urlopen(Request(url, headers={"User-Agent": "OpenEcon-fixed-Windows-diagnostic"}), timeout=120) as response, target.open("xb") as output:
            if response.status != 200 or not response.geturl().startswith("https://"):
                raise RuntimeError("Diagnostic download failed its HTTPS identity")
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > maximum:
                    raise RuntimeError("Diagnostic download exceeds its bound")
                checksum.update(chunk)
                output.write(chunk)
        if checksum.hexdigest() != expected_sha or (expected_bytes is not None and size != expected_bytes):
            raise RuntimeError("Diagnostic download differs from its immutable SHA-256/size")
        return size
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def download_artifact(artifact: tuple, directory: Path) -> Path:
    artifact_id, name, size, digest = artifact
    metadata = api_json(f"/actions/artifacts/{artifact_id}")
    run = metadata.get("workflow_run", {})
    if (metadata.get("id") != artifact_id or metadata.get("name") != name
            or metadata.get("size_in_bytes") != size or metadata.get("digest") != "sha256:" + digest
            or metadata.get("expired") is not False or run.get("id") != RUN_ID
            or run.get("head_sha") != SOURCE_SHA or run.get("repository_id") != 1398131385
            or run.get("head_repository_id") != 1398131385):
        raise RuntimeError("The fixed diagnostic artifact identity changed")
    try:
        build_opener(NoRedirect).open(github_request(f"/actions/artifacts/{artifact_id}/zip"), timeout=60)
    except HTTPError as error:
        if error.code != 302 or not error.headers.get("Location", "").startswith("https://"):
            raise RuntimeError("The fixed artifact download did not return its HTTPS storage redirect") from None
        location = error.headers["Location"]
        error.close()
    else:
        raise RuntimeError("The fixed artifact download unexpectedly bypassed its storage redirect")
    archive = directory / f"artifact-{artifact_id}.zip"
    download(location, archive, digest, size + 1024 * 1024, size)
    return archive


def read_exact(archive: zipfile.ZipFile, name: str, size: int, digest: str) -> bytes:
    entries = archive.infolist()
    if len(entries) > 128 or len({entry.filename for entry in entries}) != len(entries):
        raise RuntimeError("Diagnostic archive has duplicate or excessive members")
    matches = [entry for entry in entries if entry.filename == name]
    if len(matches) != 1 or matches[0].is_dir() or matches[0].file_size != size:
        raise RuntimeError("An exact diagnostic member is absent or differs in size")
    data = archive.read(matches[0])
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("An exact diagnostic member differs in SHA-256")
    return data


def cleanup(temporary: Path) -> None:
    raw = os.environ.get("WINDOWS_DIAGNOSTIC_INPUT_ROOT")
    if not raw:
        return
    directory = Path(raw)
    if (directory.parent.resolve(strict=True) != temporary or directory.is_symlink() or directory.is_junction()
            or not directory.name.startswith("windows-producer-replay-")
            or len(directory.name.removeprefix("windows-producer-replay-")) != 32):
        raise RuntimeError("Diagnostic input cleanup ownership is invalid")
    uuid.UUID(hex=directory.name.removeprefix("windows-producer-replay-"))
    allowed = {"OpenEconometrics_0.3.45_x64-setup.exe", "OpenEconometrics_0.3.44_x64-setup.exe", "owned-diagnostic-inputs.json"}
    if not (directory / "owned-diagnostic-inputs.json").is_file() or {entry.name for entry in directory.iterdir()} - allowed:
        raise RuntimeError("Diagnostic input cleanup has foreign contents or no owned marker")
    for entry in directory.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise RuntimeError("Diagnostic input cleanup refuses links or non-files")
    shutil.rmtree(directory)


def main() -> None:
    workspace, temporary = guarded_workspace()
    if sys.argv[1:] == ["--cleanup"]:
        cleanup(temporary)
        return
    if sys.argv[1:]:
        raise RuntimeError("No diagnostic input overrides are accepted")
    run = api_json(f"/actions/runs/{RUN_ID}")
    if (run.get("head_sha") != SOURCE_SHA or run.get("path") != ".github/workflows/openecon-desktop.yml"
            or run.get("event") != "workflow_dispatch" or run.get("run_attempt") != 1
            or run.get("status") != "completed" or run.get("conclusion") != "failure"
            or run.get("repository", {}).get("full_name") != REPOSITORY):
        raise RuntimeError("The fixed failed Windows producer run changed identity")
    directory = temporary / ("windows-producer-replay-" + uuid.uuid4().hex)
    directory.mkdir()
    output = workspace / "artifacts/windows"
    output.mkdir(parents=True, exist_ok=True)
    try:
        archives = [download_artifact(artifact, directory) for artifact in ARTIFACTS]
        with zipfile.ZipFile(archives[0]) as archive:
            entries = archive.infolist()
            if len(entries) != 1 or entries[0].filename != "OpenEconometrics_0.3.45_x64-setup.exe" or not 150_000_000 < entries[0].file_size < 250_000_000:
                raise RuntimeError("The unverified producer installer archive is not its exact single member")
            installer_bytes = read_exact(archive, entries[0].filename, entries[0].file_size, INSTALLER_SHA)
            installer = directory / entries[0].filename
            installer.write_bytes(installer_bytes)
        with zipfile.ZipFile(archives[1]) as archive:
            for name, (size, digest) in EVIDENCE.items():
                target_name = "diagnostic-producer-acceptance.json" if name == "windows-installed-acceptance.json" else name
                target = output / target_name
                if target.exists():
                    raise RuntimeError("Diagnostic producer evidence would replace an existing file")
                target.write_bytes(read_exact(archive, name, size, digest))
        for archive in archives:
            archive.unlink()
        previous = directory / "OpenEconometrics_0.3.44_x64-setup.exe"
        download(PREVIOUS_URL, previous, PREVIOUS_SHA, 300 * 1024 * 1024, 191756254)
        record = {"kind": "diagnostic-only replay of fixed unverified older installer", "producer_run_id": RUN_ID,
                  "producer_source_sha": SOURCE_SHA, "controller_source_sha": os.environ["GITHUB_SHA"],
                  "installer_sha256": INSTALLER_SHA, "installer_bytes": len(installer_bytes),
                  "previous_installer_sha256": PREVIOUS_SHA,
                  "artifact_sha256": {str(item[0]): item[3] for item in ARTIFACTS},
                  "producer_evidence_sha256": {name: spec[1] for name, spec in EVIDENCE.items()},
                  "release_acceptance": False}
        text = json.dumps(record, indent=2) + "\n"
        (output / "diagnostic-inputs.json").write_text(text, encoding="utf-8")
        (directory / "owned-diagnostic-inputs.json").write_text(text, encoding="utf-8")
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as environment:
            for key, value in {"WINDOWS_DIAGNOSTIC_INPUT_ROOT": directory, "DIAGNOSTIC_WINDOWS_INSTALLER": installer, "PREVIOUS_WINDOWS_INSTALLER": previous}.items():
                environment.write(key + "=" + str(value) + "\n")
        print(json.dumps(record))
    except BaseException:
        # Everything in this UUID directory was created by this invocation.
        shutil.rmtree(directory)
        raise RuntimeError("Pinned Windows diagnostic input acquisition failed; remote redirect/token details omitted") from None


if __name__ == "__main__":
    main()
