"""Install the exact CRAN survival reference into a fresh CI-only R library."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request

URL = "https://cran.r-project.org/src/contrib/Archive/survival/survival_3.8-6.tar.gz"
SHA256 = "9a4c0a35d86c64ec5a6e6cf179ab22dcffb5ce1d313d2880a09f1e53422699d7"
SIZE = 9549370


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    library = directory / "library"
    library.mkdir()
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript and the system Matrix package must be provisioned first")
    with urllib.request.urlopen(URL, timeout=30) as response:
        raw = response.read(SIZE + 1)
    if len(raw) != SIZE or hashlib.sha256(raw).hexdigest() != SHA256:
        raise RuntimeError("The complete pinned CRAN survival source differs")
    archive = directory / "survival_3.8-6.tar.gz"
    archive.write_bytes(raw)
    environment = dict(os.environ)
    environment.pop("R_LIBS", None)
    environment["R_LIBS_USER"] = str(library)
    program = (
        "stopifnot(requireNamespace('Matrix', quietly=TRUE)); "
        f"install.packages({json.dumps(str(archive))}, lib={json.dumps(str(library))}, "
        "repos=NULL, type='source', "
        "INSTALL_opts=c('--no-docs','--no-html','--no-help','--no-demo')); "
        f"stopifnot(normalizePath(find.package('survival')) == normalizePath({json.dumps(str(library / 'survival'))})); "
        "stopifnot(as.character(packageVersion('survival')) == '3.8.6'); "
        "cat('R_VERSION=', R.version.string, '\n', sep=''); "
        "cat('MATRIX_VERSION=', as.character(packageVersion('Matrix')), '\n', sep=''); "
        "cat('SURVIVAL_VERSION=', as.character(packageVersion('survival')), '\n', sep=''); "
        "cat('SURVIVAL_PATH=', find.package('survival'), '\n', sep='')"
    )
    command = [rscript, "--vanilla", "-e", program]
    remaining = 300 - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("Reference preparation exceeded its 300-second limit")
    with (directory / "complete-install.log").open("wb") as log:
        process = subprocess.Popen(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=remaining)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            raise
    elapsed = time.monotonic() - started
    receipt = {
        "status": "INSTALLED_REFERENCE_PASS" if code == 0 and elapsed < 300 else "FAILED",
        "url": URL, "source_bytes": len(raw), "source_sha256": SHA256,
        "Rscript": rscript, "library": str(library), "command": command,
        "OS_exit": code, "whole_seconds": elapsed,
        "scope": "CI original-author reference only; no production R dependency or model acceptance",
    }
    (directory / "install-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if code != 0:
        raise RuntimeError("Pinned survival installation failed; inspect complete-install.log")
    if time.monotonic() - started >= 300:
        raise TimeoutError("Reference installation exceeded its 300-second limit")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
