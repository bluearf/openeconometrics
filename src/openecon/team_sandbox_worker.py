"""Single-use worker. This module may run only inside a platform sandbox.

Stdin contains this execution's input URL only. The trusted broker's completion
capability, cancellation capability, headers, and environment are never supplied.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from openecon.sandbox_privileges import reduce_guest_privileges


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Metadata isolation could not be established.")


def drop_privileges() -> None:
    # The managed guest has a fixed UID 0 without SETUID/GID. Remove every
    # active capability and prevent exec privilege gains before consuming input.
    reduce_guest_privileges()


def clean_environment() -> None:
    os.environ.clear()
    os.environ.update({"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp",
                       "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "OPENECON_WORKSPACE": "/tmp/workspace"})


def verify_metadata_blocked() -> None:
    """Defense in depth, not a substitute for the live platform isolation test.

    Read only a non-secret instance identifier. Never request an access token.
    A reachable endpoint or an unexpected protocol response fails closed.
    """
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    for host in ("169.254.169.254", "metadata.google.internal"):
        request = Request(f"http://{host}/computeMetadata/v1/instance/id",
                          headers={"Metadata-Flavor": "Google"})
        try:
            with opener.open(request, timeout=1):
                raise RuntimeError("Metadata access is not isolated.")
        except HTTPError as exc:
            if exc.code not in (403,):
                raise RuntimeError("Metadata isolation could not be established.") from None
        except (URLError, TimeoutError):
            pass


def main() -> int:
    try:
        drop_privileges()
        clean_environment()
        # The marker is an installation guard. Managed process/filesystem
        # isolation and the broker's caller authentication provide the boundary.
        if not Path("/.openecon-clean-rootfs").is_file():
            return 1
        raw = sys.stdin.buffer.read(16386)
        if len(raw) > 16385 or not raw.endswith(b"\n") or raw.count(b"\n") != 1:
            return 1
        input_url = raw[:-1].decode("ascii")
        from openecon.team_job import (MAX_MANIFEST_BYTES, download, run_manifest,
                                      storage_url, upload_result, validate_manifest)
        storage_url(input_url, generation=True)
        match = re.fullmatch(r"/[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]/staging/([0-9a-f]{32})/([0-9a-f]{32})/input\.json",
                             urlsplit(input_url).path)
        if match is None:
            return 1
        verify_metadata_blocked()
        manifest = validate_manifest(json.loads(download(input_url, MAX_MANIFEST_BYTES)))
        if manifest["execution_id"] != match.group(2):
            return 1
        with tempfile.TemporaryDirectory(prefix="openecon-run-") as temporary:
            result = run_manifest(manifest, Path(temporary) / "workspace")
            upload_result(manifest["output_upload"], result)
        return 0
    except Exception:
        # No submitted Python, signed URLs, data, tracebacks or credentials enter
        # platform logs. The parent will report a generic failed completion.
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
