"""Run the installed package's exact public Python examples in isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

PROGRAM = r'''
import importlib.metadata as metadata
import json
import socket
import sys

request = json.loads(sys.argv[1])
assert metadata.metadata(request['name']).get_payload() == request['description']
def denied(*args, **kwargs):
    raise AssertionError('Package README example attempted network access')
class OfflineSocket(socket.socket):
    connect = denied
    connect_ex = denied
socket.socket = OfflineSocket
socket.create_connection = denied
exec(compile(request['code'], '<public package README>', 'exec'), {'__name__': '__main__'})
'''


def verify(python: Path, name: str, readme: Path) -> dict:
    raw = readme.read_bytes()
    description = raw.decode("utf-8")
    blocks = re.findall(r"```python\r?\n(.*?)\r?\n```", description, re.DOTALL)
    if not blocks:
        raise ValueError("Package README must contain a real Python example")
    for code in blocks:
        with tempfile.TemporaryDirectory(prefix="openecon-readme-") as directory:
            result = subprocess.run(
                [str(python.absolute()), "-I", "-c", PROGRAM,
                 json.dumps({"name": name, "description": description, "code": code + "\n"})],
                cwd=directory, capture_output=True, text=True, timeout=90,
            )
        if result.returncode:
            raise ValueError(f"Installed README example failed: {result.stderr[-8000:]}")
    return {"status": "passed", "package": name, "python_blocks": len(blocks),
            "description_sha256": hashlib.sha256(raw).hexdigest(),
            "code_sha256": [hashlib.sha256(code.encode()).hexdigest() for code in blocks],
            "isolated_installed_examples": True, "offline": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = verify(args.python, args.name, args.readme)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
