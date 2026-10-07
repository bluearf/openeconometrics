"""Export exact runtime dependency versions without changing the project lock."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib

DESKTOP = Path(__file__).resolve().parents[1]
ROOT = DESKTOP.parent


def main() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    torch = [item["version"] for item in lock["package"] if item["name"] == "torch"]
    if len(torch) != 1:
        raise SystemExit("The runtime lock must contain one unambiguous PyTorch version.")
    output = DESKTOP / "build" / "locked-runtime-dependencies.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Torch is installed separately from its official CPU index. All of its
    # dependencies still come from the lock, with Windows markers preserved.
    subprocess.run(
        [
            sys.executable, "-m", "uv", "export", "--locked", "--no-dev", "--extra", "desktop",
            "--no-hashes", "--no-emit-workspace", "--no-emit-package", "torch",
            "--output-file", str(output), "--quiet",
        ],
        cwd=ROOT,
        check=True,
    )
    print(json.dumps({"torch_version": torch[0], "requirements": str(output)}))


if __name__ == "__main__":
    main()
