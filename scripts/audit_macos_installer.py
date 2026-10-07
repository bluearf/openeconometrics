"""Read-only installed-bundle identity, payload and minimum-OS audit.

Compare this result for the mounted installer and installed copy. No signing,
installation, login, network requests or user-profile writes are performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "desktop/scripts"))
from macos_contract import audit  # noqa: E402

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.app)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "binaries"}))
    if result["minimum_os_violations"]:
        raise SystemExit("The application minimum OS is below a bundled binary's requirement.")


if __name__ == "__main__":
    main()
