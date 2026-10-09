"""Verify isolated installed wheel source identity and complete eight-target state."""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import runpy
import sys
import zlib

from verify_spatial_diagnostics_eight_runtime import MARKER, METHODS, MODULES, ROOT


def verify(site: Path, wheel: Path):
    site = site.resolve(strict=True)
    sys.path.insert(0, str(site))
    identities = {}
    for name in MODULES:
        module_name = ("openecon." + name).removesuffix(".__init__")
        module = importlib.import_module(module_name)
        installed = Path(module.__file__).resolve(strict=True)
        assert installed.is_relative_to(site), installed
        source = ROOT / "src/openecon" / (name.replace(".", "/") + ".py")
        assert installed.read_bytes() == source.read_bytes(), module_name
        identities[module_name] = hashlib.sha256(source.read_bytes()).hexdigest()
    runs = []
    for part in METHODS:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            runpy.run_path(str(ROOT / f"docs/examples/spatial_diagnostics_eight_{part}.py"))
        lines = [line[len(MARKER):] for line in stdout.getvalue().splitlines() if line.startswith(MARKER)]
        assert len(lines) == 1
        proof = json.loads(zlib.decompress(base64.b64decode(lines[0])))
        assert proof["frozen"] is False and proof["all_complete_state_restorations_equal"]
        assert proof["complete_source_model_retained"] and proof["restored_ols_used_without_original_refit"]
        assert not proof["third_party_estimation_imports"]
        runs.append(proof)
    return dict(status="passed", wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                source_files_equal_installed_wheel=identities, runs=runs,
                complete_output_tables=sum(len(proof["output_contracts"]) for proof in runs),
                source_path_injected=False, primary_environment_changed=False,
                native_window_verified=False, public_release_delivered=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.site, args.wheel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "complete_output_tables")}))
